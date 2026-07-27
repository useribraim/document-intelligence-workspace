import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from sqlalchemy import select
from sqlalchemy.orm import Session

from diw.core.agent import AgentContext, AgentDecision, ToolCall, run_bounded_agent
from diw.core.embeddings import LocalHashingEmbeddingProvider
from diw.core.ingestion import ingest_file
from diw.db.models import AgentRunStep, ApprovalRequest, Base
from diw.db.repository import (
    create_tenant,
    create_workspace_user,
    grant_tenant_document_access,
    save_ingested_document,
)
from diw.db.session import build_engine


class RepeatingDecisionProvider:
    def decide(self, context: AgentContext) -> AgentDecision:
        return AgentDecision(
            tool_call=ToolCall(
                tool_name="search_documents",
                arguments={"query": context.query, "top_k": 1, "mode": "hybrid"},
            )
        )


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.engine = build_engine("sqlite+pysqlite:///:memory:")
        Base.metadata.create_all(self.engine)
        self.provider = LocalHashingEmbeddingProvider(dimensions=32)

    def tearDown(self):
        Base.metadata.drop_all(self.engine)
        self.engine.dispose()

    def _seed_tenant_documents(self, session: Session):
        with TemporaryDirectory() as tmp:
            alpha_path = Path(tmp) / "alpha-paper.md"
            beta_path = Path(tmp) / "beta-paper.md"
            alpha_path.write_text(
                "# Alpha Paper\n\n## Method\n\nThe method uses grounded retrieval.\n",
                encoding="utf-8",
            )
            beta_path.write_text(
                "# Beta Paper\n\n## Secret\n\nThe beta-only secret is never shared.\n",
                encoding="utf-8",
            )
            alpha_document = ingest_file(alpha_path, target_chars=220, overlap_chars=0)
            beta_document = ingest_file(beta_path, target_chars=220, overlap_chars=0)
            save_ingested_document(session, alpha_document)
            save_ingested_document(session, beta_document)
            from diw.db.repository import embed_missing_chunks

            embed_missing_chunks(session, self.provider)
            alpha = create_tenant(session, slug="alpha-agent", name="Alpha Agent")
            beta = create_tenant(session, slug="beta-agent", name="Beta Agent")
            researcher = create_workspace_user(
                session,
                tenant_id=alpha.id,
                subject="researcher-1",
                email="researcher@alpha.example",
                display_name="Researcher",
                role="researcher",
            )
            grant_tenant_document_access(
                session, tenant_id=alpha.id, document_id=alpha_document.document_id
            )
            grant_tenant_document_access(
                session, tenant_id=beta.id, document_id=beta_document.document_id
            )
            session.commit()
            return alpha, researcher

    def test_agent_returns_tenant_scoped_cited_answer(self):
        with Session(self.engine) as session:
            tenant, researcher = self._seed_tenant_documents(session)

            output = run_bounded_agent(
                session,
                tenant_id=tenant.id,
                actor_user_id=researcher.id,
                query="What method does the paper use?",
                embedding_provider=self.provider,
                trace_id="agent-trace-1",
            )
            session.commit()

            self.assertEqual(output["status"], "completed")
            self.assertTrue(output["citation_validation"]["valid"])
            self.assertIn("grounded retrieval", output["answer"]["answer"].lower())
            self.assertTrue(
                all("beta" not in chunk["text"].lower() for chunk in output["retrieved_chunks"])
            )

    def test_agent_requests_approval_before_creating_a_study_task(self):
        with Session(self.engine) as session:
            tenant, researcher = self._seed_tenant_documents(session)

            output = run_bounded_agent(
                session,
                tenant_id=tenant.id,
                actor_user_id=researcher.id,
                query="Create a study task for the grounded retrieval method.",
                embedding_provider=self.provider,
                trace_id="agent-trace-2",
            )
            session.commit()

            approval = session.get(ApprovalRequest, output["approval_request_id"])
            self.assertEqual(output["status"], "approval_pending")
            self.assertEqual(approval.status, "pending")
            self.assertEqual(approval.action_type, "create_study_task")

    def test_agent_stops_repeated_tool_calls(self):
        with Session(self.engine) as session:
            tenant, researcher = self._seed_tenant_documents(session)

            output = run_bounded_agent(
                session,
                tenant_id=tenant.id,
                actor_user_id=researcher.id,
                query="What method does the paper use?",
                embedding_provider=self.provider,
                decision_provider=RepeatingDecisionProvider(),
                trace_id="agent-trace-3",
            )
            session.commit()

            steps = session.scalars(
                select(AgentRunStep).where(AgentRunStep.agent_run_id == output["agent_run_id"])
            ).all()
            self.assertEqual(output["status"], "refused")
            self.assertEqual(len(steps), 2)
            self.assertEqual(steps[-1].error_code, "duplicate_tool_call")


if __name__ == "__main__":
    unittest.main()
