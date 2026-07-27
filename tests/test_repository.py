import gc
import unittest
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

from sqlalchemy import select
from sqlalchemy.orm import Session

from diw.core.ingestion import ingest_file
from diw.db.models import (
    AgentRunStep,
    AIRun,
    AISuggestion,
    Base,
    Chunk,
    DocumentVersion,
    SourceDocument,
)
from diw.db.repository import (
    count_ai_runs,
    count_ai_suggestions,
    count_chunks,
    count_documents,
    count_review_decisions,
    count_versions,
    create_agent_run,
    create_approval_request,
    create_study_task_from_approval,
    create_tenant,
    create_workspace_user,
    decide_approval_request,
    get_research_record_for_tenant,
    record_review_decision,
    save_agent_run_step,
    save_ai_run,
    save_ai_suggestion,
    save_ingested_document,
    save_research_record,
)
from diw.db.session import build_engine


class RepositoryTests(unittest.TestCase):
    @contextmanager
    def _session(self):
        engine = build_engine("sqlite+pysqlite:///:memory:")
        Base.metadata.create_all(engine)
        session = Session(engine)
        try:
            yield session
        finally:
            session.close()
            Base.metadata.drop_all(engine)
            engine.dispose()
            gc.collect()

    def test_save_ingested_document_persists_document_version_and_chunks(self):
        with TemporaryDirectory() as tmp, self._session() as session:
            path = Path(tmp) / "paper.md"
            path.write_text("# Paper\n\n## Method\n\nChunk this.\n", encoding="utf-8")
            document = ingest_file(path, target_chars=220, overlap_chars=0)

            save_ingested_document(session, document)
            session.commit()

            self.assertEqual(count_documents(session), 1)
            self.assertEqual(count_versions(session), 1)
            self.assertEqual(count_chunks(session), len(document.chunks))

            saved_version = session.get(DocumentVersion, document.version_id)
            self.assertIsNotNone(saved_version)
            self.assertEqual(saved_version.content_hash, document.content_hash)
            self.assertEqual(saved_version.normalisation_report["normalised_line_count"], 5)

    def test_save_same_version_is_idempotent(self):
        with TemporaryDirectory() as tmp, self._session() as session:
            path = Path(tmp) / "paper.md"
            path.write_text("# Paper\n\nSame content.\n", encoding="utf-8")
            document = ingest_file(path, target_chars=220, overlap_chars=0)

            save_ingested_document(session, document)
            save_ingested_document(session, document)
            session.commit()

            self.assertEqual(count_documents(session), 1)
            self.assertEqual(count_versions(session), 1)
            self.assertEqual(count_chunks(session), len(document.chunks))

    def test_modified_file_adds_new_version_for_same_document(self):
        with TemporaryDirectory() as tmp, self._session() as session:
            path = Path(tmp) / "paper.md"
            path.write_text("# Paper\n\nFirst content.\n", encoding="utf-8")
            first = ingest_file(path, target_chars=220, overlap_chars=0)
            save_ingested_document(session, first)

            path.write_text("# Paper\n\nSecond content.\n", encoding="utf-8")
            second = ingest_file(path, target_chars=220, overlap_chars=0)
            save_ingested_document(session, second)
            session.commit()

            self.assertEqual(first.document_id, second.document_id)
            self.assertEqual(count_documents(session), 1)
            self.assertEqual(count_versions(session), 2)

            saved_document = session.get(SourceDocument, first.document_id)
            self.assertEqual(len(saved_document.versions), 2)

    def test_chunk_heading_path_round_trips(self):
        with TemporaryDirectory() as tmp, self._session() as session:
            path = Path(tmp) / "paper.md"
            path.write_text("# Paper\n\n## Method\n\nChunk this.\n", encoding="utf-8")
            document = ingest_file(path, target_chars=220, overlap_chars=0)
            save_ingested_document(session, document)
            session.commit()

            chunks = session.scalars(select(Chunk).order_by(Chunk.chunk_index)).all()
            self.assertTrue(any("Method" in chunk.heading_path for chunk in chunks))

    def test_save_ai_run_persists_audit_metadata(self):
        with self._session() as session:
            run = save_ai_run(
                session,
                run_type="answer_llm",
                query="What does the paper say?",
                retrieval_mode="hybrid",
                embedding_model="local-hashing-v1",
                llm_provider="local",
                llm_model="deterministic-structured-v1",
                prompt_version="source-cited-qa-v1",
                retrieved_chunk_ids=["chunk-1", "chunk-2"],
                citation_valid=True,
                insufficient_evidence=False,
                output={"answer": "Cited answer"},
                metrics={"retrieved_chunk_count": 2},
            )
            session.commit()

            saved = session.get(AIRun, run.id)
            self.assertEqual(count_ai_runs(session), 1)
            self.assertEqual(saved.query, "What does the paper say?")
            self.assertEqual(saved.retrieved_chunk_ids, ["chunk-1", "chunk-2"])
            self.assertTrue(saved.citation_valid)
            self.assertEqual(saved.metrics["retrieved_chunk_count"], 2)

    def test_ai_suggestion_review_decision_updates_status(self):
        with self._session() as session:
            run = save_ai_run(
                session,
                run_type="answer_llm",
                query="What is the method?",
                output={"answer": {"answer": "A cited answer"}},
            )
            suggestion = save_ai_suggestion(
                session,
                suggestion_type="source_cited_answer",
                title="What is the method?",
                ai_run_id=run.id,
                payload={"answer": {"answer": "A cited answer"}},
            )
            session.commit()

            review = record_review_decision(
                session,
                suggestion_id=suggestion.id,
                decision="accept",
                reviewer="reviewer@example.com",
                note="Evidence checks out.",
            )
            session.commit()

            saved = session.get(AISuggestion, suggestion.id)
            self.assertEqual(count_ai_suggestions(session), 1)
            self.assertEqual(count_review_decisions(session), 1)
            self.assertEqual(saved.status, "accepted")
            self.assertIsNotNone(saved.reviewed_at)
            self.assertEqual(review.decision, "accept")
            self.assertEqual(review.reviewer, "reviewer@example.com")

    def test_tenant_scoped_research_records_and_approved_study_tasks(self):
        with self._session() as session:
            alpha = create_tenant(session, slug="alpha-research", name="Alpha Research")
            beta = create_tenant(session, slug="beta-research", name="Beta Research")
            researcher = create_workspace_user(
                session,
                tenant_id=alpha.id,
                subject="researcher-1",
                email="researcher@alpha.example",
                display_name="Researcher",
                role="researcher",
            )
            manager = create_workspace_user(
                session,
                tenant_id=alpha.id,
                subject="manager-1",
                email="manager@alpha.example",
                display_name="Manager",
                role="manager",
            )
            record = save_research_record(
                session,
                tenant_id=alpha.id,
                record_type="paper",
                title="Grounded Agent Evaluation",
                payload={"reading_status": "in_progress"},
                created_by_user_id=researcher.id,
            )
            run = create_agent_run(
                session,
                tenant_id=alpha.id,
                actor_user_id=researcher.id,
                query="Create a follow-up task.",
                tool_policy_version="agent-policy-v1",
                max_steps=5,
                trace_id="trace-alpha-1",
            )
            step = save_agent_run_step(
                session,
                agent_run_id=run.id,
                sequence=1,
                tool_name="get_research_record",
                tool_args={"record_id": record.id},
                observation={"record_found": True},
                status="completed",
                latency_ms=12,
            )
            request = create_approval_request(
                session,
                tenant_id=alpha.id,
                requested_by_user_id=researcher.id,
                agent_run_id=run.id,
                action_type="create_study_task",
                action_payload={"title": "Review evaluation section", "details": "Read section 4."},
                idempotency_key="create-task-alpha-1",
            )
            duplicate = create_approval_request(
                session,
                tenant_id=alpha.id,
                requested_by_user_id=researcher.id,
                action_type="create_study_task",
                action_payload={"title": "Ignored duplicate", "details": ""},
                idempotency_key="create-task-alpha-1",
            )
            session.commit()

            self.assertIsNotNone(session.get(AgentRunStep, step.id))
            self.assertEqual(request.id, duplicate.id)
            self.assertEqual(
                get_research_record_for_tenant(session, tenant_id=alpha.id, record_id=record.id).id,
                record.id,
            )
            self.assertIsNone(
                get_research_record_for_tenant(session, tenant_id=beta.id, record_id=record.id)
            )

            decide_approval_request(
                session,
                tenant_id=alpha.id,
                approval_request_id=request.id,
                approver_user_id=manager.id,
                decision="approve",
                note="Useful follow-up.",
            )
            task = create_study_task_from_approval(
                session,
                tenant_id=alpha.id,
                approval_request_id=request.id,
            )
            duplicate_task = create_study_task_from_approval(
                session,
                tenant_id=alpha.id,
                approval_request_id=request.id,
            )
            session.commit()

            self.assertEqual(task.id, duplicate_task.id)
            self.assertEqual(task.title, "Review evaluation section")
            with self.assertRaisesRegex(ValueError, "unknown approval request for tenant"):
                create_study_task_from_approval(
                    session,
                    tenant_id=beta.id,
                    approval_request_id=request.id,
                )


if __name__ == "__main__":
    unittest.main()
