import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from diw.api import create_app
from diw.auth import AuthenticatedPrincipal
from diw.core.embeddings import LocalHashingEmbeddingProvider
from diw.core.ingestion import ingest_file
from diw.db.models import Base
from diw.db.repository import (
    create_tenant,
    create_workspace_user,
    embed_missing_chunks,
    grant_tenant_document_access,
    save_ingested_document,
)
from diw.db.session import build_engine


class AgentApiTests(unittest.TestCase):
    def test_agent_run_endpoint_returns_answer_and_trace(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            database_url = f"sqlite+pysqlite:///{root / 'agent.db'}"
            engine = build_engine(database_url)
            Base.metadata.create_all(engine)
            source = root / "paper.md"
            source.write_text(
                "# Paper\n\n## Method\n\nThe method uses grounded retrieval.\n",
                encoding="utf-8",
            )
            provider = LocalHashingEmbeddingProvider(dimensions=32)
            with Session(engine) as session:
                document = ingest_file(source, target_chars=220, overlap_chars=0)
                save_ingested_document(session, document)
                embed_missing_chunks(session, provider)
                tenant = create_tenant(session, slug="api-agent", name="API Agent")
                user = create_workspace_user(
                    session,
                    tenant_id=tenant.id,
                    subject="api-researcher",
                    email="researcher@example.com",
                    display_name="Researcher",
                    role="researcher",
                )
                grant_tenant_document_access(
                    session, tenant_id=tenant.id, document_id=document.document_id
                )
                session.commit()
                tenant_id = tenant.id
                user_id = user.id
            engine.dispose()

            with TestClient(create_app(database_url)) as client:
                response = client.post(
                    "/agent-runs",
                    json={
                        "tenant_id": tenant_id,
                        "actor_user_id": user_id,
                        "query": "What method does the paper use?",
                        "dimensions": 32,
                    },
                )
                self.assertEqual(response.status_code, 200)
                payload = response.json()
                self.assertEqual(payload["status"], "completed")
                self.assertTrue(payload["citation_validation"]["valid"])
                self.assertIn("agent_run_id", payload)

                details = client.get(
                    f"/agent-runs/{payload['agent_run_id']}",
                    params={"tenant_id": tenant_id},
                )
                self.assertEqual(details.status_code, 200)
                details_payload = details.json()
                self.assertEqual(details_payload["trace_id"], payload["trace_id"])
                self.assertEqual(details_payload["steps"][0]["tool_name"], "search_documents")

                cross_tenant = client.get(
                    f"/agent-runs/{payload['agent_run_id']}",
                    params={"tenant_id": "different-tenant"},
                )
                self.assertEqual(cross_tenant.status_code, 404)

            class FakeAuthenticator:
                def authenticate(self, token: str) -> AuthenticatedPrincipal:
                    if token != "valid-token":
                        raise ValueError("invalid")
                    return AuthenticatedPrincipal(
                        subject="api-researcher",
                        tenant_id=None,
                        claims={
                            "email": "researcher@example.com",
                            "email_verified": True,
                        },
                    )

            with patch.dict(
                "os.environ",
                {"GOOGLE_OAUTH_CLIENT_ID": "client.apps.googleusercontent.com"},
            ), TestClient(
                create_app(database_url, authenticator=FakeAuthenticator())
            ) as authenticated_client:
                sign_in = authenticated_client.get("/signin")
                self.assertEqual(sign_in.status_code, 200)
                self.assertIn(
                    "client.apps.googleusercontent.com",
                    sign_in.text,
                )

                whoami = authenticated_client.get(
                    "/auth/whoami",
                    headers={"Authorization": "Bearer valid-token"},
                )
                self.assertEqual(whoami.status_code, 200)
                self.assertEqual(whoami.json()["email"], "researcher@example.com")

                missing_token = authenticated_client.post(
                    "/agent-runs",
                    json={
                        "tenant_id": tenant_id,
                        "actor_user_id": user_id,
                        "query": "What method does the paper use?",
                        "dimensions": 32,
                    },
                )
                self.assertEqual(missing_token.status_code, 401)

                authenticated = authenticated_client.post(
                    "/agent-runs",
                    headers={"Authorization": "Bearer valid-token"},
                    json={
                        "tenant_id": tenant_id,
                        "actor_user_id": user_id,
                        "query": "What method does the paper use?",
                        "dimensions": 32,
                    },
                )
                self.assertEqual(authenticated.status_code, 200)

                unscoped_route = authenticated_client.get(
                    "/documents",
                    headers={"Authorization": "Bearer valid-token"},
                )
                self.assertEqual(unscoped_route.status_code, 403)


if __name__ == "__main__":
    unittest.main()
