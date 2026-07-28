import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi.testclient import TestClient

from diw.api import create_app
from diw.auth import AuthenticatedPrincipal
from diw.web_views import _workspace_html


class FakeAuthenticator:
    def authenticate(self, token: str) -> AuthenticatedPrincipal:
        if token != "valid-token":
            raise ValueError("invalid token")
        return AuthenticatedPrincipal(
            subject="reviewer",
            tenant_id=None,
            claims={"email": "reviewer@example.com", "email_verified": True},
        )


class PublicDemoTests(unittest.TestCase):
    def test_workspace_template_is_packaged(self):
        workspace = _workspace_html()

        self.assertIn("<title>Document Intelligence Workspace</title>", workspace)
        self.assertIn('id="paperWorkspace"', workspace)

    def _build_corpus(self, root: Path) -> Path:
        corpus = root / "corpus"
        corpus.mkdir()
        (corpus / "retrieval-paper.md").write_text(
            "# Retrieval Paper\n\n"
            "## Method\n\n"
            "Hybrid retrieval combines lexical filtering with vector search.\n\n"
            "## Policy\n\n"
            "The assistant refuses a question when retrieved evidence does not support it.\n",
            encoding="utf-8",
        )
        return corpus

    def test_public_routes_need_no_google_login_while_data_routes_fail_closed(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            app = create_app(
                f"sqlite+pysqlite:///{root / 'demo.db'}",
                public_demo_corpus_dir=self._build_corpus(root),
                authenticator=FakeAuthenticator(),
            )
            with TestClient(app) as client:
                landing = client.get("/")
                self.assertEqual(landing.status_code, 200)
                self.assertIn('href="/demo"', landing.text)
                self.assertIn("Google sign-in", landing.text)

                demo = client.get("/demo")
                self.assertEqual(demo.status_code, 200)
                self.assertIn("Public and read-only", demo.text)

                evidence = client.get("/evidence")
                self.assertEqual(evidence.status_code, 200)
                self.assertIn("0.3022", evidence.text)
                self.assertIn("uncertainty intervals include zero", evidence.text)
                self.assertIn("No human-calibrated accuracy", evidence.text)

                protected = client.get(
                    "/documents",
                    headers={"Authorization": "Bearer valid-token"},
                )
                self.assertEqual(protected.status_code, 403)

                missing_agent_token = client.post(
                    "/agent-runs",
                    json={
                        "tenant_id": "tenant",
                        "actor_user_id": "reviewer",
                        "query": "What is the method?",
                    },
                )
                self.assertEqual(missing_agent_token.status_code, 401)

    def test_public_demo_returns_valid_exact_citations_and_no_writes(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            app = create_app(
                f"sqlite+pysqlite:///{root / 'demo.db'}",
                public_demo_corpus_dir=self._build_corpus(root),
                authenticator=FakeAuthenticator(),
            )
            with TestClient(app) as client:
                response = client.post(
                    "/demo/ask",
                    json={"query": "How does hybrid retrieval work?"},
                )

            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertFalse(payload["answer"]["insufficient_evidence"])
            self.assertTrue(payload["citation_validation"]["valid"])
            self.assertGreaterEqual(len(payload["answer"]["citations"]), 1)
            self.assertEqual(payload["trace"]["access"], "public_read_only")
            self.assertFalse(payload["trace"]["write_tools_available"])
            self.assertEqual(payload["trace"]["writes_performed"], 0)
            self.assertFalse(payload["trace"]["external_model_request"])

            chunks = {
                chunk["chunk_id"]: chunk["text"]
                for chunk in payload["retrieved_chunks"]
            }
            for citation in payload["answer"]["citations"]:
                self.assertIn(citation["quote"], chunks[citation["chunk_id"]])

    def test_public_demo_refuses_when_corpus_has_no_matching_evidence(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            app = create_app(
                f"sqlite+pysqlite:///{root / 'demo.db'}",
                public_demo_corpus_dir=self._build_corpus(root),
            )
            with TestClient(app) as client:
                response = client.post(
                    "/demo/ask",
                    json={"query": "What quantum entanglement result was reported?"},
                )

            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertTrue(payload["answer"]["insufficient_evidence"])
            self.assertEqual(payload["answer"]["citations"], [])
            self.assertTrue(payload["citation_validation"]["valid"])


if __name__ == "__main__":
    unittest.main()
