from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from fastapi.testclient import TestClient

from diw.api import create_app
from diw.db.models import Base
from diw.db.session import build_engine


class ApiTests(unittest.TestCase):
    def test_api_supports_ingest_ask_review_loop(self):
        with TemporaryDirectory() as tmp:
            db = Path(tmp) / "diw.db"
            source = Path(tmp) / "paper.md"
            database_url = f"sqlite+pysqlite:///{db}"
            source.write_text(
                "# Paper\n\n"
                "## Cited Section\n\n"
                "Method: hybrid retrieval.\n\n"
                "Dataset: local paper excerpts.\n\n"
                "Metric: recall@5.\n\n"
                "Limitation: small test set.\n",
                encoding="utf-8",
            )

            with TestClient(create_app(database_url)) as client:
                health = client.get("/health")
                self.assertEqual(health.status_code, 200)
                self.assertEqual(health.json()["documents"], 0)

                ingest = client.post(
                    "/documents/ingest",
                    json={"path": str(source), "target_chars": 500, "overlap_chars": 0},
                )
                self.assertEqual(ingest.status_code, 200)
                chunk_count = ingest.json()["chunk_count"]
                self.assertGreaterEqual(chunk_count, 1)

                documents = client.get("/documents").json()["documents"]
                self.assertEqual(len(documents), 1)

                embeddings = client.post("/embeddings")
                self.assertEqual(embeddings.status_code, 200)
                self.assertEqual(embeddings.json()["embeddings_total"], chunk_count)

                answer = client.post(
                    "/ask",
                    json={
                        "query": "Extract the method, dataset, metric, and limitation.",
                        "top_k": 1,
                    },
                )
                self.assertEqual(answer.status_code, 200)
                answer_payload = answer.json()
                self.assertTrue(answer_payload["citation_validation"]["valid"])
                self.assertIn("ai_run_id", answer_payload)
                self.assertIn("suggestion_id", answer_payload)

                suggestions = client.get("/review/suggestions", params={"status": "pending"})
                self.assertEqual(suggestions.status_code, 200)
                suggestion_id = suggestions.json()["suggestions"][0]["id"]
                self.assertEqual(suggestion_id, answer_payload["suggestion_id"])

                review = client.post(
                    f"/review/suggestions/{suggestion_id}/decision",
                    json={
                        "decision": "accept",
                        "reviewer": "ivan",
                        "note": "Citations checked.",
                    },
                )
                self.assertEqual(review.status_code, 200)
                self.assertEqual(review.json()["decision"], "accept")

                accepted = client.get("/review/suggestions", params={"status": "accepted"})
                self.assertEqual(accepted.json()["suggestions"][0]["id"], suggestion_id)

                runs = client.get("/ai-runs")
                self.assertEqual(runs.status_code, 200)
                self.assertEqual(runs.json()["ai_runs"][0]["id"], answer_payload["ai_run_id"])

                dashboard = client.get("/")
                self.assertEqual(dashboard.status_code, 200)
                self.assertIn("Document Intelligence Workspace", dashboard.text)

                workspace = client.get("/workspace")
                self.assertEqual(workspace.status_code, 200)
                self.assertIn("Evidence and review inspector", workspace.text)
                self.assertIn("review/suggestions", workspace.text)

            engine = build_engine(database_url)
            try:
                Base.metadata.drop_all(engine)
            finally:
                engine.dispose()


if __name__ == "__main__":
    unittest.main()
