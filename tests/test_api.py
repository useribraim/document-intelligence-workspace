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
            eval_cases = Path(tmp) / "review_cases.jsonl"
            paper_cards_dir = Path(tmp) / "paper_cards"
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

            with TestClient(
                create_app(
                    database_url,
                    evaluation_cases_path=eval_cases,
                    paper_cards_dir=paper_cards_dir,
                )
            ) as client:
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
                versions = client.get(f"/documents/{documents[0]['id']}/versions").json()["versions"]
                self.assertEqual(len(versions), 1)

                embeddings = client.post("/embeddings")
                self.assertEqual(embeddings.status_code, 200)
                self.assertEqual(embeddings.json()["embeddings_total"], chunk_count)

                preview = client.post(
                    "/retrieval-preview",
                    json={
                        "query": "Extract the method, dataset, metric, and limitation.",
                        "top_k": 1,
                    },
                )
                self.assertEqual(preview.status_code, 200)
                self.assertEqual(len(preview.json()["retrieved_chunks"]), 1)

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

                eval_case = client.post(
                    "/evaluation-cases",
                    json={
                        "source": "review",
                        "query": answer_payload["query"],
                        "task": "structured_extraction",
                        "expected_behavior": "corrected_output",
                        "expected_fields": answer_payload["answer"]["extracted_fields"],
                        "review_note": "Regression case from review.",
                        "retrieved_chunk_ids": [
                            chunk["chunk_id"] for chunk in answer_payload["retrieved_chunks"]
                        ],
                        "ai_run_id": answer_payload["ai_run_id"],
                        "suggestion_id": answer_payload["suggestion_id"],
                    },
                )
                self.assertEqual(eval_case.status_code, 200)
                self.assertTrue(eval_cases.exists())

                eval_cases_payload = client.get("/evaluation-cases").json()
                self.assertEqual(eval_cases_payload["count"], 1)
                self.assertEqual(eval_cases_payload["cases"][0]["source"], "review")

                paper_card = client.post(
                    "/paper-cards/draft",
                    json={"version_id": versions[0]["id"]},
                )
                self.assertEqual(paper_card.status_code, 200)
                paper_card_payload = paper_card.json()
                self.assertIn("## Method", paper_card_payload["markdown"])
                self.assertIn("suggestion_id", paper_card_payload)

                saved_card = client.post(
                    "/paper-cards/save",
                    json={
                        "title": paper_card_payload["title"],
                        "markdown": paper_card_payload["markdown"],
                        "suggestion_id": paper_card_payload["suggestion_id"],
                    },
                )
                self.assertEqual(saved_card.status_code, 200)
                self.assertTrue(Path(saved_card.json()["path"]).exists())
                self.assertIn("saved_at", saved_card.json())

                paper_cards = client.get("/paper-cards").json()
                self.assertEqual(paper_cards["count"], 1)

                accepted = client.get("/review/suggestions", params={"status": "accepted"})
                self.assertEqual(accepted.json()["suggestions"][0]["id"], suggestion_id)

                runs = client.get("/ai-runs")
                self.assertEqual(runs.status_code, 200)
                run_ids = [run["id"] for run in runs.json()["ai_runs"]]
                self.assertIn(answer_payload["ai_run_id"], run_ids)

                dashboard = client.get("/")
                self.assertEqual(dashboard.status_code, 200)
                self.assertIn("Document Intelligence Workspace", dashboard.text)

                workspace = client.get("/workspace")
                self.assertEqual(workspace.status_code, 200)
                self.assertIn("Evidence and review inspector", workspace.text)
                self.assertIn("Paper workspace", workspace.text)
                self.assertIn("Find evidence", workspace.text)
                self.assertIn("Generate draft", workspace.text)
                self.assertIn("Build card", workspace.text)
                self.assertIn("Accept", workspace.text)
                self.assertIn("Save", workspace.text)
                self.assertIn("No study card yet", workspace.text)
                self.assertIn("Details", workspace.text)
                self.assertIn("Show source text", workspace.text)
                self.assertIn("needs attention", workspace.text)
                self.assertIn("workflowStatus", workspace.text)
                self.assertIn("review/suggestions", workspace.text)
                self.assertIn("retrieval-preview", workspace.text)
                self.assertIn("evaluation-cases", workspace.text)
                self.assertIn("Corpus browser", workspace.text)
                self.assertIn("documents/", workspace.text)
                self.assertIn("Paper card compiler", workspace.text)
                self.assertIn("paper-cards/draft", workspace.text)

            engine = build_engine(database_url)
            try:
                Base.metadata.drop_all(engine)
            finally:
                engine.dispose()


if __name__ == "__main__":
    unittest.main()
