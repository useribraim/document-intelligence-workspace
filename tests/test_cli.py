from pathlib import Path
import json
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from diw.cli import main
from diw.db.models import AIRun, AISuggestion, Base, ReviewDecision
from diw.db.session import build_engine
from sqlalchemy import select
from sqlalchemy.orm import Session


class CliTests(unittest.TestCase):
    def test_normalise_command_writes_output(self):
        with TemporaryDirectory() as tmp:
            source = Path(tmp) / "paper.md"
            out = Path(tmp) / "paper.normalised.md"
            source.write_text("# Paper   \n\n\nBody   \n", encoding="utf-8")

            with patch("sys.stdout"):
                exit_code = main(["normalise", str(source), "--out", str(out), "--report"])

            self.assertEqual(exit_code, 0)
            self.assertEqual(out.read_text(encoding="utf-8"), "# Paper\n\nBody")

    def test_chunk_command_writes_json(self):
        with TemporaryDirectory() as tmp:
            source = Path(tmp) / "paper.md"
            out = Path(tmp) / "paper.chunks.json"
            source.write_text("# Paper\n\n## Method\n\nChunk this document.\n", encoding="utf-8")

            with patch("sys.stdout"):
                exit_code = main(
                    [
                        "chunk",
                        str(source),
                        "--out",
                        str(out),
                        "--target-chars",
                        "220",
                        "--overlap-chars",
                        "0",
                    ]
                )

            self.assertEqual(exit_code, 0)
            output = out.read_text(encoding="utf-8")
            self.assertIn('"document_id"', output)
            self.assertIn('"chunks"', output)

    def test_load_command_creates_database_and_persists_chunks(self):
        with TemporaryDirectory() as tmp:
            source = Path(tmp) / "paper.md"
            db = Path(tmp) / "diw.db"
            source.write_text("# Paper\n\n## Method\n\nLoad this document.\n", encoding="utf-8")

            with patch("sys.stdout"):
                exit_code = main(
                    [
                        "load",
                        str(source),
                        "--database-url",
                        f"sqlite+pysqlite:///{db}",
                        "--target-chars",
                        "220",
                        "--overlap-chars",
                        "0",
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertTrue(db.exists())

    def test_embed_and_retrieve_commands_use_database(self):
        with TemporaryDirectory() as tmp:
            source = Path(tmp) / "paper.md"
            db = Path(tmp) / "diw.db"
            source.write_text(
                "# Paper\n\n## Method\n\nHybrid retrieval stores chunk embeddings.\n",
                encoding="utf-8",
            )
            database_url = f"sqlite+pysqlite:///{db}"

            with patch("sys.stdout"):
                self.assertEqual(
                    main(
                        [
                            "load",
                            str(source),
                            "--database-url",
                            database_url,
                            "--target-chars",
                            "220",
                            "--overlap-chars",
                            "0",
                        ]
                    ),
                    0,
                )
                self.assertEqual(main(["embed", "--database-url", database_url]), 0)
                self.assertEqual(
                    main(
                        [
                            "retrieve",
                            "hybrid retrieval",
                            "--database-url",
                            database_url,
                            "--top-k",
                            "1",
                        ]
                    ),
                    0,
                )


    def test_answer_command_returns_validated_source_cited_answer(self):
        with TemporaryDirectory() as tmp:
            source = Path(tmp) / "paper.md"
            db = Path(tmp) / "diw.db"
            source.write_text(
                "# Paper\n\n"
                "## Method\n\n"
                "Hybrid retrieval combines lexical matching and vector similarity over chunks.\n",
                encoding="utf-8",
            )
            database_url = f"sqlite+pysqlite:///{db}"

            with patch("sys.stdout"):
                self.assertEqual(
                    main(
                        [
                            "load",
                            str(source),
                            "--database-url",
                            database_url,
                            "--target-chars",
                            "220",
                            "--overlap-chars",
                            "0",
                        ]
                    ),
                    0,
                )

            with patch("sys.stdout") as stdout:
                exit_code = main(
                    [
                        "answer",
                        "how does hybrid retrieval work",
                        "--database-url",
                        database_url,
                        "--top-k",
                        "1",
                    ]
                )

            self.assertEqual(exit_code, 0)
            output = "".join(call.args[0] for call in stdout.write.call_args_list)
            self.assertIn('"insufficient_evidence": false', output)
            self.assertIn('"valid": true', output)
            self.assertIn('"retrieved_chunks"', output)

    def test_answer_llm_command_returns_structured_fields(self):
        with TemporaryDirectory() as tmp:
            source = Path(tmp) / "paper.md"
            db = Path(tmp) / "diw.db"
            source.write_text(
                "# Paper\n\n"
                "## Cited Section\n\n"
                "Method: contrastive retrieval.\n\n"
                "Dataset: synthetic paper excerpts.\n\n"
                "Metric: recall@5.\n\n"
                "Limitation: small corpus.\n",
                encoding="utf-8",
            )
            database_url = f"sqlite+pysqlite:///{db}"

            with patch("sys.stdout"):
                self.assertEqual(
                    main(
                        [
                            "load",
                            str(source),
                            "--database-url",
                            database_url,
                            "--target-chars",
                            "500",
                            "--overlap-chars",
                            "0",
                        ]
                    ),
                    0,
                )

            with patch("sys.stdout") as stdout:
                exit_code = main(
                    [
                        "answer-llm",
                        "Extract the method, dataset, metric, and limitation.",
                        "--database-url",
                        database_url,
                        "--top-k",
                        "1",
                    ]
                )

            self.assertEqual(exit_code, 0)
            payload = json.loads("".join(call.args[0] for call in stdout.write.call_args_list))
            self.assertTrue(payload["citation_validation"]["valid"])
            self.assertIn("ai_run_id", payload)
            self.assertIn("suggestion_id", payload)
            self.assertEqual(
                payload["answer"]["extracted_fields"]["method"],
                "contrastive retrieval.",
            )

            engine = build_engine(database_url)
            try:
                with Session(engine) as session:
                    runs = session.scalars(select(AIRun)).all()
                    suggestions = session.scalars(select(AISuggestion)).all()
                    self.assertEqual(len(runs), 1)
                    self.assertEqual(len(suggestions), 1)
                    self.assertEqual(runs[0].run_type, "answer_llm")
                    self.assertEqual(runs[0].query, "Extract the method, dataset, metric, and limitation.")
                    self.assertTrue(runs[0].citation_valid)
                    self.assertEqual(suggestions[0].status, "pending")
                    self.assertEqual(suggestions[0].ai_run_id, runs[0].id)
            finally:
                Base.metadata.drop_all(engine)
                engine.dispose()

    def test_review_list_and_decide_commands_update_suggestion(self):
        with TemporaryDirectory() as tmp:
            source = Path(tmp) / "paper.md"
            db = Path(tmp) / "diw.db"
            source.write_text(
                "# Paper\n\n"
                "## Cited Section\n\n"
                "Method: hybrid retrieval.\n\n"
                "Dataset: local paper excerpts.\n\n"
                "Metric: recall@5.\n\n"
                "Limitation: small test set.\n",
                encoding="utf-8",
            )
            database_url = f"sqlite+pysqlite:///{db}"

            with patch("sys.stdout"):
                self.assertEqual(
                    main(
                        [
                            "load",
                            str(source),
                            "--database-url",
                            database_url,
                            "--target-chars",
                            "500",
                            "--overlap-chars",
                            "0",
                        ]
                    ),
                    0,
                )

            with patch("sys.stdout") as stdout:
                self.assertEqual(
                    main(
                        [
                            "answer-llm",
                            "Extract the method, dataset, metric, and limitation.",
                            "--database-url",
                            database_url,
                            "--top-k",
                            "1",
                        ]
                    ),
                    0,
                )
            answer_payload = json.loads("".join(call.args[0] for call in stdout.write.call_args_list))
            suggestion_id = answer_payload["suggestion_id"]

            with patch("sys.stdout") as stdout:
                self.assertEqual(
                    main(["review-list", "--database-url", database_url, "--status", "pending"]),
                    0,
                )
            list_payload = json.loads("".join(call.args[0] for call in stdout.write.call_args_list))
            self.assertEqual(list_payload["suggestions"][0]["id"], suggestion_id)

            with patch("sys.stdout") as stdout:
                self.assertEqual(
                    main(
                        [
                            "review-decide",
                            suggestion_id,
                            "--database-url",
                            database_url,
                            "--decision",
                            "accept",
                            "--reviewer",
                            "ivan",
                            "--note",
                            "Citations checked.",
                        ]
                    ),
                    0,
                )
            review_payload = json.loads("".join(call.args[0] for call in stdout.write.call_args_list))
            self.assertEqual(review_payload["decision"], "accept")

            engine = build_engine(database_url)
            try:
                with Session(engine) as session:
                    suggestion = session.get(AISuggestion, suggestion_id)
                    decisions = session.scalars(select(ReviewDecision)).all()
                    self.assertEqual(suggestion.status, "accepted")
                    self.assertEqual(len(decisions), 1)
                    self.assertEqual(decisions[0].note, "Citations checked.")
            finally:
                Base.metadata.drop_all(engine)
                engine.dispose()

    def test_eval_command_scores_golden_cases(self):
        with TemporaryDirectory() as tmp:
            db = Path(tmp) / "diw.db"
            cases = Path(tmp) / "cases.jsonl"
            report = Path(tmp) / "report.json"
            markdown = Path(tmp) / "report.md"
            ml_source = Path(tmp) / "ml-paper.md"
            ablation_source = Path(tmp) / "ablation-paper.md"
            database_url = f"sqlite+pysqlite:///{db}"

            cases.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "id": "paper-method-001",
                                "task": "structured_extraction",
                                "question": (
                                    "Extract the method, dataset, metric, and limitation "
                                    "from the cited paper section."
                                ),
                                "expected_fields": [
                                    "method",
                                    "dataset",
                                    "metric",
                                    "limitation",
                                ],
                                "expected_chunk_phrases": ["contrastive retrieval"],
                            }
                        ),
                        json.dumps(
                            {
                                "id": "refusal-001",
                                "task": "refusal",
                                "question": "What does the document say about insurance obligations?",
                                "expected_behavior": "refuse_or_insufficient_evidence",
                            }
                        ),
                    ]
                ),
                encoding="utf-8",
            )
            ml_source.write_text(
                "# Paper\n\n"
                "## Cited Section\n\n"
                "Method: contrastive retrieval.\n\n"
                "Dataset: synthetic paper excerpts.\n\n"
                "Metric: recall@5.\n\n"
                "Limitation: small corpus.\n",
                encoding="utf-8",
            )
            ablation_source.write_text(
                "# Ablation Paper\n\n"
                "## Cited Section\n\n"
                "Method: ablation study over retrieval modes.\n\n"
                "Dataset: synthetic document QA cases.\n\n"
                "Metric: citation-valid answer rate.\n\n"
                "Limitation: intentionally small evaluation set.\n",
                encoding="utf-8",
            )

            with patch("sys.stdout"):
                for source in [ml_source, ablation_source]:
                    self.assertEqual(
                        main(
                            [
                                "load",
                                str(source),
                                "--database-url",
                                database_url,
                                "--target-chars",
                                "500",
                                "--overlap-chars",
                                "0",
                            ]
                        ),
                        0,
                    )

            with patch("sys.stdout") as stdout:
                exit_code = main(
                    [
                        "eval",
                        "--cases",
                        str(cases),
                        "--database-url",
                        database_url,
                        "--top-k",
                        "2",
                        "--out",
                        str(report),
                    ]
                )

            self.assertEqual(exit_code, 0)
            output = "".join(call.args[0] for call in stdout.write.call_args_list)
            self.assertIn("saved:", output)
            payload = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(payload["summary"]["total_cases"], 2)
            self.assertEqual(payload["summary"]["passed_cases"], 2)
            self.assertEqual(payload["summary"]["retrieval_hit_cases"], 1)
            self.assertIn("ai_run_id", payload)

            with patch("sys.stdout"):
                self.assertEqual(
                    main(["eval-report", str(report), "--out", str(markdown)]),
                    0,
                )

            rendered = markdown.read_text(encoding="utf-8")
            self.assertIn("# Evaluation Report", rendered)
            self.assertIn("retrieval", rendered.lower())


if __name__ == "__main__":
    unittest.main()
