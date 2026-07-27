import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from sqlalchemy import select
from sqlalchemy.orm import Session

from diw.cli import main
from diw.db.models import AIRun, AISuggestion, Base, ReviewDecision
from diw.db.session import build_engine


class CliTests(unittest.TestCase):
    def test_corpus_verify_distinguishes_manifest_only_from_strict_local_check(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.jsonl"
            missing_text = root / "paper.txt"
            manifest.write_text(
                json.dumps(
                    {
                        "document_id": "paper-1",
                        "canonical_url": "https://example.com/paper",
                        "license_url": "https://example.com/license",
                        "redistributed": False,
                        "version_identifier": "v1",
                        "text_path": str(missing_text),
                        "sha256": "0" * 64,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with patch("sys.stdout"):
                self.assertEqual(
                    main(["corpus-verify", "--manifest", str(manifest), "--allow-missing"]),
                    0,
                )
                self.assertEqual(
                    main(["corpus-verify", "--manifest", str(manifest)]),
                    1,
                )

    def test_annotation_decisions_summary_and_agreement_use_claim_pairs_only(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            annotations = root / "annotations.jsonl"
            decisions = root / "decisions.json"
            completed = root / "completed.jsonl"
            records = [
                {
                    "review_type": "answer_level",
                    "question_id": "q1",
                    "claim_id": None,
                    "citation_id": None,
                    "answer_completeness": None,
                    "refusal_appropriate": None,
                },
                {
                    "review_type": "claim_citation",
                    "question_id": "q1",
                    "claim_id": "q1_c1",
                    "citation_id": "C1",
                    "source_exists": None,
                    "citation_relevant": None,
                    "support_label": None,
                    "support_rationale": None,
                },
            ]
            annotations.write_text(
                "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
            )
            decisions.write_text(
                json.dumps(
                    {
                        "answer_level": {
                            "q1": {
                                "answer_completeness": "complete",
                                "refusal_appropriate": None,
                            }
                        },
                        "claim_citation": {
                            "q1_c1": {
                                "source_exists": True,
                                "citation_relevant": "yes",
                                "support_label": "fully_supported",
                                "support_rationale": "Exact span supports claim.",
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            with patch("sys.stdout"):
                self.assertEqual(
                    main(
                        [
                            "annotation-apply-decisions",
                            "--annotations",
                            str(annotations),
                            "--decisions",
                            str(decisions),
                            "--out",
                            str(completed),
                            "--annotator-id",
                            "test-ai",
                            "--annotation-method",
                            "test_method",
                        ]
                    ),
                    0,
                )

            with patch("sys.stdout") as stdout:
                self.assertEqual(main(["annotation-summary", "--annotations", str(completed)]), 0)
            summary = json.loads("".join(call.args[0] for call in stdout.write.call_args_list))
            self.assertEqual(summary["claim_citation_records"], 1)
            self.assertEqual(summary["support_counts"], {"fully_supported": 1})
            self.assertEqual(summary["completed_records"], 2)
            self.assertEqual(summary["pending_records"], 0)

            with patch("sys.stdout") as stdout:
                self.assertEqual(
                    main(
                        [
                            "annotation-agreement",
                            "--first",
                            str(completed),
                            "--second",
                            str(completed),
                        ]
                    ),
                    0,
                )
            agreement = json.loads("".join(call.args[0] for call in stdout.write.call_args_list))
            self.assertEqual(agreement["shared_pairs"], 1)
            self.assertEqual(agreement["cohen_kappa"], 1.0)
            self.assertTrue(agreement["gate_passed"])

            incomplete = root / "incomplete.jsonl"
            incomplete.write_text(annotations.read_text(encoding="utf-8"), encoding="utf-8")
            with patch("sys.stdout") as stdout:
                self.assertEqual(
                    main(
                        [
                            "annotation-agreement",
                            "--first",
                            str(completed),
                            "--second",
                            str(incomplete),
                            "--require-complete",
                            "--minimum-pairs",
                            "1",
                        ]
                    ),
                    1,
                )
            blocked = json.loads("".join(call.args[0] for call in stdout.write.call_args_list))
            self.assertFalse(blocked["gate_passed"])
            self.assertEqual(blocked["labeled_pairs"], 0)

    def test_claim_audit_runs_with_deterministic_provider_and_gold_metrics(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "paper.md"
            manifest = root / "manifest.jsonl"
            questions = root / "questions.jsonl"
            database = root / "audit.db"
            output = root / "run.json"
            source.write_text(
                "# Paper\n\n## Method\n\nHybrid retrieval combines lexical and vector retrieval.\n",
                encoding="utf-8",
            )
            manifest.write_text(
                json.dumps({"document_id": "paper", "text_path": str(source)}) + "\n",
                encoding="utf-8",
            )
            database_url = f"sqlite+pysqlite:///{database}"
            with patch("sys.stdout"):
                self.assertEqual(
                    main(["load", str(source), "--database-url", database_url]), 0
                )
            import sqlite3
            connection = sqlite3.connect(database)
            try:
                chunk_id = connection.execute("select id from chunks").fetchone()[0]
            finally:
                connection.close()
            questions.write_text(
                json.dumps(
                    {
                        "question_id": "q1",
                        "question": "How does hybrid retrieval work?",
                        "category": "direct_extraction",
                        "expected_evidence_status": "sufficient",
                        "source_documents": ["paper"],
                        "gold_evidence_chunk_ids": [chunk_id],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with patch("sys.stdout"):
                self.assertEqual(
                    main(
                        [
                            "claim-audit",
                            "--questions", str(questions),
                            "--manifest", str(manifest),
                            "--database-url", database_url,
                            "--require-gold-evidence",
                            "--out", str(output),
                        ]
                    ),
                    0,
                )

            run = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(run["question_count"], 1)
            self.assertEqual(run["runs"][0]["temperature"], None)
            self.assertEqual(run["summary"]["retrieval_eval_cases"], 1)

    def test_evidence_repair_reuses_saved_baseline_without_a_model_call(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline_path = root / "baseline.json"
            repaired_path = root / "repaired.json"
            chunk_id = "paper:1"
            baseline_path.write_text(
                json.dumps(
                    {
                        "run_id": "baseline",
                        "config": {"evidence_repair": False},
                        "runs": [
                            {
                                "run_id": "baseline",
                                "question_id": "q1",
                                "expected_evidence_status": "sufficient",
                                "gold_evidence_chunk_ids": [chunk_id],
                                "retrieval_recall_at_k": 1.0,
                                "retrieval_mrr": 1.0,
                                "gold_citation_recall": 1.0,
                                "expected_min_claim_count": 1,
                                "structural_complete": True,
                                "retrieval_config_hash": "old",
                                "answer": "Hybrid retrieval uses lexical and vector search [C1]",
                                "answer_sha256": "old",
                                "insufficient_evidence": False,
                                "latency_ms": 5,
                                "input_tokens": 10,
                                "cached_input_tokens": 0,
                                "output_tokens": 5,
                                "completion_attempts": 1,
                                "estimated_cost_usd": 0.01,
                                "claims": [{"claim_id": "q1_c1", "text": "Hybrid retrieval uses lexical and vector search", "citation_ids": ["C1"]}],
                                "annotations_pending": [{"claim_id": "q1_c1", "claim_text": "Hybrid retrieval uses lexical and vector search", "citation_id": "C1", "evidence_span": "Hybrid retrieval uses lexical and vector search.", "quote_alignment": "exact", "source_exists": True, "citation_relevant": "yes", "support_label": "partially_supported", "support_rationale": "Fixture partial label."}],
                                "retrieved_passages": [{"rank": 1, "chunk_id": chunk_id, "document_id": "paper", "version_id": "v1", "chunk_index": 1, "heading_path": ["Method"], "text": "Hybrid retrieval uses lexical and vector search.", "lexical_score": 1.0, "vector_score": 1.0, "score": 1.0}],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with patch("sys.stdout"):
                self.assertEqual(
                    main([
                        "audit-evidence-repair", "--run", str(baseline_path),
                        "--out", str(repaired_path), "--run-id", "repaired",
                    ]),
                    0,
                )

            repaired = json.loads(repaired_path.read_text(encoding="utf-8"))
            self.assertEqual(repaired["parent_run_id"], "baseline")
            self.assertEqual(repaired["summary"]["fully_supported_rate"], 1.0)
            self.assertEqual(repaired["summary"]["repair_incremental_cost_usd"], 0.0)

    def test_retrieval_trace_reports_rank_changes_from_saved_runs(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            left_path = root / "left.json"
            right_path = root / "right.json"
            output = root / "trace.json"
            left_path.write_text(
                json.dumps({"run_id": "left", "runs": [{"question_id": "q1", "gold_evidence_chunk_ids": ["gold"], "retrieved_passage_ids": ["gold", "other"]}]}),
                encoding="utf-8",
            )
            right_path.write_text(
                json.dumps({"run_id": "right", "runs": [{"question_id": "q1", "gold_evidence_chunk_ids": ["gold"], "retrieved_passage_ids": ["other", "gold"]}]}),
                encoding="utf-8",
            )

            with patch("sys.stdout"):
                self.assertEqual(
                    main([
                        "audit-retrieval-trace", "--left", str(left_path),
                        "--right", str(right_path), "--out", str(output),
                    ]),
                    0,
                )

            trace = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(trace["summary"]["identical_top_k_questions"], 0)
            self.assertEqual(trace["traces"][0]["left_gold_ranks"], [1])
            self.assertEqual(trace["traces"][0]["right_gold_ranks"], [2])

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
