from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

from fastapi.testclient import TestClient

from diw.annotation_app import create_annotation_app, load_jsonl


class AnnotationAppTests(unittest.TestCase):
    def test_saves_human_decision_without_overwriting_input(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.jsonl"
            output = root / "human.jsonl"
            original = {
                "source_run_id": "run-1",
                "question_id": "q1",
                "question": "What happened?",
                "review_type": "claim_citation",
                "claim_id": "q1-c1",
                "citation_id": "C1",
                "claim_text": "The method improved recall.",
                "evidence_span": "Recall improved by two points.",
                "annotation_status": "pending_human_confirmation",
            }
            source.write_text(json.dumps(original) + "\n", encoding="utf-8")
            app = create_annotation_app(
                input_path=source,
                output_path=output,
                default_annotator_id="a1",
            )
            with TestClient(app) as client:
                key = client.get("/api/state").json()["records"][0]
                record_key = "|".join(
                    str(key.get(field) or "")
                    for field in (
                        "source_run_id",
                        "question_id",
                        "review_type",
                        "claim_id",
                        "citation_id",
                    )
                )
                response = client.post(
                    "/api/decisions",
                    json={
                        "record_key": record_key,
                        "annotator_id": "human-a1",
                        "source_exists": True,
                        "citation_relevant": "yes",
                        "support_label": "fully_supported",
                        "support_rationale": "The exact improvement is stated.",
                    },
                )

            self.assertEqual(response.status_code, 200)
            saved = load_jsonl(output)[0]
            self.assertEqual(saved["annotation_status"], "completed_human")
            self.assertEqual(saved["annotator_id"], "human-a1")
            self.assertEqual(load_jsonl(source)[0], original)

    def test_requires_failure_mode_for_partial_support(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.jsonl"
            source.write_text(
                json.dumps(
                    {
                        "source_run_id": "run-1",
                        "question_id": "q1",
                        "review_type": "claim_citation",
                        "claim_id": "c1",
                        "citation_id": "C1",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            app = create_annotation_app(
                input_path=source,
                output_path=root / "out.jsonl",
                default_annotator_id="a1",
            )
            with TestClient(app) as client:
                response = client.post(
                    "/api/decisions",
                    json={
                        "record_key": "run-1|q1|claim_citation|c1|C1",
                        "annotator_id": "a1",
                        "source_exists": True,
                        "citation_relevant": "yes",
                        "support_label": "partially_supported",
                        "support_rationale": "A qualifier is missing.",
                    },
                )
            self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
