from __future__ import annotations

import hashlib
import json
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = ROOT / "data/audit/questions/v2_140_calibration.jsonl"
PRIMARY = ROOT / "data/audit/annotations/v2_primary_annotation_template.jsonl"
INDEPENDENT = ROOT / "data/audit/annotations/v2_independent_annotation_template.jsonl"
MANIFEST = ROOT / "data/audit/calibration/v2_manifest.json"


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class CalibrationV2Tests(unittest.TestCase):
    def test_question_set_is_unique_and_balanced(self):
        questions = load_jsonl(QUESTIONS)

        self.assertEqual(len(questions), 140)
        self.assertEqual(
            Counter(question["case_type"] for question in questions),
            Counter(
                {
                    "supported": 28,
                    "partial": 28,
                    "unsupported": 28,
                    "misleading_context": 28,
                    "refusal": 28,
                }
            ),
        )
        self.assertEqual(len({question["question_id"] for question in questions}), 140)
        self.assertEqual(len({question["question"] for question in questions}), 140)

    def test_packets_have_112_aligned_blank_pairs(self):
        primary = load_jsonl(PRIMARY)
        independent = load_jsonl(INDEPENDENT)

        self.assertEqual(len(primary), 252)
        self.assertEqual(len(independent), 252)

        def keys(records: list[dict]) -> set[tuple]:
            return {
                (
                    record["question_id"],
                    record["review_type"],
                    record.get("claim_id"),
                    record.get("citation_id"),
                )
                for record in records
            }

        self.assertEqual(keys(primary), keys(independent))
        self.assertEqual(
            sum(record["review_type"] == "claim_citation" for record in primary),
            112,
        )
        for records in (primary, independent):
            for record in records:
                self.assertTrue(record["blinded"])
                self.assertNotIn("case_type", record)
                self.assertNotIn("expected_evidence_status", record)
                self.assertNotIn("automation_prefill", record)
                self.assertIsNone(record["support_label"])
                self.assertIsNone(record["support_rationale"])
                self.assertIsNone(record["answer_completeness"])
                self.assertIsNone(record["refusal_appropriate"])

    def test_manifest_hashes_match_generated_artifacts(self):
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

        self.assertEqual(manifest["question_count"], 140)
        self.assertEqual(manifest["claim_citation_pairs_per_annotator"], 112)
        self.assertEqual(
            manifest["pair_case_counts"],
            {
                "misleading_context": 28,
                "partial": 28,
                "supported": 28,
                "unsupported": 28,
            },
        )
        source = manifest["source_seed"]
        source_content = (ROOT / source["path"]).read_bytes()
        self.assertEqual(hashlib.sha256(source_content).hexdigest(), source["sha256"])
        self.assertEqual(len(source_content), source["bytes"])
        for relative_path, expected in manifest["files"].items():
            content = (ROOT / relative_path).read_bytes()
            self.assertEqual(hashlib.sha256(content).hexdigest(), expected["sha256"])
            self.assertEqual(len(content), expected["bytes"])


if __name__ == "__main__":
    unittest.main()
