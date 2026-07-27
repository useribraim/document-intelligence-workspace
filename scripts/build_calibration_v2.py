from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDS_PATH = ROOT / "data/audit/calibration/v2_case_seeds.jsonl"
QUESTIONS_PATH = ROOT / "data/audit/questions/v2_140_calibration.jsonl"
PRIMARY_PATH = ROOT / "data/audit/annotations/v2_primary_annotation_template.jsonl"
INDEPENDENT_PATH = ROOT / "data/audit/annotations/v2_independent_annotation_template.jsonl"
MANIFEST_PATH = ROOT / "data/audit/calibration/v2_manifest.json"
CORPUS_MANIFEST_PATH = ROOT / "data/audit/corpus_manifest.jsonl"

CASE_TYPES = (
    "supported",
    "partial",
    "unsupported",
    "misleading_context",
    "refusal",
)
CASE_STATUS = {
    "supported": "sufficient",
    "partial": "sufficient",
    "unsupported": "insufficient",
    "misleading_context": "conflicting",
    "refusal": "insufficient",
}
PAIR_CASES = set(CASE_TYPES) - {"refusal"}
LABEL_FIELDS = (
    "source_exists",
    "citation_relevant",
    "support_label",
    "support_rationale",
    "failure_mode",
    "answer_completeness",
    "refusal_appropriate",
)


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def jsonl_bytes(records: list[dict]) -> bytes:
    return "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        for record in records
    ).encode()


def question_text(seed: dict, case_type: str) -> str:
    topic = seed["topic"]
    if case_type == "supported":
        return f"What does the cited source establish about {topic}?"
    if case_type == "partial":
        return f"What combined conclusion does the cited source establish about {topic}?"
    if case_type == "unsupported":
        return f"According to the cited source, what can be concluded about {topic}?"
    if case_type == "misleading_context":
        return f"Does this evidence justify a broad conclusion about {topic}?"
    return seed["refusal_question"]


def candidate_claim(seed: dict, case_type: str) -> str:
    return seed[f"{case_type}_claim"]


def annotation_base(
    *,
    question: dict,
    answer: str,
    annotator_id: str,
) -> dict:
    return {
        "source_run_id": "calibration-v2-controlled-stimuli",
        "question_id": question["question_id"],
        "question": question["question"],
        "review_context": {
            "answer": answer,
            "insufficient_evidence": question["case_type"] == "refusal",
        },
        "rubric_version": "v2.0",
        "annotator_id": annotator_id,
        "annotation_status": "pending_human_confirmation",
        "blinded": True,
    }


def build_packet(
    questions: list[dict],
    seeds_by_id: dict[str, dict],
    *,
    annotator_id: str,
) -> list[dict]:
    records = []
    for question in questions:
        seed = seeds_by_id[question["seed_id"]]
        case_type = question["case_type"]
        if case_type == "refusal":
            answer = (
                "The cited evidence does not provide "
                f"{seed['refusal_target']}, so this question should be refused."
            )
        else:
            answer = f"{candidate_claim(seed, case_type)} [C1]"
        base = annotation_base(
            question=question,
            answer=answer,
            annotator_id=annotator_id,
        )
        records.append(
            {
                **base,
                "review_type": "answer_level",
                "claim_id": None,
                "claim_text": None,
                "citation_id": None,
                "evidence_span": None,
                "quote_alignment": None,
                **{field: None for field in LABEL_FIELDS},
            }
        )
        if case_type in PAIR_CASES:
            records.append(
                {
                    **base,
                    "review_type": "claim_citation",
                    "claim_id": f"{question['question_id']}_c1",
                    "claim_text": candidate_claim(seed, case_type),
                    "citation_id": "C1",
                    "evidence_span": seed["evidence_span"],
                    "evidence_source": {
                        "document_id": seed["document_id"],
                        "chunk_id": seed["chunk_id"],
                    },
                    "quote_alignment": "canonicalized",
                    **{field: None for field in LABEL_FIELDS},
                }
            )
    return records


def build() -> tuple[list[dict], list[dict], list[dict], dict]:
    seeds = load_jsonl(SEEDS_PATH)
    if len(seeds) != 28:
        raise ValueError(f"expected 28 source-grounded seeds, found {len(seeds)}")
    if len({seed["seed_id"] for seed in seeds}) != len(seeds):
        raise ValueError("seed IDs must be unique")
    if len({seed["chunk_id"] for seed in seeds}) != len(seeds):
        raise ValueError("each calibration seed must use a distinct source chunk")
    if len({seed["evidence_span"] for seed in seeds}) != len(seeds):
        raise ValueError("each calibration seed must use a distinct evidence span")
    known_documents = {
        record["document_id"] for record in load_jsonl(CORPUS_MANIFEST_PATH)
    }
    unknown_documents = sorted(
        {seed["document_id"] for seed in seeds} - known_documents
    )
    if unknown_documents:
        raise ValueError(
            "calibration seeds reference unknown documents: "
            + ", ".join(unknown_documents)
        )

    questions = []
    for seed_index, seed in enumerate(seeds):
        for case_index, case_type in enumerate(CASE_TYPES):
            sequence = seed_index * len(CASE_TYPES) + case_index + 1
            question_id = f"v2_{sequence:03d}"
            questions.append(
                {
                    "question_id": question_id,
                    "seed_id": seed["seed_id"],
                    "question": question_text(seed, case_type),
                    "case_type": case_type,
                    "expected_evidence_status": CASE_STATUS[case_type],
                    "source_documents": [seed["document_id"]],
                    "gold_evidence_chunk_ids": (
                        [seed["chunk_id"]] if case_type in PAIR_CASES else []
                    ),
                    "author_notes": (
                        "Controlled calibration stimulus. The case stratum is excluded "
                        "from both blinded annotation packets."
                    ),
                }
            )

    seeds_by_id = {seed["seed_id"]: seed for seed in seeds}
    primary = build_packet(questions, seeds_by_id, annotator_id="primary-a1")
    independent = build_packet(questions, seeds_by_id, annotator_id="independent-a2")

    counts = Counter(question["case_type"] for question in questions)
    pair_count = sum(
        record["review_type"] == "claim_citation" for record in primary
    )
    pair_case_counts = Counter(
        question["case_type"]
        for question in questions
        if question["case_type"] in PAIR_CASES
    )
    manifest = {
        "version": "v2.0",
        "instrument": "controlled_claim_citation_calibration",
        "question_count": len(questions),
        "case_counts": dict(sorted(counts.items())),
        "claim_citation_pairs_per_annotator": pair_count,
        "pair_case_counts": dict(sorted(pair_case_counts.items())),
        "annotation_records_per_annotator": len(primary),
        "independent_annotators_required": 2,
        "intended_labels_are_human_pending": True,
        "source_seed": {
            "path": str(SEEDS_PATH.relative_to(ROOT)),
            "sha256": hashlib.sha256(SEEDS_PATH.read_bytes()).hexdigest(),
            "bytes": SEEDS_PATH.stat().st_size,
        },
        "files": {},
    }
    return questions, primary, independent, manifest


def validate(
    questions: list[dict],
    primary: list[dict],
    independent: list[dict],
) -> None:
    case_counts = Counter(question["case_type"] for question in questions)
    if len(questions) not in range(120, 161):
        raise ValueError("calibration must contain 120 to 160 questions")
    if case_counts != Counter({case_type: 28 for case_type in CASE_TYPES}):
        raise ValueError(f"case strata are not balanced: {dict(case_counts)}")
    if len({question["question_id"] for question in questions}) != len(questions):
        raise ValueError("question IDs must be unique")
    if len({question["question"] for question in questions}) != len(questions):
        raise ValueError("question text must be unique")

    packets = {"primary": primary, "independent": independent}
    packet_keys = {}
    for packet_name, records in packets.items():
        pairs = [
            record for record in records if record["review_type"] == "claim_citation"
        ]
        if len(pairs) < 100:
            raise ValueError(f"{packet_name} has only {len(pairs)} claim-citation pairs")
        keys = {
            (
                record["question_id"],
                record["review_type"],
                record.get("claim_id"),
                record.get("citation_id"),
            )
            for record in records
        }
        if len(keys) != len(records):
            raise ValueError(f"{packet_name} contains duplicate annotation records")
        packet_keys[packet_name] = keys
        for record in records:
            if not record.get("blinded"):
                raise ValueError(f"{packet_name} contains an unblinded record")
            if "case_type" in record or "expected_evidence_status" in record:
                raise ValueError(f"{packet_name} leaks the author-designed stratum")
            if "automation_prefill" in record:
                raise ValueError(f"{packet_name} contains automated labels")
            if any(record.get(field) is not None for field in LABEL_FIELDS):
                raise ValueError(f"{packet_name} contains a prefilled human field")
        for record in pairs:
            if len(record["evidence_span"].split()) > 55:
                raise ValueError(
                    f"{record['question_id']} exceeds the short-excerpt limit"
                )

    if packet_keys["primary"] != packet_keys["independent"]:
        raise ValueError("annotation packets do not contain the same aligned records")
    if {record["annotator_id"] for record in primary} & {
        record["annotator_id"] for record in independent
    }:
        raise ValueError("annotation packet identifiers must be distinct")


def render_outputs() -> dict[Path, bytes]:
    questions, primary, independent, manifest = build()
    validate(questions, primary, independent)
    outputs = {
        QUESTIONS_PATH: jsonl_bytes(questions),
        PRIMARY_PATH: jsonl_bytes(primary),
        INDEPENDENT_PATH: jsonl_bytes(independent),
    }
    manifest["files"] = {
        str(path.relative_to(ROOT)): {
            "sha256": hashlib.sha256(content).hexdigest(),
            "bytes": len(content),
        }
        for path, content in outputs.items()
    }
    outputs[MANIFEST_PATH] = (
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    ).encode()
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build or verify the blinded V2 human-calibration instrument."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if tracked outputs differ from deterministic source seeds.",
    )
    args = parser.parse_args()
    outputs = render_outputs()

    if args.check:
        failures = [
            str(path.relative_to(ROOT))
            for path, expected in outputs.items()
            if not path.is_file() or path.read_bytes() != expected
        ]
        if failures:
            raise SystemExit("stale calibration outputs: " + ", ".join(failures))
        print("calibration v2: valid, balanced, blinded, and reproducible")
        return 0

    for path, content in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        print(f"saved: {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
