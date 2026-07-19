from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from diw.core.llm import LLMProvider, generate_structured_answer
from diw.core.qa import validate_citations
from diw.core.retrieval import RetrievalResult


@dataclass(frozen=True)
class GoldenCase:
    id: str
    task: str
    question: str
    expected_fields: list[str]
    expected_behavior: str | None = None
    expected_chunk_phrases: list[str] | None = None


def retrieval_recall_at_k(results: list[RetrievalResult], expected_phrases: list[str]) -> float:
    if not expected_phrases:
        return 0.0
    retrieved_text = "\n".join(result.text.lower() for result in results)
    found = sum(1 for phrase in expected_phrases if phrase.lower() in retrieved_text)
    return round(found / len(expected_phrases), 4)


def retrieval_mrr(results: list[RetrievalResult], expected_phrases: list[str]) -> float:
    if not expected_phrases:
        return 0.0
    expected = [phrase.lower() for phrase in expected_phrases]
    for rank, result in enumerate(results, start=1):
        if any(phrase in result.text.lower() for phrase in expected):
            return round(1 / rank, 4)
    return 0.0


def load_golden_cases(path: Path) -> list[GoldenCase]:
    cases: list[GoldenCase] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        cases.append(
            GoldenCase(
                id=payload["id"],
                task=payload["task"],
                question=payload["question"],
                expected_fields=list(payload.get("expected_fields", [])),
                expected_behavior=payload.get("expected_behavior"),
                expected_chunk_phrases=payload.get("expected_chunk_phrases"),
            )
        )
    return cases


def score_case(
    case: GoldenCase,
    results: list[RetrievalResult],
    provider: LLMProvider,
) -> dict[str, object]:
    answer = generate_structured_answer(case.question, results, provider)
    citation_validation = validate_citations(answer, results)
    present_fields = sorted(
        field for field in case.expected_fields if answer.extracted_fields.get(field)
    )
    missing_fields = sorted(set(case.expected_fields) - set(present_fields))
    retrieved_text = "\n".join(result.text.lower() for result in results)
    expected_phrases = case.expected_chunk_phrases or []
    retrieved_phrases = [
        phrase for phrase in expected_phrases if phrase.lower() in retrieved_text
    ]
    missing_phrases = sorted(set(expected_phrases) - set(retrieved_phrases))
    retrieval_hit = bool(expected_phrases) and not missing_phrases
    recall_at_k = retrieval_recall_at_k(results, expected_phrases)
    mrr = retrieval_mrr(results, expected_phrases)

    if case.expected_behavior == "refuse_or_insufficient_evidence":
        passed = answer.insufficient_evidence and citation_validation.valid
    else:
        passed = (
            bool(present_fields or retrieved_phrases)
            and not missing_fields
            and retrieval_hit
            and citation_validation.valid
            and not answer.insufficient_evidence
        )

    return {
        "id": case.id,
        "task": case.task,
        "passed": passed,
        "citation_valid": citation_validation.valid,
        "citation_errors": citation_validation.errors,
        "insufficient_evidence": answer.insufficient_evidence,
        "expected_fields": case.expected_fields,
        "present_fields": present_fields,
        "missing_fields": missing_fields,
        "expected_chunk_phrases": expected_phrases,
        "retrieved_phrases": retrieved_phrases,
        "missing_phrases": missing_phrases,
        "retrieval_hit": retrieval_hit,
        "retrieval_recall_at_k": recall_at_k,
        "retrieval_mrr": mrr,
        "retrieved_chunk_count": len(results),
        "answer": answer.model_dump(),
    }


def summarise_eval(case_results: list[dict[str, object]]) -> dict[str, object]:
    total = len(case_results)
    passed = sum(1 for result in case_results if result["passed"])
    citation_valid = sum(1 for result in case_results if result["citation_valid"])
    retrieval_hits = sum(1 for result in case_results if result["retrieval_hit"])
    retrieval_cases = [result for result in case_results if result["expected_chunk_phrases"]]
    refusal_cases = [
        result
        for result in case_results
        if result["answer"]["insufficient_evidence"]
        or result.get("insufficient_evidence")
        or result.get("task") == "refusal"
    ]
    correct_refusals = sum(
        1
        for result in case_results
        if result.get("task") == "refusal" and result.get("insufficient_evidence")
    )
    task_summary = _summarise_by_task(case_results)
    return {
        "total_cases": total,
        "passed_cases": passed,
        "pass_rate": round(passed / total, 4) if total else 0.0,
        "citation_valid_cases": citation_valid,
        "citation_valid_rate": round(citation_valid / total, 4) if total else 0.0,
        "retrieval_hit_cases": retrieval_hits,
        "retrieval_hit_rate": round(retrieval_hits / total, 4) if total else 0.0,
        "retrieval_eval_cases": len(retrieval_cases),
        "retrieval_recall_at_k": _average(retrieval_cases, "retrieval_recall_at_k"),
        "retrieval_mrr": _average(retrieval_cases, "retrieval_mrr"),
        "refusal_cases": len(refusal_cases),
        "correct_refusals": correct_refusals,
        "refusal_accuracy": (
            round(correct_refusals / len(refusal_cases), 4) if refusal_cases else 0.0
        ),
        "task_summary": task_summary,
    }


def _average(results: list[dict[str, object]], key: str) -> float:
    if not results:
        return 0.0
    return round(sum(float(result[key]) for result in results) / len(results), 4)


def _summarise_by_task(case_results: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    tasks = sorted({str(result["task"]) for result in case_results})
    summary: dict[str, dict[str, object]] = {}
    for task in tasks:
        task_results = [result for result in case_results if result["task"] == task]
        total = len(task_results)
        passed = sum(1 for result in task_results if result["passed"])
        summary[task] = {
            "total_cases": total,
            "passed_cases": passed,
            "pass_rate": round(passed / total, 4) if total else 0.0,
        }
    return summary


def render_markdown_report(report: dict[str, object]) -> str:
    summary = report["summary"]
    lines = [
        "# Evaluation Report",
        "",
        "## Summary",
        "",
        f"- Total cases: {summary['total_cases']}",
        f"- Passed cases: {summary['passed_cases']}",
        f"- Pass rate: {summary['pass_rate']}",
        f"- Citation-validity rate: {summary['citation_valid_rate']}",
        f"- Retrieval hit rate: {summary['retrieval_hit_rate']}",
        f"- Retrieval recall@k: {summary['retrieval_recall_at_k']}",
        f"- Retrieval MRR: {summary['retrieval_mrr']}",
        f"- Refusal accuracy: {summary['refusal_accuracy']}",
        "",
        "## Task Breakdown",
        "",
    ]
    for task, task_summary in summary["task_summary"].items():
        lines.append(
            f"- {task}: {task_summary['passed_cases']}/{task_summary['total_cases']} "
            f"passed ({task_summary['pass_rate']})"
        )

    lines.extend(["", "## Cases", ""])
    for case in report["cases"]:
        status = "PASS" if case["passed"] else "FAIL"
        lines.extend(
            [
                f"### {case['id']} - {status}",
                "",
                f"- Task: {case['task']}",
                f"- Citation valid: {case['citation_valid']}",
                f"- Retrieval hit: {case['retrieval_hit']}",
                f"- Retrieval recall@k: {case['retrieval_recall_at_k']}",
                f"- Retrieval MRR: {case['retrieval_mrr']}",
                f"- Missing fields: {', '.join(case['missing_fields']) or 'none'}",
                f"- Missing phrases: {', '.join(case['missing_phrases']) or 'none'}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"
