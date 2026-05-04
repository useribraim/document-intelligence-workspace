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
        "retrieved_chunk_count": len(results),
        "answer": answer.model_dump(),
    }


def summarise_eval(case_results: list[dict[str, object]]) -> dict[str, object]:
    total = len(case_results)
    passed = sum(1 for result in case_results if result["passed"])
    citation_valid = sum(1 for result in case_results if result["citation_valid"])
    retrieval_hits = sum(1 for result in case_results if result["retrieval_hit"])
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
        "refusal_cases": len(refusal_cases),
        "correct_refusals": correct_refusals,
        "refusal_accuracy": (
            round(correct_refusals / len(refusal_cases), 4) if refusal_cases else 0.0
        ),
        "task_summary": task_summary,
    }


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
                f"- Missing fields: {', '.join(case['missing_fields']) or 'none'}",
                f"- Missing phrases: {', '.join(case['missing_phrases']) or 'none'}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"
