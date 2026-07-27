from __future__ import annotations

import re
from dataclasses import dataclass

PAPER_CARD_SCHEMA_VERSION = "paper-card-v1"


@dataclass(frozen=True)
class PaperCardChunk:
    id: str
    heading_path: list[str]
    text: str
    start_line: int
    end_line: int


@dataclass(frozen=True)
class PaperCard:
    title: str
    markdown: str
    extracted_fields: dict[str, str]
    source_chunk_ids: list[str]
    schema_version: str = PAPER_CARD_SCHEMA_VERSION


def build_paper_card(
    *,
    title: str,
    source_name: str,
    version_id: str,
    content_hash: str,
    chunks: list[PaperCardChunk],
) -> PaperCard:
    extracted_fields = {
        "core_idea": _first_sentence(chunks) or "Not identified in source.",
        "problem": _extract_field("problem", chunks, ["problem", "challenge", "motivation"]),
        "method": _extract_field("method", chunks, ["method", "approach", "pipeline", "system"]),
        "dataset": _extract_field("dataset", chunks, ["dataset", "corpus", "data"]),
        "metric": _extract_field("metric", chunks, ["metric", "evaluation", "benchmark", "score"]),
        "result": _extract_field("result", chunks, ["result", "finding", "outcome", "performance"]),
        "limitation": _extract_field("limitation", chunks, ["limitation", "constraint", "risk", "failure"]),
    }
    key_terms = _key_terms(chunks)
    evidence = _evidence_lines(chunks)
    source_chunk_ids = [chunk.id for chunk in chunks]
    markdown = "\n".join(
        [
            f"# {title}",
            "",
            "## Source",
            "",
            f"- Source file: `{source_name}`",
            f"- Version ID: `{version_id}`",
            f"- Content hash: `{content_hash}`",
            f"- Schema: `{PAPER_CARD_SCHEMA_VERSION}`",
            "",
            "## Core Idea",
            "",
            extracted_fields["core_idea"],
            "",
            "## Problem",
            "",
            extracted_fields["problem"],
            "",
            "## Method",
            "",
            extracted_fields["method"],
            "",
            "## Dataset",
            "",
            extracted_fields["dataset"],
            "",
            "## Metrics",
            "",
            extracted_fields["metric"],
            "",
            "## Results",
            "",
            extracted_fields["result"],
            "",
            "## Limitations",
            "",
            extracted_fields["limitation"],
            "",
            "## Key Terms",
            "",
            *(f"- {term}" for term in key_terms),
            "",
            "## Related Concepts",
            "",
            "- [[retrieval augmented generation]]",
            "- [[evaluation harness]]",
            "- [[source-cited generation]]",
            "",
            "## Source Evidence",
            "",
            *evidence,
            "",
        ]
    )
    return PaperCard(
        title=title,
        markdown=markdown,
        extracted_fields=extracted_fields,
        source_chunk_ids=source_chunk_ids,
    )


def _extract_labeled_value(label: str, chunks: list[PaperCardChunk]) -> str | None:
    pattern = re.compile(rf"{re.escape(label)}\s*:\s*(.+)", re.IGNORECASE)
    for chunk in chunks:
        match = pattern.search(chunk.text)
        if match:
            return match.group(1).strip().split("\n", 1)[0].strip()
    return None


def _extract_field(label: str, chunks: list[PaperCardChunk], keywords: list[str]) -> str:
    labeled_value = _extract_labeled_value(label, chunks)
    if labeled_value:
        return labeled_value

    heading_value = _extract_from_heading(chunks, keywords)
    if heading_value:
        return heading_value

    keyword_value = _extract_sentence_with_keywords(chunks, keywords)
    if keyword_value:
        return keyword_value

    return "Not identified in source."


def _extract_from_heading(chunks: list[PaperCardChunk], keywords: list[str]) -> str | None:
    for chunk in chunks:
        heading = " ".join(chunk.heading_path).lower()
        if any(keyword in heading for keyword in keywords):
            return _first_non_heading_line(chunk.text)
    return None


def _extract_sentence_with_keywords(chunks: list[PaperCardChunk], keywords: list[str]) -> str | None:
    for chunk in chunks:
        for sentence in _sentences(chunk.text):
            lowered = sentence.lower()
            if any(keyword in lowered for keyword in keywords):
                return sentence
    return None


def _sentences(text: str) -> list[str]:
    cleaned_lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    cleaned = " ".join(cleaned_lines)
    return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", cleaned) if sentence.strip()]


def _first_sentence(chunks: list[PaperCardChunk]) -> str | None:
    for chunk in chunks:
        lines = [line.strip() for line in chunk.text.splitlines() if line.strip()]
        for line in lines:
            if line.startswith("#"):
                continue
            return re.split(r"(?<=[.!?])\s+", line, maxsplit=1)[0].strip()
    return None


def _key_terms(chunks: list[PaperCardChunk]) -> list[str]:
    text = " ".join(chunk.text.lower() for chunk in chunks)
    candidates = [
        "retrieval",
        "chunking",
        "evaluation",
        "citation",
        "embedding",
        "dataset",
        "metric",
        "hybrid retrieval",
    ]
    found = [candidate for candidate in candidates if candidate in text]
    return found[:6] or ["Needs review"]


def _evidence_lines(chunks: list[PaperCardChunk]) -> list[str]:
    lines = []
    for index, chunk in enumerate(chunks, start=1):
        quote = _first_non_heading_line(chunk.text)
        heading = " / ".join(chunk.heading_path) or "Untitled section"
        lines.append(
            f"- [C{index}] `{chunk.id}` lines {chunk.start_line}-{chunk.end_line}, "
            f"{heading}: \"{quote}\""
        )
    return lines


def _first_non_heading_line(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return stripped[:260]
    return text.strip().splitlines()[0][:260] if text.strip() else ""
