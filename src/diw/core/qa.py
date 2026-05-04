from __future__ import annotations

import re

from pydantic import BaseModel, Field

from diw.core.retrieval import RetrievalResult, tokenise_query


class EvidenceCitation(BaseModel):
    label: str
    chunk_id: str
    document_id: str
    version_id: str
    heading_path: list[str]
    quote: str
    score: float


class SourceCitedAnswer(BaseModel):
    query: str
    answer: str
    insufficient_evidence: bool
    citations: list[EvidenceCitation] = Field(default_factory=list)
    extracted_fields: dict[str, str] = Field(default_factory=dict)
    model: str | None = None
    provider: str | None = None
    prompt_version: str | None = None


class CitationValidation(BaseModel):
    valid: bool
    errors: list[str] = Field(default_factory=list)


def _sentences(text: str) -> list[str]:
    candidates: list[str] = []
    for paragraph in text.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        candidates.extend(part.strip() for part in re.split(r"(?<=[.!?])\s+", paragraph) if part)
    return candidates


def _best_quote(query: str, text: str, *, max_chars: int = 360) -> str:
    query_tokens = tokenise_query(query)
    candidates = _sentences(text)
    if not candidates:
        return text[:max_chars].strip()

    def score(sentence: str) -> tuple[int, int, int]:
        sentence_tokens = tokenise_query(sentence)
        labelled_value_bonus = 2 if ":" in sentence else 0
        heading_penalty = -2 if sentence.lstrip().startswith("#") else 0
        return (len(query_tokens & sentence_tokens), labelled_value_bonus + heading_penalty, -len(sentence))

    best = max(candidates, key=score).strip()
    if len(best) <= max_chars:
        return best
    return best[: max_chars - 3].rstrip() + "..."


def compose_source_cited_answer(
    query: str,
    results: list[RetrievalResult],
    *,
    min_score: float = 0.12,
    min_lexical_score: float = 0.05,
    max_citations: int = 3,
) -> SourceCitedAnswer:
    usable_results = [
        result
        for result in results
        if result.score >= min_score and result.lexical_score >= min_lexical_score
    ]
    usable_results = usable_results[:max_citations]

    if not usable_results:
        return SourceCitedAnswer(
            query=query,
            answer=(
                "Insufficient evidence in the retrieved document chunks to answer this "
                "question reliably."
            ),
            insufficient_evidence=True,
            citations=[],
        )

    citations: list[EvidenceCitation] = []
    answer_parts: list[str] = []

    for index, result in enumerate(usable_results, start=1):
        label = f"C{index}"
        quote = _best_quote(query, result.text)
        citations.append(
            EvidenceCitation(
                label=label,
                chunk_id=result.chunk_id,
                document_id=result.document_id,
                version_id=result.version_id,
                heading_path=result.heading_path,
                quote=quote,
                score=result.score,
            )
        )
        answer_parts.append(f"{quote} [{label}]")

    return SourceCitedAnswer(
        query=query,
        answer=" ".join(answer_parts),
        insufficient_evidence=False,
        citations=citations,
    )


def validate_citations(answer: SourceCitedAnswer, results: list[RetrievalResult]) -> CitationValidation:
    chunks_by_id = {result.chunk_id: result for result in results}
    errors: list[str] = []

    if answer.insufficient_evidence and answer.citations:
        errors.append("insufficient-evidence answers must not include citations")

    labels = set()
    for citation in answer.citations:
        if citation.label in labels:
            errors.append(f"duplicate citation label: {citation.label}")
        labels.add(citation.label)

        source = chunks_by_id.get(citation.chunk_id)
        if source is None:
            errors.append(f"citation references unknown chunk: {citation.chunk_id}")
            continue

        if citation.quote not in source.text:
            errors.append(f"citation quote is not present in source chunk: {citation.label}")

    return CitationValidation(valid=not errors, errors=errors)
