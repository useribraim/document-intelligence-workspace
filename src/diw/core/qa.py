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
    quote_alignment: str = "exact"


class SourceCitedAnswer(BaseModel):
    query: str
    answer: str
    insufficient_evidence: bool
    citations: list[EvidenceCitation] = Field(default_factory=list)
    extracted_fields: dict[str, str] = Field(default_factory=dict)
    model: str | None = None
    provider: str | None = None
    prompt_version: str | None = None
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    completion_attempts: int = 0


class CitationValidation(BaseModel):
    valid: bool
    errors: list[str] = Field(default_factory=list)


def _normalise_evidence_text(text: str) -> str:
    """Make PDF-extraction whitespace irrelevant without changing evidence wording."""
    return " ".join(text.split())


def canonicalise_citation_quotes(
    answer: SourceCitedAnswer,
    results: list[RetrievalResult],
) -> SourceCitedAnswer:
    """Replace a model-paraphrased quote with a deterministic exact source span.

    PDF extraction can make exact character copying unreliable. The cited chunk remains
    model-selected; only the stored span is canonicalised from that already-selected chunk.
    """
    sources = {result.chunk_id: result for result in results}
    for citation in answer.citations:
        source = sources.get(citation.chunk_id)
        if source is None:
            continue
        if _normalise_evidence_text(citation.quote) in _normalise_evidence_text(source.text):
            continue
        citation.quote = _best_matching_source_span(citation.quote, source.text)
        citation.quote_alignment = "canonicalized"
    return answer


def prune_unused_citations(answer: SourceCitedAnswer) -> SourceCitedAnswer:
    """Keep only citation objects that are actually attached to the answer text."""
    mapped_labels = set(re.findall(r"\[(C\d+)\]", answer.answer))
    answer.citations = [citation for citation in answer.citations if citation.label in mapped_labels]
    return answer


def materialise_answer_citations(
    answer: SourceCitedAnswer,
    results: list[RetrievalResult],
) -> SourceCitedAnswer:
    """Fill missing citation objects from the answer's explicit Cn evidence labels."""
    by_label = {citation.label: citation for citation in answer.citations}
    for label in sorted(set(re.findall(r"\[(C\d+)\]", answer.answer))):
        if label in by_label:
            continue
        index = int(label[1:]) - 1
        if not 0 <= index < len(results):
            continue
        source = results[index]
        citation = EvidenceCitation(
            label=label,
            chunk_id=source.chunk_id,
            document_id=source.document_id,
            version_id=source.version_id,
            heading_path=source.heading_path,
            quote=_best_matching_source_span(answer.answer, source.text),
            score=source.score,
            quote_alignment="label_resolved",
        )
        answer.citations.append(citation)
        by_label[label] = citation
    return answer


def normalise_refusal(answer: SourceCitedAnswer) -> SourceCitedAnswer:
    """Make a model-declared insufficient-evidence response structurally explicit."""
    if not answer.insufficient_evidence:
        return answer
    answer.citations = []
    answer.answer = re.sub(r"\s*\[C\d+\]", "", answer.answer).strip()
    if not answer.answer.strip():
        answer.answer = "Insufficient evidence in the retrieved document chunks to answer reliably."
    return answer


def _best_matching_source_span(model_quote: str, source_text: str, *, max_words: int = 45) -> str:
    source_words = source_text.split()
    if not source_words:
        return ""
    quote_tokens = tokenise_query(model_quote)
    if not quote_tokens:
        return " ".join(source_words[:max_words])
    window_size = min(max_words, len(source_words))
    best_start = 0
    best_score = -1
    for start in range(len(source_words)):
        candidate = " ".join(source_words[start : start + window_size])
        score = len(quote_tokens & tokenise_query(candidate))
        if score > best_score:
            best_start, best_score = start, score
    return " ".join(source_words[best_start : best_start + window_size])


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
    if answer.insufficient_evidence and not answer.answer.strip():
        errors.append("insufficient-evidence answers must include a refusal explanation")
    if not answer.insufficient_evidence and not answer.citations:
        errors.append("evidence-backed answers must include at least one citation")

    labels = set()
    for citation in answer.citations:
        if citation.label in labels:
            errors.append(f"duplicate citation label: {citation.label}")
        labels.add(citation.label)

        source = chunks_by_id.get(citation.chunk_id)
        if source is None:
            errors.append(f"citation references unknown chunk: {citation.chunk_id}")
            continue

        if _normalise_evidence_text(citation.quote) not in _normalise_evidence_text(source.text):
            errors.append(f"citation quote is not present in source chunk: {citation.label}")
        if f"[{citation.label}]" not in answer.answer:
            errors.append(f"citation label is not mapped in answer text: {citation.label}")

    answer_labels = set(re.findall(r"\[(C\d+)\]", answer.answer))
    for label in sorted(answer_labels - labels):
        errors.append(f"answer references citation without a citation object: {label}")

    return CitationValidation(valid=not errors, errors=errors)
