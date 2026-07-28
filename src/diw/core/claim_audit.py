"""Claim-to-evidence audit primitives.

The verifier is deliberately deterministic: it checks exact evidence-span
membership and token coverage, rather than presenting an LLM judgement as
ground truth. Human annotation remains the authoritative evaluation path.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable

from pydantic import BaseModel, Field

from diw.core.qa import EvidenceCitation, SourceCitedAnswer
from diw.core.retrieval import RetrievalResult, tokenise_query

AUDIT_PROMPT_VERSION = "claim-evidence-audit-v1"
_CITATION_MARKER = re.compile(r"\s*\[C\d+\]\s*")
_CITED_SEGMENT = re.compile(r"(?s)(.*?)(?:\s*)\[(C\d+)\]")


class AtomicClaim(BaseModel):
    claim_id: str
    text: str
    citation_ids: list[str] = Field(default_factory=list)


class ClaimCitationAssessment(BaseModel):
    claim_id: str
    claim_text: str
    citation_id: str | None = None
    evidence_span: str | None = None
    quote_alignment: str | None = None
    source_exists: bool
    citation_relevant: str
    support_label: str
    support_rationale: str
    answer_completeness: str | None = None
    refusal_appropriate: bool | None = None


def config_hash(payload: object) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def extract_atomic_claims(answer: SourceCitedAnswer, *, question_id: str) -> list[AtomicClaim]:
    """Split deterministic answers into sentence-level claim units and mappings."""
    if answer.insufficient_evidence:
        return []
    claims: list[AtomicClaim] = []
    for match in _CITED_SEGMENT.finditer(answer.answer.strip()):
        text = match.group(1).strip()
        label = match.group(2)
        if not text:
            continue
        claims.append(
            AtomicClaim(
                claim_id=f"{question_id}_c{len(claims) + 1}",
                text=_CITATION_MARKER.sub(" ", text).strip(),
                citation_ids=[label],
            )
        )
    if not claims and answer.answer.strip():
        claims.append(
            AtomicClaim(
                claim_id=f"{question_id}_c1",
                text=answer.answer.strip(),
                citation_ids=[],
            )
        )
    return claims


def _coverage(claim_text: str, evidence: str) -> float:
    tokens = tokenise_query(claim_text)
    if not tokens:
        return 0.0
    return len(tokens & tokenise_query(evidence)) / len(tokens)


def assess_claims(
    claims: Iterable[AtomicClaim],
    citations: Iterable[EvidenceCitation],
    results: Iterable[RetrievalResult],
    *,
    expected_evidence_status: str | None = None,
) -> list[ClaimCitationAssessment]:
    citations_by_label = {citation.label: citation for citation in citations}
    chunks = {result.chunk_id: result for result in results}
    assessments: list[ClaimCitationAssessment] = []
    for claim in claims:
        if not claim.citation_ids:
            assessments.append(ClaimCitationAssessment(claim_id=claim.claim_id, claim_text=claim.text, source_exists=False, citation_relevant="no", support_label="unsupported", support_rationale="No citation is mapped to this atomic claim."))
            continue
        for label in claim.citation_ids:
            citation = citations_by_label.get(label)
            source = chunks.get(citation.chunk_id) if citation else None
            exists = citation is not None and source is not None
            evidence = citation.quote if citation else None
            coverage = _coverage(claim.text, evidence or "")
            relevance = "yes" if coverage >= 0.25 else "no"
            if not exists:
                support, rationale = "unsupported", "Mapped citation is absent from retrieved evidence."
            elif coverage >= 0.85:
                support, rationale = "fully_supported", "All or nearly all material claim tokens occur in the cited evidence span."
            elif coverage >= 0.4:
                support, rationale = "partially_supported", "The evidence overlaps the claim but omits material detail or qualification."
            else:
                support, rationale = "unsupported", "The citation is real but does not provide enough claim-level overlap."
            assessments.append(ClaimCitationAssessment(
                claim_id=claim.claim_id, claim_text=claim.text, citation_id=label, evidence_span=evidence,
                quote_alignment=citation.quote_alignment if citation else None,
                source_exists=exists, citation_relevant=relevance, support_label=support,
                support_rationale=rationale, answer_completeness=None,
                # Whether declining was appropriate is an answer-level judgement. It
                # cannot be inferred from token overlap on a cited span.
                refusal_appropriate=None,
            ))
    return assessments


def summarise_claim_audit(assessments: list[ClaimCitationAssessment]) -> dict[str, float | int]:
    """Summarise deterministic token-overlap labels without calling them human support.

    The underlying labels remain useful to the conservative evidence gate, but an
    extractive generator can score highly simply by copying its selected span.  These
    fields are therefore diagnostics for that gate, not citation-validity or answer-
    quality metrics and must not be compared as human-evaluation results.
    """
    total = len(assessments)
    counts = Counter(item.support_label for item in assessments)
    labels = ("fully_supported", "partially_supported", "unsupported", "contradicted")
    return {
        "claim_citation_pairs": total,
        **{
            f"automated_overlap_{label}_rate": (
                round(counts[label] / total, 4) if total else 0.0
            )
            for label in labels
        },
    }


def apply_claim_verification_gate(
    answer: SourceCitedAnswer,
    claims: list[AtomicClaim],
    assessments: list[ClaimCitationAssessment],
    *,
    policy: str = "strict",
) -> SourceCitedAnswer:
    """Retain claims according to an explicit deterministic support policy."""
    permitted_labels = {
        "strict": {"fully_supported"},
        "supported": {"fully_supported", "partially_supported"},
    }.get(policy)
    if permitted_labels is None:
        raise ValueError(f"unknown verification gate policy: {policy}")
    if answer.insufficient_evidence:
        return answer
    assessments_by_claim: dict[str, list[ClaimCitationAssessment]] = {}
    for assessment in assessments:
        assessments_by_claim.setdefault(assessment.claim_id, []).append(assessment)
    retained = [
        claim
        for claim in claims
        if claim.citation_ids
        and assessments_by_claim.get(claim.claim_id)
        and all(item.support_label in permitted_labels for item in assessments_by_claim[claim.claim_id])
    ]
    if not retained:
        return SourceCitedAnswer(
            query=answer.query,
            answer="Insufficient evidence in the retrieved document chunks to answer this question reliably.",
            insufficient_evidence=True,
            model=answer.model,
            provider=answer.provider,
            prompt_version=answer.prompt_version,
            input_tokens=answer.input_tokens,
            cached_input_tokens=answer.cached_input_tokens,
            output_tokens=answer.output_tokens,
            completion_attempts=answer.completion_attempts,
        )
    retained_labels = {label for claim in retained for label in claim.citation_ids}
    rendered = " ".join(
        f"{claim.text} {' '.join(f'[{label}]' for label in claim.citation_ids)}"
        for claim in retained
    )
    return answer.model_copy(
        update={
            "answer": rendered,
            "citations": [item for item in answer.citations if item.label in retained_labels],
        }
    )


def apply_evidence_repair(
    answer: SourceCitedAnswer,
    claims: list[AtomicClaim],
    assessments: list[ClaimCitationAssessment],
) -> tuple[SourceCitedAnswer, list[dict[str, str]]]:
    """Rewrite partial claims to exact cited spans and remove unsupported claims.

    This is intentionally conservative. A fully-supported claim remains in its
    readable generated form; a partially-supported claim becomes the exact
    evidence span selected for it. If no defensible claim remains, the result
    is a transparent insufficiency response.
    """
    if answer.insufficient_evidence:
        return answer, [{"action": "unchanged_refusal", "reason": "generator_refusal"}]
    citations_by_label = {citation.label: citation for citation in answer.citations}
    assessments_by_claim: dict[str, list[ClaimCitationAssessment]] = {}
    for assessment in assessments:
        assessments_by_claim.setdefault(assessment.claim_id, []).append(assessment)

    rendered: list[str] = []
    retained_labels: set[str] = set()
    actions: list[dict[str, str]] = []
    for claim in claims:
        claim_assessments = assessments_by_claim.get(claim.claim_id, [])
        labels = claim.citation_ids
        if not labels or not claim_assessments:
            actions.append({"claim_id": claim.claim_id, "action": "removed", "reason": "uncited"})
            continue
        if all(item.support_label == "fully_supported" for item in claim_assessments):
            rendered.append(f"{claim.text} {' '.join(f'[{label}]' for label in labels)}")
            retained_labels.update(labels)
            actions.append({"claim_id": claim.claim_id, "action": "kept", "reason": "fully_supported"})
            continue
        if all(item.support_label == "partially_supported" for item in claim_assessments):
            citation = citations_by_label.get(labels[0])
            if citation and citation.quote.strip():
                rendered.append(f"{citation.quote.strip()} [{citation.label}]")
                retained_labels.add(citation.label)
                actions.append(
                    {
                        "claim_id": claim.claim_id,
                        "action": "rewritten_to_evidence",
                        "reason": "partially_supported",
                    }
                )
                continue
        actions.append(
            {"claim_id": claim.claim_id, "action": "removed", "reason": "unsupported_or_mixed"}
        )

    if not rendered:
        return (
            SourceCitedAnswer(
                query=answer.query,
                answer=(
                    "Insufficient evidence in the retrieved document chunks to answer this "
                    "question reliably after claim-level evidence repair."
                ),
                insufficient_evidence=True,
                model=answer.model,
                provider=answer.provider,
                prompt_version=answer.prompt_version,
                input_tokens=answer.input_tokens,
                cached_input_tokens=answer.cached_input_tokens,
                output_tokens=answer.output_tokens,
                completion_attempts=answer.completion_attempts,
            ),
            actions,
        )
    return (
        answer.model_copy(
            update={
                "answer": " ".join(rendered),
                "citations": [
                    citation for citation in answer.citations if citation.label in retained_labels
                ],
            }
        ),
        actions,
    )


def retrieval_gold_metrics(
    results: Iterable[RetrievalResult], gold_chunk_ids: Iterable[str]
) -> tuple[float | None, float | None]:
    """Return gold-chunk Recall@k and reciprocal rank for a fixed retrieval list."""
    gold = set(gold_chunk_ids)
    if not gold:
        return None, None
    retrieved = list(results)
    hits = [rank for rank, item in enumerate(retrieved, start=1) if item.chunk_id in gold]
    recall = round(sum(item.chunk_id in gold for item in retrieved) / len(gold), 4)
    return recall, round(1 / hits[0], 4) if hits else 0.0


def cohen_kappa(first: list[str], second: list[str]) -> float | None:
    """Unweighted Cohen's kappa for aligned categorical labels."""
    if len(first) != len(second) or not first:
        return None
    observed = sum(a == b for a, b in zip(first, second)) / len(first)
    first_counts, second_counts = Counter(first), Counter(second)
    expected = sum(
        (first_counts[label] / len(first)) * (second_counts[label] / len(second))
        for label in set(first_counts) | set(second_counts)
    )
    if expected == 1:
        return 1.0 if observed == 1 else None
    return round((observed - expected) / (1 - expected), 4)
