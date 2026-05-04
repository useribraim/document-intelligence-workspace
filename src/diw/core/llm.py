from __future__ import annotations

from dataclasses import dataclass
import json
import os
import re
from typing import Protocol

from diw.core.qa import SourceCitedAnswer, compose_source_cited_answer
from diw.core.retrieval import RetrievalResult


PROMPT_VERSION = "source-cited-qa-v1"


@dataclass(frozen=True)
class LLMRequest:
    query: str
    evidence: list[RetrievalResult]
    prompt_version: str = PROMPT_VERSION


@dataclass(frozen=True)
class LLMResponse:
    raw_text: str
    model: str
    provider: str
    prompt_version: str


class LLMProvider(Protocol):
    model: str
    provider: str

    def complete(self, request: LLMRequest) -> LLMResponse:
        pass


def evidence_payload(results: list[RetrievalResult]) -> list[dict[str, object]]:
    return [
        {
            "label": f"C{index}",
            "chunk_id": result.chunk_id,
            "document_id": result.document_id,
            "version_id": result.version_id,
            "heading_path": result.heading_path,
            "text": result.text,
            "score": result.score,
        }
        for index, result in enumerate(results, start=1)
    ]


def build_source_cited_prompt(request: LLMRequest) -> str:
    schema = {
        "query": "string",
        "answer": "string",
        "insufficient_evidence": "boolean",
        "citations": [
            {
                "label": "C1",
                "chunk_id": "string",
                "document_id": "string",
                "version_id": "string",
                "heading_path": ["string"],
                "quote": "exact quote copied from evidence text",
                "score": "number",
            }
        ],
        "extracted_fields": {"field_name": "value copied or inferred from cited evidence"},
    }
    return (
        "You answer questions over retrieved document chunks. Use only the evidence. "
        "If the evidence is insufficient, set insufficient_evidence to true and return no "
        "citations. Return strict JSON matching this schema.\n\n"
        f"Schema:\n{json.dumps(schema, indent=2)}\n\n"
        f"Question:\n{request.query}\n\n"
        f"Evidence:\n{json.dumps(evidence_payload(request.evidence), indent=2)}"
    )


class DeterministicStructuredProvider:
    model = "deterministic-structured-v1"
    provider = "local"

    def complete(self, request: LLMRequest) -> LLMResponse:
        answer = compose_source_cited_answer(request.query, request.evidence)
        answer.extracted_fields.update(_extract_fields(request.query, request.evidence))
        return LLMResponse(
            raw_text=answer.model_dump_json(),
            model=self.model,
            provider=self.provider,
            prompt_version=request.prompt_version,
        )


class OpenAIChatProvider:
    provider = "openai"

    def __init__(self, *, model: str = "gpt-4.1-mini", api_key: str | None = None) -> None:
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAIChatProvider")

    def complete(self, request: LLMRequest) -> LLMResponse:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key)
        completion = client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "Return only strict JSON. Do not include markdown fences.",
                },
                {"role": "user", "content": build_source_cited_prompt(request)},
            ],
            temperature=0,
        )
        text = completion.choices[0].message.content or "{}"
        return LLMResponse(
            raw_text=text,
            model=self.model,
            provider=self.provider,
            prompt_version=request.prompt_version,
        )


def generate_structured_answer(
    query: str,
    results: list[RetrievalResult],
    provider: LLMProvider,
    *,
    prompt_version: str = PROMPT_VERSION,
) -> SourceCitedAnswer:
    response = provider.complete(
        LLMRequest(query=query, evidence=results, prompt_version=prompt_version)
    )
    payload = json.loads(response.raw_text)
    answer = SourceCitedAnswer.model_validate(payload)
    answer.model = response.model
    answer.provider = response.provider
    answer.prompt_version = response.prompt_version
    return answer


def _extract_fields(query: str, results: list[RetrievalResult]) -> dict[str, str]:
    text = "\n".join(result.text for result in results)
    requested = _requested_fields(query)
    extracted: dict[str, str] = {}
    for field in requested:
        value = _extract_field_value(field, text)
        if value:
            extracted[field] = value
    return extracted


def _requested_fields(query: str) -> list[str]:
    lower_query = query.lower()
    known_fields = [
        "method",
        "dataset",
        "metric",
        "limitation",
    ]
    return [field for field in known_fields if field.replace("_", " ") in lower_query]


def _extract_field_value(field: str, text: str) -> str | None:
    label = field.replace("_", r"[_\s-]")
    pattern = re.compile(rf"{label}\s*:\s*(.+)", re.IGNORECASE)
    match = pattern.search(text)
    if not match:
        return None
    return match.group(1).strip().split("\n", 1)[0].strip()
