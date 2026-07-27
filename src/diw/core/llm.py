from __future__ import annotations

from dataclasses import dataclass
import importlib
import json
import os
import re
from typing import Protocol

from diw.core.qa import (
    SourceCitedAnswer,
    canonicalise_citation_quotes,
    compose_source_cited_answer,
    materialise_answer_citations,
    normalise_refusal,
    prune_unused_citations,
    validate_citations,
)
from diw.core.retrieval import RetrievalResult


PROMPT_VERSION = "source-cited-qa-v1"


@dataclass(frozen=True)
class LLMRequest:
    query: str
    evidence: list[RetrievalResult]
    prompt_version: str = PROMPT_VERSION
    validation_feedback: str | None = None


@dataclass(frozen=True)
class LLMResponse:
    raw_text: str
    model: str
    provider: str
    prompt_version: str
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0


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
    prefix = (
        "You answer questions over retrieved document chunks. Use only the evidence. Every "
        "factual sentence in answer MUST end with one or more citation labels such as [C1]. "
        "For every label used in answer, include a matching citations object with an exact quote "
        "copied from that evidence item. If the evidence is insufficient, set insufficient_evidence "
        "to true, give a short refusal explanation, and return no citations. Return strict JSON "
        "matching this schema. Keep the answer concise, use at most three citations, and keep every "
        "exact evidence quote to 180 characters or fewer.\n\n"
        f"Schema:\n{json.dumps(schema, indent=2)}\n\n"
        f"Question:\n{request.query}\n\n"
    )
    feedback = (
        f"Previous output failed validation: {request.validation_feedback}. Repair it.\n\n"
        if request.validation_feedback
        else ""
    )
    return prefix + feedback + f"Evidence:\n{json.dumps(evidence_payload(request.evidence), indent=2)}"


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

    def __init__(
        self,
        *,
        model: str = "gpt-5-mini-2025-08-07",
        api_key: str | None = None,
        max_output_tokens: int = 1200,
    ) -> None:
        self.model = model
        self.api_key = os.getenv("OPENAI_API_KEY") if api_key is None else api_key
        self.max_output_tokens = max_output_tokens
        self.temperature = None if model.startswith("gpt-5") else 0
        if max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAIChatProvider")

    def complete(self, request: LLMRequest) -> LLMResponse:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, timeout=45.0, max_retries=1)
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "Return only strict JSON. Do not include markdown fences.",
                },
                {"role": "user", "content": build_source_cited_prompt(request)},
            ],
            "max_completion_tokens": self.max_output_tokens,
            "response_format": {"type": "json_object"},
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.model.startswith("gpt-5"):
            payload["reasoning_effort"] = "minimal"
        completion = client.chat.completions.create(**payload)
        message = completion.choices[0].message
        text = message.content
        if not text:
            raise ValueError(
                "model returned no JSON content "
                f"(finish_reason={completion.choices[0].finish_reason}, "
                f"refusal={getattr(message, 'refusal', None)!r})"
            )
        usage = completion.usage
        prompt_details = getattr(usage, "prompt_tokens_details", None)
        return LLMResponse(
            raw_text=text,
            model=self.model,
            provider=self.provider,
            prompt_version=request.prompt_version,
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            cached_input_tokens=getattr(prompt_details, "cached_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )


class VertexAIGeminiProvider:
    """Structured Gemini requests on Vertex AI using Application Default Credentials."""

    provider = "vertex"

    def __init__(
        self,
        *,
        model: str | None = None,
        project: str | None = None,
        location: str | None = None,
        max_output_tokens: int = 1200,
    ) -> None:
        self.model = model or os.getenv("VERTEX_CHAT_MODEL", "")
        self.project = project or os.getenv("GOOGLE_CLOUD_PROJECT")
        self.location = location or os.getenv("GOOGLE_CLOUD_LOCATION", "global")
        self.max_output_tokens = max_output_tokens
        if not self.model:
            raise ValueError(
                "VERTEX_CHAT_MODEL or an explicit Vertex model is required; "
                "model lifecycle must be chosen deliberately"
            )
        if not self.project:
            raise ValueError("GOOGLE_CLOUD_PROJECT is required for VertexAIGeminiProvider")
        if max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")
        self._client = None

    def complete(self, request: LLMRequest) -> LLMResponse:
        if self._client is None:
            try:
                genai = importlib.import_module("google.genai")
            except ImportError as error:
                raise RuntimeError(
                    "Vertex AI support requires the 'cloud' extra: pip install -e '.[cloud]'"
                ) from error
            self._client = genai.Client(
                vertexai=True,
                project=self.project,
                location=self.location,
                http_options={"api_version": "v1"},
            )
        response = self._client.models.generate_content(
            model=self.model,
            contents=build_source_cited_prompt(request),
            config={
                "response_mime_type": "application/json",
                "max_output_tokens": self.max_output_tokens,
                "temperature": 0,
            },
        )
        raw_text = getattr(response, "text", None)
        if not raw_text:
            raise ValueError("Vertex AI returned no JSON content")
        usage = getattr(response, "usage_metadata", None)
        return LLMResponse(
            raw_text=raw_text,
            model=self.model,
            provider=self.provider,
            prompt_version=request.prompt_version,
            input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
            cached_input_tokens=getattr(usage, "cached_content_token_count", 0) or 0,
            output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
        )


def estimate_openai_cost_usd(
    *, model: str, input_tokens: int, cached_input_tokens: int, output_tokens: int
) -> float | None:
    """Estimate request cost from recorded OpenAI token usage.

    Rates are deliberately version-pinned here so a saved run remains interpretable.
    ``cached_input_tokens`` is included in ``input_tokens`` and receives its own rate.
    """
    rates = {
        "gpt-5-mini-2025-08-07": {
            "input": 0.25 / 1_000_000,
            "cached_input": 0.025 / 1_000_000,
            "output": 2.00 / 1_000_000,
        }
    }
    pricing = rates.get(model)
    if pricing is None:
        return None
    uncached_input = max(input_tokens - cached_input_tokens, 0)
    return round(
        uncached_input * pricing["input"]
        + cached_input_tokens * pricing["cached_input"]
        + output_tokens * pricing["output"],
        8,
    )


def generate_structured_answer(
    query: str,
    results: list[RetrievalResult],
    provider: LLMProvider,
    *,
    prompt_version: str = PROMPT_VERSION,
) -> SourceCitedAnswer:
    feedback = None
    failures: list[str] = []
    total_input_tokens = 0
    total_cached_input_tokens = 0
    total_output_tokens = 0
    for attempt in range(1, 4):
        response = provider.complete(
            LLMRequest(
                query=query,
                evidence=results,
                prompt_version=prompt_version,
                validation_feedback=feedback,
            )
        )
        total_input_tokens += response.input_tokens
        total_cached_input_tokens += response.cached_input_tokens
        total_output_tokens += response.output_tokens
        try:
            payload = json.loads(response.raw_text)
            answer = SourceCitedAnswer.model_validate(payload)
        except (json.JSONDecodeError, ValueError) as error:
            feedback = f"invalid JSON or schema: {error}"
            failures.append(feedback)
            continue
        answer = normalise_refusal(answer)
        if not answer.insufficient_evidence:
            answer = materialise_answer_citations(answer, results)
            answer = prune_unused_citations(answer)
            answer = canonicalise_citation_quotes(answer, results)
        validation = validate_citations(answer, results)
        if validation.valid:
            answer.model = response.model
            answer.provider = response.provider
            answer.prompt_version = response.prompt_version
            answer.input_tokens = total_input_tokens
            answer.cached_input_tokens = total_cached_input_tokens
            answer.output_tokens = total_output_tokens
            answer.completion_attempts = attempt
            return answer
        feedback = "; ".join(validation.errors)
        failures.append(feedback)
    raise ValueError(
        "model failed citation validation after 3 attempts: " + " | ".join(failures)
    )


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
