"""Google ADK multi-agent orchestration with auditable request economics.

The existing deterministic agent remains the fail-safe write workflow.  This
module adds a separate read-only ADK path in which a coordinator delegates to
retrieval and citation-verification specialists through ``AgentTool``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

ADK_APP_NAME = "diw_adk_research"
ADK_WORKFLOW_VERSION = "google-adk-hierarchical-v1"
DEFAULT_ADK_MODEL = "gemini-2.5-flash"
VERTEX_PRICING_SNAPSHOT = "2026-07-28"
RESULT_PREFIX = "DIW_ADK_RESULT="

SearchDocuments = Callable[[str, int], dict[str, Any]]


class RetrievalRequest(BaseModel):
    query: str = Field(min_length=2, max_length=2_000)
    top_k: int = Field(default=5, ge=1, le=5)


class VerificationRequest(BaseModel):
    claim: str = Field(min_length=1, max_length=4_000)
    citation_quote: str = Field(min_length=1, max_length=2_000)
    evidence_text: str = Field(min_length=1, max_length=8_000)


@dataclass(frozen=True)
class ModelCallEconomics:
    agent: str
    model: str
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    thinking_tokens: int
    latency_ms: float
    output_tokens_per_second: float | None
    estimated_cost_usd: float | None


class ADKEconomicsRecorder:
    """Collect model usage through ADK callbacks, including delegated agents."""

    def __init__(self) -> None:
        self._started: dict[tuple[str, str], float] = {}
        self.calls: list[ModelCallEconomics] = []

    def before_model(self, callback_context, llm_request) -> None:
        del llm_request
        self._started[
            (callback_context.agent_name, callback_context.invocation_id)
        ] = perf_counter()

    def after_model(self, callback_context, llm_response) -> None:
        started = self._started.pop(
            (callback_context.agent_name, callback_context.invocation_id),
            perf_counter(),
        )
        latency_ms = round((perf_counter() - started) * 1_000, 2)
        usage = llm_response.usage_metadata
        input_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
        cached_input_tokens = int(getattr(usage, "cached_content_token_count", 0) or 0)
        output_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
        thinking_tokens = int(getattr(usage, "thoughts_token_count", 0) or 0)
        output_rate = None
        if output_tokens and latency_ms > 0:
            output_rate = round(output_tokens / (latency_ms / 1_000), 2)
        model = llm_response.model_version or DEFAULT_ADK_MODEL
        self.calls.append(
            ModelCallEconomics(
                agent=callback_context.agent_name,
                model=model,
                input_tokens=input_tokens,
                cached_input_tokens=cached_input_tokens,
                output_tokens=output_tokens,
                thinking_tokens=thinking_tokens,
                latency_ms=latency_ms,
                output_tokens_per_second=output_rate,
                estimated_cost_usd=estimate_vertex_cost_usd(
                    model=model,
                    input_tokens=input_tokens,
                    cached_input_tokens=cached_input_tokens,
                    output_tokens=output_tokens,
                    thinking_tokens=thinking_tokens,
                ),
            )
        )

    def summary(self, total_latency_ms: float) -> dict[str, Any]:
        calls = [asdict(call) for call in self.calls]
        known_costs = [
            call.estimated_cost_usd
            for call in self.calls
            if call.estimated_cost_usd is not None
        ]
        total_output_tokens = sum(call.output_tokens for call in self.calls)
        generation_seconds = sum(call.latency_ms for call in self.calls) / 1_000
        return {
            "pricing_snapshot": VERTEX_PRICING_SNAPSHOT,
            "model_calls": calls,
            "model_call_count": len(calls),
            "input_tokens": sum(call.input_tokens for call in self.calls),
            "cached_input_tokens": sum(call.cached_input_tokens for call in self.calls),
            "output_tokens": total_output_tokens,
            "thinking_tokens": sum(call.thinking_tokens for call in self.calls),
            "total_latency_ms": round(total_latency_ms, 2),
            "aggregate_output_tokens_per_second": (
                round(total_output_tokens / generation_seconds, 2)
                if total_output_tokens and generation_seconds > 0
                else None
            ),
            "estimated_cost_usd": (
                round(sum(known_costs), 8)
                if len(known_costs) == len(self.calls) and self.calls
                else None
            ),
        }


def estimate_vertex_cost_usd(
    *,
    model: str,
    input_tokens: int,
    cached_input_tokens: int,
    output_tokens: int,
    thinking_tokens: int = 0,
) -> float | None:
    """Estimate Vertex request cost using a version-pinned pricing snapshot.

    Gemini 2.5 Flash prices below are per one million tokens.  The estimate
    intentionally returns ``None`` for unknown model versions rather than
    silently applying the wrong rate.
    """

    normalised = model.lower()
    if "gemini-2.5-flash" not in normalised or "lite" in normalised:
        return None
    uncached_input = max(input_tokens - cached_input_tokens, 0)
    non_thinking_output = max(output_tokens - thinking_tokens, 0)
    return round(
        (uncached_input * 0.15 / 1_000_000)
        + (cached_input_tokens * 0.0375 / 1_000_000)
        + (non_thinking_output * 0.60 / 1_000_000)
        + (thinking_tokens * 3.50 / 1_000_000),
        8,
    )


def exact_quote_verifier(
    claim: str,
    citation_quote: str,
    evidence_text: str,
) -> dict[str, Any]:
    """Check whether the proposed citation is an exact span of retrieved evidence."""

    quote = citation_quote.strip()
    return {
        "claim": claim,
        "citation_quote": quote,
        "exact_quote_valid": bool(quote) and quote in evidence_text,
        "limitation": (
            "Exact-span validation does not establish semantic support; "
            "human calibration remains separate."
        ),
    }


def build_adk_research_system(
    *,
    search_documents: SearchDocuments,
    model: str = DEFAULT_ADK_MODEL,
    recorder: ADKEconomicsRecorder | None = None,
):
    """Build the ADK coordinator and two delegated specialist agents."""

    try:
        from google.adk.agents import Agent
        from google.adk.tools import AgentTool
    except ImportError as error:
        raise RuntimeError(
            "Google ADK support requires: pip install -e '.[adk]'"
        ) from error

    economics = recorder or ADKEconomicsRecorder()

    def search_documents_tool(query: str, top_k: int = 5) -> dict[str, Any]:
        """Retrieve tenant-safe or bundled read-only evidence for a research question."""

        request = RetrievalRequest(query=query, top_k=top_k)
        return search_documents(request.query, request.top_k)

    def verify_citation_tool(
        claim: str,
        citation_quote: str,
        evidence_text: str,
    ) -> dict[str, Any]:
        """Validate that a citation quote occurs exactly in the retrieved evidence."""

        request = VerificationRequest(
            claim=claim,
            citation_quote=citation_quote,
            evidence_text=evidence_text,
        )
        return exact_quote_verifier(
            request.claim,
            request.citation_quote,
            request.evidence_text,
        )

    shared_callbacks = {
        "before_model_callback": economics.before_model,
        "after_model_callback": economics.after_model,
    }
    retrieval_agent = Agent(
        name="retrieval_specialist",
        model=model,
        description=(
            "Retrieves evidence chunks for a research question and reports their exact text "
            "without answering beyond the evidence."
        ),
        instruction=(
            "You are the retrieval specialist. Always call search_documents_tool exactly once. "
            "Return the query and retrieved chunks, including chunk_id, heading_path, exact text, "
            "and retrieval scores. Do not invent or paraphrase evidence."
        ),
        tools=[search_documents_tool],
        **shared_callbacks,
    )
    verification_agent = Agent(
        name="citation_verification_specialist",
        model=model,
        description=(
            "Checks a proposed claim and citation against retrieved evidence before the "
            "coordinator may return it."
        ),
        instruction=(
            "You are the citation-verification specialist. Call verify_citation_tool for every "
            "proposed claim/citation pair. Approve only exact quotes. If validation fails, return "
            "REFUSED and explain that the evidence gate failed. Never credit outside knowledge."
        ),
        tools=[verify_citation_tool],
        **shared_callbacks,
    )
    coordinator = Agent(
        name="research_coordinator",
        model=model,
        description="Coordinates retrieval and citation verification for evidence-grounded answers.",
        instruction=(
            "Use a ReAct-style cycle: plan briefly, act through a specialist, inspect the "
            "observation, then decide the next action. First delegate the user's question to "
            "retrieval_specialist. Draft a concise answer only from returned exact text. Then "
            "delegate every proposed claim, quote, and evidence text to "
            "citation_verification_specialist. Return the answer only after the verifier reports "
            "exact_quote_valid=true. Otherwise refuse. Never call write tools; none are exposed."
        ),
        tools=[
            AgentTool(agent=retrieval_agent),
            AgentTool(agent=verification_agent),
        ],
        **shared_callbacks,
    )
    return coordinator, economics


async def run_adk_research(
    query: str,
    *,
    search_documents: SearchDocuments,
    model: str = DEFAULT_ADK_MODEL,
) -> dict[str, Any]:
    """Run one hierarchical ADK research request and return its trace/economics."""

    try:
        from google.adk.runners import InMemoryRunner
    except ImportError as error:
        raise RuntimeError(
            "Google ADK support requires: pip install -e '.[adk]'"
        ) from error

    clean_query = RetrievalRequest(query=query).query
    recorder = ADKEconomicsRecorder()
    coordinator, recorder = build_adk_research_system(
        search_documents=search_documents,
        model=model,
        recorder=recorder,
    )
    runner = InMemoryRunner(agent=coordinator, app_name=ADK_APP_NAME)
    started = perf_counter()
    events = await runner.run_debug(
        clean_query,
        user_id="diw-adk-user",
        session_id=str(uuid4()),
        quiet=True,
    )
    total_latency_ms = (perf_counter() - started) * 1_000
    final_text = _last_model_text(events)
    tool_calls = _tool_call_trace(events)
    await runner.close()
    return {
        "schema_version": ADK_WORKFLOW_VERSION,
        "query": clean_query,
        "model": model,
        "answer": final_text,
        "delegations": tool_calls,
        "specialists": [
            "retrieval_specialist",
            "citation_verification_specialist",
        ],
        "write_tools_available": False,
        "economics": recorder.summary(total_latency_ms),
    }


def _last_model_text(events: list[Any]) -> str:
    for event in reversed(events):
        content = getattr(event, "content", None)
        parts = getattr(content, "parts", None) or []
        text = "\n".join(
            part.text for part in parts if getattr(part, "text", None) and not part.thought
        )
        if text:
            return text
    return ""


def _tool_call_trace(events: list[Any]) -> list[dict[str, Any]]:
    calls = []
    for event in events:
        content = getattr(event, "content", None)
        for part in getattr(content, "parts", None) or []:
            function_call = getattr(part, "function_call", None)
            if function_call:
                calls.append(
                    {
                        "agent": getattr(event, "author", None),
                        "tool": function_call.name,
                        "arguments": function_call.args,
                    }
                )
    return calls


def _bundled_demo_search(corpus_dir: Path) -> SearchDocuments:
    def search(query: str, top_k: int) -> dict[str, Any]:
        from diw.api import _retrieve_public_demo_chunks
        from diw.core.retrieval import retrieval_results_as_dicts

        results, document_count = _retrieve_public_demo_chunks(
            query,
            corpus_dir=corpus_dir,
            top_k=top_k,
        )
        return {
            "query": query,
            "corpus": "bundled synthetic ML-paper excerpts",
            "document_count": document_count,
            "chunks": retrieval_results_as_dicts(results),
        }

    return search


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Google ADK multi-agent research slice.")
    parser.add_argument("query")
    parser.add_argument(
        "--model",
        default=os.getenv("ADK_MODEL", DEFAULT_ADK_MODEL),
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=Path("data/demo/raw"),
    )
    parser.add_argument("--out", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    payload = asyncio.run(
        run_adk_research(
            args.query,
            search_documents=_bundled_demo_search(args.corpus_dir),
            model=args.model,
        )
    )
    rendered = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")
    print(RESULT_PREFIX + json.dumps(payload, separators=(",", ":"), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
