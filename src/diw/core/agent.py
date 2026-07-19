from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Literal, Protocol
from uuid import uuid4

from pydantic import BaseModel, Field, ValidationError
from sqlalchemy.orm import Session

from diw.core.embeddings import EmbeddingProvider
from diw.core.qa import compose_source_cited_answer, validate_citations
from diw.core.retrieval import RetrievalResult, retrieval_results_as_dicts, retrieve_chunks
from diw.db.repository import (
    complete_agent_run,
    create_agent_run,
    create_approval_request,
    get_research_record_for_tenant,
    list_tenant_document_ids,
    save_agent_run_step,
)


AGENT_POLICY_VERSION = "research-agent-policy-v1"


class ToolCall(BaseModel):
    tool_name: Literal[
        "search_documents",
        "get_research_record",
        "request_human_approval",
    ]
    arguments: dict = Field(default_factory=dict)


class FinalResponse(BaseModel):
    status: Literal["completed", "approval_pending", "refused"]


class AgentDecision(BaseModel):
    tool_call: ToolCall | None = None
    final_response: FinalResponse | None = None

    def model_post_init(self, __context) -> None:
        if (self.tool_call is None) == (self.final_response is None):
            raise ValueError("agent decision requires exactly one of tool_call or final_response")


class SearchDocumentsArgs(BaseModel):
    query: str = Field(min_length=1, max_length=2_000)
    top_k: int = Field(default=3, ge=1, le=5)
    mode: Literal["lexical", "vector", "hybrid"] = "hybrid"


class GetResearchRecordArgs(BaseModel):
    record_id: str = Field(min_length=1, max_length=36)


class RequestHumanApprovalArgs(BaseModel):
    title: str = Field(min_length=1, max_length=500)
    details: str = Field(default="", max_length=4_000)


@dataclass(frozen=True)
class AgentContext:
    query: str
    agent_run_id: str
    completed_steps: list[dict]


class AgentDecisionProvider(Protocol):
    def decide(self, context: AgentContext) -> AgentDecision:
        pass


class DeterministicResearchDecisionProvider:
    """A repeatable stand-in until ADK/Gemini owns decision selection."""

    def decide(self, context: AgentContext) -> AgentDecision:
        if not context.completed_steps:
            return AgentDecision(
                tool_call=ToolCall(
                    tool_name="search_documents",
                    arguments={"query": context.query, "top_k": 3, "mode": "hybrid"},
                )
            )

        previous_tool = context.completed_steps[-1]["tool_name"]
        if previous_tool == "search_documents" and _requests_study_task(context.query):
            return AgentDecision(
                tool_call=ToolCall(
                    tool_name="request_human_approval",
                    arguments={
                        "title": _study_task_title(context.query),
                        "details": "Follow-up task proposed after source-cited document retrieval.",
                    },
                )
            )

        if previous_tool == "request_human_approval":
            return AgentDecision(final_response=FinalResponse(status="approval_pending"))
        return AgentDecision(final_response=FinalResponse(status="completed"))


def run_bounded_agent(
    session: Session,
    *,
    tenant_id: str,
    actor_user_id: str,
    query: str,
    embedding_provider: EmbeddingProvider,
    decision_provider: AgentDecisionProvider | None = None,
    max_steps: int = 5,
    trace_id: str | None = None,
) -> dict:
    provider = decision_provider or DeterministicResearchDecisionProvider()
    run = create_agent_run(
        session,
        tenant_id=tenant_id,
        actor_user_id=actor_user_id,
        query=query,
        tool_policy_version=AGENT_POLICY_VERSION,
        max_steps=max_steps,
        trace_id=trace_id or str(uuid4()),
    )
    results: list[RetrievalResult] = []
    completed_steps: list[dict] = []
    seen_calls: set[str] = set()
    approval_request_id: str | None = None

    for sequence in range(1, max_steps + 1):
        decision = provider.decide(
            AgentContext(query=query, agent_run_id=run.id, completed_steps=completed_steps)
        )
        if decision.final_response is not None:
            return _finish_agent_run(
                session,
                run_id=run.id,
                query=query,
                status=decision.final_response.status,
                results=results,
                approval_request_id=approval_request_id,
                step_count=len(completed_steps),
            )

        assert decision.tool_call is not None
        signature = _tool_call_signature(decision.tool_call)
        if signature in seen_calls:
            _save_failed_step(
                session,
                run_id=run.id,
                sequence=sequence,
                call=decision.tool_call,
                error_code="duplicate_tool_call",
            )
            return _finish_agent_run(
                session,
                run_id=run.id,
                query=query,
                status="refused",
                results=results,
                refusal_reason="The agent stopped because it attempted the same tool call twice.",
                step_count=len(completed_steps) + 1,
            )
        seen_calls.add(signature)

        try:
            observation, results, approval_request_id = _execute_tool_call(
                session,
                tenant_id=tenant_id,
                actor_user_id=actor_user_id,
                agent_run_id=run.id,
                call=decision.tool_call,
                embedding_provider=embedding_provider,
                results=results,
            )
        except (ValidationError, ValueError) as error:
            _save_failed_step(
                session,
                run_id=run.id,
                sequence=sequence,
                call=decision.tool_call,
                error_code="invalid_tool_arguments",
            )
            return _finish_agent_run(
                session,
                run_id=run.id,
                query=query,
                status="refused",
                results=results,
                refusal_reason=f"The agent stopped because the selected tool was invalid: {error}",
                step_count=len(completed_steps) + 1,
            )

        save_agent_run_step(
            session,
            agent_run_id=run.id,
            sequence=sequence,
            tool_name=decision.tool_call.tool_name,
            tool_args=decision.tool_call.arguments,
            observation=observation,
            status="completed",
        )
        completed_steps.append({"tool_name": decision.tool_call.tool_name, "observation": observation})

    return _finish_agent_run(
        session,
        run_id=run.id,
        query=query,
        status="refused",
        results=results,
        refusal_reason="The agent stopped after reaching its tool-call limit.",
        step_count=len(completed_steps),
    )


def _execute_tool_call(
    session: Session,
    *,
    tenant_id: str,
    actor_user_id: str,
    agent_run_id: str,
    call: ToolCall,
    embedding_provider: EmbeddingProvider,
    results: list[RetrievalResult],
) -> tuple[dict, list[RetrievalResult], str | None]:
    if call.tool_name == "search_documents":
        args = SearchDocumentsArgs.model_validate(call.arguments)
        document_ids = set(list_tenant_document_ids(session, tenant_id=tenant_id))
        retrieved = retrieve_chunks(
            session,
            args.query,
            embedding_provider,
            top_k=args.top_k,
            mode=args.mode,
            document_ids=document_ids,
        )
        return (
            {
                "result_count": len(retrieved),
                "chunk_ids": [result.chunk_id for result in retrieved],
                "document_ids": sorted({result.document_id for result in retrieved}),
            },
            retrieved,
            None,
        )

    if call.tool_name == "get_research_record":
        args = GetResearchRecordArgs.model_validate(call.arguments)
        record = get_research_record_for_tenant(
            session, tenant_id=tenant_id, record_id=args.record_id
        )
        if record is None:
            return ({"record_found": False}, results, None)
        return (
            {"record_found": True, "record_id": record.id, "record_type": record.record_type},
            results,
            None,
        )

    args = RequestHumanApprovalArgs.model_validate(call.arguments)
    approval = create_approval_request(
        session,
        tenant_id=tenant_id,
        requested_by_user_id=actor_user_id,
        agent_run_id=agent_run_id,
        action_type="create_study_task",
        action_payload={"title": args.title, "details": args.details},
        idempotency_key=f"{agent_run_id}:create_study_task",
    )
    return (
        {"approval_request_id": approval.id, "status": approval.status},
        results,
        approval.id,
    )


def _finish_agent_run(
    session: Session,
    *,
    run_id: str,
    query: str,
    status: str,
    results: list[RetrievalResult],
    approval_request_id: str | None = None,
    refusal_reason: str | None = None,
    step_count: int,
) -> dict:
    answer = compose_source_cited_answer(query, results)
    validation = validate_citations(answer, results)
    if refusal_reason is not None:
        answer.answer = refusal_reason
        answer.insufficient_evidence = True
        answer.citations = []
        validation = validate_citations(answer, results)

    output = {
        "answer": answer.model_dump(),
        "citation_validation": validation.model_dump(),
        "retrieved_chunks": retrieval_results_as_dicts(results),
        "approval_request_id": approval_request_id,
    }
    run = complete_agent_run(
        session,
        agent_run_id=run_id,
        status=status,
        output=output,
        metrics={"tool_call_count": step_count, "citation_count": len(answer.citations)},
    )
    output["agent_run_id"] = run.id
    output["trace_id"] = run.trace_id
    output["status"] = run.status
    return output


def _save_failed_step(
    session: Session,
    *,
    run_id: str,
    sequence: int,
    call: ToolCall,
    error_code: str,
) -> None:
    save_agent_run_step(
        session,
        agent_run_id=run_id,
        sequence=sequence,
        tool_name=call.tool_name,
        tool_args=call.arguments,
        observation={},
        status="failed",
        error_code=error_code,
    )


def _tool_call_signature(call: ToolCall) -> str:
    return f"{call.tool_name}:{json.dumps(call.arguments, sort_keys=True, separators=(',', ':'))}"


def _requests_study_task(query: str) -> bool:
    lower = query.lower()
    return "study task" in lower or "follow-up task" in lower or "follow up task" in lower


def _study_task_title(query: str) -> str:
    compact_query = " ".join(query.split())
    return f"Follow up: {compact_query[:420]}"
