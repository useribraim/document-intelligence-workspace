from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from diw.core.embeddings import EmbeddingProvider, embed_documents
from diw.core.ingestion import IngestedDocument
from diw.db.models import (
    AgentRun,
    AgentRunStep,
    AIRun,
    AISuggestion,
    ApprovalRequest,
    Chunk,
    ChunkEmbedding,
    DocumentVersion,
    ResearchRecord,
    ReviewDecision,
    SourceDocument,
    StudyTask,
    Tenant,
    TenantDocumentAccess,
    WorkspaceUser,
)


def _parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value)


def chunk_id(version_id: str, chunk_index: int) -> str:
    return f"{version_id}:{chunk_index}"


def embedding_id(chunk_id_value: str, embedding_model: str) -> str:
    return f"{chunk_id_value}:{embedding_model}"


def save_ingested_document(session: Session, document: IngestedDocument) -> None:
    existing_document = session.get(SourceDocument, document.document_id)
    if existing_document is None:
        session.add(
            SourceDocument(
                id=document.document_id,
                source_path=document.source_path,
                source_name=document.source_name,
                source_type=document.source_type,
                created_at=_parse_datetime(document.ingested_at),
            )
        )

    existing_version = session.get(DocumentVersion, document.version_id)
    if existing_version is not None:
        return

    version = DocumentVersion(
        id=document.version_id,
        document_id=document.document_id,
        content_hash=document.content_hash,
        normalised_text=document.normalised_text,
        normalisation_report=asdict(document.normalisation_report),
        ingested_at=_parse_datetime(document.ingested_at),
    )
    session.add(version)

    for chunk in document.chunks:
        session.add(
            Chunk(
                id=chunk_id(document.version_id, int(chunk["chunk_index"])),
                version_id=document.version_id,
                chunk_index=int(chunk["chunk_index"]),
                text=str(chunk["text"]),
                heading_path=list(chunk["heading_path"]),
                content_hash=str(chunk["content_hash"]),
                start_line=int(chunk["start_line"]),
                end_line=int(chunk["end_line"]),
            )
        )


def count_documents(session: Session) -> int:
    return len(session.scalars(select(SourceDocument.id)).all())


def count_versions(session: Session) -> int:
    return len(session.scalars(select(DocumentVersion.id)).all())


def count_chunks(session: Session) -> int:
    return len(session.scalars(select(Chunk.id)).all())


def count_embeddings(session: Session) -> int:
    return len(session.scalars(select(ChunkEmbedding.id)).all())


def count_ai_runs(session: Session) -> int:
    return len(session.scalars(select(AIRun.id)).all())


def count_ai_suggestions(session: Session) -> int:
    return len(session.scalars(select(AISuggestion.id)).all())


def count_review_decisions(session: Session) -> int:
    return len(session.scalars(select(ReviewDecision.id)).all())


def create_tenant(session: Session, *, slug: str, name: str) -> Tenant:
    existing = session.scalar(select(Tenant).where(Tenant.slug == slug))
    if existing is not None:
        return existing

    tenant = Tenant(
        id=str(uuid4()),
        slug=slug,
        name=name,
        created_at=datetime.now(UTC),
    )
    session.add(tenant)
    return tenant


def create_workspace_user(
    session: Session,
    *,
    tenant_id: str,
    subject: str,
    email: str,
    display_name: str,
    role: str,
) -> WorkspaceUser:
    if session.get(Tenant, tenant_id) is None:
        raise ValueError(f"unknown tenant_id: {tenant_id}")

    statement = select(WorkspaceUser).where(
        WorkspaceUser.tenant_id == tenant_id,
        WorkspaceUser.subject == subject,
    )
    existing = session.scalar(statement)
    if existing is not None:
        return existing

    user = WorkspaceUser(
        id=str(uuid4()),
        tenant_id=tenant_id,
        subject=subject,
        email=email,
        display_name=display_name,
        role=role,
        created_at=datetime.now(UTC),
    )
    session.add(user)
    return user


def save_research_record(
    session: Session,
    *,
    tenant_id: str,
    record_type: str,
    title: str,
    payload: dict,
    created_by_user_id: str | None = None,
) -> ResearchRecord:
    if session.get(Tenant, tenant_id) is None:
        raise ValueError(f"unknown tenant_id: {tenant_id}")
    if created_by_user_id is not None:
        user = session.get(WorkspaceUser, created_by_user_id)
        if user is None or user.tenant_id != tenant_id:
            raise ValueError("research record user must belong to the same tenant")

    now = datetime.now(UTC)
    record = ResearchRecord(
        id=str(uuid4()),
        tenant_id=tenant_id,
        record_type=record_type,
        title=title,
        payload=payload,
        created_by_user_id=created_by_user_id,
        created_at=now,
        updated_at=now,
    )
    session.add(record)
    return record


def get_research_record_for_tenant(
    session: Session,
    *,
    tenant_id: str,
    record_id: str,
) -> ResearchRecord | None:
    statement = select(ResearchRecord).where(
        ResearchRecord.id == record_id,
        ResearchRecord.tenant_id == tenant_id,
    )
    return session.scalar(statement)


def grant_tenant_document_access(
    session: Session,
    *,
    tenant_id: str,
    document_id: str,
) -> TenantDocumentAccess:
    if session.get(Tenant, tenant_id) is None:
        raise ValueError(f"unknown tenant_id: {tenant_id}")
    if session.get(SourceDocument, document_id) is None:
        raise ValueError(f"unknown document_id: {document_id}")

    statement = select(TenantDocumentAccess).where(
        TenantDocumentAccess.tenant_id == tenant_id,
        TenantDocumentAccess.document_id == document_id,
    )
    existing = session.scalar(statement)
    if existing is not None:
        return existing

    access = TenantDocumentAccess(
        id=str(uuid4()),
        tenant_id=tenant_id,
        document_id=document_id,
        granted_at=datetime.now(UTC),
    )
    session.add(access)
    return access


def list_tenant_document_ids(session: Session, *, tenant_id: str) -> list[str]:
    statement = select(TenantDocumentAccess.document_id).where(
        TenantDocumentAccess.tenant_id == tenant_id
    )
    return list(session.scalars(statement).all())


def create_agent_run(
    session: Session,
    *,
    tenant_id: str,
    actor_user_id: str,
    query: str,
    tool_policy_version: str,
    max_steps: int,
    trace_id: str,
) -> AgentRun:
    user = session.get(WorkspaceUser, actor_user_id)
    if user is None or user.tenant_id != tenant_id:
        raise ValueError("agent run actor must belong to the same tenant")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")

    run = AgentRun(
        id=str(uuid4()),
        tenant_id=tenant_id,
        actor_user_id=actor_user_id,
        query=query,
        status="running",
        tool_policy_version=tool_policy_version,
        max_steps=max_steps,
        trace_id=trace_id,
        output=None,
        metrics={},
        created_at=datetime.now(UTC),
        completed_at=None,
    )
    session.add(run)
    session.flush()
    return run


def save_agent_run_step(
    session: Session,
    *,
    agent_run_id: str,
    sequence: int,
    tool_name: str,
    tool_args: dict,
    observation: dict,
    status: str,
    latency_ms: int | None = None,
    error_code: str | None = None,
    metrics: dict | None = None,
) -> AgentRunStep:
    run = session.get(AgentRun, agent_run_id)
    if run is None:
        raise ValueError(f"unknown agent_run_id: {agent_run_id}")
    if sequence < 1 or sequence > run.max_steps:
        raise ValueError("agent step sequence is outside the run step budget")

    step = AgentRunStep(
        id=str(uuid4()),
        agent_run_id=agent_run_id,
        sequence=sequence,
        tool_name=tool_name,
        tool_args=tool_args,
        observation=observation,
        status=status,
        error_code=error_code,
        latency_ms=latency_ms,
        metrics=metrics or {},
        created_at=datetime.now(UTC),
    )
    session.add(step)
    return step


def complete_agent_run(
    session: Session,
    *,
    agent_run_id: str,
    status: str,
    output: dict,
    metrics: dict | None = None,
) -> AgentRun:
    run = session.get(AgentRun, agent_run_id)
    if run is None:
        raise ValueError(f"unknown agent_run_id: {agent_run_id}")
    if run.status != "running":
        raise ValueError(f"agent run is already {run.status}")
    if status not in {"completed", "approval_pending", "refused", "failed"}:
        raise ValueError(f"unsupported agent run status: {status}")

    run.status = status
    run.output = output
    run.metrics = metrics or {}
    run.completed_at = datetime.now(UTC)
    return run


def create_approval_request(
    session: Session,
    *,
    tenant_id: str,
    requested_by_user_id: str,
    action_type: str,
    action_payload: dict,
    idempotency_key: str,
    agent_run_id: str | None = None,
) -> ApprovalRequest:
    user = session.get(WorkspaceUser, requested_by_user_id)
    if user is None or user.tenant_id != tenant_id:
        raise ValueError("approval requester must belong to the same tenant")
    if action_type != "create_study_task":
        raise ValueError(f"unsupported approval action: {action_type}")
    if agent_run_id is not None:
        run = session.get(AgentRun, agent_run_id)
        if run is None or run.tenant_id != tenant_id:
            raise ValueError("approval run must belong to the same tenant")

    statement = select(ApprovalRequest).where(
        ApprovalRequest.tenant_id == tenant_id,
        ApprovalRequest.idempotency_key == idempotency_key,
    )
    existing = session.scalar(statement)
    if existing is not None:
        return existing

    request = ApprovalRequest(
        id=str(uuid4()),
        tenant_id=tenant_id,
        requested_by_user_id=requested_by_user_id,
        agent_run_id=agent_run_id,
        action_type=action_type,
        action_payload=action_payload,
        idempotency_key=idempotency_key,
        status="pending",
        approved_by_user_id=None,
        decision_note=None,
        created_at=datetime.now(UTC),
        decided_at=None,
    )
    session.add(request)
    return request


def decide_approval_request(
    session: Session,
    *,
    tenant_id: str,
    approval_request_id: str,
    approver_user_id: str,
    decision: str,
    note: str | None = None,
) -> ApprovalRequest:
    request = session.get(ApprovalRequest, approval_request_id)
    if request is None or request.tenant_id != tenant_id:
        raise ValueError("unknown approval request for tenant")
    approver = session.get(WorkspaceUser, approver_user_id)
    if approver is None or approver.tenant_id != tenant_id or approver.role != "manager":
        raise ValueError("approval requires a manager from the same tenant")
    if request.status != "pending":
        raise ValueError(f"approval request is already {request.status}")
    if decision not in {"approve", "reject"}:
        raise ValueError(f"unsupported approval decision: {decision}")

    request.status = "approved" if decision == "approve" else "rejected"
    request.approved_by_user_id = approver_user_id
    request.decision_note = note
    request.decided_at = datetime.now(UTC)
    return request


def create_study_task_from_approval(
    session: Session,
    *,
    tenant_id: str,
    approval_request_id: str,
) -> StudyTask:
    request = session.get(ApprovalRequest, approval_request_id)
    if request is None or request.tenant_id != tenant_id:
        raise ValueError("unknown approval request for tenant")
    if request.status != "approved" or request.action_type != "create_study_task":
        raise ValueError("study task requires an approved create_study_task request")

    existing = session.scalar(
        select(StudyTask).where(StudyTask.approval_request_id == approval_request_id)
    )
    if existing is not None:
        return existing

    title = request.action_payload.get("title")
    details = request.action_payload.get("details", "")
    if not isinstance(title, str) or not title.strip() or not isinstance(details, str):
        raise ValueError("study task approval payload requires title and string details")

    task = StudyTask(
        id=str(uuid4()),
        tenant_id=tenant_id,
        approval_request_id=approval_request_id,
        title=title.strip(),
        details=details,
        status="open",
        created_at=datetime.now(UTC),
    )
    session.add(task)
    return task


def list_source_documents(session: Session) -> list[SourceDocument]:
    return list(session.scalars(select(SourceDocument).order_by(SourceDocument.created_at.desc())).all())


def list_document_versions(session: Session, document_id: str) -> list[DocumentVersion]:
    statement = (
        select(DocumentVersion)
        .where(DocumentVersion.document_id == document_id)
        .order_by(DocumentVersion.ingested_at.desc())
    )
    return list(session.scalars(statement).all())


def list_chunks_for_version(session: Session, version_id: str) -> list[Chunk]:
    statement = select(Chunk).where(Chunk.version_id == version_id).order_by(Chunk.chunk_index)
    return list(session.scalars(statement).all())


def list_ai_runs(session: Session, *, limit: int = 50) -> list[AIRun]:
    statement = select(AIRun).order_by(AIRun.created_at.desc()).limit(limit)
    return list(session.scalars(statement).all())


def count_pgvector_embeddings(session: Session) -> int:
    if session.bind is None or session.bind.dialect.name != "postgresql":
        return 0
    return int(session.execute(text("SELECT count(*) FROM chunk_embedding_vectors")).scalar_one())


def embed_missing_chunks(session: Session, provider: EmbeddingProvider) -> int:
    chunks = session.scalars(select(Chunk).order_by(Chunk.id)).all()
    missing_chunks: list[Chunk] = []
    for chunk in chunks:
        existing = session.get(ChunkEmbedding, embedding_id(chunk.id, provider.model_name))
        if existing is not None and existing.content_hash == chunk.content_hash:
            continue

        if existing is not None:
            session.delete(existing)
            session.flush()
        missing_chunks.append(chunk)

    vectors = embed_documents(provider, [chunk.text for chunk in missing_chunks])
    for chunk, vector in zip(missing_chunks, vectors):
        created_at = datetime.now(UTC)
        embedding = ChunkEmbedding(
            id=embedding_id(chunk.id, provider.model_name),
            chunk_id=chunk.id,
            embedding_model=provider.model_name,
            dimensions=provider.dimensions,
            content_hash=chunk.content_hash,
            vector=vector,
            created_at=created_at,
        )
        session.add(embedding)
        session.flush()
        _upsert_pgvector_embedding(session, embedding, vector)

    return len(missing_chunks)


def _upsert_pgvector_embedding(
    session: Session,
    embedding: ChunkEmbedding,
    vector: list[float],
) -> None:
    if session.bind is None or session.bind.dialect.name != "postgresql":
        return

    session.execute(
        text(
            """
            INSERT INTO chunk_embedding_vectors (
                embedding_id,
                chunk_id,
                embedding_model,
                dimensions,
                content_hash,
                vector,
                created_at
            )
            VALUES (
                :embedding_id,
                :chunk_id,
                :embedding_model,
                :dimensions,
                :content_hash,
                CAST(:vector AS vector),
                :created_at
            )
            ON CONFLICT (embedding_id) DO UPDATE SET
                chunk_id = EXCLUDED.chunk_id,
                embedding_model = EXCLUDED.embedding_model,
                dimensions = EXCLUDED.dimensions,
                content_hash = EXCLUDED.content_hash,
                vector = EXCLUDED.vector,
                created_at = EXCLUDED.created_at
            """
        ),
        {
            "embedding_id": embedding.id,
            "chunk_id": embedding.chunk_id,
            "embedding_model": embedding.embedding_model,
            "dimensions": embedding.dimensions,
            "content_hash": embedding.content_hash,
            "vector": _vector_literal(vector),
            "created_at": embedding.created_at,
        },
    )


def _vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(f"{value:.12g}" for value in vector) + "]"


def save_ai_run(
    session: Session,
    *,
    run_type: str,
    output: dict,
    query: str | None = None,
    retrieval_mode: str | None = None,
    embedding_model: str | None = None,
    llm_provider: str | None = None,
    llm_model: str | None = None,
    prompt_version: str | None = None,
    retrieved_chunk_ids: list[str] | None = None,
    citation_valid: bool | None = None,
    insufficient_evidence: bool | None = None,
    metrics: dict | None = None,
) -> AIRun:
    run = AIRun(
        id=str(uuid4()),
        run_type=run_type,
        query=query,
        retrieval_mode=retrieval_mode,
        embedding_model=embedding_model,
        llm_provider=llm_provider,
        llm_model=llm_model,
        prompt_version=prompt_version,
        retrieved_chunk_ids=retrieved_chunk_ids or [],
        citation_valid=citation_valid,
        insufficient_evidence=insufficient_evidence,
        output=output,
        metrics=metrics or {},
        created_at=datetime.now(UTC),
    )
    session.add(run)
    return run


def save_ai_suggestion(
    session: Session,
    *,
    suggestion_type: str,
    title: str,
    payload: dict,
    ai_run_id: str | None = None,
    status: str = "pending",
) -> AISuggestion:
    suggestion = AISuggestion(
        id=str(uuid4()),
        ai_run_id=ai_run_id,
        suggestion_type=suggestion_type,
        status=status,
        title=title,
        payload=payload,
        created_at=datetime.now(UTC),
        reviewed_at=None,
    )
    session.add(suggestion)
    return suggestion


def list_ai_suggestions(session: Session, *, status: str | None = None) -> list[AISuggestion]:
    statement = select(AISuggestion).order_by(AISuggestion.created_at.desc())
    if status is not None:
        statement = statement.where(AISuggestion.status == status)
    return list(session.scalars(statement).all())


def record_review_decision(
    session: Session,
    *,
    suggestion_id: str,
    decision: str,
    reviewer: str = "local-user",
    note: str | None = None,
    edited_payload: dict | None = None,
) -> ReviewDecision:
    status_by_decision = {
        "accept": "accepted",
        "reject": "rejected",
        "edit": "edited",
    }
    if decision not in status_by_decision:
        raise ValueError(f"unsupported review decision: {decision}")

    suggestion = session.get(AISuggestion, suggestion_id)
    if suggestion is None:
        raise ValueError(f"unknown suggestion_id: {suggestion_id}")

    now = datetime.now(UTC)
    if edited_payload is not None:
        suggestion.payload = edited_payload
    suggestion.status = status_by_decision[decision]
    suggestion.reviewed_at = now

    review = ReviewDecision(
        id=str(uuid4()),
        suggestion_id=suggestion_id,
        decision=decision,
        reviewer=reviewer,
        note=note,
        edited_payload=edited_payload,
        created_at=now,
    )
    session.add(review)
    return review
