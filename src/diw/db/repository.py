from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from diw.core.embeddings import EmbeddingProvider
from diw.core.ingestion import IngestedDocument
from diw.db.models import (
    AIRun,
    AISuggestion,
    Chunk,
    ChunkEmbedding,
    DocumentVersion,
    ReviewDecision,
    SourceDocument,
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
    created = 0

    for chunk in chunks:
        existing = session.get(ChunkEmbedding, embedding_id(chunk.id, provider.model_name))
        if existing is not None and existing.content_hash == chunk.content_hash:
            continue

        if existing is not None:
            session.delete(existing)
            session.flush()

        vector = provider.embed(chunk.text)
        created_at = datetime.now(timezone.utc)
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
        created += 1

    return created


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
        created_at=datetime.now(timezone.utc),
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
        created_at=datetime.now(timezone.utc),
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

    now = datetime.now(timezone.utc)
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
