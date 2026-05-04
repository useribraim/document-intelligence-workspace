from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class SourceDocument(Base):
    __tablename__ = "source_documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    source_path: Mapped[str] = mapped_column(Text, nullable=False)
    source_name: Mapped[str] = mapped_column(String(512), nullable=False)
    source_type: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    versions: Mapped[list[DocumentVersion]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
    )


class DocumentVersion(Base):
    __tablename__ = "document_versions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    document_id: Mapped[str] = mapped_column(
        ForeignKey("source_documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    normalised_text: Mapped[str] = mapped_column(Text, nullable=False)
    normalisation_report: Mapped[dict] = mapped_column(JSON, nullable=False)
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    document: Mapped[SourceDocument] = relationship(back_populates="versions")
    chunks: Mapped[list[Chunk]] = relationship(
        back_populates="version",
        cascade="all, delete-orphan",
        order_by="Chunk.chunk_index",
    )


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    version_id: Mapped[str] = mapped_column(
        ForeignKey("document_versions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    heading_path: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    start_line: Mapped[int] = mapped_column(Integer, nullable=False)
    end_line: Mapped[int] = mapped_column(Integer, nullable=False)

    version: Mapped[DocumentVersion] = relationship(back_populates="chunks")
    embeddings: Mapped[list[ChunkEmbedding]] = relationship(
        back_populates="chunk",
        cascade="all, delete-orphan",
    )


class ChunkEmbedding(Base):
    __tablename__ = "chunk_embeddings"

    id: Mapped[str] = mapped_column(String(160), primary_key=True)
    chunk_id: Mapped[str] = mapped_column(
        ForeignKey("chunks.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    embedding_model: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    dimensions: Mapped[int] = mapped_column(Integer, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    vector: Mapped[list[float]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    chunk: Mapped[Chunk] = relationship(back_populates="embeddings")


class AIRun(Base):
    __tablename__ = "ai_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    run_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    query: Mapped[str | None] = mapped_column(Text, nullable=True)
    retrieval_mode: Mapped[str | None] = mapped_column(String(32), nullable=True)
    embedding_model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    llm_provider: Mapped[str | None] = mapped_column(String(64), nullable=True)
    llm_model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    prompt_version: Mapped[str | None] = mapped_column(String(128), nullable=True)
    retrieved_chunk_ids: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    citation_valid: Mapped[bool | None] = mapped_column(nullable=True)
    insufficient_evidence: Mapped[bool | None] = mapped_column(nullable=True)
    output: Mapped[dict] = mapped_column(JSON, nullable=False)
    metrics: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    suggestions: Mapped[list[AISuggestion]] = relationship(
        back_populates="ai_run",
        cascade="all, delete-orphan",
    )


class AISuggestion(Base):
    __tablename__ = "ai_suggestions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    ai_run_id: Mapped[str | None] = mapped_column(
        ForeignKey("ai_runs.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    suggestion_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    ai_run: Mapped[AIRun | None] = relationship(back_populates="suggestions")
    review_decisions: Mapped[list[ReviewDecision]] = relationship(
        back_populates="suggestion",
        cascade="all, delete-orphan",
        order_by="ReviewDecision.created_at",
    )


class ReviewDecision(Base):
    __tablename__ = "review_decisions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    suggestion_id: Mapped[str] = mapped_column(
        ForeignKey("ai_suggestions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    decision: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    reviewer: Mapped[str] = mapped_column(String(128), nullable=False)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    edited_payload: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    suggestion: Mapped[AISuggestion] = relationship(back_populates="review_decisions")
