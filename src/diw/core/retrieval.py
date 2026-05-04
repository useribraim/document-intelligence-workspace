from __future__ import annotations

from dataclasses import dataclass
import re

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from diw.core.embeddings import EmbeddingProvider, cosine_similarity
from diw.db.models import Chunk, ChunkEmbedding, DocumentVersion


QUERY_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
QUERY_STOPWORDS = {
    "a",
    "about",
    "an",
    "and",
    "are",
    "does",
    "document",
    "documents",
    "from",
    "how",
    "is",
    "it",
    "of",
    "say",
    "the",
    "this",
    "to",
    "what",
}


@dataclass(frozen=True)
class RetrievalResult:
    chunk_id: str
    document_id: str
    version_id: str
    chunk_index: int
    heading_path: list[str]
    text: str
    lexical_score: float
    vector_score: float
    score: float


def tokenise_query(text: str) -> set[str]:
    return {token for token in QUERY_TOKEN_RE.findall(text.lower()) if token not in QUERY_STOPWORDS}


def lexical_score(query: str, chunk: Chunk) -> float:
    query_tokens = tokenise_query(query)
    if not query_tokens:
        return 0.0

    searchable = " ".join([*chunk.heading_path, chunk.text]).lower()
    searchable_tokens = set(QUERY_TOKEN_RE.findall(searchable))
    matched = sum(1 for token in query_tokens if token in searchable_tokens)
    return matched / len(query_tokens)


def retrieve_chunks(
    session: Session,
    query: str,
    provider: EmbeddingProvider,
    *,
    top_k: int = 5,
    mode: str = "hybrid",
) -> list[RetrievalResult]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if mode not in {"lexical", "vector", "hybrid"}:
        raise ValueError("mode must be one of: lexical, vector, hybrid")

    if (
        mode in {"vector", "hybrid"}
        and session.bind is not None
        and session.bind.dialect.name == "postgresql"
    ):
        return _retrieve_chunks_postgres_pgvector(
            session,
            query,
            provider,
            top_k=top_k,
            mode=mode,
        )

    query_vector = provider.embed(query)
    rows = session.execute(
        select(Chunk, ChunkEmbedding, DocumentVersion)
        .join(ChunkEmbedding, ChunkEmbedding.chunk_id == Chunk.id)
        .join(DocumentVersion, DocumentVersion.id == Chunk.version_id)
        .where(ChunkEmbedding.embedding_model == provider.model_name)
        .where(ChunkEmbedding.dimensions == provider.dimensions)
        .where(ChunkEmbedding.content_hash == Chunk.content_hash)
    ).all()

    results: list[RetrievalResult] = []
    for chunk, embedding, version in rows:
        lex = lexical_score(query, chunk)
        vec = cosine_similarity(query_vector, embedding.vector)
        if mode == "lexical":
            score = lex
        elif mode == "vector":
            score = vec
        else:
            score = (0.55 * lex) + (0.45 * max(vec, 0.0))

        results.append(
            RetrievalResult(
                chunk_id=chunk.id,
                document_id=version.document_id,
                version_id=chunk.version_id,
                chunk_index=chunk.chunk_index,
                heading_path=list(chunk.heading_path),
                text=chunk.text,
                lexical_score=round(lex, 6),
                vector_score=round(vec, 6),
                score=round(score, 6),
            )
        )

    return sorted(results, key=lambda result: result.score, reverse=True)[:top_k]


def _retrieve_chunks_postgres_pgvector(
    session: Session,
    query: str,
    provider: EmbeddingProvider,
    *,
    top_k: int,
    mode: str,
) -> list[RetrievalResult]:
    query_vector = provider.embed(query)
    candidate_limit = top_k if mode == "vector" else max(top_k * 8, 20)
    rows = session.execute(
        text(
            """
            SELECT
                c.id AS chunk_id,
                dv.document_id AS document_id,
                c.version_id AS version_id,
                c.chunk_index AS chunk_index,
                c.heading_path AS heading_path,
                c.text AS text,
                1 - (cev.vector <=> CAST(:query_vector AS vector)) AS vector_score
            FROM chunk_embedding_vectors cev
            JOIN chunks c ON c.id = cev.chunk_id
            JOIN document_versions dv ON dv.id = c.version_id
            WHERE cev.embedding_model = :embedding_model
              AND cev.dimensions = :dimensions
              AND cev.content_hash = c.content_hash
            ORDER BY cev.vector <=> CAST(:query_vector AS vector)
            LIMIT :candidate_limit
            """
        ),
        {
            "query_vector": _vector_literal(query_vector),
            "embedding_model": provider.model_name,
            "dimensions": provider.dimensions,
            "candidate_limit": candidate_limit,
        },
    ).mappings()

    results: list[RetrievalResult] = []
    for row in rows:
        chunk = _ChunkLike(
            text=str(row["text"]),
            heading_path=list(row["heading_path"]),
        )
        lex = lexical_score(query, chunk)
        vec = float(row["vector_score"])
        if mode == "vector":
            score = vec
        else:
            score = (0.55 * lex) + (0.45 * max(vec, 0.0))

        results.append(
            RetrievalResult(
                chunk_id=str(row["chunk_id"]),
                document_id=str(row["document_id"]),
                version_id=str(row["version_id"]),
                chunk_index=int(row["chunk_index"]),
                heading_path=list(row["heading_path"]),
                text=str(row["text"]),
                lexical_score=round(lex, 6),
                vector_score=round(vec, 6),
                score=round(score, 6),
            )
        )

    return sorted(results, key=lambda result: result.score, reverse=True)[:top_k]


@dataclass(frozen=True)
class _ChunkLike:
    text: str
    heading_path: list[str]


def _vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(f"{value:.12g}" for value in vector) + "]"


def retrieval_results_as_dicts(results: list[RetrievalResult]) -> list[dict[str, object]]:
    return [
        {
            "chunk_id": result.chunk_id,
            "document_id": result.document_id,
            "version_id": result.version_id,
            "chunk_index": result.chunk_index,
            "heading_path": result.heading_path,
            "text": result.text,
            "lexical_score": result.lexical_score,
            "vector_score": result.vector_score,
            "score": result.score,
        }
        for result in results
    ]
