from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from math import log

from sqlalchemy import bindparam, select, text
from sqlalchemy.orm import Session

from diw.core.embeddings import EmbeddingProvider, cosine_similarity, embed_query
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


def bm25_scores(query: str, chunks: list[Chunk | _ChunkLike]) -> list[float]:
    """Return query-normalized BM25 scores for one candidate collection.

    Normalizing within the candidate collection keeps the existing weighted
    fusion interface bounded while replacing unweighted token-set overlap with
    term-frequency, document-frequency, and length normalization.
    """

    query_tokens = tokenise_query(query)
    if not query_tokens:
        return [0.0] * len(chunks)

    token_lists = [
        QUERY_TOKEN_RE.findall(" ".join([*chunk.heading_path, chunk.text]).lower())
        for chunk in chunks
    ]
    if not token_lists:
        return []
    document_frequency = Counter(
        token for tokens in token_lists for token in set(tokens) if token in query_tokens
    )
    average_length = sum(len(tokens) for tokens in token_lists) / len(token_lists)
    k1, b = 1.2, 0.75
    raw_scores: list[float] = []
    for tokens in token_lists:
        term_frequency = Counter(tokens)
        length = max(len(tokens), 1)
        score = 0.0
        for token in query_tokens:
            frequency = term_frequency[token]
            if not frequency:
                continue
            inverse_document_frequency = log(
                1 + (len(token_lists) - document_frequency[token] + 0.5)
                / (document_frequency[token] + 0.5)
            )
            denominator = frequency + k1 * (1 - b + b * length / average_length)
            score += inverse_document_frequency * frequency * (k1 + 1) / denominator
        raw_scores.append(score)
    maximum = max(raw_scores, default=0.0)
    return [score / maximum if maximum else 0.0 for score in raw_scores]


def lexical_score(query: str, chunk: Chunk | _ChunkLike) -> float:
    """Return the one-chunk BM25-compatible lexical score.

    Retrieval uses collection-aware BM25 scores. This compatibility helper is
    retained for callers that only have one chunk.
    """

    return bm25_scores(query, [chunk])[0]


def retrieve_chunks(
    session: Session,
    query: str,
    provider: EmbeddingProvider,
    *,
    top_k: int = 5,
    mode: str = "hybrid",
    reranker: str = "weighted",
    document_ids: set[str] | None = None,
) -> list[RetrievalResult]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if mode not in {"lexical", "vector", "hybrid"}:
        raise ValueError("mode must be one of: lexical, vector, hybrid")
    if reranker not in {"weighted", "rrf"}:
        raise ValueError("reranker must be one of: weighted, rrf")
    if document_ids is not None and not document_ids:
        return []

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
            reranker=reranker,
            document_ids=document_ids,
        )

    query_vector = embed_query(provider, query)
    statement = (
        select(Chunk, ChunkEmbedding, DocumentVersion)
        .join(ChunkEmbedding, ChunkEmbedding.chunk_id == Chunk.id)
        .join(DocumentVersion, DocumentVersion.id == Chunk.version_id)
        .where(ChunkEmbedding.embedding_model == provider.model_name)
        .where(ChunkEmbedding.dimensions == provider.dimensions)
        .where(ChunkEmbedding.content_hash == Chunk.content_hash)
    )
    if document_ids is not None:
        statement = statement.where(DocumentVersion.document_id.in_(document_ids))
    rows = session.execute(statement).all()

    lexical_scores = bm25_scores(query, [chunk for chunk, _, _ in rows])
    results: list[RetrievalResult] = []
    for (chunk, embedding, version), lex in zip(rows, lexical_scores, strict=True):
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

    return _rank_results(results, top_k=top_k, mode=mode, reranker=reranker)


def _retrieve_chunks_postgres_pgvector(
    session: Session,
    query: str,
    provider: EmbeddingProvider,
    *,
    top_k: int,
    mode: str,
    reranker: str,
    document_ids: set[str] | None,
) -> list[RetrievalResult]:
    query_vector = embed_query(provider, query)
    candidate_limit = top_k if mode == "vector" else max(top_k * 8, 20)
    document_filter = ""
    if document_ids is not None:
        document_filter = "AND dv.document_id IN :document_ids"
    statement = text(
        f"""
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
              {document_filter}
            ORDER BY cev.vector <=> CAST(:query_vector AS vector)
            LIMIT :candidate_limit
            """
    )
    if document_ids is not None:
        statement = statement.bindparams(bindparam("document_ids", expanding=True))
    params = {
        "query_vector": _vector_literal(query_vector),
        "embedding_model": provider.model_name,
        "dimensions": provider.dimensions,
        "candidate_limit": candidate_limit,
    }
    if document_ids is not None:
        params["document_ids"] = sorted(document_ids)
    rows = session.execute(
        statement,
        params,
    ).mappings()

    row_values = list(rows)
    chunks = [
        _ChunkLike(
            text=str(row["text"]),
            heading_path=list(row["heading_path"]),
        )
        for row in row_values
    ]
    lexical_scores = bm25_scores(query, chunks)
    results: list[RetrievalResult] = []
    for row, chunk, lex in zip(row_values, chunks, lexical_scores, strict=True):
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

    return _rank_results(results, top_k=top_k, mode=mode, reranker=reranker)


def _rank_results(
    results: list[RetrievalResult],
    *,
    top_k: int,
    mode: str,
    reranker: str,
) -> list[RetrievalResult]:
    if mode != "hybrid" or reranker == "weighted":
        return sorted(
            results,
            key=lambda result: (-result.score, result.chunk_id),
        )[:top_k]

    lexical_ranks = _tie_aware_ranks(results, "lexical_score")
    vector_ranks = _tie_aware_ranks(results, "vector_score")
    rrf_k = 60
    maximum_rrf_score = 2 / (rrf_k + 1)
    fused: list[RetrievalResult] = []
    for result in results:
        raw_score = (1 / (rrf_k + lexical_ranks[result.chunk_id])) + (
            1 / (rrf_k + vector_ranks[result.chunk_id])
        )
        score = raw_score / maximum_rrf_score
        fused.append(
            RetrievalResult(
                chunk_id=result.chunk_id,
                document_id=result.document_id,
                version_id=result.version_id,
                chunk_index=result.chunk_index,
                heading_path=result.heading_path,
                text=result.text,
                lexical_score=result.lexical_score,
                vector_score=result.vector_score,
                score=round(score, 8),
            )
        )
    return sorted(fused, key=lambda result: (-result.score, result.chunk_id))[:top_k]


def _tie_aware_ranks(
    results: list[RetrievalResult], score_field: str
) -> dict[str, float]:
    """Assign average ranks so equal scores do not inherit chunk-ID ordering."""

    ordered = sorted(results, key=lambda result: -float(getattr(result, score_field)))
    ranks: dict[str, float] = {}
    start = 0
    while start < len(ordered):
        score = getattr(ordered[start], score_field)
        end = start + 1
        while end < len(ordered) and getattr(ordered[end], score_field) == score:
            end += 1
        average_rank = ((start + 1) + end) / 2
        for result in ordered[start:end]:
            ranks[result.chunk_id] = average_rank
        start = end
    return ranks


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
