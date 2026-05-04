# RAG Build Plan

## Definition of Done

The project can claim "RAG" when this loop exists end to end:

```text
chunk -> embed -> store vector -> retrieve lexical + semantic candidates
  -> hybrid score -> select evidence -> generate cited answer
  -> validate citations -> log provenance -> evaluate retrieval
```

## Implementation Steps

1. Add Postgres schema and Alembic migrations.
2. Store chunks with source document version, heading path, content hash, and line offsets. Done in v0.2.
3. Generate deterministic local embeddings for chunks. Done in v0.3.
4. Store embedding model and chunk hash next to vectors. Done in v0.3.
5. Implement lexical retrieval. Done in v0.3 as application-level term matching.
6. Implement vector retrieval with cosine similarity. Done in v0.3.
7. Merge candidates with hybrid scoring. Done in v0.3.
8. Enable `vector` extension through pgvector.
9. Replace local vector scanning with pgvector-backed vector search.
10. Add evidence thresholding and insufficient-evidence fallback.
11. Generate answers with citations over retrieved chunks only.
12. Validate that cited chunk IDs exist and were retrieved for the run.
13. Store AI-run provenance.
14. Add retrieval recall@5 evaluation.

## First Retrieval API

```http
POST /api/retrieval/query
{
  "query": "Which paper compares reranking and hybrid retrieval?",
  "top_k": 5,
  "mode": "hybrid"
}
```

Expected response:

```json
{
  "query": "...",
  "mode": "hybrid",
  "results": [
    {
      "chunk_id": "...",
      "document_id": "...",
      "heading_path": ["Paper", "Method"],
      "score": 0.82,
      "text": "..."
    }
  ]
}
```

## Interview-Safe Claim

Once implemented, the resume can say:

> Implemented hybrid RAG using PostgreSQL full-text search plus pgvector embeddings, retrieving and reranking document chunks before generating source-cited answers with fallback behaviour when evidence is insufficient.
