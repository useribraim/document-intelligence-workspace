# Learning Note 03: Embeddings And Retrieval

## What Problem This Step Solves

v0.2 stored chunks in a database, but stored chunks are not useful until the system can find the right evidence for a query.

v0.3 adds retrieval:

```text
query -> lexical score
      -> query embedding
      -> vector score
      -> hybrid score
      -> ranked chunks
```

## Why Use A Local Embedding Provider First

The current provider is `local-hashing-v1`. It is deterministic and API-free. It is not a real semantic model, but it lets the project build the retrieval pipeline without hiding behaviour behind an external API.

This is useful for learning because every part is inspectable:

- tokenisation
- vector construction
- L2 normalisation
- cosine similarity
- hybrid score calculation

Later, this provider can be replaced with OpenAI or Azure OpenAI embeddings.

## Lexical Retrieval

Lexical retrieval checks whether query terms appear in the chunk text or heading path. It is simple but important because exact words often matter in professional/legal documents.

Example:

```text
query: termination notice period
```

Lexical search should strongly prefer chunks that actually contain those terms.

## Vector Retrieval

Vector retrieval compares the query embedding with stored chunk embeddings using cosine similarity.

In production, embeddings should come from a semantic embedding model. In this local version, embeddings are deterministic hashing vectors so the retrieval code can be tested without external services.

## Hybrid Retrieval

Hybrid retrieval combines lexical and vector scores:

```text
hybrid_score = 0.55 * lexical_score + 0.45 * positive_vector_score
```

The weights are intentionally simple for now. Later evaluation should tune or replace them.

## Why This Still Is Not Full Production RAG

v0.3 retrieves ranked chunks, but it does not yet generate answers from them. Full RAG needs:

- stronger embeddings
- pgvector-backed vector search
- retrieval run logging
- source-cited answer generation
- citation validation
- recall@5 evaluation

## Interview Explanation

I built retrieval in stages. First I created stable chunks and database records. Then I added a deterministic local embedding provider so I could test the retrieval pipeline without API dependencies. The current system supports lexical, vector, and hybrid retrieval over stored chunks. The next step is replacing local vectors with pgvector-backed semantic embeddings and measuring retrieval recall.
