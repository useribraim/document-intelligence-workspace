# Architecture

## Core request path

```text
authenticated or public request
  -> route policy and tenant resolution
  -> lexical/vector candidate retrieval
  -> weighted fusion or reciprocal-rank fusion
  -> structured generation over retrieved chunks
  -> citation materialization and exact-quote validation
  -> refusal normalization
  -> AI-run provenance and optional human review
```

The public demo stops after validated read-only output. The local agent path can create an approval
request, but study-task creation requires manager approval and an idempotency key.

## Components

| Component | Responsibility |
|---|---|
| `core/ingestion.py` | Normalize Markdown/text, create deterministic versions and heading-aware chunks. |
| `core/embeddings.py` | Local, OpenAI, and Vertex embedding providers with model/dimension isolation. |
| `core/retrieval.py` | Lexical/vector candidates, tenant document filters, weighted fusion, and RRF. |
| `core/llm.py` | Deterministic, OpenAI, and Vertex structured-generation providers. |
| `core/qa.py` | Citation materialization, exact-source alignment, pruning, and refusal normalization. |
| `core/agent.py` | Typed tools, step budget, duplicate-call guard, approval request, and idempotent task flow. |
| `db/` | SQLAlchemy models, tenant-scoped repositories, SQLite fallback, and PostgreSQL/pgvector path. |
| `api.py` | HTTP schemas, route policy, orchestration, and public demo boundary. |
| `mcp_server.py` | Two tenant-pinned read-only stdio tools. |

## Identity and tenancy

In Google-auth mode, bearer tokens are verified for issuer, audience, signature, expiry, and
subject. Tenant membership is resolved from server-owned records. Model-facing tool schemas do not
accept a tenant identifier.

Repository queries enforce tenant ownership or tenant-document grants. A missing or cross-tenant
record is returned as absent rather than leaking existence.

## Persistence

SQLite is the credential-free local and public-demo fallback. PostgreSQL stores relational records
and mirrors embedding vectors into pgvector for vector and hybrid retrieval. The public Cloud Run
service does not claim durable Cloud SQL persistence.

## Audit record

An AI run can record:

- query and retrieval mode;
- embedding provider, model, and dimensions;
- generation provider/model and prompt version;
- retrieved chunk identifiers;
- citation-validity and refusal state;
- token usage, latency, and estimated cost;
- structured output and timestamp.

External validation artifacts are redacted before being checked into `results/evidence/`.
