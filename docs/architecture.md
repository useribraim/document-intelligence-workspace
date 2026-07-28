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

The public demo stops after validated read-only output. The authenticated local workflow is a
deterministic policy demonstration: it always retrieves first and may create an approval request
when a rule matches a study-task query. It is not an LLM planner. Study-task creation still requires
manager approval and an idempotency key. The Google ADK workflow is a separate read-only Cloud Run
Job validation slice, not the public API request path.

## Components

| Component | Responsibility |
|---|---|
| `core/ingestion.py` | Normalize Markdown/text, create deterministic versions and heading-aware chunks. |
| `core/embeddings.py` | Local, OpenAI, and Vertex embedding providers with model/dimension isolation. |
| `core/retrieval.py` | Lexical/vector candidates, tenant document filters, weighted fusion, and RRF. |
| `core/llm.py` | Deterministic, OpenAI, and Vertex structured-generation providers. |
| `core/qa.py` | Citation materialization, exact-source alignment, pruning, and refusal normalization. |
| `core/agent.py` | Rule-based, bounded approval-workflow demonstration with typed tools, duplicate-call guard, and idempotent task flow; not an autonomous planner. |
| `adk_workflow.py` | Google ADK ReAct-style coordinator, retrieval and citation-verification specialist delegation, and per-model-call economics. |
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
and mirrors embedding vectors into pgvector for vector and hybrid retrieval. Hybrid BM25 scoring
uses the full tenant-filtered embedding corpus on both backends so the lexical candidate universe
does not depend on SQLite versus PostgreSQL; this favors parity over large-corpus efficiency. The
public Cloud Run service does not claim durable Cloud SQL persistence.

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
