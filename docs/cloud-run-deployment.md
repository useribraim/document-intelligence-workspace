# Cloud Run Deployment Notes

This document is a study guide for the next deployment slice. It is not a claim that the
service has already been deployed.

## Current Container Contract

The `Dockerfile` starts the FastAPI app with Uvicorn on `0.0.0.0:${PORT}`. Cloud Run supplies
the `PORT` value. The application still defaults to SQLite, so a real deployment must provide
`DATABASE_URL` for Cloud SQL PostgreSQL before it can be called production-ready.

## Planned Runtime Shape

```text
Cloud Run service
  -> FastAPI API
  -> Cloud SQL PostgreSQL + pgvector
  -> Vertex AI embeddings and Gemini provider
  -> Secret Manager for credentials
```

Document ingestion will later move to Cloud Storage and Pub/Sub with a separate worker. The
API should remain responsible for request validation, tenant policy, agent orchestration, and
returning trace IDs; the worker should own retryable ingestion work.

## Deployment Gate

Do not call the deployment production-ready until all of these are true:

- PostgreSQL integration tests pass against the target pgvector version.
- Database schema changes run through migrations rather than implicit `create_all`.
- The API image starts with a non-local provider configuration.
- Authentication and tenant checks execute before document or research-record access.
- Secrets are injected by Secret Manager and are absent from logs.
- A smoke test proves question -> evidence -> cited answer -> trace inspection.
- The public URL has a documented cost limit and a way to stop paid resources.

## Interview Explanation

Cloud Run is appropriate for the first deployment because the workload is request-driven and
low-volume. Cloud SQL keeps relational business records, approvals, audit metadata, and
pgvector retrieval in one system. The tradeoff is that Cloud Run does not remove database
connection, migration, or observability responsibilities; those remain explicit engineering
work in this project.
