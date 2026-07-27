# Cloud Run Deployment Notes

This document describes both the executable deployment scaffold and the currently verified
bounded Cloud Run service. It does not claim that Cloud SQL or the full persistent workflow has
been deployed.

## Current Container Contract

The `Dockerfile` starts the FastAPI app with Uvicorn on `0.0.0.0:${PORT}`. Cloud Run supplies
the `PORT` value. The application still defaults to SQLite, so a real deployment must provide
`DATABASE_URL` for Cloud SQL PostgreSQL before it can be called production-ready.

`scripts/deploy_cloud_run.sh` supports two deliberately distinct smoke-test boundaries:

- `DIW_AUTH_MODE=off` makes the Cloud Run service private with IAM.
- `DIW_AUTH_MODE=google` makes the HTTP endpoint reachable and requires the application to verify
  Google OAuth ID tokens for authenticated routes. The browser sign-in proof is at `/signin`.

The Google mode is suitable for proving the deployed identity boundary. It is not, by itself,
authorization for unscoped document routes: those routes fail closed until they have tenant-aware
policies.

The deployed service also exposes three deliberately public routes:

- `/` — public landing page with the product scope and measured facts.
- `/demo` plus `POST /demo/ask` — deterministic, read-only retrieval over bundled synthetic
  sources. It has no write tools, database mutation, or external model request.
- `/evidence` — the frozen retrieval comparison and explicit pending human-calibration boundary.

These public routes are independent of the ephemeral SQLite database and remain safe on a
scale-to-zero revision. The persistent document, approval, and action workflow still requires
Cloud SQL before it can be called production-ready.

The script defaults to dedicated `diw-cloud-run-builder` and `diw-cloud-run-runtime` service
accounts. The build identity has `roles/run.builder`; the runtime identity has
`roles/aiplatform.user` so the separate Vertex smoke job can make Google Gen AI requests.
Override `DIW_BUILD_SERVICE_ACCOUNT` or `DIW_RUNTIME_SERVICE_ACCOUNT` only when the target project
deliberately uses different identities.

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

## Live Vertex Job

`scripts/run_vertex_cloud_smoke.sh` deploys a scale-to-zero Cloud Run Job from the same image and
runtime service account as the public service. It uses explicit `gemini-embedding-001` and
`gemini-2.5-flash` model IDs, an ephemeral SQLite database, and bundled synthetic sources. The job
records a redacted result from Cloud Logging without writing credentials.

This closes the live-provider gate but does not make the public deterministic demo a Vertex-backed
or persistent workflow. See [the validation record](integrations/vertex-ai-validation.md).

## Deployment Gate

Do not call the deployment production-ready until all of these are true:

- PostgreSQL integration tests pass against the target pgvector version.
- Database schema changes run through migrations rather than implicit `create_all`.
- The API image starts with a non-local provider configuration.
- Authentication and tenant checks execute before document or research-record access.
- Secrets are injected by Secret Manager and are absent from logs.
- A smoke test records question -> evidence -> cited answer -> trace inspection.
- The public URL has a documented cost limit and a way to stop paid resources.

## Basic Private Smoke Deployment

This is a container and request-path smoke test, not the production gate above:

```bash
export DIW_GCP_PROJECT="your-project-id"
export DIW_CLOUD_RUN_SERVICE="document-intelligence-workspace"
export DIW_CLOUD_RUN_REGION="europe-west1"
./scripts/deploy_cloud_run.sh
```

The script uses `--no-allow-unauthenticated`, caps the service at one 512 MiB instance, and prints the
resulting URL. Because the default database is ephemeral SQLite, do not present data persistence
from this smoke deployment as production-ready.

## Google OIDC Smoke Deployment

Create a Google OAuth web client for the deployed URL, add the service URL to its authorized
JavaScript origins, and deploy with:

```bash
export DIW_GCP_PROJECT="your-project-id"
export DIW_CLOUD_RUN_SERVICE="document-intelligence-workspace"
export DIW_CLOUD_RUN_REGION="europe-west1"
export DIW_AUTH_MODE="google"
export GOOGLE_OAUTH_CLIENT_ID="your-client-id.apps.googleusercontent.com"
./scripts/deploy_cloud_run.sh
```

Open `${SERVICE_URL}/signin`, complete Google sign-in, and confirm `/auth/whoami` displays the
verified account. Cloud Run is public at the transport layer in this mode because the application
performs the end-user identity check. Keep every non-public application route behind an explicit
authorization policy.

## Design Rationale

Cloud Run is appropriate for the first deployment because the workload is request-driven and
low-volume. Cloud SQL keeps relational business records, approvals, audit metadata, and
pgvector retrieval in one system. The tradeoff is that Cloud Run does not remove database
connection, migration, or observability responsibilities; those remain explicit engineering
work in this project.
