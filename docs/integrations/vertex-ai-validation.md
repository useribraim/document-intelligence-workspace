# Vertex AI Validation

## Validated Boundary

On 2026-07-27, Cloud Run Job execution `diw-vertex-smoke-z5m7b` completed a bounded
retrieval-augmented generation workflow in the deployed Google Cloud project.

- Vertex AI API: enabled.
- Billing: enabled for the project.
- Runtime identity:
  `diw-cloud-run-runtime@<redacted-project-id>.iam.gserviceaccount.com`.
- Runtime role: `roles/aiplatform.user`.
- Runtime image: the same immutable image digest used by the deployed service.
- Vertex location and API version: `global`, stable `v1`.
- Embedding model: `gemini-embedding-001`, 768 output dimensions.
- Generation model: `gemini-2.5-flash`.
- Result: eight document vectors, two query-embedding calls, and two successful generation calls.
- Supported case: five chunks retrieved, one exact-aligned citation, citation validation passed.
- Unsupported case: five chunks retrieved, the model set `insufficient_evidence=true`, returned no
  citations, and citation validation passed.
- Usage: 3,063 input tokens and 566 output tokens across the two generations.
- Trace: workflow run ID, two AI-run IDs, provider/model, token use, retrieved chunks, citations,
  per-step latency, total latency, and an empty error list.

The credential-free-at-rest evidence is
[`vertex-cloud-run-smoke.json`](../../results/evidence/vertex-cloud-run-smoke.json).
It excludes tokens, request headers, credential paths, and embedding vectors.

## Reproduce

The runtime service account must have the Vertex AI User role and the Vertex AI API must be
enabled. Then run:

```bash
export DIW_GCP_PROJECT="your-project-id"
export DIW_CLOUD_RUN_SERVICE="document-intelligence-workspace"
export DIW_CLOUD_RUN_REGION="europe-west1"
export DIW_VERTEX_LOCATION="global"
export DIW_VERTEX_CHAT_MODEL="gemini-2.5-flash"
export DIW_VERTEX_EMBEDDING_MODEL="gemini-embedding-001"
./scripts/run_vertex_cloud_smoke.sh
```

The script deploys and executes `diw-vertex-smoke`, waits for success, reads only the result marker
from Cloud Logging, and writes the redacted JSON artifact.

## Honest Limitation

This is a real cloud provider check over bundled synthetic sources and an ephemeral SQLite
database. It does not validate Cloud SQL persistence, asynchronous ingestion, production traffic, or
that the public interactive demo uses Vertex.

## Validated Scope

> Ran a Cloud Run RAG smoke workflow using Vertex AI `gemini-embedding-001` embeddings and
> `gemini-2.5-flash` generation, with token/model provenance, retrieved chunks, exact-quote
> citation validation, and unsupported-query refusal recorded per run.
