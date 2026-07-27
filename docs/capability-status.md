# Capability Status

This file separates code that exists from infrastructure or evaluation that has actually been
validated. A passing unit test is not a claim that a paid cloud request, deployment, identity
provider, or human study has run.

| Capability | Current evidence | Safe claim |
|---|---|---|
| Stronger retrieval | A frozen 40-question comparison records hashing + weighted retrieval against OpenAI `text-embedding-3-small` + RRF. Recall@5 changed from 0.2609 to 0.2826, MRR from 0.1935 to 0.3022, and gold-citation recall from 0.1087 to 0.2536. | Semantic embeddings plus RRF improved the frozen retrieval benchmark. RRF alone did not improve it. |
| Vertex AI generation and embeddings | Cloud Run Job execution `diw-vertex-smoke-z5m7b` used the deployed service image and runtime identity to create eight 768-dimensional `gemini-embedding-001` document vectors, make two query-embedding calls, and make two `gemini-2.5-flash` generations. The supported answer passed exact-quote citation validation; the unsupported legal query refused. Token use, model/provider, retrieved chunks, prompt version, AI-run IDs, workflow-run ID, latency, and an error-free trace are recorded. | Live Vertex AI embeddings and Gemini generation are cloud-validated in a bounded RAG smoke workflow. This does not claim that the public interactive demo or persistent workflow uses Vertex. |
| Cloud Run | Public revision `document-intelligence-workspace-00004-26q` serves a zero-login landing page, read-only cited demo, evidence page, and optional `/signin` in `europe-west1`. External checks returned 200 for the public routes, 401 for missing-token identity/agent routes, and 403 for unscoped data routes. A separate scale-to-zero Cloud Run Job runs the Vertex smoke workflow under the same runtime service account. | Stable public Cloud Run URL and bounded recruiter demo are verified. The public demo is deterministic and read-only, not the persistent production workflow. |
| MCP | An official Python MCP SDK `ClientSession` (1.28.1) launched the server as a separate stdio process, discovered both read-only tools, invoked evidence search and research-record lookup, hid a cross-tenant record, and safely ignored a model-supplied tenant override. | A thin, tenant-pinned MCP stdio server is implemented and validated through an external client process. This does not claim a remote MCP transport or write tools. |
| OIDC | A real external Google OAuth web client is configured for the Cloud Run origin. The test user completed sign-in; deployed `/auth/whoami` accepted the Google ID token and rendered the verified email. Tenant membership is resolved server-side from the verified subject; unscoped routes fail closed. | Google OAuth/OIDC is implemented, cloud-configured, and live-verified for the configured test user. |
| Human calibration | Annotation packets and rubric exist, but labels remain pending. | Human calibration is not complete. Never report automated diagnostics as human results. |
| Tenant isolation | Repository queries, agent retrieval, API actor checks, and MCP tools filter by tenant. Cross-tenant tests exist. | Tenant isolation is implemented and locally tested on these paths. |

## Reproducible Commands

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/ruff check src tests
.venv/bin/python -m diw.cli corpus-verify
```

Install optional integrations only when they are needed:

```bash
python -m pip install -e ".[cloud]"
python -m pip install -e ".[auth]"
python -m pip install -e ".[mcp]"
```

## Vertex Cloud Run Smoke Test

The reproducible smoke job uses explicit stable model IDs and Application Default Credentials
from the runtime service account:

```bash
export DIW_GCP_PROJECT="your-project-id"
export DIW_CLOUD_RUN_SERVICE="document-intelligence-workspace"
export DIW_CLOUD_RUN_REGION="europe-west1"
export DIW_VERTEX_LOCATION="global"
export DIW_VERTEX_CHAT_MODEL="gemini-2.5-flash"
export DIW_VERTEX_EMBEDDING_MODEL="gemini-embedding-001"
./scripts/run_vertex_cloud_smoke.sh
```

The checked-in result is
[`vertex-cloud-run-smoke.json`](../results/integrations/vertex/vertex-cloud-run-smoke.json);
the validation record and limitations are in
[`vertex-ai-validation.md`](integrations/vertex-ai-validation.md).

## MCP

The MCP process is pinned to one tenant through process configuration. The model cannot pass a
different tenant ID to either tool.

```bash
export DIW_MCP_TENANT_ID="tenant-uuid"
diw-mcp
```

The server uses the official Python SDK's stdio transport. It intentionally exposes no write tool.
Run the credential-free external-client check with:

```bash
make validate-mcp-stdio
```

The checked-in transcript is
[`mcp-stdio-validation.json`](../results/integrations/mcp/mcp-stdio-validation.json);
the client configuration and limitations are in
[`mcp-stdio-validation.md`](integrations/mcp-stdio-validation.md).

## OIDC

Set `AUTH_MODE=oidc` plus the issuer metadata:

```bash
export AUTH_MODE=oidc
export OIDC_ISSUER="https://issuer.example"
export OIDC_AUDIENCE="document-intelligence-workspace"
export OIDC_JWKS_URL="https://issuer.example/.well-known/jwks.json"
export OIDC_TENANT_CLAIM="tenant_id"
```

In this mode, `/agent-runs` requires a bearer token. The token tenant must equal the request tenant,
and the token subject must equal the stored actor subject. Other application routes fail closed
until each receives tenant-aware authorization.
