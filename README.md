# Document Intelligence Workspace

A FastAPI research assistant that retrieves tenant-scoped evidence, produces structured answers
with exact citations, enters an insufficient-evidence state when its current retrieval gate finds
no usable chunk, and records auditable model runs. That gate is not presented as calibrated refusal
performance.

[Live demo](https://document-intelligence-workspace-312779789755.europe-west1.run.app/) ·
[Architecture](docs/architecture.md) ·
[Evaluation](docs/evaluation.md) ·
[Validation matrix](docs/validation-matrix.md)

The public demo is deliberately narrow: it requires no account, uses bundled synthetic paper
excerpts, performs no writes, and shows the retrieved quote and execution trace behind every
answer. Google sign-in and tenant-aware application routes are separate from this public surface.

## Why this project exists

Most document-QA demos stop after returning plausible text. This project focuses on the harder
boundaries around that answer:

- Can the cited quote be found in the retrieved source chunk?
- Does the system refuse when the corpus does not support the request?
- Can one tenant retrieve another tenant's document or research record?
- Can a model perform a write without a human approval record?
- Can a run be reconstructed from its model, prompt, evidence, token usage, and trace?

## Architecture

```mermaid
flowchart LR
    U["Browser or API client"] --> A["FastAPI application"]
    A --> P["OIDC and tenant policy"]
    P --> R["Lexical + vector retrieval"]
    P --> C["ADK ReAct coordinator"]
    C --> RS["Retrieval specialist"]
    C --> VS["Citation-verification specialist"]
    RS --> R
    R --> G["Structured generation"]
    G --> V["Citation and refusal validation"]
    V --> Q["Human review queue"]
    Q --> W["Approval-gated write"]
    A --> T["AI-run and tool trace"]
    R --> D["SQLite or PostgreSQL + pgvector"]
    M["External MCP client"] --> S["Tenant-pinned stdio server"]
    S --> R
```

The deterministic local providers keep development and CI credential-free. Optional providers add
OpenAI embeddings/chat or Vertex AI embeddings/Gemini generation through the same retrieval,
validation, and audit interfaces.

## Verified evidence

| Area | Recorded result |
|---|---|
| Retrieval | A 23-gold-question pilot observed higher point estimates for semantic embeddings plus RRF, but paired 95% bootstrap intervals for Recall@5 and MRR include zero; no superiority claim is made. |
| Citation safety | Supported answers require exact source-aligned quotes; unsupported answers return no citations. |
| Cloud runtime | A scale-to-zero Cloud Run service exposes the public demo while protected routes reject missing or unscoped identity. |
| Vertex AI | A Cloud Run Job completed real `gemini-embedding-001` and `gemini-2.5-flash` calls with model, token, evidence, citation, refusal, and run provenance. |
| Google ADK | A ReAct-style coordinator delegated to retrieval and citation-verification `AgentTool` specialists on Vertex AI; one recorded run used 7 model calls, 13.82 s, 69.14 output tokens/s, and an estimated $0.002634. |
| MCP | An external Python MCP client discovered and invoked both read-only tools; cross-tenant record lookup and tenant-argument injection failed safely. |
| Automated verification | 108 tests pass locally; one PostgreSQL integration test is conditional on a test database. |

The compact artifacts behind these statements are in [`results/evidence/`](results/evidence/).
Automated diagnostics are not presented as human judgments.

## Safety boundaries

- Tenant identifiers come from authenticated or server-owned context, never model tool arguments.
- Retrieval is filtered to documents granted to the active tenant.
- MCP exposes two read-only tools and no write surface.
- Study-task creation requires a manager approval bound to the exact proposal.
- Idempotency keys prevent duplicate task creation.
- Agent runs have a five-step budget and reject repeated tool calls.
- Citation validation checks quotes against the retrieved chunks used for generation.
- Public demo routes cannot create runs, suggestions, reviews, approvals, or tasks.

## Quickstart

Python 3.11 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest -q
uvicorn diw.api:app --reload
```

Open:

- `http://127.0.0.1:8000/` — public product page
- `http://127.0.0.1:8000/demo` — deterministic cited-answer demo
- `http://127.0.0.1:8000/workspace` — local research workspace
- `http://127.0.0.1:8000/docs` — API schema

The default database is local SQLite. To exercise PostgreSQL and pgvector:

```bash
docker compose up -d
export DATABASE_URL="postgresql+psycopg://diw:diw_local_password@localhost:5433/diw"
python -m diw.cli init-db
POSTGRES_TEST_DATABASE_URL="$DATABASE_URL" pytest tests/test_postgres_runtime.py -q
```

## Optional integrations

Install only the integration extras you need:

```bash
python -m pip install -e ".[cloud]"  # Vertex AI
python -m pip install -e ".[auth]"   # OIDC verification
python -m pip install -e ".[mcp]"    # MCP stdio server/client
python -m pip install -e ".[adk]"    # Google ADK multi-agent workflow
```

Credentials belong in environment variables or an ignored `.env.local`; tracked examples contain
names only. Useful validation commands:

```bash
make test
make calibration-check
make validate-mcp-stdio
./scripts/run_vertex_cloud_smoke.sh
./scripts/run_adk_cloud_smoke.sh
```

The Vertex command requires an enabled Google Cloud project and Application Default Credentials.
The MCP validation is local and creates a temporary two-tenant database.

## Evaluation

Evaluation has two deliberately separate layers:

- the measured retrieval benchmark has 40 frozen questions with predeclared gold chunk IDs;
- the V2 human-calibration instrument has 140 questions, with 28 each for supported,
  partial-support, unsupported, misleading-context, and refusal cases.

Each blinded calibration packet contains 140 answer records and 112 aligned claim-citation pairs.
The current V2 bank deliberately reuses each evidence span across controlled variants. It is useful
for rubric development, but it is not an independent 140-item human-evaluation sample and must not
be used for naive confidence intervals or headline accuracy claims.

The current source tree excludes raw paper text. However, earlier Git history contains the removed
files, so this public repository must not be described as non-redistributing until that history is
rewritten. The corpus manifest records canonical versions, licence URLs, local paths, and SHA-256
hashes so a legally obtained local corpus can be verified:

```bash
python -m diw.cli corpus-verify
```

Human labeling remains open. The repository publishes the reproducible instrument and two blank
packets, but no agreement or human-accuracy number until two people independently complete all
records and preserve a separate adjudication trail.

## Current limitations

- The public demo uses synthetic sources and deterministic extractive generation.
- The deployed service uses ephemeral SQLite; Cloud SQL persistence is not claimed.
- Vertex AI is validated through a bounded Cloud Run Job, not the public interactive route.
- MCP is validated through an external local stdio client, not a remote transport.
- Human-calibrated agreement is incomplete.
- The V2 calibration bank is clustered by source seed; an independent-item V3 study is required
  before publishing calibration confidence intervals or accuracy claims.
- ADK is validated as a bounded Cloud Run Job, not a public interactive multi-agent endpoint.
- Asynchronous ingestion and production observability remain future work.

## Repository map

```text
src/diw/core/          retrieval, generation, validation, evaluation, agent policy
src/diw/db/            SQLAlchemy models, repositories, schema, sessions
src/diw/api.py         FastAPI routes and orchestration
src/diw/mcp_server.py  tenant-pinned read-only MCP boundary
tests/                 deterministic and integration-oriented tests
data/demo/             synthetic public demo inputs
data/audit/            evaluation definitions and blank annotation templates
results/evidence/      curated, credential-free validation artifacts
docs/                  architecture, operations, evaluation, and limitations
```

## Technical notes

- [Architecture](docs/architecture.md)
- [Evaluation methodology](docs/evaluation.md)
- [Validation matrix](docs/validation-matrix.md)
- [Cloud Run boundary](docs/cloud-run-deployment.md)
- [Vertex AI validation](docs/integrations/vertex-ai-validation.md)
- [Retrieval uncertainty analysis](results/evidence/retrieval-uncertainty.md)
- [Retrieval-pilot incident note](docs/incidents/retrieval-pilot-overclaim.md)
- [MCP stdio validation](docs/integrations/mcp-stdio-validation.md)
- [Google ADK validation](docs/integrations/adk-validation.md)
- [Human-calibration runbook](docs/human-calibration-runbook.md)
- [Corpus provenance and licensing](docs/corpus-provenance.md)

## License

Project code is available under the [MIT License](LICENSE). Third-party research papers are not
covered by that licence. The current tree excludes them, but the repository history requires a
separate purge before this project can make a non-redistribution claim.
