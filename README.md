# Document Intelligence Workspace

Markdown-first document intelligence system for ML/NLP research sources and technical documents.

**Live Cloud Run demo:** [Open the public recruiter page](https://document-intelligence-workspace-312779789755.europe-west1.run.app/)

The stable service URL has three deliberate trust levels:

- `/` and `/demo` are public and require no account; the demo is read-only and uses bundled
  synthetic paper excerpts.
- `/signin` proves live Google OIDC and is optional for a reviewer.
- Data and agent routes remain protected; unscoped application routes fail closed in Google-auth
  mode.

The project is built to prove a controlled document AI loop, not a generic chat-with-PDFs app:

1. Ingest long-form sources.
2. Clean and chunk text with document/version provenance.
3. Retrieve source chunks using lexical search first, then hybrid retrieval with embeddings.
4. Generate structured, source-cited outputs.
5. Validate outputs against schemas.
6. Log AI-run provenance for auditability.
7. Route AI outputs through human review decisions.
8. Evaluate retrieval quality, citation validity, refusal behaviour, and structured extraction.

## Evidence Snapshot

- **Retrieval:** on the frozen 40-question set, OpenAI `text-embedding-3-small` plus RRF
  improved Recall@5 from 0.2609 to 0.2826, MRR from 0.1935 to 0.3022, and
  gold-citation recall from 0.1087 to 0.2536 versus local hashing plus weighted fusion.
  [Comparison report](results/reports/v1-hashing-weighted-vs-te3s-rrf.md)
- **Safety and identity:** the deployed Google OAuth flow verified a real ID token and rendered
  the authenticated account. Server-side tenant membership, approval-gated writes, idempotency,
  and cross-tenant denial are tested.
- **Vertex AI:** Cloud Run Job execution `diw-vertex-smoke-z5m7b` made real
  `gemini-embedding-001` and `gemini-2.5-flash` calls, produced a cited supported answer, and
  refused an unsupported legal query. The redacted trace records token use, chunks, citations,
  prompt/run IDs, and no errors.
- **MCP:** the official Python SDK, running as a separate client process, discovered and invoked
  both read-only stdio tools; its transcript proves cross-tenant lookup denial and safe handling
  of a model-supplied tenant override.
- **Verification:** 91 tests pass locally; one optional PostgreSQL integration test is skipped
  unless its database URL is configured.
- **Deployment:** a public, scale-to-zero Cloud Run service presents a zero-login recruiter page,
  read-only cited retrieval demo, and optional Google sign-in; protected identity and agent routes
  reject missing tokens.
- **Not yet claimed:** human-calibrated agreement metrics remain evidence-gated until the primary
  and independent blind labels are complete.

The exact resume-safe wording and its release gates are tracked in
[resume-claim-gates.md](docs/resume-claim-gates.md).

## v1.8.2 Scope

Implemented:

- Project scaffold for a Python/FastAPI/PostgreSQL document intelligence system.
- Conservative text normalisation.
- Heading-aware deterministic chunking.
- Document and version provenance.
- Content hashes for normalised documents and chunks.
- JSON export of chunk records.
- CLI commands for normalisation and chunking.
- Local database persistence for source documents, document versions, and chunks.
- CLI commands for database initialisation and loading documents.
- Deterministic local embedding provider for development and tests.
- Optional OpenAI text-embedding provider for measured semantic retrieval runs.
- Optional Vertex AI `gemini-embedding-001` provider with retrieval-query/document task types.
- Reciprocal-rank fusion reranking for hybrid retrieval.
- Chunk embedding storage with per-model isolation, so embeddings from different providers coexist in one database.
- Lexical, vector, and hybrid retrieval over stored chunks.
- CLI commands for embedding chunks and retrieving ranked evidence.
- Source-cited QA composer over retrieved chunks.
- Insufficient-evidence refusal behaviour when retrieval confidence is too low.
- Citation validation that checks quoted evidence is present in retrieved source chunks.
- CLI command for source-cited answers.
- LLM provider interface for structured source-cited answers.
- Deterministic structured provider for local development and tests.
- OpenAI-compatible chat provider for real LLM runs when `OPENAI_API_KEY` is available.
- Vertex AI Gemini structured-output provider through Application Default Credentials.
- Prompt version metadata on generated answers.
- Golden-case evaluation runner for structured extraction, citation validity, and refusal behaviour.
- Persistent AI run records for answer generation and evaluation reports.
- Local bounded agent core with typed tools, a five-step budget, duplicate-call protection,
  tenant-scoped document retrieval, and durable per-step audit records.
- Read-only, tenant-pinned MCP stdio server for evidence search and research-record lookup.
- Optional OIDC verification for tenant-scoped agent routes with issuer, audience, expiry,
  subject, and tenant-claim enforcement.
- Tenant, user, research-record, approval-request, and idempotent study-task persistence for
  the forthcoming research action workflow.
- Manager approval gate before a study task can be created.
- Audit metadata for query, retrieval mode, embedding model, LLM provider/model, prompt version, retrieved chunks, citation validity, refusal status, output payload, and metrics.
- PostgreSQL runtime path through Docker Compose and `DATABASE_URL`.
- Optional PostgreSQL integration test for ingest, embedding, retrieval, structured answer generation, and citation validation.
- `pgvector` extension creation when running against PostgreSQL.
- `chunk_embedding_vectors` table for pgvector-backed embedding storage.
- Embedding writes mirrored to JSON storage for SQLite compatibility and pgvector storage for PostgreSQL.
- PostgreSQL/pgvector-backed vector retrieval for `vector` and `hybrid` retrieval modes.
- SQLite/Python retrieval fallback for local unit tests and API-free development.
- Expanded 12-case evaluation set covering structured extraction, supported QA, near-miss retrieval, conflicting evidence, and refusal behaviour.
- JSON evaluation report output with pass rate, citation-validity rate, retrieval-hit rate, refusal accuracy, and task-level breakdowns.
- Markdown evaluation report renderer.
- Human review queue for AI-generated suggestions.
- Review decision records for accepted, rejected, and edited suggestions.
- CLI commands for listing suggestions and recording review decisions.
- FastAPI backend exposing ingestion, embedding, question answering, AI-run inspection, and human review.
- Public recruiter landing page and deterministic read-only cited demo with a visible execution
  trace, exact-quote validation, no external model call, and no write tools.
- Public evidence page that distinguishes automated retrieval measurements from pending human
  calibration.
- Minimal browser dashboard for documents, pending review suggestions, and recent AI runs.
- Agent-style review workspace with answer thread, evidence inspector, AI-run metadata, and review actions.
- Evidence-first workspace flow with retrieval preview before answer generation.
- Task mode selector, run history tab, and richer evidence-based review notes.
- Review-to-evaluation loop for saving reviewed outputs as JSONL regression cases.
- Evaluation-case API and workspace Eval tab for review-derived cases.
- Corpus browser in the workspace for documents, versions, chunks, and chunk provenance.
- Markdown paper-card compiler for turning reviewed document versions into durable study artifacts.
- Paper-card draft/save API.
- Paper Workspace view that keeps the selected paper, chunks, evidence state, generated answer, review state, and study artifact actions in one flow.
- Direct workspace actions for finding evidence, generating a study card, reviewing the card, saving the note, and creating review-derived eval cases.
- Heading-aware paper-card field extraction for unlabeled source sections.
- Visible workflow status updates for evidence extraction, answer generation, card drafting, saving, review, and eval-case creation.
- Workspace draft action avoids creating duplicate pending review suggestions for repeated paper-card drafts.
- Saved paper-card confirmations include path and timestamp.
- Rendered study-card editor with editable Core Idea, Problem, Method, Dataset, Metrics, Results, and Limitations fields.
- Missing or weak fields are marked as needing attention instead of being presented as finished output.
- Technical provenance is hidden behind a disclosure in the main workflow instead of dominating the study card.
- Card review lifecycle with Build card, Accept card, and Save note actions.
- Save note is gated until required fields are present and the card is accepted.
- Source text is collapsed by default behind a Show source text disclosure.
- Study-card status badges show lifecycle state, required missing count, and needs-attention count.
- Guided five-step workflow row: Find evidence, Generate draft, Build card, Accept, Save.
- Only the next valid action is styled as the primary step.
- Dense answer/citation/save metadata is collapsed under Details in the study-card column.
- Empty study-card state explains the next workflow steps.
- The study-card column gets more space once a card exists.
- Workspace import flow for uploading `.md` and `.txt` sources directly from the browser.
- Imported documents are saved under `data/demo/raw/uploads`, ingested, persisted, and selected automatically.
- Demo ML/NLP extraction sources.
- Unit tests for normalisation, ingestion, chunking, CLI behaviour, persistence, embeddings, and retrieval.
- Starter golden evaluation cases.
- Architecture and evaluation documentation.

Next build slice:

- Evaluation execution from saved review cases, richer answer editing, and concept-note generation.

## Design Goals

The workspace is designed for long-form technical documents where answers need to stay tied to source evidence. It stores document versions, chunk hashes, retrieval metadata, generated outputs, and review decisions so the system remains inspectable after the initial model call.

## Flagship Evolution

The project is evolving into a Research Knowledge & Action Assistant. The [flagship project guide](docs/flagship-project-guide.md) is the plain-language source of truth for its architecture, milestones, evaluation strategy, and interview demonstration. It distinguishes the implemented evidence-first baseline from the planned agent, identity, MCP, and Google Cloud work.

Evaluation details are documented in [evaluation.md](docs/evaluation.md), and the current
container/deployment boundary is recorded in [cloud-run-deployment.md](docs/cloud-run-deployment.md).
The local bounded agent, RRF retrieval, live Vertex smoke workflow, externally validated thin MCP
server, OIDC-protected agent boundary, and public Cloud Run deployment are implemented. Cloud SQL
deployment, the persistent public workflow, remote MCP transport, and human calibration remain
unverified. See [capability-status.md](docs/capability-status.md) for the exact claim boundary.

## Claim-to-evidence audit pilot

The workspace now has a claim-level audit path for testing whether a cited span supports the exact sentence-level claim. See [claim-evidence-audit.md](docs/claim-evidence-audit.md) for the reproducible pilot workflow and its deliberately narrow limits.

## Repository Layout

```text
document-intelligence-workspace/
  src/diw/core/        # Domain logic that should remain framework-light
  src/diw/api.py       # FastAPI routes, auth boundaries, and orchestration
  src/diw/web_views.py # Public, sign-in, dashboard, and workspace HTML/JavaScript
  tests/               # Unit tests for deterministic core logic
  docs/                # Architecture, evaluation, and deployment notes
  data/demo/           # Public/synthetic demo sources and eval cases
```

## Development

### Local API credentials

For local OpenAI-backed commands, copy `.env.local.example` to `.env.local` and add your
own `OPENAI_API_KEY`. The local file is ignored by Git and takes effect automatically when
you run the CLI from the repository root. Do not put credentials in tracked files.

Create and activate a virtual environment if desired:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Run unit tests:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

Start the local PostgreSQL runtime:

```bash
docker compose up -d
```

Use PostgreSQL instead of the default local SQLite file:

```bash
export DATABASE_URL="postgresql+psycopg://diw:diw_local_password@localhost:5433/diw"
PYTHONPATH=src python3 -m diw.cli init-db
```

Run optional PostgreSQL integration tests:

```bash
POSTGRES_TEST_DATABASE_URL="$DATABASE_URL" \
  PYTHONPATH=src python3 -m unittest tests.test_postgres_runtime -v
```

In PostgreSQL, `embed` stores vectors in both `chunk_embeddings` and the pgvector-backed `chunk_embedding_vectors` table. `vector` and `hybrid` retrieval modes use pgvector similarity search when running against PostgreSQL.

## v1.1.0 Demo

Start the local API and browser dashboard:

```bash
PYTHONPATH=src python3 -m uvicorn diw.api:app --reload
```

Open:

```text
http://127.0.0.1:8000/
http://127.0.0.1:8000/workspace
http://127.0.0.1:8000/docs
```

Normalise a demo Markdown source:

```bash
PYTHONPATH=src python3 -m diw.cli normalise data/demo/raw/retrieval-notes.md --report
```

Write normalised text to a file:

```bash
PYTHONPATH=src python3 -m diw.cli normalise \
  data/demo/raw/retrieval-notes.md \
  --out data/demo/raw/retrieval-notes.normalised.md \
  --report
```

Create chunk JSON with provenance:

```bash
PYTHONPATH=src python3 -m diw.cli chunk \
  data/demo/raw/retrieval-notes.md \
  --out data/demo/processed/retrieval-notes.chunks.json \
  --target-chars 500 \
  --overlap-chars 0
```

The exported JSON includes:

- `document_id`
- `version_id`
- `source_path`
- `source_type`
- `content_hash`
- `normalisation_report`
- `chunk_count`
- chunk text, heading path, line range, and chunk hash

Initialise a local SQLite database:

```bash
PYTHONPATH=src python3 -m diw.cli init-db
```

Load a document into the local database:

```bash
PYTHONPATH=src python3 -m diw.cli load \
  data/demo/raw/retrieval-notes.md \
  --target-chars 500 \
  --overlap-chars 0
```

By default this creates `diw_local.db`. Use `DATABASE_URL` or `--database-url` to point at PostgreSQL.

Create deterministic local embeddings for stored chunks:

```bash
PYTHONPATH=src python3 -m diw.cli embed
```

Retrieve ranked evidence:

```bash
PYTHONPATH=src python3 -m diw.cli retrieve \
  "how does the workspace preserve source evidence" \
  --mode hybrid \
  --top-k 3
```

The local embedding provider is intentionally deterministic and API-free. It builds and tests the retrieval pipeline without credentials. For measured semantic retrieval runs, an OpenAI embedding provider is available behind `--embedding-provider openai` (requires `OPENAI_API_KEY`, defaults to `text-embedding-3-small` at 1536 dimensions):

```bash
PYTHONPATH=src python3 -m diw.cli embed --embedding-provider openai
PYTHONPATH=src python3 -m diw.cli retrieve \
  "how does the workspace preserve source evidence" \
  --mode hybrid \
  --embedding-provider openai \
  --top-k 3
```

Embeddings are stored per model name and dimension, so OpenAI and local hashing embeddings coexist in the same database and retrieval only uses vectors from the selected provider.

Produce a source-cited answer:

```bash
PYTHONPATH=src python3 -m diw.cli answer \
  "how does the workspace preserve source evidence" \
  --mode hybrid \
  --top-k 3
```

The answer payload includes:

- structured answer fields
- citation labels
- cited chunk IDs, document IDs, version IDs, and heading paths
- exact source quotes
- citation validation results
- retrieved chunks used to produce the answer

Load the structured extraction demo sources:

```bash
PYTHONPATH=src python3 -m diw.cli load data/demo/raw/ml-paper-excerpt.md
PYTHONPATH=src python3 -m diw.cli load data/demo/raw/ablation-paper-excerpt.md
```

Generate an LLM-style structured answer with the deterministic local provider:

```bash
PYTHONPATH=src python3 -m diw.cli answer-llm \
  "Extract the method, dataset, metric, and limitation from the cited paper section." \
  --top-k 3
```

By default this creates both an `ai_runs` audit record and a pending `ai_suggestions` review item. Use `--no-create-suggestion` for throwaway checks.

List pending AI suggestions:

```bash
PYTHONPATH=src python3 -m diw.cli review-list --status pending
```

Accept, reject, or edit a suggestion:

```bash
PYTHONPATH=src python3 -m diw.cli review-decide SUGGESTION_ID \
  --decision accept \
  --reviewer ivan \
  --note "Citations checked against the retrieved chunks."
```

Run the golden-case evaluation set:

```bash
PYTHONPATH=src python3 -m diw.cli eval \
  --top-k 4 \
  --out data/demo/evals/report.json
```

Render a Markdown evaluation report:

```bash
PYTHONPATH=src python3 -m diw.cli eval-report \
  data/demo/evals/report.json \
  --out docs/eval-report.md
```

The eval report includes:

- total cases
- pass rate
- citation-validity rate
- missing structured fields
- refusal correctness for unsupported questions

`answer-llm` and `eval` persist an `ai_runs` record by default. `answer-llm` also creates a pending review suggestion by default. Use `--no-log-run` and `--no-create-suggestion` for throwaway local checks.

## Roadmap

- Expand the golden evaluation set.
- Add PostgreSQL full-text search to the hybrid retrieval path.
- Report retrieval recall@5, citation validity, schema validity, refusal correctness, latency, and estimated cost.
- Add a richer review UI for document, chunk, answer, citation, and evaluation inspection.
