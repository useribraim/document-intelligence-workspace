# Engineering Roadmap: Research Knowledge & Action Assistant

## One-Minute Explanation

This project began as a way to study ML papers without losing source evidence. It now evolves into a bounded assistant that can search papers, inspect structured research records, draft useful study artifacts, and create an approved follow-up task.

The central rule is simple: the model may propose a tool call, but application policy decides whether it is allowed and the database records what happened.

## Engineering Scope

Document QA demonstrates grounding. An action assistant must additionally validate identity, tenant
isolation, safe tool use, approvals, observability, evaluation, and recovery from failure. The
project keeps the reliable evidence-first `/ask` path and adds the agent separately, so
experimentation cannot silently weaken the baseline.

## What Exists Today

- Versioned document ingestion, heading-aware chunks, hybrid retrieval, citations, refusals, review records, and evaluation cases.
- A paper workspace for building and reviewing study cards.
- PostgreSQL/pgvector support and a FastAPI application.
- A bounded local agent core with typed tool calls, a five-step budget, duplicate-call
  protection, and durable step observations. It can search only documents granted to its
  tenant, return a source-cited answer, and create an approval request for a study task.
- Local persistence for tenants, users, tenant-scoped research records, agent-run steps,
  approval requests, and idempotent study-task creation. Automated tests verify that a
  research record cannot be read through another tenant, another tenant's document is never
  retrieved, a repeated tool call is stopped, and a task cannot be created before manager
  approval.
- A tenant-scoped agent API, read-only MCP stdio server, Google OAuth/OIDC verifier, and public
  Cloud Run deployment. The stable service URL provides a zero-login public page, deterministic
  cited demo, public evidence page, and optional live Google sign-in.
- A frozen 40-question retrieval evaluation and a measured semantic-embedding plus RRF comparison.

Not implemented yet: Google ADK orchestration, Cloud SQL deployment, the persistent public write
workflow, BigQuery metrics export, remote MCP transport, and completed human calibration. A
bounded live Vertex workflow and external stdio MCP-client validation are complete.

## Target Architecture

```text
User signs in with Google OAuth/OIDC
  -> Cloud Run API verifies identity, tenant, and role
  -> Google ADK agent chooses one typed MCP tool at a time
  -> policy validates arguments, permissions, and step budget
  -> tool reads tenant-scoped documents or research records
  -> approval is required before creating a study task
  -> every step is traced, measured, and stored
```

### Initial Tools

| Tool | Purpose | Can change data? |
| --- | --- | --- |
| `search_documents` | Find source-cited passages in papers and notes. | No |
| `get_research_record` | Read an allowlisted paper, experiment, or reading-status record. | No |
| `draft_study_note` | Turn grounded evidence into a structured draft. | No |
| `request_human_approval` | Create a durable request to perform a write. | Yes, approval request only |
| `create_study_task` | Create a follow-up task after approval. | Yes, approval required |

The model never writes SQL, accesses another tenant, bypasses approval, or runs without a limit. The first version stops after five tool calls or 30 seconds, refuses when evidence is insufficient, and prevents repeated identical calls.

## Architecture Decisions To Know

| Decision | Choice | Reason |
| --- | --- | --- |
| Existing QA path | Keep `/ask` deterministic | Gives a stable, testable baseline. |
| Agent framework | Google ADK | Shows Google-native agent orchestration and Gemini tool use. |
| Tool protocol | MCP | Makes application tools reusable by other compatible clients. |
| Authentication | Google Identity Platform with OAuth/OIDC | Produces verifiable user identity and role claims. |
| Data boundary | `tenant_id` on every tenant-owned record | Enforced in repositories, never trusted from model input. |
| Write safety | Human approval plus idempotency key | Prevents accidental or duplicate actions. |
| Primary database | Cloud SQL PostgreSQL with pgvector | Builds on the existing data model at suitable portfolio scale. |
| Runtime | Cloud Run API and ingestion worker | Low operational burden and independently scalable services. |
| Observability | OpenTelemetry, Cloud Trace, BigQuery | Supports debugging with evidence, latency, and cost data. |

## Delivery Roadmap

### 1. Local Agent Foundation

Add tenant, user, research-record, approval, agent-run-step, and study-task data models. Implement typed tools, a bounded local agent loop, durable approval state, and deterministic tests. `/ask` remains unchanged.

**Progress:** tenant, approval, task-idempotency, scoped document access, typed tools, a bounded
agent loop, and a dedicated tenant-checked agent API are implemented and tested.

**Done when:** a seeded user can ask about a paper, see each tool call, request a task, approve it, and create exactly one task even if the request is retried.

### 2. Identity And MCP

Add OAuth/OIDC token verification, role policy, tenant-scoped repository access, and an MCP server exposing the read tools and approval-gated write proposal.

**Done when:** cross-tenant reads and unapproved writes are rejected in automated tests.

**Progress:** Google ID-token verification and the tenant-pinned read-only MCP server are
implemented. The deployed OAuth flow is live-verified for the configured test user; unscoped
routes fail closed. The official Python MCP SDK has discovered and invoked both read tools through
a separate stdio process; cross-tenant record lookup and a model-supplied tenant override fail
safely. The write-proposal tool and remote transport remain.

### 3. Google Cloud Vertical Slice

Deploy Cloud Run API/worker, Cloud SQL, Cloud Storage, Pub/Sub, Vertex AI Gemini and embeddings, Secret Manager, and Terraform. Use a public, licensed paper corpus plus personal study material only where access remains private.

**Done when:** the live system ingests a document asynchronously and completes a traced, cited agent run.

**Progress:** the bounded API is deployed to Cloud Run with a stable public URL, scale-to-zero,
one-instance cap, public landing page, and read-only cited demo. A separate Cloud Run Job using
the same runtime identity has completed real Vertex embeddings, Gemini generation, cited-answer
validation, and refusal checks. Cloud SQL, worker ingestion, Secret Manager, Terraform, and
putting Vertex behind the persistent interactive workflow are still pending.

### 4. Evaluation And Operations

Build 40-60 labeled cases for retrieval, groundedness, answer quality, tool choice, refusal, action safety, and tenant isolation. Export redacted trace and cost records to BigQuery. Add load tests and failure experiments.

**Done when:** CI runs deterministic regression gates, the dashboard reports p50/p95 latency and cost, and each deliberately induced failure has a recorded fix and regression test.

## Metrics That Matter

- Retrieval recall@k and MRR: did the required evidence appear?
- Groundedness and citation validity: did the answer remain tied to evidence?
- Answer correctness: did it solve the labeled task?
- Tool-selection correctness: did it choose an allowed, useful tool?
- Action safety: were approval, role, tenant, and idempotency rules honoured?
- Refusal accuracy: did it decline when evidence or permission was missing?
- p50/p95 latency, tool error rate, tokens, and estimated cost per run.

Use labeled cases as the initial CI gate. Run LLM-as-judge alongside those labels until its agreement is measured; do not let an uncalibrated judge make builds flaky.

## Failure Experiments

| Failure | Evidence to inspect | Fix to demonstrate |
| --- | --- | --- |
| Endless tool loop | repeated tool names and step count in trace | step limit and duplicate-call guard |
| Tool timeout | span duration and timeout error | deadline, retry policy, and clear terminal response |
| Stale state | wrong session or approval reference | versioned state and approval binding |
| Retrieved prompt injection | source chunk and policy decision | treat retrieved text as data, never instructions |
| Duplicate action | repeated idempotency key | unique action record and safe replay response |
| Cost amplification | token/cost trace totals | per-run budget and bounded retries |

## End-to-End Validation Scenario

1. Sign in as a research assistant and ask a question about a paper.
2. Show retrieved evidence and the agent trace.
3. Ask for a study task; show that the assistant creates an approval request, not the task.
4. Sign in as a manager, approve the exact proposal, then create the task.
5. Show the trace, audit records, latency, cost, and the corresponding regression case.
6. Attempt a cross-tenant lookup and an unapproved write; show both being blocked.

## Incident Note Template

For each failure experiment, record the date, user-visible symptom, trace ID, measured impact,
root cause, policy or code change, before/after metric, and regression test.

## Honest Claim Rule

Only mark a component complete after it is implemented, tested, and demonstrable. This document
intentionally separates the current baseline from the target system.
