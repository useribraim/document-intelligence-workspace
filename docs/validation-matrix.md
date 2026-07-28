# Validation Matrix

This matrix separates implemented code from behavior that has been tested, measured, or exercised
against a live external system.

| Capability | Status | Validated scope | Evidence |
|---|---|---|---|
| Retrieval comparison | Measured; inconclusive | The combined semantic-plus-RRF arm had higher point estimates on 23 gold-scored questions, but paired 95% bootstrap intervals for Recall@5 and MRR include zero. | [`retrieval-comparison.md`](../results/evidence/retrieval-comparison.md) and [`retrieval-uncertainty.md`](../results/evidence/retrieval-uncertainty.md) |
| Cloud Run | Live-validated | The bounded FastAPI service exposes public read-only routes and rejects missing or unscoped identity on protected routes. | Public `/`, `/demo`, `/evidence`, `/signin`, `/startup`, and `/health`; external route checks in [`deployment-evidence.md`](deployment-evidence.md) |
| Google OAuth/OIDC | Live-validated | The deployed verifier accepted a Google ID token for the configured audience and resolved tenant membership server-side. | Redacted validation record in [`deployment-evidence.md`](deployment-evidence.md) |
| Vertex AI | Live-validated | A Cloud Run Job used `gemini-embedding-001` and `gemini-2.5-flash`, recorded token/model provenance and retrieved chunks, validated an exact citation, and refused an unsupported query. | [`vertex-cloud-run-smoke.json`](../results/evidence/vertex-cloud-run-smoke.json) and [validation notes](integrations/vertex-ai-validation.md) |
| Google ADK | Live-validated | A ReAct-style coordinator used two real ADK `AgentTool` specialists for hierarchical retrieval and citation-verification delegation on Vertex AI, with per-call token, latency, throughput, and estimated-cost records. | [`adk-cloud-run-smoke.json`](../results/evidence/adk-cloud-run-smoke.json) and [validation notes](integrations/adk-validation.md) |
| MCP | Client-validated | An external MCP client discovered both read-only stdio tools, invoked evidence and record lookup, and confirmed that cross-tenant access and tenant-argument injection fail safely. | [`mcp-stdio-validation.json`](../results/evidence/mcp-stdio-validation.json) and [validation notes](integrations/mcp-stdio-validation.md) |
| Human calibration | Controlled bank only; labels incomplete | V2 has 28 source seeds expressed as five controlled variants each. It is useful for rubric development but is not an independent 140-item sample. No human accuracy, agreement, or Cohen's kappa is published. | [Calibration manifest](../data/audit/calibration/v2_manifest.json) and [runbook](human-calibration-runbook.md) |

## Interpretation boundaries

- Do not describe the current public demo as the persistent production workflow: it uses bundled
  synthetic sources, deterministic extractive generation, and zero write tools.
- Do not call the Cloud Run service public until an unauthenticated browser can reach the landing
  page and protected actions still require a verified token.
- Do not call a mocked provider test a live OAuth configuration.
- Do not call deterministic verifier labels, model-assisted review, or self-review independent
  human calibration.
- Do not imply that the public interactive demo uses Vertex: the live proof is a separate Cloud
  Run Job over bundled sources and an ephemeral database.
- Do not imply that ADK is the public interactive request path or that one smoke run is a
  production benchmark; the live proof is a bounded Cloud Run Job.
- Do not imply a remote MCP deployment or MCP write tools: the validation uses an official
  external SDK client talking to a separate local stdio process.
- Do not invent or round an agreement statistic before the aligned labels exist.
- Do not claim that the observed combined configuration improved retrieval or attribute an effect
  to RRF or embeddings until a complete factorial, held-out evaluation reports uncertainty.
