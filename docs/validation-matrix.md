# Validation Matrix

This matrix separates implemented code from behavior that has been tested, measured, or exercised
against a live external system.

| Capability | Status | Validated scope | Evidence |
|---|---|---|---|
| Retrieval comparison | Measured | Semantic embeddings plus RRF increased Recall@5 from 0.261 to 0.283 and MRR from 0.194 to 0.302 on the frozen 40-question set. RRF alone did not produce the improvement. | [`retrieval-comparison.md`](../results/evidence/retrieval-comparison.md) |
| Cloud Run | Live-validated | The bounded FastAPI service exposes public read-only routes and rejects missing or unscoped identity on protected routes. | Public `/`, `/demo`, `/evidence`, `/signin`, `/startup`, and `/health`; external route checks in [`deployment-evidence.md`](deployment-evidence.md) |
| Google OAuth/OIDC | Live-validated | The deployed verifier accepted a Google ID token for the configured audience and resolved tenant membership server-side. | Redacted validation record in [`deployment-evidence.md`](deployment-evidence.md) |
| Vertex AI | Live-validated | A Cloud Run Job used `gemini-embedding-001` and `gemini-2.5-flash`, recorded token/model provenance and retrieved chunks, validated an exact citation, and refused an unsupported query. | [`vertex-cloud-run-smoke.json`](../results/evidence/vertex-cloud-run-smoke.json) and [validation notes](integrations/vertex-ai-validation.md) |
| MCP | Client-validated | An external MCP client discovered both read-only stdio tools, invoked evidence and record lookup, and confirmed that cross-tenant access and tenant-argument injection fail safely. | [`mcp-stdio-validation.json`](../results/evidence/mcp-stdio-validation.json) and [validation notes](integrations/mcp-stdio-validation.md) |
| Human calibration | Incomplete | The rubric and two blank 72-record annotation templates exist. No human accuracy, agreement, or Cohen's kappa is published. | [Calibration runbook](human-calibration-runbook.md) |

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
- Do not imply a remote MCP deployment or MCP write tools: the validation uses an official
  external SDK client talking to a separate local stdio process.
- Do not invent or round an agreement statistic before the aligned labels exist.
- Keep the negative result: RRF by itself degraded the hashing baseline; the measured improvement
  came from the combined semantic-embedding plus RRF configuration.
