# Resume Claim Gates

This file keeps ambitious resume language tied to evidence. A claim moves from `blocked` to
`released` only when its acceptance evidence exists in the repository or in a reproducible live
check. Skeleton code is useful project evidence, but it is not the same as a deployed or
human-validated result.

| Capability | Status | Resume wording after release | Release evidence |
|---|---|---|---|
| Stronger retrieval | Released | Improved frozen-set retrieval with semantic embeddings and RRF, increasing Recall@5 from 0.261 to 0.283 and MRR from 0.194 to 0.302 across 40 questions. | `results/reports/v1-hashing-weighted-vs-te3s-rrf.md` and its trace |
| Cloud Run | Released | Deployed the bounded FastAPI service to Cloud Run and published a public read-only cited demo. | Public `/`, `/demo`, `/evidence`, `/signin`, `/startup`, and `/health`; recorded revision `document-intelligence-workspace-00004-26q`; external route smoke results |
| Google OAuth/OIDC | Released | Configured and live-verified Google OAuth on Cloud Run, with ID-token audience checks and server-side tenant membership enforcement. | Deployed `/signin` completed and `/auth/whoami` rendered the verified test account |
| Vertex AI | Released | Ran a Cloud Run RAG smoke workflow using Vertex AI `gemini-embedding-001` embeddings and `gemini-2.5-flash` generation, with token/model provenance, retrieved chunks, exact-quote citation validation, and unsupported-query refusal recorded per run. | Cloud Run Job execution `diw-vertex-smoke-z5m7b`; `results/integrations/vertex/vertex-cloud-run-smoke.json`; [validation record](integrations/vertex-ai-validation.md) |
| MCP | Released | Built a tenant-pinned, read-only MCP stdio server and validated tool discovery, evidence search, record lookup, and cross-tenant denial through an external MCP client. | `results/integrations/mcp/mcp-stdio-validation.json`; `configs/mcp-client.example.json`; [validation record](integrations/mcp-stdio-validation.md) |
| Human calibration | Blocked | Built a 10-paper, 40-question evaluation with 32 claim-citation pairs independently labeled by two humans; report raw agreement and Cohen's kappa exactly as measured. | 72/72 primary and 72/72 independent records completed; 32/32 aligned claim pairs; `results/reports/v1-human-agreement.json`; dated adjudication record; [release runbook](human-calibration-runbook.md) |

## Non-negotiable wording boundaries

- Do not describe the current public demo as the persistent production workflow: it uses bundled
  synthetic sources, deterministic extractive generation, and zero write tools.
- Do not call the Cloud Run service public until an unauthenticated browser can reach the landing
  page and protected actions still require a verified token.
- Do not call a mocked provider test a live OAuth configuration.
- Do not call deterministic verifier labels, model-assisted review, or self-review independent
  human calibration.
- Do not imply that the public interactive demo uses Vertex: the live proof is a separate Cloud
  Run Job over bundled sources and an ephemeral database.
- Do not imply a remote MCP deployment or MCP write tools: the released proof is an official
  external SDK client talking to a separate local stdio process.
- Do not invent or round an agreement statistic before the aligned labels exist.
- Keep the negative result: RRF by itself degraded the hashing baseline; the measured improvement
  came from the combined semantic-embedding plus RRF configuration.
