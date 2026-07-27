# Evaluation Method

Evaluation is split into deterministic regression checks, frozen retrieval measurements, external
integration smoke tests, and unfinished human calibration. Those categories are not interchangeable.

## Deterministic regression suite

The test suite covers:

- ingestion, normalization, chunking, and persistence;
- lexical, vector, hybrid, and reciprocal-rank-fusion retrieval;
- exact-quote citation validation and unsupported-evidence refusal;
- schema-shaped generation and AI-run persistence;
- tenant-scoped document and research-record access;
- approval and idempotency boundaries;
- OIDC claim verification;
- MCP tool schemas and tenant pinning;
- the bounded Vertex workflow with injected deterministic providers.

The optional PostgreSQL integration test runs only when `POSTGRES_TEST_DATABASE_URL` is configured.

## Frozen retrieval comparison

The 40-question definition includes direct extraction, synthesis, multi-claim, conflicting
evidence, misleadingly relevant evidence, insufficient evidence, and refusal-required cases.
Twenty-three questions contain predeclared gold chunk identifiers and contribute to Recall@5 and
MRR.

The public comparison is
[`retrieval-comparison.md`](../results/evidence/retrieval-comparison.md). The per-question trace
contains identifiers and ranks but no full paper text.

## External integrations

Vertex AI and MCP are validated separately from unit tests:

- the Vertex Cloud Run Job records provider/model identifiers, tokens, chunks, citations, refusal,
  prompt/run IDs, latency, and errors;
- the external MCP client records initialization, tool discovery, arguments, results, cross-tenant
  denial, and tenant-argument injection behavior.

Both redacted records are under [`results/evidence/`](../results/evidence/).

## Human calibration

Human support labels are pending. Automated support diagnostics and model-assisted review are not
used as substitutes. The completion gate is defined in
[`human-calibration-runbook.md`](human-calibration-runbook.md).

## Anti-gaming rules

- Questions and gold chunk IDs are frozen before comparison.
- Failing cases remain visible.
- A combined intervention is not attributed to one component without a control.
- Exact citations are validated against the chunks used for generation.
- Agreement is reported before adjudication and without target-driven rounding.
