# Evaluation Method

Evaluation is split into deterministic regression checks, frozen retrieval measurements, external
integration smoke tests, and unfinished human calibration. Those categories are not interchangeable.

## Deterministic regression suite

The test suite covers:

- ingestion, normalization, chunking, and persistence;
- lexical, vector, hybrid, and reciprocal-rank-fusion retrieval;
- exact-quote citation validation and insufficient-evidence gate behavior;
- schema-shaped generation and AI-run persistence;
- tenant-scoped document and research-record access;
- approval and idempotency boundaries;
- OIDC claim verification;
- MCP tool schemas and tenant pinning;
- the bounded Vertex workflow with injected deterministic providers.

The optional PostgreSQL integration test runs only when `POSTGRES_TEST_DATABASE_URL` is configured.

## Frozen retrieval comparison

The 40-question definition includes direct extraction, synthesis, multi-claim, conflicting
evidence, misleadingly relevant evidence, insufficient evidence, and refusal-required cases. Future
automated refusal recall is computed only on the explicit `refusal_required` category; refusal
precision remains a human-judgement field.
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

The V2 controlled bank contains 140 questions: 28 each for supported, partially supported,
unsupported, misleading-context, and refusal cases. Each blinded packet has 140 answer-level
records and 112 aligned claim-citation pairs. Its five variants deliberately share each of 28 source
seeds, so it is useful for rubric development but not an independent 140-item evaluation sample.
Deterministic generation checks counts, hashes, case balance, packet alignment, label blankness, and
the absence of author strata from reviewer files.

Human support labels remain pending. The author-designed strata and automated diagnostics are not
used as substitutes. Any publishable agreement or calibration result must use the independent-item
[V3 design](calibration-v3-design.md), not a naive V2 item-level interval.

## Anti-gaming rules

- Questions and gold chunk IDs are frozen before comparison.
- Controlled calibration strata are removed from the annotator packets.
- Failing cases remain visible.
- A combined intervention is not attributed to one component without a control.
- Exact citations are validated against the chunks used for generation.
- Agreement is reported before adjudication and without target-driven rounding.
