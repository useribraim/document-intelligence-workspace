# Claim-to-Evidence Audit

The claim audit tests a narrower question than general answer quality: does each cited source span
support the exact sentence-level claim attached to it?

## Frozen definition

- Ten versioned research papers recorded in
  [`corpus_manifest.jsonl`](../data/audit/corpus_manifest.jsonl).
- Forty predeclared questions in
  [`v1_40_gold.jsonl`](../data/audit/questions/v1_40_gold.jsonl).
- Six task families plus explicit refusal-required cases.
- Gold chunk identifiers for the 23 retrieval-scored questions.
- A versioned human rubric in [`annotation-rubric.md`](annotation-rubric.md).

Raw paper text is not redistributed. See [`corpus-provenance.md`](corpus-provenance.md) for the
licence boundary and local hash-verification process.

## Recorded retrieval experiment

The curated comparison evaluates:

1. local 64-dimensional hashing embeddings with weighted hybrid fusion;
2. `text-embedding-3-small` embeddings with reciprocal-rank fusion.

Recall@5, MRR, gold-citation recall, latency, per-question top-k identifiers, and gold ranks are in
[`results/evidence/`](../results/evidence/). The result applies to the combined semantic-plus-RRF
configuration; a separate control found no improvement from RRF alone.

## Support diagnostics

The code can split answers into cited claims, align citation labels to retrieved chunks, verify
source existence, and apply deterministic token-overlap diagnostics. These checks find broken
references and obvious support failures, but they are not human labels.

Model-assisted review artifacts are generated locally and are excluded from the public repository.
They may help prioritize inspection but cannot be used as ground truth or to report Cohen's kappa.

## Human boundary

The V2 calibration instrument includes two blank 252-record templates. Agreement is calculated
only after two distinct people complete all records, including the 112 aligned claim-citation
pairs. See [`human-calibration-runbook.md`](human-calibration-runbook.md).

Until that gate passes:

- no human-calibrated accuracy is published;
- no inter-annotator agreement is published;
- automated support rates are described only as diagnostics;
- evidence repair is not described as a human-validated improvement.
