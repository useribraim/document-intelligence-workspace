# Human Calibration Runbook

Automated citation checks, author-designed case strata, and model-assisted review are not human
ground truth. This runbook defines the work required before publishing agreement or calibrated
support numbers.

## Instrument

| Item | Count |
|---|---:|
| Questions | 140 |
| Supported cases | 28 |
| Partial-support cases | 28 |
| Unsupported cases | 28 |
| Misleading-context cases | 28 |
| Refusal cases | 28 |
| Answer-level records per annotator | 140 |
| Claim-citation pairs per annotator | 112 |
| Total records per annotator | 252 |

The deterministic source is
[`v2_case_seeds.jsonl`](../data/audit/calibration/v2_case_seeds.jsonl). The generated manifest
records file hashes and counts. Short canonicalized excerpts are used instead of redistributing
paper text.

Rebuild or verify the instrument with:

```bash
make calibration-build
make calibration-check
```

The verifier fails if the suite leaves the 120–160-question range, any of the five strata is
imbalanced, either packet has fewer than 100 pairs, records are not aligned, author strata leak
into the packets, or a human label is prefilled.

## Independent annotation

Run the sessions separately:

```bash
make annotate-primary
make annotate-independent
```

Each annotator must see only their generated template and the
[`v2.0 rubric`](annotation-rubric.md). Outputs are written under the ignored
`data/audit/annotations/local/` directory.

The second annotator must not see:

- the case-seed file or question metadata;
- the primary labels or rationales;
- automated support diagnostics;
- comparison reports or target agreement values.

## Agreement and adjudication

After both people complete all 252 records:

```bash
make annotation-agreement
```

The command fails closed unless at least 100 aligned claim-citation pairs are labeled and the
annotator identifiers are distinct. Save the pre-adjudication report before discussion. It contains
raw agreement, Cohen's kappa, a confusion matrix, input hashes, and every disagreement.

Adjudication is a separate dated record containing both original labels, both rationales, the final
decision, and the reason. Never overwrite either source packet.

## What completion will and will not prove

Completing this instrument can support:

- agreement on 112 controlled claim-citation pairs;
- label distributions and confusion patterns across the five balanced strata;
- calibration of automated support diagnostics against adjudicated human labels.

It does not by itself measure end-to-end retrieval recall, live-model answer quality, or production
performance. Those require separate system-run evaluations over the 140-question definition.

Until two people finish and adjudicate the packet, no human accuracy, agreement, or Cohen's kappa
is published.
