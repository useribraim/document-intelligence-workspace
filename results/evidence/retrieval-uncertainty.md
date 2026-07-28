# Retrieval Comparison: Paired Uncertainty Analysis

This analysis uses the 23 questions with gold chunk annotations in
[`retrieval-comparison.trace.json`](retrieval-comparison.trace.json). It samples those questions
as paired units 200,000 times with a fixed seed (`20260728`) and reports the percentile 95%
interval for the right-run minus left-run difference.

| Metric | Point difference | Paired-bootstrap 95% interval |
| --- | ---: | ---: |
| Recall@5 | +0.0217 | [-0.1159, +0.1594] |
| MRR | +0.1087 | [-0.0391, +0.2457] |

Six of the 23 gold-scored questions changed Recall@5. The intervals include zero, so this pilot
does **not** establish that the semantic-plus-RRF configuration is superior to the hashing-plus-
weighted baseline. The repository reports the point estimates as observed configuration results,
not as a reliable improvement.

The trace does not contain per-question emitted-citation data, so no uncertainty interval is
published for gold-citation recall. A complete factorial comparison (hashing/semantic × weighted/
RRF) is also not currently published; it is required before attributing any difference to one
intervention.

Reproduce:

```bash
.venv/bin/python scripts/analyze_retrieval_uncertainty.py \
  --trace results/evidence/retrieval-comparison.trace.json
```
