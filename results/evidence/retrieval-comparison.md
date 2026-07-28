# Frozen Retrieval Comparison

These are retrieval diagnostics over a fixed 40-question definition, not human judgments of
answer quality.

- Baseline: 64-dimensional local hashing embeddings with weighted hybrid fusion.
- Intervention: OpenAI `text-embedding-3-small` embeddings with reciprocal-rank fusion.
- Gold retrieval cases: 23 of the 40 questions contain predeclared gold chunk identifiers.

| Metric | Hashing + weighted | Semantic + RRF |
| --- | ---: | ---: |
| retrieval_eval_cases | 23 | 23 |
| retrieval_recall_at_k | 0.2609 | 0.2826 |
| retrieval_mrr | 0.1935 | 0.3022 |
| gold_citation_recall | 0.1087 | 0.2536 |
| mean_latency_ms | 31.77 | 311.88 |

The combined semantic-embedding plus RRF configuration produced higher point estimates at the cost
of higher latency. The paired uncertainty intervals for Recall@5 and MRR include zero, so this
small pilot does not establish a reliable improvement. See
[`retrieval-uncertainty.md`](retrieval-uncertainty.md). A complete published factorial comparison
is required before attributing any difference to RRF or semantic embeddings in isolation.

[`retrieval-comparison.trace.json`](retrieval-comparison.trace.json) records per-question top-k
chunk identifiers, gold ranks, and overlap without redistributing source text.
