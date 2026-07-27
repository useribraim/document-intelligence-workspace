# V1: Hashing Weighted versus TE3S RRF

> These are deterministic diagnostic metrics, not human-annotation results.

- Hashing+weighted: `v1-hashing-weighted-deterministic-001`
- TE3S+RRF: `v1-te3s-rrf-deterministic-001`

| Metric | Hashing+weighted | TE3S+RRF |
| --- | ---: | ---: |
| retrieval_eval_cases | 23 | 23 |
| retrieval_recall_at_k | 0.2609 | 0.2826 |
| retrieval_mrr | 0.1935 | 0.3022 |
| gold_citation_recall | 0.1087 | 0.2536 |
| fully_supported_rate | 0.9917 | 0.9917 |
| partially_supported_rate | 0.0083 | 0.0083 |
| unsupported_rate | 0.0 | 0.0 |
| structural_completeness_rate | 0.575 | 0.575 |
| refusal_precision | 0.0 | 0.0 |
| appropriate_refusal_recall | 0.0 | 0.0 |
| answer_claim_count | 120 | 120 |
| mean_latency_ms | 31.77 | 311.88 |
| input_tokens | 0 | 0 |
| output_tokens | 0 | 0 |
| estimated_cost_usd | 0.0 | 0.0 |
