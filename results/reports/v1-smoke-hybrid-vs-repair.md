# V1 smoke: Hybrid versus Evidence Repair

> These are deterministic diagnostic metrics, not human-annotation results.

- Hybrid: `v1-hybrid-deterministic-smoke`
- Hybrid+repair: `v1-hybrid-repair-deterministic-smoke`

| Metric | Hybrid | Hybrid+repair |
| --- | ---: | ---: |
| retrieval_eval_cases | 23 | 23 |
| retrieval_recall_at_k | 0.2609 | 0.2609 |
| retrieval_mrr | 0.1935 | 0.1935 |
| gold_citation_recall | 0.1087 | 0.1087 |
| fully_supported_rate | 0.9917 | 1.0 |
| partially_supported_rate | 0.0083 | 0.0 |
| unsupported_rate | 0.0 | 0.0 |
| structural_completeness_rate | 0.575 | 0.575 |
| refusal_precision | 0.0 | 0.0 |
| appropriate_refusal_recall | 0.0 | 0.0 |
| answer_claim_count | 120 | 120 |
| mean_latency_ms | 52.08 | 28.62 |
| input_tokens | 0 | 0 |
| output_tokens | 0 | 0 |
| estimated_cost_usd | 0.0 | 0.0 |
