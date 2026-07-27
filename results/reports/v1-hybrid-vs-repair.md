# V1: Hybrid versus Hybrid plus Evidence Repair

> These are deterministic diagnostic metrics, not human-annotation results.

- Hybrid: `v1-hybrid-gpt5mini-001`
- Hybrid+repair: `v1-hybrid-repair-gpt5mini-001`

| Metric | Hybrid | Hybrid+repair |
| --- | ---: | ---: |
| retrieval_eval_cases | 23 | 23 |
| retrieval_recall_at_k | 0.2609 | 0.2609 |
| retrieval_mrr | 0.1935 | 0.1935 |
| gold_citation_recall | 0.1957 | 0.1087 |
| fully_supported_rate | 0.0312 | 1.0 |
| partially_supported_rate | 0.5 | 0.0 |
| unsupported_rate | 0.4688 | 0.0 |
| structural_completeness_rate | 0.8 | 0.65 |
| refusal_precision | 0.8421 | 0.64 |
| appropriate_refusal_recall | 0.9412 | 0.9412 |
| answer_claim_count | 32 | 17 |
| mean_latency_ms | 4753.45 | 4753.45 |
| input_tokens | 88092 | 88092 |
| output_tokens | 10896 | 10896 |
| estimated_cost_usd | 0.043815 | 0.043815 |
