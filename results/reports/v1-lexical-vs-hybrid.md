# V1: Lexical versus Hybrid Retrieval

> These are deterministic diagnostic metrics, not human-annotation results.

- Lexical: `v1-lexical-gpt5mini-001`
- Hybrid: `v1-hybrid-gpt5mini-001`

| Metric | Lexical | Hybrid |
| --- | ---: | ---: |
| retrieval_eval_cases | 23 | 23 |
| retrieval_recall_at_k | 0.2609 | 0.2609 |
| retrieval_mrr | 0.2688 | 0.1935 |
| gold_citation_recall | 0.2246 | 0.1957 |
| fully_supported_rate | 0.0357 | 0.0312 |
| partially_supported_rate | 0.6786 | 0.5 |
| unsupported_rate | 0.2857 | 0.4688 |
| structural_completeness_rate | 0.775 | 0.8 |
| refusal_precision | 0.8889 | 0.8421 |
| appropriate_refusal_recall | 0.9412 | 0.9412 |
| answer_claim_count | 28 | 32 |
| mean_latency_ms | 4848.12 | 4753.45 |
| input_tokens | 88103 | 88092 |
| output_tokens | 11510 | 10896 |
| estimated_cost_usd | 0.04504575 | 0.043815 |
