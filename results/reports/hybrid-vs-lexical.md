# Hybrid versus Lexical Retrieval

> These are deterministic diagnostic metrics, not human-annotation results.

- Hybrid: `pilot-hybrid-gpt5mini-001`
- Lexical: `pilot-lexical-gpt5mini-001`

| Metric | Hybrid | Lexical |
| --- | ---: | ---: |
| fully_supported_rate | 0.0909 | 0.0 |
| partially_supported_rate | 0.5455 | 0.5 |
| unsupported_rate | 0.3636 | 0.5 |
| refusal_precision | 0.8182 | 0.9 |
| appropriate_refusal_recall | 0.8182 | 0.8182 |
| answer_claim_count | 11 | 12 |
| mean_latency_ms | 4978.4 | 4942.75 |
| input_tokens | 44930 | 46690 |
| output_tokens | 4606 | 5368 |
| estimated_cost_usd | 0.0121789 | 0.0224085 |
