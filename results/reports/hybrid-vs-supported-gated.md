# Hybrid versus Supported Claim Gate

> These are deterministic diagnostic metrics, not human-annotation results.

- Hybrid: `pilot-hybrid-gpt5mini-001`
- Supported-gate: `pilot-hybrid-supported-gated-gpt5mini-001`

| Metric | Hybrid | Supported-gate |
| --- | ---: | ---: |
| fully_supported_rate | 0.0909 | 0.0 |
| partially_supported_rate | 0.5455 | 1.0 |
| unsupported_rate | 0.3636 | 0.0 |
| refusal_precision | 0.8182 | 0.7143 |
| appropriate_refusal_recall | 0.8182 | 0.9091 |
| answer_claim_count | 11 | 6 |
| mean_latency_ms | 4978.4 | 3842.05 |
| input_tokens | 44930 | 44930 |
| output_tokens | 4606 | 4274 |
| estimated_cost_usd | 0.0121789 | 0.0197805 |
