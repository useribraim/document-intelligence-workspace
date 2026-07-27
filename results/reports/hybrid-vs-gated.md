# Automated Claim-Verification Gate Comparison

> These are deterministic diagnostic metrics, not human-annotation results.

- Baseline: `pilot-hybrid-gpt5mini-001`
- Verification gate: `pilot-hybrid-gated-gpt5mini-001`

| Metric | Baseline | Verification gate |
| --- | ---: | ---: |
| fully_supported_rate | 0.0909 | 0.0 |
| partially_supported_rate | 0.5455 | 0.0 |
| unsupported_rate | 0.3636 | 0.0 |
| refusal_precision | 0.8182 | 0.55 |
| appropriate_refusal_recall | 0.8182 | 1.0 |
| answer_claim_count | 11 | 0 |
| mean_latency_ms | 4978.4 | 3956.9 |
| input_tokens | 44930 | 44930 |
| output_tokens | 4606 | 4246 |
| estimated_cost_usd | 0.0121789 | 0.0100477 |
