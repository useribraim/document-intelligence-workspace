# Google ADK Validation

## Live result

On 2026-07-28, Cloud Run Job execution `diw-adk-smoke-bpdk8` completed a real
Google ADK workflow on Vertex AI using `gemini-2.5-flash`.

The `research_coordinator` followed a bounded ReAct-style cycle:

1. plan the evidence request;
2. delegate retrieval to the `retrieval_specialist`;
3. observe the retrieved chunk;
4. delegate exact-span checking to the `citation_verification_specialist`;
5. return the verified answer.

Both specialists are ADK `AgentTool` instances, not ordinary functions renamed as agents.
The specialists can call deterministic read-only tools; no write tool is exposed to this workflow.

## Recorded economics

| Measure | One smoke execution |
|---|---:|
| Model calls | 7 |
| Input tokens | 3,771 |
| Output tokens | 695 |
| Thinking tokens | 505 |
| End-to-end workflow latency | 13,823.18 ms |
| Aggregate output throughput | 69.14 tokens/s |
| Estimated model cost | $0.00263435 |

Cost uses a version-pinned 2026-07-28 Vertex AI pricing snapshot. It is an estimate derived from
recorded ADK usage metadata, not a Cloud Billing export. Throughput divides output tokens by
measured model-call latency and does not include Cloud Run provisioning time.

The redacted result, including every model call and both delegation arguments, is stored in
[`adk-cloud-run-smoke.json`](../../results/evidence/adk-cloud-run-smoke.json).

## Reproduction

```bash
python -m pip install -e ".[adk]"
export DIW_GCP_PROJECT="your-project"
export DIW_CLOUD_RUN_SERVICE="document-intelligence-workspace"
./scripts/run_adk_cloud_smoke.sh
```

The script resolves the deployed service image, deploys an isolated scale-to-zero Cloud Run Job,
runs the coordinator, reads its structured result from Cloud Logging, and writes a compact
credential-free evidence artifact.

## Claim boundary

This validates real Google ADK orchestration, hierarchical specialist delegation, Vertex model
calls, and per-request token/latency/cost instrumentation. It does not establish a production
multi-agent benchmark, human-rated answer quality, durable Cloud SQL state, or a public
interactive ADK endpoint.
