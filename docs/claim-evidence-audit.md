# Claim-to-Evidence Audit

A citation can be real, relevant, and still fail to support the claim it appears to justify. This workspace now contains a reproducible pilot measurement path for that gap.

## Frozen pilot corpus

The corpus is frozen at 10 open-access ML/RAG papers. Their canonical URLs, fixed arXiv versions, access basis, retrieval date, text paths, and SHA-256 values are in [corpus_manifest.jsonl](../data/audit/corpus_manifest.jsonl). The repository retains extracted text, not PDFs.

Verify that no pilot source changed:

```bash
.venv/bin/python -m diw.cli corpus-verify
```

## What is implemented

- Atomic claim extraction from source-cited answers.
- Claim-to-citation mappings with exact evidence spans and document/version provenance.
- A deterministic triage verifier that separates source existence, topical relevance, and support labels.
- Saved JSON run artifacts with exact retrieved passages, passage and answer SHA-256 hashes, prompt/retrieval hashes, token usage, retry count, timing, and pinned-rate cost estimates.
- A 45-second API timeout and per-question checkpoint/resume option for reliable paid runs.
- Frozen pilot questions, paired lexical/hybrid configurations, and unit tests.
- An optional OpenAI text-embedding provider (`--embedding-provider openai`, default `text-embedding-3-small`) whose vectors are stored under their own model name alongside the local hashing embeddings.

Run the fixed hybrid baseline first, then lexical, after loading all frozen texts into the same database:

```bash
.venv/bin/python -m diw.cli claim-audit --mode hybrid --top-k 5 --out results/runs/pilot-hybrid.json
.venv/bin/python -m diw.cli claim-audit --mode lexical --top-k 5 --out results/runs/pilot-lexical.json
```

The pinned OpenAI runs used in this pilot are available as Make targets. They use the same frozen SQLite corpus, GPT-5 mini snapshot, prompt, top-k, and completion cap; only retrieval mode or the predeclared gate changes:

```bash
make run-hybrid-openai
make run-lexical-openai
make run-hybrid-gated-openai
make compare-hybrid-lexical
make compare-hybrid-gated
```

For a meaningful claim-support experiment, use the same pinned external generator for both configurations, for example `--llm-provider openai --llm-model <pinned-model-id>`. GPT-5 snapshots use the provider-default temperature because the API rejects an explicit `0`; preserve the snapshot, prompt hash, completion cap, and all retrieval settings. The default deterministic provider copies source text and is suitable for pipeline checks, not for measuring generative citation failures.

The generated `annotations_pending` entries are review aids, not final labels. `quote_alignment` makes explicit whether the model copied a span exactly, a source span was canonicalized after a model paraphrase, or a cited source was resolved from a model-used label. Human annotation must use `fully_supported`, `partially_supported`, `unsupported`, `contradicted`, or `not_applicable`, record rationale and adjudications, and remain the reported ground truth.

Create human-owned annotation files and compare two completed passes:

```bash
.venv/bin/python -m diw.cli annotation-template --run results/runs/pilot-hybrid.json --questions data/audit/questions/pilot_v0.jsonl --out data/audit/annotations/pilot_hybrid_a1.jsonl
.venv/bin/python -m diw.cli annotation-agreement --first data/audit/annotations/pilot_hybrid_a1.jsonl --second data/audit/annotations/pilot_hybrid_reannotation_a1.jsonl
```

## Current pilot artifacts (automated diagnostics only)

The completed frozen 20-question runs are in `results/runs/`; their comparison tables are in `results/reports/`.

| Run | Claim pairs | Automated unsupported rate | Refusal precision | Estimated API cost |
| --- | ---: | ---: | ---: | ---: |
| Hybrid baseline | 11 | 0.3636 | 0.8182 | $0.01217890 |
| Lexical control | 12 | 0.5000 | 0.9000 | $0.02240850 |
| Hybrid strict gate | 0 | n/a | 0.5500 | $0.01004770 |
| Hybrid supported gate | 6 | 0.0000 | 0.7143 | $0.01978050 |

The strict gate removed every generated claim because none of its deterministic pre-labels was `fully_supported`; it increased automated refusal recall to 1.0 but is not an improvement result. The separately named supported gate removed the automated-unsupported claims (11 claims to 6) but retained partially-supported ones, so it too is only a calibration diagnostic. Do not put any of these automated rates on a resume or describe them as human annotation.

## V1 controlled evaluation: 40 questions with gold chunks

`data/audit/questions/v1_40_gold.jsonl` doubles the question set while leaving the frozen 10-paper corpus untouched. It contains six cases each for direct extraction, synthesis, multi-claim, conflicting evidence, misleadingly relevant evidence, and insufficient evidence, plus four refusal-required cases. Every answerable case has pre-recorded gold chunk IDs; `--require-gold-evidence` fails fast if that invariant is broken.

The three controlled artifacts below use the same pinned GPT-5 mini snapshot, prompt, top-k (5), frozen SQLite corpus, and 40 questions. Retrieval is the only variable in the lexical/hybrid comparison. Evidence repair is deliberately an offline transformation of the saved hybrid artifact: it reuses the exact answers and passages, rewrites partial claims to their canonical cited spans, and removes unsupported claims. It therefore adds no API cost and does not confound the repair comparison with another sampled generation run.

```bash
make run-v1-lexical-openai
make run-v1-hybrid-openai
make run-v1-hybrid-repair-openai
make compare-v1-lexical-hybrid
make compare-v1-hybrid-repair
```

| Automated diagnostic | Lexical | Hybrid | Hybrid + offline repair |
| --- | ---: | ---: | ---: |
| Gold chunk Recall@5 | 0.2609 | 0.2609 | 0.2609 |
| Gold chunk MRR | 0.2688 | 0.1935 | 0.1935 |
| Gold-evidence citation recall | 0.2246 | 0.1957 | 0.1087 |
| Fully-supported claim-pair rate | 0.0357 | 0.0312 | 1.0000 |
| Structural completeness proxy | 0.7750 | 0.8000 | 0.6500 |
| Appropriate-refusal recall | 0.9412 | 0.9412 | 0.9412 |
| Refusal precision | 0.8889 | 0.8421 | 0.6400 |
| Estimated API cost | $0.04504575 | $0.04381500 | $0 incremental |

These are deterministic diagnostics, not adjudicated outcomes. **The repair's 100% fully-supported rate is circular as an effectiveness result:** repair uses this deterministic verifier's labels to retain or rewrite claims, then that same verifier scores the transformed claims. It proves conformance to this verifier's definition of support, not that the definition agrees with human judgement. The V0 model-assisted review already disagreed materially with the deterministic verifier, so absolute V1 claim-support rates are currently measures of verifier behaviour as well as model behaviour. Independent human annotation is the calibration gate that can break this circularity; until then, do not call the repair result factuality improvement.

Repair reaches 100% only over the 17 claim-citation pairs it retains; it raises refusal rate from 47.5% to 62.5% and reduces the preregistered structural-completeness proxy from 80.0% to 65.0%. The result is therefore a selectivity diagnostic, not a validated factuality result.

`structural_completeness_rate` is explicitly a preregistered claim-count proxy (one claim for extraction/conflict, two for synthesis, three for multi-claim, or a refusal for insufficient-evidence cases). Human `answer_completeness` remains the authoritative semantic label. The saved artifacts also report passage-level Recall@5, MRR, citation coverage, latency, token usage, and version-pinned cost.

The lexical/hybrid comparison is a single sampled generation per arm because the pinned GPT-5 snapshot accepts only provider-default temperature. Retrieval metrics are deterministic, but generated-answer and verifier-derived metrics need a duplicate-run variance check before being described as stable. `make run-v1-hybrid-replica-openai` produces that second sample with a distinct run ID and saved provenance.

The completed duplicate hybrid sample confirms that warning: retrieval Recall@5/MRR and gold-citation recall are identical, but the verifier-derived fully-supported rate changes from 0.0312 to 0.0000, refusal rate from 0.4750 to 0.4500, and refusal precision from 0.8421 to 0.8889. The pair is saved as `v1-hybrid-gpt5mini-001` and `v1-hybrid-gpt5mini-002`; run `make compare-v1-hybrid-replica` for the exact variance table. Treat this as evidence of sampling variation, not a confidence interval or model-quality result.

For retrieval diagnosis, `make trace-v1-lexical-hybrid` writes a per-question trace with gold ranks and both top-5 lists. On V1, hybrid changes the top-5 list on all 40 questions (mean overlap 2.875/5); it gains one gold hit and loses three, leaving aggregate Recall@5 tied while degrading MRR. The current vector component is a 64-dimensional local hashing embedding intended for API-free development, so the trace supports a concrete hypothesis—noisy or weak semantic ranks—not a claim that hybrid fusion is inherently worse. Test a stronger embedding or calibrated fusion on the unchanged V1 set after human calibration.

The stronger-embedding arm is now prepared: the OpenAI embedding provider is implemented and tested, and `make run-v1-hybrid-te3s-openai`, `make compare-v1-hybrid-embeddings`, and `make trace-v1-hybrid-embeddings` pin the same frozen corpus, questions, generator, prompt, top-k, and completion cap, changing only the embedding model. The saved lexical arm remains the valid control because lexical ranking ignores embeddings. The arm is deliberately held until the V1 human calibration (blind re-annotation, second annotator, and adjudication) is complete, so the predeclared hypothesis is evaluated against adjudicated labels rather than the uncalibrated verifier.

## Model-assisted adjudication pass

`pilot_hybrid_ai_adjudication_v1.jsonl` is a completed, rationale-bearing review of all 20 answer-level and 11 claim-citation records. It is explicitly labelled `model_assisted_exact_span_review_v1`, and its decisions are reproduced from [the decision map](../data/audit/annotations/pilot_hybrid_ai_adjudication_v1.decisions.json) rather than silently overwriting the original packet.

Its descriptive result is 3 fully supported, 7 partially supported, and 1 unsupported claim-citation pair. This is useful for debugging and interview walkthroughs, but it is not independent human ground truth and must not be described as a human annotation result or used to report Cohen's kappa. The `annotation-apply-decisions` and `annotation-summary` commands make this distinction machine-readable.

## Limits

This is a completed reproducible pilot run, not a completed benchmark: no LLM judge is ground truth, and the model-assisted labels still require independent human annotation and the planned blind re-annotation/second-annotator check before results are publishable. The corpus remains frozen and questions/rubric were authored before these saved runs.
