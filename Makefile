.PHONY: test reproduce-pilot run-hybrid-openai run-lexical-openai run-hybrid-gated-openai run-hybrid-supported-gated-openai compare-hybrid-lexical compare-hybrid-gated compare-hybrid-supported-gated run-v1-lexical-openai run-v1-hybrid-openai run-v1-hybrid-replica-openai run-v1-hybrid-repair-openai compare-v1-lexical-hybrid compare-v1-hybrid-repair compare-v1-hybrid-replica trace-v1-lexical-hybrid freeze-v1-evaluation materialize-v1-verifier-reference run-v1-hybrid-te3s-openai compare-v1-hybrid-embeddings trace-v1-hybrid-embeddings validate-mcp-stdio prepare-v1-second-full annotate-v1-primary annotate-v1-second annotate-v1-recheck annotate-v1-repair summarize-v1-primary agree-v1-annotators

test:
	.venv/bin/python -m unittest discover -s tests -v

reproduce-pilot:
	.venv/bin/python -m diw.cli claim-audit --mode lexical --out results/runs/pilot-lexical.json
	.venv/bin/python -m diw.cli claim-audit --mode hybrid --out results/runs/pilot-hybrid.json

run-hybrid-openai:
	.venv/bin/python -m diw.cli claim-audit --database-url sqlite:////private/tmp/claim_evidence_audit_pilot.db --mode hybrid --top-k 5 --llm-provider openai --llm-model gpt-5-mini-2025-08-07 --max-output-tokens 1200 --run-id pilot-hybrid-gpt5mini-001 --out results/runs/pilot-hybrid-gpt5mini-001.json

run-lexical-openai:
	.venv/bin/python -m diw.cli claim-audit --database-url sqlite:////private/tmp/claim_evidence_audit_pilot.db --mode lexical --top-k 5 --llm-provider openai --llm-model gpt-5-mini-2025-08-07 --max-output-tokens 1200 --run-id pilot-lexical-gpt5mini-001 --out results/runs/pilot-lexical-gpt5mini-001.json

run-hybrid-gated-openai:
	.venv/bin/python -m diw.cli claim-audit --database-url sqlite:////private/tmp/claim_evidence_audit_pilot.db --mode hybrid --top-k 5 --llm-provider openai --llm-model gpt-5-mini-2025-08-07 --max-output-tokens 1200 --verification-gate --resume --run-id pilot-hybrid-gated-gpt5mini-001 --out results/runs/pilot-hybrid-gated-gpt5mini-001.json

run-hybrid-supported-gated-openai:
	.venv/bin/python -m diw.cli claim-audit --database-url sqlite:////private/tmp/claim_evidence_audit_pilot.db --mode hybrid --top-k 5 --llm-provider openai --llm-model gpt-5-mini-2025-08-07 --max-output-tokens 1200 --verification-gate --verification-gate-policy supported --resume --run-id pilot-hybrid-supported-gated-gpt5mini-001 --out results/runs/pilot-hybrid-supported-gated-gpt5mini-001.json

compare-hybrid-lexical:
	.venv/bin/python -m diw.cli audit-comparison --baseline results/runs/pilot-hybrid-gpt5mini-001.json --intervention results/runs/pilot-lexical-gpt5mini-001.json --title "Hybrid versus Lexical Retrieval" --left-label Hybrid --right-label Lexical --out results/reports/hybrid-vs-lexical.md

compare-hybrid-gated:
	.venv/bin/python -m diw.cli audit-comparison --baseline results/runs/pilot-hybrid-gpt5mini-001.json --intervention results/runs/pilot-hybrid-gated-gpt5mini-001.json --out results/reports/hybrid-vs-gated.md

compare-hybrid-supported-gated:
	.venv/bin/python -m diw.cli audit-comparison --baseline results/runs/pilot-hybrid-gpt5mini-001.json --intervention results/runs/pilot-hybrid-supported-gated-gpt5mini-001.json --title "Hybrid versus Supported Claim Gate" --left-label Hybrid --right-label Supported-gate --out results/reports/hybrid-vs-supported-gated.md

run-v1-lexical-openai:
	.venv/bin/python -m diw.cli claim-audit --questions data/audit/questions/v1_40_gold.jsonl --database-url sqlite:////private/tmp/claim_evidence_audit_pilot.db --mode lexical --top-k 5 --llm-provider openai --llm-model gpt-5-mini-2025-08-07 --max-output-tokens 1200 --require-gold-evidence --resume --run-id v1-lexical-gpt5mini-001 --out results/runs/v1-lexical-gpt5mini-001.json

run-v1-hybrid-openai:
	.venv/bin/python -m diw.cli claim-audit --questions data/audit/questions/v1_40_gold.jsonl --database-url sqlite:////private/tmp/claim_evidence_audit_pilot.db --mode hybrid --top-k 5 --llm-provider openai --llm-model gpt-5-mini-2025-08-07 --max-output-tokens 1200 --require-gold-evidence --resume --run-id v1-hybrid-gpt5mini-001 --out results/runs/v1-hybrid-gpt5mini-001.json

run-v1-hybrid-replica-openai:
	.venv/bin/python -m diw.cli claim-audit --questions data/audit/questions/v1_40_gold.jsonl --database-url sqlite:////private/tmp/claim_evidence_audit_pilot.db --mode hybrid --top-k 5 --llm-provider openai --llm-model gpt-5-mini-2025-08-07 --max-output-tokens 1200 --require-gold-evidence --resume --run-id v1-hybrid-gpt5mini-002 --out results/runs/v1-hybrid-gpt5mini-002.json

run-v1-hybrid-repair-openai:
	.venv/bin/python -m diw.cli audit-evidence-repair --run results/runs/v1-hybrid-gpt5mini-001.json --run-id v1-hybrid-repair-gpt5mini-001 --out results/runs/v1-hybrid-repair-gpt5mini-001.json

compare-v1-lexical-hybrid:
	.venv/bin/python -m diw.cli audit-comparison --baseline results/runs/v1-lexical-gpt5mini-001.json --intervention results/runs/v1-hybrid-gpt5mini-001.json --title "V1: Lexical versus Hybrid Retrieval" --left-label Lexical --right-label Hybrid --out results/reports/v1-lexical-vs-hybrid.md

compare-v1-hybrid-repair:
	.venv/bin/python -m diw.cli audit-comparison --baseline results/runs/v1-hybrid-gpt5mini-001.json --intervention results/runs/v1-hybrid-repair-gpt5mini-001.json --title "V1: Hybrid versus Hybrid plus Evidence Repair" --left-label Hybrid --right-label Hybrid+repair --out results/reports/v1-hybrid-vs-repair.md

compare-v1-hybrid-replica:
	.venv/bin/python -m diw.cli audit-comparison --baseline results/runs/v1-hybrid-gpt5mini-001.json --intervention results/runs/v1-hybrid-gpt5mini-002.json --title "V1: Hybrid Sampling Variance" --left-label Hybrid-001 --right-label Hybrid-002 --out results/reports/v1-hybrid-sampling-variance.md

trace-v1-lexical-hybrid:
	.venv/bin/python -m diw.cli audit-retrieval-trace --left results/runs/v1-lexical-gpt5mini-001.json --right results/runs/v1-hybrid-gpt5mini-001.json --out results/reports/v1-lexical-vs-hybrid.trace.json

freeze-v1-evaluation:
	.venv/bin/python -m diw.cli corpus-verify
	.venv/bin/python -m diw.cli evaluation-freeze --freeze-version v1.0 --artifact data/audit/corpus_manifest.jsonl --artifact data/audit/questions/v1_40_gold.jsonl --artifact docs/annotation-rubric.md --artifact results/runs/v1-hybrid-gpt5mini-001.json --artifact results/runs/v1-hybrid-repair-gpt5mini-001.json --out data/audit/freeze/v1.0.json

materialize-v1-verifier-reference:
	.venv/bin/python -m diw.cli annotation-prefill-reference --annotations data/audit/annotations/v1_hybrid_primary_human.jsonl --out data/audit/annotations/v1_hybrid_verifier_reference.jsonl

# Historical generated-answer arm. The frozen deterministic retrieval-only benchmark is recorded
# separately in results/runs/v1-*-deterministic-001.json.
run-v1-hybrid-te3s-openai:
	.venv/bin/python -m diw.cli claim-audit --questions data/audit/questions/v1_40_gold.jsonl --database-url sqlite:////private/tmp/claim_evidence_audit_pilot.db --mode hybrid --top-k 5 --embedding-provider openai --embedding-model text-embedding-3-small --llm-provider openai --llm-model gpt-5-mini-2025-08-07 --max-output-tokens 1200 --require-gold-evidence --resume --run-id v1-hybrid-te3s-gpt5mini-001 --out results/runs/v1-hybrid-te3s-gpt5mini-001.json

compare-v1-hybrid-embeddings:
	.venv/bin/python -m diw.cli audit-comparison --baseline results/runs/v1-hybrid-gpt5mini-001.json --intervention results/runs/v1-hybrid-te3s-gpt5mini-001.json --title "V1: Hashing versus text-embedding-3-small Embeddings" --left-label Hashing --right-label TE3S --out results/reports/v1-hashing-vs-te3s.md

trace-v1-hybrid-embeddings:
	.venv/bin/python -m diw.cli audit-retrieval-trace --left results/runs/v1-hybrid-gpt5mini-001.json --right results/runs/v1-hybrid-te3s-gpt5mini-001.json --out results/reports/v1-hashing-vs-te3s.trace.json

validate-mcp-stdio:
	PYTHONPATH=src .venv/bin/python scripts/validate_mcp_stdio.py

prepare-v1-second-full:
	.venv/bin/python -m diw.cli annotation-blind-sample --run results/runs/v1-hybrid-gpt5mini-001.json --questions data/audit/questions/v1_40_gold.jsonl --question-count 40 --seed 20260727 --annotator-id independent-a2 --rubric-version v1.0 --out data/audit/annotations/v1_hybrid_second_annotator_full_blind.jsonl

annotate-v1-primary:
	.venv/bin/python -m diw.annotation_app --input data/audit/annotations/v1_hybrid_primary_human.jsonl --output data/audit/annotations/v1_hybrid_primary_ibraim_a1.jsonl --annotator-id ibraim-a1 --host 127.0.0.1 --port 8765

annotate-v1-second:
	.venv/bin/python -m diw.annotation_app --input data/audit/annotations/v1_hybrid_second_annotator_full_blind.jsonl --output data/audit/annotations/v1_hybrid_second_annotator_human.jsonl --annotator-id independent-a2 --host 127.0.0.1 --port 8766

annotate-v1-recheck:
	.venv/bin/python -m diw.annotation_app --input data/audit/annotations/v1_hybrid_reannotation_blind.jsonl --output data/audit/annotations/v1_hybrid_reannotation_ibraim_a1.jsonl --annotator-id ibraim-a1-recheck --host 127.0.0.1 --port 8767

annotate-v1-repair:
	.venv/bin/python -m diw.annotation_app --input data/audit/annotations/v1_hybrid_repair_primary_human.jsonl --output data/audit/annotations/v1_hybrid_repair_ibraim_a1.jsonl --annotator-id ibraim-a1-repair --host 127.0.0.1 --port 8768

summarize-v1-primary:
	.venv/bin/python -m diw.cli annotation-summary --annotations data/audit/annotations/v1_hybrid_primary_ibraim_a1.jsonl

agree-v1-annotators:
	.venv/bin/python -m diw.cli annotation-agreement --first data/audit/annotations/v1_hybrid_primary_ibraim_a1.jsonl --second data/audit/annotations/v1_hybrid_second_annotator_human.jsonl --require-complete --require-distinct-annotators --minimum-pairs 32 --out results/reports/v1-human-agreement.json
