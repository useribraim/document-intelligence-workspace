PYTHON := .venv/bin/python

.PHONY: test lint verify serve corpus-manifest validate-mcp-stdio vertex-smoke \
	annotate-primary annotate-independent annotation-agreement

test:
	$(PYTHON) -m pytest -q

lint:
	.venv/bin/ruff check src tests scripts

corpus-manifest:
	$(PYTHON) -m diw.cli corpus-verify --allow-missing

verify: lint test corpus-manifest
	$(PYTHON) scripts/verify_public_evidence.py
	bash -n scripts/deploy_cloud_run.sh
	bash -n scripts/run_vertex_cloud_smoke.sh

serve:
	$(PYTHON) -m uvicorn diw.api:app --reload

validate-mcp-stdio:
	PYTHONPATH=src $(PYTHON) scripts/validate_mcp_stdio.py

vertex-smoke:
	./scripts/run_vertex_cloud_smoke.sh

annotate-primary:
	$(PYTHON) -m diw.annotation_app \
		--input data/audit/annotations/v1_primary_annotation_template.jsonl \
		--output data/audit/annotations/local/v1_primary_a1.jsonl \
		--annotator-id primary-a1 --host 127.0.0.1 --port 8765

annotate-independent:
	$(PYTHON) -m diw.annotation_app \
		--input data/audit/annotations/v1_independent_annotation_template.jsonl \
		--output data/audit/annotations/local/v1_independent_a2.jsonl \
		--annotator-id independent-a2 --host 127.0.0.1 --port 8766

annotation-agreement:
	$(PYTHON) -m diw.cli annotation-agreement \
		--first data/audit/annotations/local/v1_primary_a1.jsonl \
		--second data/audit/annotations/local/v1_independent_a2.jsonl \
		--require-complete --require-distinct-annotators --minimum-pairs 32 \
		--out results/local/v1-human-agreement.json
