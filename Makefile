PYTHON := .venv/bin/python

.PHONY: test lint verify serve corpus-manifest calibration-build calibration-check \
	validate-mcp-stdio vertex-smoke annotate-primary annotate-independent \
	annotation-agreement

test:
	$(PYTHON) -m pytest -q

lint:
	.venv/bin/ruff check src tests scripts

corpus-manifest:
	$(PYTHON) -m diw.cli corpus-verify --allow-missing

calibration-build:
	$(PYTHON) scripts/build_calibration_v2.py

calibration-check:
	$(PYTHON) scripts/build_calibration_v2.py --check

verify: lint test corpus-manifest
	$(PYTHON) scripts/verify_public_evidence.py
	$(PYTHON) scripts/build_calibration_v2.py --check
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
		--input data/audit/annotations/v2_primary_annotation_template.jsonl \
		--output data/audit/annotations/local/v2_primary_a1.jsonl \
		--annotator-id primary-a1 --host 127.0.0.1 --port 8765

annotate-independent:
	$(PYTHON) -m diw.annotation_app \
		--input data/audit/annotations/v2_independent_annotation_template.jsonl \
		--output data/audit/annotations/local/v2_independent_a2.jsonl \
		--annotator-id independent-a2 --host 127.0.0.1 --port 8766

annotation-agreement:
	$(PYTHON) -m diw.cli annotation-agreement \
		--first data/audit/annotations/local/v2_primary_a1.jsonl \
		--second data/audit/annotations/local/v2_independent_a2.jsonl \
		--require-complete --require-distinct-annotators --minimum-pairs 100 \
		--out results/local/v2-human-agreement.json
