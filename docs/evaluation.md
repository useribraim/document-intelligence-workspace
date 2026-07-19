# Evaluation Plan

## Why Evaluation Comes Early

The project should not rely on spot-checking or "looks good" demos. Evaluation is the proof that the document AI workflow is controlled rather than decorative.

## First Metrics

- `retrieval_recall_at_k`: the proportion of expected evidence phrases found in the retrieved top-k chunks.
- `retrieval_mrr`: the reciprocal rank of the first retrieved chunk containing expected evidence.
- `citation_validity_rate`: whether citations point to chunks that support the generated claim.
- `schema_validity_rate`: whether structured outputs pass validation.
- `extraction_accuracy`: field-level match against golden answers.
- `refusal_accuracy`: whether unsupported questions trigger insufficient-evidence behaviour.
- `latency_ms`: runtime per workflow.
- `estimated_cost`: model and embedding cost per workflow.

## Starter Evaluation Sets

Initial target:

- 20 ML/NLP paper cases.
- 20 professional document extraction cases.
- 10 refusal/insufficient-evidence cases.

Release target:

- 50-100 total golden cases.
- At least one saved evaluation report in `docs/eval-results.md`.
- At least three diagnosed failures with fixes or tradeoff notes.

The current runner reports both `retrieval_recall_at_k` and `retrieval_mrr` in its JSON and
Markdown output. The existing 12-case report is a baseline, not the final release target.

## Evaluation Case Shape

Each case should include:

- `id`
- `corpus`
- `task`
- `question` or extraction instruction
- `expected_answer` or `expected_fields`
- `expected_chunk_ids` once sources are indexed
- `expected_behavior`

## Anti-Gaming Rule

Do not tune the system only until demo examples pass. Keep failing examples in the report and explain what they reveal about retrieval, chunking, prompting, or validation.
