# Evaluation Harness Paper

## Benchmark Design

Method: golden-case evaluation for structured extraction, supported QA, refusal, and citation validation.

Dataset: manually written synthetic cases covering research-document workflows.

Metric: pass rate by task, retrieval hit rate, field coverage, citation validity, and refusal accuracy.

Limitation: the benchmark is deterministic and should later be expanded with harder adversarial cases.

## Conflicting Evidence

Earlier note: vector-only retrieval is sufficient for all questions.

Corrected note: hybrid retrieval is preferred because lexical evidence helps prevent unsupported answers.
