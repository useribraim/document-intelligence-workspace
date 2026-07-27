# Human Calibration Runbook

Automated citation checks and model-assisted labels are diagnostics, not human ground truth. This
runbook defines the remaining work before publishing agreement or human-accuracy numbers.

## Inputs

| Pass | Template | Records |
|---|---|---:|
| Primary | `data/audit/annotations/v1_primary_annotation_template.jsonl` | 72 |
| Independent | `data/audit/annotations/v1_independent_annotation_template.jsonl` | 72 |

Both templates are intentionally blank. The independent annotator must not see the primary labels,
automated support labels, comparison results, or any desired agreement value.

## Run the annotation sessions

```bash
make annotate-primary
make annotate-independent
```

Outputs are written under `data/audit/annotations/local/`, which is ignored by Git. Each annotator
must complete the answer-level records and the 32 aligned claim-citation records using
[`annotation-rubric.md`](annotation-rubric.md).

## Calculate agreement

```bash
make annotation-agreement
```

The command fails closed unless both inputs are complete, annotator identifiers differ, and all 32
claim-citation pairs align. The local report includes:

- record counts and input hashes;
- raw agreement;
- Cohen's kappa;
- confusion matrix;
- every disagreement with both rationales.

Preserve the pre-adjudication report. Discuss disagreements in a separate dated adjudication
record; never overwrite either annotator's original labels.

## Completion gate

Human calibration remains incomplete until all of the following are true:

- both outputs contain 72 completed human records;
- all 32 shared claim-citation pairs have labels from distinct people;
- the pre-adjudication agreement report is saved;
- disagreements retain both rationales;
- adjudication is recorded separately;
- public documentation reports the observed values exactly, including a low or negative kappa.

Repair effectiveness is a separate experiment and is not implied by completing calibration.
