# Human Calibration Release Runbook

This runbook closes the final evidence gap without turning automated diagnostics, model review,
or self-review into “human ground truth.” The V1 corpus, questions, model run, and rubric are
already frozen. As of 2026-07-27, no V1 human labels are complete.

## Release claim

After the gate passes, report the observed values exactly:

> Built a 10-paper, 40-question evaluation with 32 claim-citation pairs independently labeled by
> two humans; raw agreement was X and Cohen's kappa was Y before adjudication.

Do not preselect, round toward, or promise a target kappa. A low or negative value is still a real
result: preserve it, inspect the confusion matrix, clarify ambiguous rubric boundaries in a new
version, and run a new labeled pass.

## Roles and packets

| Pass | Owner | Packet | Records |
|---|---|---|---:|
| Primary baseline | Ibraim | `v1_hybrid_primary_human.jsonl` | 72 |
| Independent baseline | A real second person | `v1_hybrid_second_annotator_full_blind.jsonl` | 72 |
| Delayed self-recheck | Ibraim, after about one week | `v1_hybrid_reannotation_blind.jsonl` | 10 |
| Repair review | Ibraim, after baseline labels are locked | `v1_hybrid_repair_primary_human.jsonl` | 57 |

The second annotator must not be Ibraim, Codex, another model, or someone copying Ibraim's labels.
Give them only the annotation UI and frozen rubric. Do not show automated labels, retrieval
comparison results, or the desired resume wording.

## Execution

1. Start the primary UI and complete every record:

   ```bash
   make annotate-v1-primary
   ```

2. Have the independent person start the full blind packet and complete every record:

   ```bash
   make annotate-v1-second
   ```

3. Produce the fail-closed agreement artifact:

   ```bash
   make agree-v1-annotators
   ```

   This command fails unless all 32 shared claim-citation pairs are labeled by distinct annotator
   IDs. On success it writes `results/reports/v1-human-agreement.json` with input hashes, raw
   agreement, Cohen's kappa, the confusion matrix, and both rationales for every disagreement.

4. Preserve the pre-adjudication report. Discuss every disagreement and write a separate dated
   adjudication record. Never overwrite either annotator's original JSONL.

5. After about one week, complete the blind self-recheck:

   ```bash
   make annotate-v1-recheck
   ```

6. Only if claiming that evidence repair helped under human judgment, complete the separate repair
   packet and compare it with the locked baseline labels:

   ```bash
   make annotate-v1-repair
   ```

## Release gate

Human calibration moves from `Blocked` to `Released` only when all of these exist:

- primary output: 72/72 `completed_human`;
- independent output: 72/72 `completed_human`;
- 32/32 aligned claim-citation pairs with distinct annotator identities;
- saved pre-adjudication raw agreement, Cohen's kappa, and confusion matrix;
- saved disagreements with both human rationales;
- dated adjudication record that preserves the original labels;
- README and resume wording use the measured values exactly.

Repair effectiveness is a separate gate. Calibration can be released without a positive repair
result, but repair improvement cannot be claimed until the repair packet is human-scored.
