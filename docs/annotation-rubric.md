# Claim-to-Evidence Annotation Rubric v1.0

This rubric is frozen for the V1 annotation pass. The reviewer judges exact cited spans, not paper titles or general topic similarity. Automated suggestions are comparison aids only; they are not labels and must not be treated as ground truth.

## Answer-level records

- `answer_completeness`: `complete` if all material, evidence-supported parts of the request are addressed; `incomplete` if a material part or qualification is missing; `not_applicable` for a proper refusal.
- `refusal_appropriate`: `true` when the frozen corpus cannot support the request or refusal is required; `false` when sufficient retrieved evidence existed and the system should have answered; otherwise `null`.

## Claim-citation records

- `source_exists`: the cited chunk/span is present in the saved retrieved evidence.
- `citation_relevant`: `yes` only when the exact span addresses the claim's topic.
- `support_label`:
  - `fully_supported`: every material part of the claim is entailed by the exact span.
  - `partially_supported`: the central idea is supported but a material qualifier, number, scope, causality, or other detail is missing.
  - `unsupported`: the span is real and possibly related but does not establish the claim.
  - `contradicted`: the span materially conflicts with the claim.
  - `not_applicable`: no claim-citation judgement is applicable.
- `support_rationale`: one concise sentence naming the supported, missing, or contradictory material part.

Example: a paper reports an improvement on one task, while the answer claims universal superiority. Label `unsupported`; rationale: “The span is task-specific and does not establish universal superiority.”

## Failure mode

For every non-fully-supported, non-`not_applicable` claim, assign exactly one `failure_mode`:

- `overgeneralization`: evidence is narrower than the claim.
- `unsupported_specificity`: the claim adds an absent number, version, entity, or detail.
- `missing_qualification`: conditional evidence is stated categorically.
- `citation_misattribution`: the citation supports another nearby claim, not this one.
- `claim_bundling`: one sentence has multiple claims and the span supports only some.
- `retrieval_miss`: relevant frozen-corpus evidence exists but was not retrieved.
- `out_of_scope`: the frozen corpus cannot support the request.

Do not choose a dominant failure mode before counting completed labels.

## Reliability and revision protocol

Complete the primary packet without consulting `automation_prefill` where practical. After about one week, complete the blinded five-question reannotation without viewing original labels. A second annotator independently completes the separate five-question packet without project results or automated labels.

Report aligned claim-citation-pair count, raw agreement, Cohen’s kappa, confusion matrix, and every disagreement with both rationales. Preserve original labels. If disagreements cluster at a boundary, create a dated rubric v1.1 with examples, record the reason, and re-label affected records; never silently overwrite V1.0 labels.

Human labels calibrate the deterministic verifier and model-assisted reviewer. The repair artifact requires its own review under this same rubric; its deterministic 100% rate is not a human-effectiveness result.
