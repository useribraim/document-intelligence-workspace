# Claim-to-Evidence Annotation Rubric v2.0

This rubric governs the V2 controlled calibration pass. Reviewers judge the candidate answer and
the exact cited excerpt shown in the packet. They must not consult the case-seed file, author notes,
another annotator's work, automated diagnostics, or a desired agreement value.

## Answer-level records

- `answer_completeness`: `complete` when the candidate answer addresses every material,
  evidence-supported part of the question; `incomplete` when a supported part or qualification is
  missing; `not_applicable` for an appropriate refusal.
- `refusal_appropriate`: `true` when the displayed evidence cannot answer the question; `false`
  when it can; otherwise `null`.

## Claim-citation records

- `source_exists`: whether the cited excerpt and recorded source identifier exist in the packet.
- `citation_relevant`: `yes` only when the excerpt concerns the candidate claim's subject.
- `support_label`:
  - `fully_supported`: the excerpt entails every material part of the claim;
  - `partially_supported`: the central assertion is supported but a material qualifier, number,
    scope, cause, or additional assertion is not;
  - `unsupported`: the excerpt is real and may be topically related, but does not establish the
    claim;
  - `contradicted`: the excerpt materially conflicts with the claim;
  - `not_applicable`: no claim-citation judgment is possible.
- `support_rationale`: one concise sentence identifying the supported, missing, or contradictory
  material.

Related subject matter is not entailment. A result on one dataset does not support “always,”
“every,” “guarantees,” or a new implementation detail unless the excerpt states that scope.

## Failure mode

For each label other than `fully_supported` or `not_applicable`, choose exactly one:

- `overgeneralization`: the evidence is narrower than the claim;
- `unsupported_specificity`: the claim adds an absent number, version, entity, or detail;
- `missing_qualification`: conditional evidence is stated categorically;
- `citation_misattribution`: the excerpt supports a different nearby claim;
- `claim_bundling`: one sentence combines claims and the excerpt supports only part;
- `retrieval_miss`: relevant frozen-corpus evidence exists but the displayed citation misses it;
- `out_of_scope`: the frozen evidence cannot answer the request.

## Independence protocol

The primary and independent annotators each receive only their V2 template. Both packets contain
the same 140 answer records and 112 claim-citation pairs, but neither exposes the author-designed
case stratum or a proposed label.

Annotators must:

1. work separately and use distinct identifiers;
2. complete all records before seeing the other packet;
3. preserve their original files and rationales;
4. compute agreement before discussing disagreements;
5. record adjudication separately rather than overwriting either label set.

Case strata balance the instrument; they are not human ground truth. Only completed independent
labels and documented adjudication can support an agreement or calibrated-accuracy claim.
