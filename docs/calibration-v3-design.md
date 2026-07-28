# Independent Human Calibration: V3 Design

V2 is retained as a controlled rubric-development bank. Its five variants share one evidence seed,
so it must not be analyzed as 140 independent examples or used for ordinary item-level confidence
intervals. V3 replaces it for any published human-calibration result.

## Unit of sampling

One V3 item is one independently selected source span and one system-produced claim/citation pair.
No source span, generated answer, or paraphrase family may appear more than once in the evaluation
split. Questions from the same paper may appear in development and evaluation only when their source
spans are disjoint and the split is recorded before annotation.

## Minimum protocol

- Build at least 60 independent items across supported, partial, unsupported, misleading-context,
  and refusal categories.
- Freeze a development split for threshold/prompt/reranker decisions and a held-out evaluation split
  for reporting.
- Randomize packet order independently for each annotator.
- Hide seed IDs, expected labels, automated diagnostics, and the other annotator's work.
- Require two distinct annotators, preserve pre-adjudication labels, and compute agreement over the
  independent evaluation items only.
- Report a cluster-robust or paper-stratified interval if multiple items come from the same paper.

## Publication gate

Do not publish human accuracy, Cohen's kappa, or a calibration confidence interval until the
independent-item manifest, both immutable annotation packets, pre-adjudication report, and separate
adjudication record are present. A negative or low-agreement result remains publishable evidence.
