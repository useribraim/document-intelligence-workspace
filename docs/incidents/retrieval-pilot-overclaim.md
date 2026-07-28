# Incident Note: Retrieval Pilot Overclaim

**Date discovered:** 2026-07-28  
**Severity:** evidence integrity; no customer-data or availability impact

## Symptom

The public repository described a 23-gold-question retrieval pilot as an improvement because the
combined semantic-plus-RRF configuration had higher point estimates than the hashing-plus-weighted
baseline. The published trace showed only six Recall@5 changes and did not include a paired
uncertainty analysis or a complete factorial comparison.

## Impact

The wording could lead a reader to infer retrieval superiority and a causal RRF/embedding effect
that the available evidence did not establish. The historical point estimates and trace remain
available; the unsupported conclusion does not.

## Diagnosis

The paired bootstrap over the 23 gold-scored questions produced intervals crossing zero for both
Recall@5 and MRR. The old retrieval code also turned tied lexical/vector scores into distinct ranks
by sorting chunk identifiers, introducing an identifier-dependent input to RRF. The local public
demo further used token hashing while its explanatory copy implied semantic retrieval.

## Fix

- Added a deterministic paired-bootstrap script and a published uncertainty artifact.
- Reframed all public project copy as an inconclusive pilot.
- Replaced lexical token-set overlap with BM25 for current runtime retrieval.
- Assigned average ranks to tied component scores before RRF.
- Corrected public copy to distinguish local hashing vectors from semantic embeddings.
- Added a V3 independent-calibration design; V2 remains a controlled rubric-development bank.

## Regression checks

`make verify` now runs the uncertainty calculation, full test suite, public-evidence verifier, and
calibration integrity check. Retrieval tests assert that RRF does not derive unequal component ranks
from chunk-ID ordering.

## What remains open

The old trace cannot establish a four-arm causal result or a citation-recall interval because those
per-question artifacts were not recorded. A future result requires a preregistered development/
held-out split, all four configurations, per-question emitted citations, and uncertainty reported
with the headline metric.
