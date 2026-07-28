# Incident Note: Corpus History Disclosure

**Date discovered:** 2026-07-28  
**Severity:** provenance and repository-publication integrity

## Symptom

The current source tree omits the ten frozen paper-text inputs, while project copy stated that the
repository did not redistribute them.

## Diagnosis

The files were removed in commit `1f554a2`, but they remain reachable through earlier public Git
history (introduced in `7415f8f`). A normal clone therefore still receives them.

## Impact

The working-tree statement was incomplete and a public non-redistribution claim was inaccurate.
This does not change the recorded retrieval trace, but it does affect the provenance boundary and
what the project may claim about distribution.

## Fix

Public copy now says that the current tree excludes paper text and explicitly discloses the history
issue. The manifest and strict SHA-256 verifier remain the reproduction mechanism for a locally
obtained corpus.

## Required follow-up

Rewrite the canonical remote's reachable history to remove the eleven corpus files, force-push the
rewritten refs, invalidate any project-controlled clones/forks as appropriate, and verify the
remote before considering a narrower current-repository claim. A rewrite cannot retract copies
already cloned, forked, cached, or mirrored by others, so it cannot support an absolute historical
non-redistribution claim. This is an externally visible destructive operation and is intentionally
not performed automatically.
