# Local audit corpus

The evaluation uses fixed text extracted from the paper versions listed in
`data/audit/corpus_manifest.jsonl`. Full paper text is intentionally not redistributed in this
repository.

To reproduce the frozen evaluation:

1. Obtain each paper from its canonical source under the licence shown in the manifest.
2. Extract plain text into the exact `text_path` listed for that record.
3. Run `python -m diw.cli corpus-verify`.

The command compares each local file with the recorded SHA-256 value. A mismatch means the
extraction is not the frozen evaluation input and should not be used to reproduce the published
comparison.
