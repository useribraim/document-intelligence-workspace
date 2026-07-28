# Audit Corpus Provenance

The frozen retrieval evaluation references ten public arXiv paper versions. The repository records
their canonical URLs, displayed licence URLs, expected local text paths, and SHA-256 hashes in
[`corpus_manifest.jsonl`](../data/audit/corpus_manifest.jsonl).

Full paper text is not included in the current source tree. Earlier Git history contains removed
corpus files, so the public repository cannot yet be described as non-redistributing; it requires
a history rewrite before that claim is accurate. “Available without a paywall” is not treated as
permission to redistribute a paper under this project's MIT licence.

The V2 calibration packet contains only short, canonicalized excerpts tied to recorded chunk
identifiers. They let independent reviewers judge the displayed claim-citation relationship
without receiving or redistributing complete papers.

## Recorded licences

| Paper version | Licence displayed by arXiv |
|---|---|
| `2005.11401v4` | [arXiv non-exclusive distribution licence](https://arxiv.org/licenses/nonexclusive-distrib/1.0/) |
| `2004.04906v3` | [arXiv non-exclusive distribution licence](https://arxiv.org/licenses/nonexclusive-distrib/1.0/) |
| `2208.03299v3` | [arXiv non-exclusive distribution licence](https://arxiv.org/licenses/nonexclusive-distrib/1.0/) |
| `2302.00083v3` | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) |
| `2305.14627v2` | [arXiv non-exclusive distribution licence](https://arxiv.org/licenses/nonexclusive-distrib/1.0/) |
| `2309.15217v2` | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| `2310.11511v1` | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| `2401.00396v2` | [arXiv non-exclusive distribution licence](https://arxiv.org/licenses/nonexclusive-distrib/1.0/) |
| `2401.15884v3` | [arXiv non-exclusive distribution licence](https://arxiv.org/licenses/nonexclusive-distrib/1.0/) |
| `2401.18059v1` | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |

The licence URLs were read from the corresponding arXiv abstract pages on 2026-07-27. This record is
provenance, not legal advice; a reproducer remains responsible for complying with the applicable
licence.

## Local reproduction

Place legally obtained, identically extracted text at the manifest's `text_path` values, then run:

```bash
python -m diw.cli corpus-verify
```

The strict command fails if any file is missing or its hash differs. CI uses
`corpus-verify --allow-missing` to validate manifest structure and any present local files without
requiring the current source tree to contain the corpus. This does not remedy the historical files
described above.
