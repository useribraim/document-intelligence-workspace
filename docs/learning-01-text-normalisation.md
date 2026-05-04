# Learning Note 01: Text Normalisation

## What Problem This Step Solves

Raw documents arrive with inconsistent line endings, trailing spaces, and noisy blank-line runs. If we chunk or hash that raw text directly, the same document can produce different chunks and hashes depending on where it came from.

Text normalisation creates a stable input for later stages:

```text
raw text -> normalised text -> chunks -> hashes -> retrieval -> citations
```

## What The Current Normaliser Does

The normaliser is deliberately conservative:

- Converts Windows/macOS line endings to `\n`.
- Removes trailing whitespace at the end of each line.
- Collapses repeated blank lines into a single blank line.
- Strips blank space from the very start and end of the document.
- Produces a small report showing how many safe changes were made.

## What It Does Not Do

It does not:

- lowercase text
- remove punctuation
- remove headings
- rewrite sentence spacing
- remove stop words
- change Markdown heading markers
- alter legal or technical terms

Those transformations might be useful for some NLP tasks, but they are risky for document intelligence because they can destroy meaning, citations, or source provenance.

## Why This Matters For Legal/Professional Documents

In legal-style documents, formatting can carry meaning. Clause numbers, headings, indentation, and exact wording can matter. A document AI system should clean only what is safe to clean.

That is why this project treats normalisation as a narrow stability step, not an aggressive text-preprocessing step.

## Interview Explanation

I normalise text before chunking so that the same source document produces stable chunks, hashes, and citations across runs. The normaliser is intentionally conservative: it fixes line endings, trailing whitespace, and repeated blank lines, but it does not rewrite terms, headings, punctuation, or casing because those can be meaningful in professional and legal documents.
