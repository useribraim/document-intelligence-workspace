# Learning Note 02: Database Persistence

## What Problem This Step Solves

v0.1 could normalise and chunk a document, then export JSON. That is useful, but later RAG needs stable storage. Embeddings, retrieval runs, citations, evaluations, and audit logs all need to refer back to persistent document and chunk records.

v0.2 adds the first database layer:

```text
source file -> ingested document -> source_documents
            -> document_versions
            -> chunks
```

## Why Split Documents And Versions

A source document is the logical file:

```text
data/demo/raw/retrieval-notes.md
```

A document version is a specific content snapshot of that file. If the file changes, it should keep the same `document_id` but receive a new `version_id`.

This matters because citations and embeddings should point to the exact version of the source text used at the time.

## Tables

### `source_documents`

Stores file-level identity:

- document ID
- source path
- source name
- source type
- created timestamp

### `document_versions`

Stores immutable source snapshots:

- version ID
- document ID
- content hash
- normalised text
- normalisation report
- ingestion timestamp

### `chunks`

Stores retrieval-ready passages:

- chunk ID
- version ID
- chunk index
- text
- heading path
- content hash
- start/end lines

## Why This Is Not pgvector Yet

This step deliberately stops before embeddings. The database first needs stable document, version, and chunk records. In v0.3, embeddings can reference those chunk IDs instead of floating around as disconnected vectors.

## Interview Explanation

I separated source documents from document versions because document AI systems need stable provenance. A document path identifies the logical source, while the version hash identifies the exact text snapshot used for chunking, retrieval, embeddings, and citations. That gives later AI outputs a reliable source trail.
