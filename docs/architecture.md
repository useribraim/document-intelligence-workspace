# Architecture

## Product Thesis

Long-form knowledge work should not disappear into chat history. The workspace turns raw sources into durable, source-cited, versioned, inspectable knowledge artifacts.

The system is deliberately not an "AI tutor" or generic PDF chatbot. It treats LLMs as unreliable components inside a controlled document pipeline.

## Core Loop

```text
source document
  -> parse and clean text
  -> store document version and content hash
  -> heading-aware chunking
  -> lexical index and embedding index
  -> hybrid retrieval
  -> structured/source-cited generation
  -> schema validation
  -> human review decision
  -> AI-run provenance and audit event
  -> evaluation report
```

## Domain Objects

- `source_documents`: imported sources such as papers, articles, repos, dataset notes, or technical documents.
- `document_versions`: immutable source snapshots with hashes.
- `chunks`: heading-aware passages with source offsets and version references.
- `chunk_embeddings`: vector representations tied to chunk hash and embedding model.
- `generated_pages`: Markdown outputs such as paper summaries, concept pages, comparison notes, and glossaries.
- `ai_runs`: one model or evaluation operation with query, retrieval mode, prompt version, model metadata, retrieved chunk IDs, validation result, output payload, metrics, and timestamp.
- `ai_suggestions`: model outputs awaiting review.
- `review_decisions`: accept, reject, or edit actions.
- `audit_events`: security- and provenance-relevant events.
- `evaluation_cases`: golden cases for retrieval, extraction, citation, and refusal behaviour.
- `evaluation_results`: metric outputs from repeatable evaluation runs.

## Retrieval Strategy

The project should become real RAG before it is pitched heavily as RAG:

1. Lexical retrieval with PostgreSQL full-text search.
2. Semantic retrieval with pgvector embeddings.
3. Hybrid scoring across lexical and vector results.
4. Optional reranking of the top candidates.
5. Evidence thresholding before answer generation.
6. Source-cited generation over retrieved chunks only.

If retrieval evidence is weak or missing, the system should produce an insufficient-evidence response instead of improvising.

## Structured Output Strategy

LLM workflows must return schema-shaped outputs:

- paper summary and method/dataset/metric extraction
- document comparison
- structured document extraction
- cited question answering

Raw model output and validated output are stored separately. Validation failures become review items and evaluation failures.

## Auditability

Every AI-generated output must be traceable to:

- source document version/hash
- retrieved chunk IDs
- prompt version
- model name
- model parameters
- output schema version
- validation result
- latency and estimated cost
- user review decision
- timestamp

This is the core high-stakes document AI differentiator.
