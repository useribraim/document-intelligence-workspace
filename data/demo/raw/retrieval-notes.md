# Retrieval Notes

## Problem

Long-form technical and professional documents are difficult to use when notes stay disconnected from source evidence.

## Method

The workspace normalises input text, preserves headings, creates deterministic chunks, and stores provenance metadata for each document version.

## Evaluation

The first milestone checks stable normalisation, heading-aware chunking, content hashes, and JSON export. Later milestones add PostgreSQL storage, pgvector embeddings, hybrid retrieval, source-cited generation, and evaluation metrics.
