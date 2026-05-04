# RAG Systems Paper

## Retrieval Pipeline

Method: hybrid retrieval with lexical filtering, vector search, and evidence thresholding.

Dataset: synthetic long-form research notes and paper excerpts.

Metric: recall@3 for retrieval and citation-validity rate for answers.

Limitation: retrieval quality depends on chunk boundaries and embedding quality.

## Failure Analysis

Near miss: lexical retrieval can over-rank chunks that share generic words but miss the key concept.

Unsupported query policy: answer generation should refuse when retrieved evidence does not mention the requested obligation.
