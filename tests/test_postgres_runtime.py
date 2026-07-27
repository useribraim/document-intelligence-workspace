import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from sqlalchemy.orm import Session

from diw.core.embeddings import LocalHashingEmbeddingProvider
from diw.core.ingestion import ingest_file
from diw.core.llm import DeterministicStructuredProvider, generate_structured_answer
from diw.core.qa import validate_citations
from diw.core.retrieval import retrieve_chunks
from diw.db.repository import (
    count_ai_suggestions,
    count_embeddings,
    count_pgvector_embeddings,
    count_review_decisions,
    embed_missing_chunks,
    record_review_decision,
    save_ai_run,
    save_ai_suggestion,
    save_ingested_document,
)
from diw.db.schema import create_schema, drop_schema
from diw.db.session import build_engine

POSTGRES_TEST_DATABASE_URL = os.getenv("POSTGRES_TEST_DATABASE_URL")


@unittest.skipUnless(
    POSTGRES_TEST_DATABASE_URL,
    "set POSTGRES_TEST_DATABASE_URL to run PostgreSQL runtime integration tests",
)
class PostgresRuntimeTests(unittest.TestCase):
    def test_postgres_runtime_supports_ingest_retrieve_answer_flow(self):
        engine = build_engine(POSTGRES_TEST_DATABASE_URL)
        provider = LocalHashingEmbeddingProvider(dimensions=32)
        try:
            drop_schema(engine)
            create_schema(engine)
            with TemporaryDirectory() as tmp, Session(engine) as session:
                source = Path(tmp) / "paper.md"
                source.write_text(
                    "# Paper\n\n"
                    "## Method\n\n"
                    "Method: hybrid retrieval over chunked research notes.\n\n"
                    "Dataset: synthetic paper excerpts.\n\n"
                    "Metric: citation-valid answer rate.\n\n"
                    "Limitation: small local demo corpus.\n",
                    encoding="utf-8",
                )

                document = ingest_file(source, target_chars=500, overlap_chars=0)
                save_ingested_document(session, document)
                embed_missing_chunks(session, provider)
                session.commit()

                self.assertEqual(count_embeddings(session), len(document.chunks))
                self.assertEqual(count_pgvector_embeddings(session), len(document.chunks))

                results = retrieve_chunks(
                    session,
                    "Extract the method, dataset, metric, and limitation.",
                    provider,
                    top_k=2,
                    mode="hybrid",
                )
                vector_results = retrieve_chunks(
                    session,
                    "hybrid retrieval research notes",
                    provider,
                    top_k=1,
                    mode="vector",
                )
                answer = generate_structured_answer(
                    "Extract the method, dataset, metric, and limitation.",
                    results,
                    DeterministicStructuredProvider(),
                )
                validation = validate_citations(answer, results)
                run = save_ai_run(
                    session,
                    run_type="answer_llm",
                    query="Extract the method, dataset, metric, and limitation.",
                    retrieval_mode="hybrid",
                    embedding_model=provider.model_name,
                    llm_provider="deterministic",
                    llm_model="deterministic-structured-v1",
                    prompt_version=answer.prompt_version,
                    retrieved_chunk_ids=[result.chunk_id for result in results],
                    citation_valid=validation.valid,
                    insufficient_evidence=answer.insufficient_evidence,
                    output={"answer": answer.model_dump()},
                )
                suggestion = save_ai_suggestion(
                    session,
                    suggestion_type="source_cited_answer",
                    title="Extract the method, dataset, metric, and limitation.",
                    ai_run_id=run.id,
                    payload={"answer": answer.model_dump()},
                )
                record_review_decision(
                    session,
                    suggestion_id=suggestion.id,
                    decision="accept",
                    reviewer="integration-test",
                    note="Citation validation passed.",
                )
                session.commit()

                self.assertTrue(results)
                self.assertEqual(len(vector_results), 1)
                self.assertGreater(vector_results[0].vector_score, 0)
                self.assertTrue(validation.valid)
                self.assertEqual(count_ai_suggestions(session), 1)
                self.assertEqual(count_review_decisions(session), 1)
                self.assertEqual(
                    answer.extracted_fields["method"],
                    "hybrid retrieval over chunked research notes.",
                )
        finally:
            drop_schema(engine)
            engine.dispose()


if __name__ == "__main__":
    unittest.main()
