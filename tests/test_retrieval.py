import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from sqlalchemy.orm import Session

from diw.core.embeddings import LocalHashingEmbeddingProvider
from diw.core.ingestion import ingest_file
from diw.core.retrieval import RetrievalResult, _rank_results, bm25_scores, retrieve_chunks
from diw.db.models import Base
from diw.db.repository import (
    count_embeddings,
    create_tenant,
    embed_missing_chunks,
    grant_tenant_document_access,
    save_ingested_document,
)
from diw.db.session import build_engine


class _ForeignEmbeddingProvider:
    """Stands in for a different embedding model (e.g. an API provider)."""

    model_name = "foreign-embedding-v1"
    dimensions = 32

    def embed(self, text: str) -> list[float]:
        return [0.0] * self.dimensions


class RetrievalTests(unittest.TestCase):
    def test_rrf_balances_lexical_and_vector_ranks(self):
        results = [
            RetrievalResult(
                chunk_id=chunk_id,
                document_id="doc",
                version_id="version",
                chunk_index=index,
                heading_path=[],
                text=chunk_id,
                lexical_score=lexical,
                vector_score=vector,
                score=0,
            )
            for index, (chunk_id, lexical, vector) in enumerate(
                [
                    ("lexical-only", 1.0, 0.0),
                    ("balanced", 0.8, 0.8),
                    ("vector-only", 0.0, 1.0),
                    ("weak", 0.1, 0.1),
                ]
            )
        ]

        ranked = _rank_results(results, top_k=4, mode="hybrid", reranker="rrf")

        self.assertEqual(ranked[0].chunk_id, "balanced")
        self.assertGreater(ranked[0].score, ranked[-1].score)

    def test_unknown_reranker_is_rejected(self):
        engine = build_engine("sqlite+pysqlite:///:memory:")
        Base.metadata.create_all(engine)
        try:
            with (
                Session(engine) as session,
                self.assertRaisesRegex(ValueError, "reranker"),
            ):
                retrieve_chunks(
                    session,
                    "query",
                    LocalHashingEmbeddingProvider(),
                    reranker="mystery",
                )
        finally:
            engine.dispose()

    def test_rrf_does_not_use_chunk_id_to_break_component_score_ties(self):
        results = [
            RetrievalResult(
                chunk_id=chunk_id,
                document_id="doc",
                version_id="version",
                chunk_index=index,
                heading_path=[],
                text=chunk_id,
                lexical_score=0.2,
                vector_score=0.0,
                score=0.0,
            )
            for index, chunk_id in enumerate(["a-first", "z-last"])
        ]

        ranked = _rank_results(results, top_k=2, mode="hybrid", reranker="rrf")

        self.assertEqual(ranked[0].score, ranked[1].score)

    def test_bm25_rewards_a_distinctive_query_term(self):
        chunks = [
            SimpleNamespace(heading_path=[], text="common common common"),
            SimpleNamespace(heading_path=[], text="common distinctive"),
            SimpleNamespace(heading_path=[], text="common"),
        ]

        scores = bm25_scores("common distinctive", chunks)

        self.assertGreater(scores[1], scores[0])
        self.assertGreater(scores[1], scores[2])

    def test_embed_missing_chunks_is_idempotent(self):
        engine = build_engine("sqlite+pysqlite:///:memory:")
        Base.metadata.create_all(engine)
        provider = LocalHashingEmbeddingProvider(dimensions=16)
        try:
            with TemporaryDirectory() as tmp, Session(engine) as session:
                path = Path(tmp) / "paper.md"
                path.write_text("# Paper\n\n## Method\n\nHybrid retrieval uses chunks.\n")
                document = ingest_file(path, target_chars=220, overlap_chars=0)
                save_ingested_document(session, document)
                session.commit()

                first_created = embed_missing_chunks(session, provider)
                second_created = embed_missing_chunks(session, provider)
                session.commit()

                self.assertEqual(first_created, len(document.chunks))
                self.assertEqual(second_created, 0)
                self.assertEqual(count_embeddings(session), len(document.chunks))
        finally:
            engine.dispose()

    def test_hybrid_retrieval_returns_relevant_chunk(self):
        engine = build_engine("sqlite+pysqlite:///:memory:")
        Base.metadata.create_all(engine)
        provider = LocalHashingEmbeddingProvider(dimensions=32)
        try:
            with TemporaryDirectory() as tmp, Session(engine) as session:
                path = Path(tmp) / "paper.md"
                path.write_text(
                    "# Paper\n\n"
                    "## Method\n\n"
                    "The method uses hybrid retrieval over document chunks.\n\n"
                    "## Results\n\n"
                    "The result reports citation validity.\n",
                    encoding="utf-8",
                )
                document = ingest_file(path, target_chars=220, overlap_chars=0)
                save_ingested_document(session, document)
                embed_missing_chunks(session, provider)
                session.commit()

                results = retrieve_chunks(
                    session,
                    "hybrid retrieval method",
                    provider,
                    top_k=1,
                    mode="hybrid",
                )

                self.assertEqual(len(results), 1)
                self.assertIn("Method", results[0].heading_path)
                self.assertIn("hybrid retrieval", results[0].text.lower())
        finally:
            engine.dispose()

    def test_retrieval_ignores_embeddings_from_other_models_or_dimensions(self):
        engine = build_engine("sqlite+pysqlite:///:memory:")
        Base.metadata.create_all(engine)
        embedded_provider = LocalHashingEmbeddingProvider(dimensions=32)
        try:
            with TemporaryDirectory() as tmp, Session(engine) as session:
                path = Path(tmp) / "paper.md"
                path.write_text("# Paper\n\nHybrid retrieval evidence.\n", encoding="utf-8")
                document = ingest_file(path, target_chars=220, overlap_chars=0)
                save_ingested_document(session, document)
                embed_missing_chunks(session, embedded_provider)
                session.commit()

                different_dimensions = LocalHashingEmbeddingProvider(dimensions=16)
                self.assertEqual(
                    retrieve_chunks(
                        session, "hybrid retrieval", different_dimensions, top_k=5
                    ),
                    [],
                )

                foreign_model = _ForeignEmbeddingProvider()
                self.assertEqual(
                    retrieve_chunks(session, "hybrid retrieval", foreign_model, top_k=5),
                    [],
                )
        finally:
            engine.dispose()

    def test_document_scope_prevents_cross_tenant_retrieval(self):
        engine = build_engine("sqlite+pysqlite:///:memory:")
        provider = LocalHashingEmbeddingProvider(dimensions=32)
        Base.metadata.create_all(engine)
        try:
            with TemporaryDirectory() as tmp, Session(engine) as session:
                alpha_path = Path(tmp) / "alpha.md"
                beta_path = Path(tmp) / "beta.md"
                alpha_path.write_text("# Alpha\n\nPrivate alpha retrieval evidence.\n", encoding="utf-8")
                beta_path.write_text("# Beta\n\nPrivate beta retrieval evidence.\n", encoding="utf-8")
                alpha_document = ingest_file(alpha_path, target_chars=220, overlap_chars=0)
                beta_document = ingest_file(beta_path, target_chars=220, overlap_chars=0)
                save_ingested_document(session, alpha_document)
                save_ingested_document(session, beta_document)
                embed_missing_chunks(session, provider)
                alpha_tenant = create_tenant(session, slug="alpha", name="Alpha")
                beta_tenant = create_tenant(session, slug="beta", name="Beta")
                grant_tenant_document_access(
                    session, tenant_id=alpha_tenant.id, document_id=alpha_document.document_id
                )
                grant_tenant_document_access(
                    session, tenant_id=beta_tenant.id, document_id=beta_document.document_id
                )
                session.commit()

                results = retrieve_chunks(
                    session,
                    "private evidence",
                    provider,
                    top_k=5,
                    document_ids={alpha_document.document_id},
                )

                self.assertEqual({result.document_id for result in results}, {alpha_document.document_id})
                self.assertTrue(all("alpha" in result.text.lower() for result in results))
        finally:
            engine.dispose()


if __name__ == "__main__":
    unittest.main()
