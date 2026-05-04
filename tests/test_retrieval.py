from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from diw.core.embeddings import LocalHashingEmbeddingProvider
from diw.core.ingestion import ingest_file
from diw.core.retrieval import retrieve_chunks
from diw.db.models import Base
from diw.db.repository import count_embeddings, embed_missing_chunks, save_ingested_document
from diw.db.session import build_engine
from sqlalchemy.orm import Session


class RetrievalTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
