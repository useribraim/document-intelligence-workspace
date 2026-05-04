from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from contextlib import contextmanager
import gc

from diw.core.ingestion import ingest_file
from diw.db.models import AIRun, AISuggestion, Base, Chunk, DocumentVersion, SourceDocument
from diw.db.repository import (
    count_ai_runs,
    count_ai_suggestions,
    count_chunks,
    count_documents,
    count_review_decisions,
    count_versions,
    record_review_decision,
    save_ai_run,
    save_ai_suggestion,
    save_ingested_document,
)
from diw.db.session import build_engine
from sqlalchemy import select
from sqlalchemy.orm import Session


class RepositoryTests(unittest.TestCase):
    @contextmanager
    def _session(self):
        engine = build_engine("sqlite+pysqlite:///:memory:")
        Base.metadata.create_all(engine)
        session = Session(engine)
        try:
            yield session
        finally:
            session.close()
            Base.metadata.drop_all(engine)
            engine.dispose()
            gc.collect()

    def test_save_ingested_document_persists_document_version_and_chunks(self):
        with TemporaryDirectory() as tmp, self._session() as session:
            path = Path(tmp) / "paper.md"
            path.write_text("# Paper\n\n## Method\n\nChunk this.\n", encoding="utf-8")
            document = ingest_file(path, target_chars=220, overlap_chars=0)

            save_ingested_document(session, document)
            session.commit()

            self.assertEqual(count_documents(session), 1)
            self.assertEqual(count_versions(session), 1)
            self.assertEqual(count_chunks(session), len(document.chunks))

            saved_version = session.get(DocumentVersion, document.version_id)
            self.assertIsNotNone(saved_version)
            self.assertEqual(saved_version.content_hash, document.content_hash)
            self.assertEqual(saved_version.normalisation_report["normalised_line_count"], 5)

    def test_save_same_version_is_idempotent(self):
        with TemporaryDirectory() as tmp, self._session() as session:
            path = Path(tmp) / "paper.md"
            path.write_text("# Paper\n\nSame content.\n", encoding="utf-8")
            document = ingest_file(path, target_chars=220, overlap_chars=0)

            save_ingested_document(session, document)
            save_ingested_document(session, document)
            session.commit()

            self.assertEqual(count_documents(session), 1)
            self.assertEqual(count_versions(session), 1)
            self.assertEqual(count_chunks(session), len(document.chunks))

    def test_modified_file_adds_new_version_for_same_document(self):
        with TemporaryDirectory() as tmp, self._session() as session:
            path = Path(tmp) / "paper.md"
            path.write_text("# Paper\n\nFirst content.\n", encoding="utf-8")
            first = ingest_file(path, target_chars=220, overlap_chars=0)
            save_ingested_document(session, first)

            path.write_text("# Paper\n\nSecond content.\n", encoding="utf-8")
            second = ingest_file(path, target_chars=220, overlap_chars=0)
            save_ingested_document(session, second)
            session.commit()

            self.assertEqual(first.document_id, second.document_id)
            self.assertEqual(count_documents(session), 1)
            self.assertEqual(count_versions(session), 2)

            saved_document = session.get(SourceDocument, first.document_id)
            self.assertEqual(len(saved_document.versions), 2)

    def test_chunk_heading_path_round_trips(self):
        with TemporaryDirectory() as tmp, self._session() as session:
            path = Path(tmp) / "paper.md"
            path.write_text("# Paper\n\n## Method\n\nChunk this.\n", encoding="utf-8")
            document = ingest_file(path, target_chars=220, overlap_chars=0)
            save_ingested_document(session, document)
            session.commit()

            chunks = session.scalars(select(Chunk).order_by(Chunk.chunk_index)).all()
            self.assertTrue(any("Method" in chunk.heading_path for chunk in chunks))

    def test_save_ai_run_persists_audit_metadata(self):
        with self._session() as session:
            run = save_ai_run(
                session,
                run_type="answer_llm",
                query="What does the paper say?",
                retrieval_mode="hybrid",
                embedding_model="local-hashing-v1",
                llm_provider="local",
                llm_model="deterministic-structured-v1",
                prompt_version="source-cited-qa-v1",
                retrieved_chunk_ids=["chunk-1", "chunk-2"],
                citation_valid=True,
                insufficient_evidence=False,
                output={"answer": "Cited answer"},
                metrics={"retrieved_chunk_count": 2},
            )
            session.commit()

            saved = session.get(AIRun, run.id)
            self.assertEqual(count_ai_runs(session), 1)
            self.assertEqual(saved.query, "What does the paper say?")
            self.assertEqual(saved.retrieved_chunk_ids, ["chunk-1", "chunk-2"])
            self.assertTrue(saved.citation_valid)
            self.assertEqual(saved.metrics["retrieved_chunk_count"], 2)

    def test_ai_suggestion_review_decision_updates_status(self):
        with self._session() as session:
            run = save_ai_run(
                session,
                run_type="answer_llm",
                query="What is the method?",
                output={"answer": {"answer": "A cited answer"}},
            )
            suggestion = save_ai_suggestion(
                session,
                suggestion_type="source_cited_answer",
                title="What is the method?",
                ai_run_id=run.id,
                payload={"answer": {"answer": "A cited answer"}},
            )
            session.commit()

            review = record_review_decision(
                session,
                suggestion_id=suggestion.id,
                decision="accept",
                reviewer="reviewer@example.com",
                note="Evidence checks out.",
            )
            session.commit()

            saved = session.get(AISuggestion, suggestion.id)
            self.assertEqual(count_ai_suggestions(session), 1)
            self.assertEqual(count_review_decisions(session), 1)
            self.assertEqual(saved.status, "accepted")
            self.assertIsNotNone(saved.reviewed_at)
            self.assertEqual(review.decision, "accept")
            self.assertEqual(review.reviewer, "reviewer@example.com")


if __name__ == "__main__":
    unittest.main()
