import unittest

from diw.core.qa import compose_source_cited_answer, validate_citations
from diw.core.retrieval import RetrievalResult


class SourceCitedQATests(unittest.TestCase):
    def test_answer_uses_quotes_from_retrieved_chunks(self):
        result = RetrievalResult(
            chunk_id="chunk-1",
            document_id="doc-1",
            version_id="version-1",
            chunk_index=0,
            heading_path=["Paper", "Method"],
            text="Hybrid retrieval combines lexical matching and vector similarity over chunks.",
            lexical_score=1.0,
            vector_score=0.8,
            score=0.91,
        )

        answer = compose_source_cited_answer("how does hybrid retrieval work", [result])
        validation = validate_citations(answer, [result])

        self.assertFalse(answer.insufficient_evidence)
        self.assertEqual(len(answer.citations), 1)
        self.assertIn("[C1]", answer.answer)
        self.assertIn(answer.citations[0].quote, result.text)
        self.assertTrue(validation.valid)

    def test_low_scoring_retrieval_refuses_to_answer(self):
        result = RetrievalResult(
            chunk_id="chunk-1",
            document_id="doc-1",
            version_id="version-1",
            chunk_index=0,
            heading_path=["Paper", "Background"],
            text="This chunk discusses unrelated background material.",
            lexical_score=0.0,
            vector_score=0.01,
            score=0.01,
        )

        answer = compose_source_cited_answer("what is citation validation", [result])
        validation = validate_citations(answer, [result])

        self.assertTrue(answer.insufficient_evidence)
        self.assertEqual(answer.citations, [])
        self.assertIn("Insufficient evidence", answer.answer)
        self.assertTrue(validation.valid)

    def test_validation_rejects_quote_not_present_in_source_chunk(self):
        result = RetrievalResult(
            chunk_id="chunk-1",
            document_id="doc-1",
            version_id="version-1",
            chunk_index=0,
            heading_path=["Paper"],
            text="The source chunk contains only this sentence.",
            lexical_score=1.0,
            vector_score=1.0,
            score=1.0,
        )
        answer = compose_source_cited_answer("source chunk", [result])
        answer.citations[0].quote = "This quote was never retrieved."

        validation = validate_citations(answer, [result])

        self.assertFalse(validation.valid)
        self.assertIn("citation quote is not present", validation.errors[0])


if __name__ == "__main__":
    unittest.main()
