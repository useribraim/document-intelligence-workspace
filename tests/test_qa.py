import unittest

from diw.core.qa import (
    SourceCitedAnswer,
    canonicalise_citation_quotes,
    compose_source_cited_answer,
    materialise_answer_citations,
    normalise_refusal,
    prune_unused_citations,
    validate_citations,
)
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

    def test_validation_rejects_an_uncited_non_refusal_answer(self):
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
        answer.citations = []
        answer.answer = "A claim without evidence mapping."

        validation = validate_citations(answer, [result])

        self.assertFalse(validation.valid)
        self.assertTrue(
            any("must include at least one citation" in error for error in validation.errors)
        )

    def test_validation_accepts_quote_with_normalised_pdf_whitespace(self):
        result = RetrievalResult(
            chunk_id="chunk-1",
            document_id="doc-1",
            version_id="version-1",
            chunk_index=0,
            heading_path=["Paper"],
            text="The source contains a line-broken\nquote for validation.",
            lexical_score=1.0,
            vector_score=1.0,
            score=1.0,
        )
        answer = compose_source_cited_answer("line broken quote", [result])
        answer.citations[0].quote = "line-broken quote for validation."

        self.assertTrue(validate_citations(answer, [result]).valid)

    def test_canonicalisation_replaces_paraphrased_quote_with_source_span(self):
        result = RetrievalResult(
            chunk_id="chunk-1",
            document_id="doc-1",
            version_id="version-1",
            chunk_index=0,
            heading_path=["Paper"],
            text="Dense retrieval uses learned passage and question representations for matching.",
            lexical_score=1.0,
            vector_score=1.0,
            score=1.0,
        )
        answer = compose_source_cited_answer("dense retrieval", [result])
        answer.citations[0].quote = "Learned representations match questions to passages."

        canonicalise_citation_quotes(answer, [result])

        self.assertEqual(answer.citations[0].quote_alignment, "canonicalized")
        self.assertTrue(validate_citations(answer, [result]).valid)

    def test_pruning_removes_citations_not_attached_to_answer(self):
        first = RetrievalResult(
            chunk_id="chunk-1",
            document_id="doc-1",
            version_id="version-1",
            chunk_index=0,
            heading_path=["Paper"],
            text="First cited source.",
            lexical_score=1.0,
            vector_score=1.0,
            score=1.0,
        )
        second = RetrievalResult(
            chunk_id="chunk-2",
            document_id="doc-1",
            version_id="version-1",
            chunk_index=1,
            heading_path=["Paper"],
            text="Second cited source.",
            lexical_score=1.0,
            vector_score=1.0,
            score=1.0,
        )
        answer = compose_source_cited_answer("cited source", [first, second])
        answer.answer = "Only the first source is used. [C1]"

        prune_unused_citations(answer)

        self.assertEqual([citation.label for citation in answer.citations], ["C1"])
        self.assertTrue(validate_citations(answer, [first, second]).valid)

    def test_materialisation_resolves_answer_label_to_retrieved_source(self):
        result = RetrievalResult(
            chunk_id="chunk-1",
            document_id="doc-1",
            version_id="version-1",
            chunk_index=0,
            heading_path=["Paper"],
            text="The retrieved source states a verifiable fact.",
            lexical_score=1.0,
            vector_score=1.0,
            score=1.0,
        )
        answer = compose_source_cited_answer("verifiable fact", [result])
        answer.answer = "A verifiable fact is present. [C1]"
        answer.citations = []

        materialise_answer_citations(answer, [result])

        self.assertEqual(answer.citations[0].label, "C1")
        self.assertEqual(answer.citations[0].quote_alignment, "label_resolved")
        self.assertTrue(validate_citations(answer, [result]).valid)

    def test_normalised_refusal_has_message_and_no_citations(self):
        result = RetrievalResult(
            chunk_id="chunk-1",
            document_id="doc-1",
            version_id="version-1",
            chunk_index=0,
            heading_path=["Paper"],
            text="Some retrieved text.",
            lexical_score=1.0,
            vector_score=1.0,
            score=1.0,
        )
        answer = compose_source_cited_answer("retrieved text", [result])
        answer.insufficient_evidence = True
        answer.answer = ""

        normalise_refusal(answer)

        self.assertTrue(answer.answer.startswith("Insufficient evidence"))
        self.assertEqual(answer.citations, [])
        self.assertTrue(validate_citations(answer, [result]).valid)

    def test_normalised_refusal_removes_model_citation_labels(self):
        answer = SourceCitedAnswer(
            query="q",
            answer="The corpus cannot establish that claim. [C1]",
            insufficient_evidence=True,
            citations=[],
        )

        normalised = normalise_refusal(answer)

        self.assertEqual(normalised.answer, "The corpus cannot establish that claim.")


if __name__ == "__main__":
    unittest.main()
