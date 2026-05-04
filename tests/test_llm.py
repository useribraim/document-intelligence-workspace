import unittest

from diw.core.llm import (
    DeterministicStructuredProvider,
    LLMRequest,
    OpenAIChatProvider,
    generate_structured_answer,
)
from diw.core.qa import validate_citations
from diw.core.retrieval import RetrievalResult


class LLMProviderTests(unittest.TestCase):
    def test_deterministic_provider_returns_structured_answer_with_fields(self):
        result = RetrievalResult(
            chunk_id="chunk-1",
            document_id="doc-1",
            version_id="version-1",
            chunk_index=0,
            heading_path=["Paper", "Method"],
            text=(
                "Method: contrastive retrieval.\n\n"
                "Dataset: synthetic paper excerpts.\n\n"
                "Metric: recall@5.\n\n"
                "Limitation: small demo corpus."
            ),
            lexical_score=1.0,
            vector_score=0.8,
            score=0.91,
        )
        provider = DeterministicStructuredProvider()

        answer = generate_structured_answer(
            "Extract the method, dataset, metric, and limitation.",
            [result],
            provider,
        )
        validation = validate_citations(answer, [result])

        self.assertFalse(answer.insufficient_evidence)
        self.assertEqual(answer.provider, "local")
        self.assertEqual(answer.model, "deterministic-structured-v1")
        self.assertEqual(answer.extracted_fields["method"], "contrastive retrieval.")
        self.assertEqual(answer.extracted_fields["dataset"], "synthetic paper excerpts.")
        self.assertTrue(validation.valid)

    def test_provider_can_be_called_directly_for_raw_json(self):
        provider = DeterministicStructuredProvider()
        response = provider.complete(LLMRequest(query="Unsupported question", evidence=[]))

        self.assertEqual(response.provider, "local")
        self.assertIn("Insufficient evidence", response.raw_text)

    def test_openai_provider_requires_api_key(self):
        with self.assertRaises(ValueError):
            OpenAIChatProvider(api_key="")


if __name__ == "__main__":
    unittest.main()
