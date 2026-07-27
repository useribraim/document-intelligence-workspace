import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from diw.core.llm import (
    DeterministicStructuredProvider,
    LLMRequest,
    OpenAIChatProvider,
    VertexAIGeminiProvider,
    estimate_openai_cost_usd,
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

    def test_estimate_openai_cost_uses_pinned_gpt5_mini_rates(self):
        cost = estimate_openai_cost_usd(
            model="gpt-5-mini-2025-08-07",
            input_tokens=1_000_000,
            cached_input_tokens=0,
            output_tokens=1_000_000,
        )
        self.assertEqual(cost, 2.25)

    def test_vertex_provider_makes_structured_json_request(self):
        recorded: dict = {}
        client_kwargs: dict = {}

        def generate_content(**kwargs):
            recorded.update(kwargs)
            return SimpleNamespace(
                text='{"answer":"ok","insufficient_evidence":true,"citations":[]}',
                usage_metadata=SimpleNamespace(
                    prompt_token_count=10,
                    cached_content_token_count=2,
                    candidates_token_count=4,
                ),
            )

        client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))

        def build_client(**kwargs):
            client_kwargs.update(kwargs)
            return client

        fake_genai = SimpleNamespace(Client=build_client)
        with patch.dict(sys.modules, {"google.genai": fake_genai}):
            provider = VertexAIGeminiProvider(
                model="test-gemini",
                project="test-project",
            )
            response = provider.complete(LLMRequest(query="Question?", evidence=[]))

        self.assertEqual(recorded["model"], "test-gemini")
        self.assertEqual(recorded["config"]["response_mime_type"], "application/json")
        self.assertEqual(response.provider, "vertex")
        self.assertEqual(response.input_tokens, 10)
        self.assertEqual(client_kwargs["http_options"], {"api_version": "v1"})

    def test_vertex_provider_requires_explicit_model_and_project(self):
        with patch.dict(
            "os.environ",
            {"VERTEX_CHAT_MODEL": "", "GOOGLE_CLOUD_PROJECT": ""},
            clear=False,
        ), self.assertRaises(ValueError):
            VertexAIGeminiProvider()


if __name__ == "__main__":
    unittest.main()
