import os
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from diw.core.embeddings import (
    LocalHashingEmbeddingProvider,
    OpenAIEmbeddingProvider,
    VertexAIEmbeddingProvider,
    build_embedding_provider,
    cosine_similarity,
)


class EmbeddingTests(unittest.TestCase):
    def test_local_hashing_embeddings_are_deterministic(self):
        provider = LocalHashingEmbeddingProvider(dimensions=16)

        first = provider.embed("retrieval evaluation")
        second = provider.embed("retrieval evaluation")

        self.assertEqual(first, second)
        self.assertEqual(len(first), 16)

    def test_cosine_similarity_prefers_same_text(self):
        provider = LocalHashingEmbeddingProvider(dimensions=32)
        query = provider.embed("structured extraction")
        same = provider.embed("structured extraction")
        different = provider.embed("banana calendar")

        self.assertGreater(cosine_similarity(query, same), cosine_similarity(query, different))

    def test_empty_text_returns_zero_vector(self):
        provider = LocalHashingEmbeddingProvider(dimensions=8)
        self.assertEqual(provider.embed(""), [0.0] * 8)


class BuildEmbeddingProviderTests(unittest.TestCase):
    def test_local_is_the_default_with_64_dimensions(self):
        provider = build_embedding_provider()

        self.assertIsInstance(provider, LocalHashingEmbeddingProvider)
        self.assertEqual(provider.dimensions, 64)
        self.assertEqual(provider.model_name, "local-hashing-v1")

    def test_openai_defaults_to_text_embedding_3_small(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            provider = build_embedding_provider("openai")

        self.assertIsInstance(provider, OpenAIEmbeddingProvider)
        self.assertEqual(provider.model_name, "text-embedding-3-small")
        self.assertEqual(provider.dimensions, 1536)

    def test_openai_unknown_model_requires_explicit_dimensions(self):
        with self.assertRaises(ValueError):
            OpenAIEmbeddingProvider(model_name="future-model", api_key="test-key")

    def test_openai_requires_api_key(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            with self.assertRaises(ValueError):
                build_embedding_provider("openai")

    def test_unknown_provider_raises(self):
        with self.assertRaises(ValueError):
            build_embedding_provider("anthropic")


class OpenAIEmbeddingProviderTests(unittest.TestCase):
    def _fake_openai_module(self, recorded: dict) -> SimpleNamespace:
        def create(**kwargs):
            recorded.update(kwargs)
            return SimpleNamespace(data=[SimpleNamespace(embedding=[0.1, 0.2, 0.3])])

        client = SimpleNamespace(embeddings=SimpleNamespace(create=create))
        return SimpleNamespace(OpenAI=lambda **kwargs: client)

    def test_embed_returns_api_vector_with_model_and_dimensions(self):
        recorded: dict = {}
        fake_openai = self._fake_openai_module(recorded)
        with patch.dict(sys.modules, {"openai": fake_openai}):
            provider = OpenAIEmbeddingProvider(dimensions=3, api_key="test-key")
            vector = provider.embed("semantic retrieval evidence")

        self.assertEqual(vector, [0.1, 0.2, 0.3])
        self.assertEqual(recorded["model"], "text-embedding-3-small")
        self.assertEqual(recorded["dimensions"], 3)
        self.assertEqual(recorded["input"], "semantic retrieval evidence")

    def test_empty_text_returns_zero_vector_without_api_client(self):
        provider = OpenAIEmbeddingProvider(dimensions=4, api_key="test-key")

        self.assertEqual(provider.embed("   "), [0.0] * 4)
        self.assertIsNone(provider._client)

    def test_embed_documents_batches_and_preserves_input_order(self):
        calls: list[list[str]] = []

        def create(**kwargs):
            calls.append(kwargs["input"])
            return SimpleNamespace(
                data=[
                    SimpleNamespace(index=index, embedding=[float(text[-1]), 0.0])
                    for index, text in enumerate(kwargs["input"])
                ]
            )

        client = SimpleNamespace(embeddings=SimpleNamespace(create=create))
        fake_openai = SimpleNamespace(OpenAI=lambda **kwargs: client)
        with patch.dict(sys.modules, {"openai": fake_openai}):
            provider = OpenAIEmbeddingProvider(dimensions=2, api_key="test-key")
            vectors = provider.embed_documents(
                ["document 1", "", "document 2", "document 3"],
                batch_size=2,
            )

        self.assertEqual(calls, [["document 1", "document 2"], ["document 3"]])
        self.assertEqual(vectors, [[1.0, 0.0], [0.0, 0.0], [2.0, 0.0], [3.0, 0.0]])


class VertexAIEmbeddingProviderTests(unittest.TestCase):
    def test_query_and_document_use_distinct_retrieval_tasks(self):
        calls: list[dict] = []
        client_kwargs: dict = {}

        def embed_content(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(embeddings=[SimpleNamespace(values=[0.1, 0.2, 0.3])])

        client = SimpleNamespace(models=SimpleNamespace(embed_content=embed_content))

        def build_client(**kwargs):
            client_kwargs.update(kwargs)
            return client

        fake_genai = SimpleNamespace(Client=build_client)
        with patch.dict(sys.modules, {"google.genai": fake_genai}):
            provider = VertexAIEmbeddingProvider(
                project="test-project",
                dimensions=3,
            )
            self.assertEqual(provider.embed_document("document evidence"), [0.1, 0.2, 0.3])
            self.assertEqual(provider.embed_query("search query"), [0.1, 0.2, 0.3])

        self.assertEqual(calls[0]["config"]["task_type"], "RETRIEVAL_DOCUMENT")
        self.assertEqual(calls[1]["config"]["task_type"], "RETRIEVAL_QUERY")
        self.assertFalse(calls[0]["config"]["auto_truncate"])
        self.assertEqual(client_kwargs["http_options"], {"api_version": "v1"})

    def test_vertex_requires_project(self):
        with patch.dict(os.environ, {"GOOGLE_CLOUD_PROJECT": ""}, clear=False):
            with self.assertRaises(ValueError):
                VertexAIEmbeddingProvider()


if __name__ == "__main__":
    unittest.main()
