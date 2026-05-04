import unittest

from diw.core.embeddings import LocalHashingEmbeddingProvider, cosine_similarity


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


if __name__ == "__main__":
    unittest.main()
