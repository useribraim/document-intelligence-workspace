import unittest

from diw.core.evaluation import retrieval_mrr, retrieval_recall_at_k
from diw.core.retrieval import RetrievalResult


def result(text: str) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=text,
        document_id="doc-1",
        version_id="version-1",
        chunk_index=0,
        heading_path=[],
        text=text,
        lexical_score=1.0,
        vector_score=1.0,
        score=1.0,
    )


class EvaluationMetricTests(unittest.TestCase):
    def test_recall_at_k_counts_expected_phrases(self):
        results = [result("the hybrid retrieval method"), result("a small dataset")]

        self.assertEqual(
            retrieval_recall_at_k(results, ["hybrid retrieval", "small dataset"]),
            1.0,
        )
        self.assertEqual(retrieval_recall_at_k(results[:1], ["hybrid retrieval", "small dataset"]), 0.5)

    def test_mrr_uses_first_relevant_rank(self):
        results = [result("unrelated passage"), result("the hybrid retrieval method")]

        self.assertEqual(retrieval_mrr(results, ["hybrid retrieval"]), 0.5)
        self.assertEqual(retrieval_mrr(results, ["not present"]), 0.0)


if __name__ == "__main__":
    unittest.main()
