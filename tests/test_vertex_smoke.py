from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from diw.core.embeddings import LocalHashingEmbeddingProvider
from diw.core.llm import DeterministicStructuredProvider
from diw.vertex_smoke import run_vertex_smoke


class VertexSmokeTests(unittest.TestCase):
    def test_smoke_workflow_records_supported_and_refusal_controls(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "paper.md"
            source.write_text(
                "# Evidence\n\n"
                "## Policy\n\n"
                "When retrieved evidence does not support a request, answer generation "
                "must refuse rather than invent an answer.\n",
                encoding="utf-8",
            )
            payload = run_vertex_smoke(
                project="test-project",
                location="global",
                embedding_model="local-hashing-v1",
                generation_model="deterministic-structured-v1",
                dimensions=64,
                source_paths=(source,),
                embedding_provider=LocalHashingEmbeddingProvider(),
                llm_provider=DeterministicStructuredProvider(),
            )

        self.assertEqual(payload["schema_version"], "vertex-smoke-v1")
        self.assertEqual(len(payload["cases"]), 2)
        self.assertTrue(payload["cases"][0]["citation_validation"]["valid"])
        self.assertFalse(payload["cases"][0]["answer"]["insufficient_evidence"])
        self.assertTrue(payload["cases"][1]["citation_validation"]["valid"])
        self.assertTrue(payload["cases"][1]["answer"]["insufficient_evidence"])
        self.assertEqual(payload["errors"], [])
        self.assertIn("workflow_run_id", payload)


if __name__ == "__main__":
    unittest.main()
