import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from diw.core.ingestion import ingest_file


class IngestionTests(unittest.TestCase):
    def test_ingest_file_returns_document_provenance_and_chunks(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "paper.md"
            path.write_text(
                "# Paper\n\n## Method\n\nRetrieval is evaluated with recall at k.\n",
                encoding="utf-8",
            )

            document = ingest_file(path, target_chars=220, overlap_chars=0)

            self.assertEqual(document.source_name, "paper.md")
            self.assertEqual(document.source_type, "markdown")
            self.assertTrue(document.document_id)
            self.assertTrue(document.version_id)
            self.assertEqual(len(document.content_hash), 64)
            self.assertGreaterEqual(len(document.chunks), 1)
            self.assertIn("normalisation_report", document.as_dict())

    def test_version_id_changes_when_content_changes(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "paper.md"
            path.write_text("# Paper\n\nFirst version.\n", encoding="utf-8")
            first = ingest_file(path, target_chars=220, overlap_chars=0)

            path.write_text("# Paper\n\nSecond version.\n", encoding="utf-8")
            second = ingest_file(path, target_chars=220, overlap_chars=0)

            self.assertEqual(first.document_id, second.document_id)
            self.assertNotEqual(first.version_id, second.version_id)
            self.assertNotEqual(first.content_hash, second.content_hash)

    def test_document_json_is_valid(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "note.txt"
            path.write_text("Plain text note.\n", encoding="utf-8")

            document = ingest_file(path, target_chars=220, overlap_chars=0)
            parsed = json.loads(document.to_json())

            self.assertEqual(parsed["source_type"], "plain_text")
            self.assertEqual(parsed["chunk_count"], len(parsed["chunks"]))


if __name__ == "__main__":
    unittest.main()
