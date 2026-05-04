import unittest

from diw.core.chunking import chunk_markdown
from diw.core.normalisation import normalise_text, normalise_text_with_report


class ChunkingTests(unittest.TestCase):
    def test_normalise_text_compacts_blank_lines(self):
        raw = "Title\r\n\r\n\r\nBody   \r\n\r\n"
        self.assertEqual(normalise_text(raw), "Title\n\nBody")

    def test_normalise_text_report_counts_safe_changes(self):
        raw = "Title  \r\n\r\n\r\nBody\t\n"
        result = normalise_text_with_report(raw)

        self.assertEqual(result.text, "Title\n\nBody")
        self.assertEqual(result.report.original_line_count, 5)
        self.assertEqual(result.report.normalised_line_count, 3)
        self.assertEqual(result.report.trailing_whitespace_lines, 2)
        self.assertEqual(result.report.collapsed_blank_lines, 1)

    def test_normalise_text_preserves_markdown_heading_markers(self):
        raw = "# Paper Title   \n\n## Method\nThe method is described here."
        self.assertEqual(
            normalise_text(raw),
            "# Paper Title\n\n## Method\nThe method is described here.",
        )

    def test_chunk_markdown_preserves_heading_path(self):
        text = """# Paper A

Intro text.

## Method

The method uses retrieval and structured extraction.

## Results

The benchmark reports improved citation validity.
"""
        chunks = chunk_markdown(text, target_chars=220, overlap_chars=0)

        self.assertGreaterEqual(len(chunks), 1)
        method_chunks = [chunk for chunk in chunks if "Method" in chunk.heading_path]
        self.assertTrue(method_chunks)
        self.assertIn("Paper A", method_chunks[0].heading_path)

    def test_chunk_hashes_are_stable(self):
        text = "# A\n\n" + "repeatable text\n" * 40
        first = chunk_markdown(text, target_chars=240, overlap_chars=20)
        second = chunk_markdown(text, target_chars=240, overlap_chars=20)

        self.assertEqual(
            [chunk.content_hash for chunk in first],
            [chunk.content_hash for chunk in second],
        )

    def test_rejects_invalid_overlap(self):
        with self.assertRaises(ValueError):
            chunk_markdown("text", target_chars=300, overlap_chars=300)


if __name__ == "__main__":
    unittest.main()
