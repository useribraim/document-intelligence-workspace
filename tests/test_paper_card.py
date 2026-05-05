import unittest

from diw.core.paper_card import PaperCardChunk, build_paper_card


class PaperCardTests(unittest.TestCase):
    def test_build_paper_card_extracts_fields_and_source_evidence(self):
        card = build_paper_card(
            title="Retrieval Paper",
            source_name="retrieval.md",
            version_id="version-1",
            content_hash="hash-1",
            chunks=[
                PaperCardChunk(
                    id="chunk-1",
                    heading_path=["Paper", "Method"],
                    text=(
                        "Problem: citations drift without provenance.\n"
                        "Method: hybrid retrieval with source-cited generation.\n"
                        "Dataset: local paper excerpts.\n"
                    ),
                    start_line=1,
                    end_line=3,
                ),
                PaperCardChunk(
                    id="chunk-2",
                    heading_path=["Paper", "Evaluation"],
                    text="Metric: citation-valid answer rate.\nLimitation: small corpus.",
                    start_line=4,
                    end_line=5,
                ),
            ],
        )

        self.assertIn("# Retrieval Paper", card.markdown)
        self.assertEqual(card.extracted_fields["method"], "hybrid retrieval with source-cited generation.")
        self.assertEqual(card.extracted_fields["metric"], "citation-valid answer rate.")
        self.assertIn("chunk-1", card.source_chunk_ids)
        self.assertIn("## Source Evidence", card.markdown)

    def test_build_paper_card_uses_heading_fallbacks(self):
        card = build_paper_card(
            title="Retrieval Notes",
            source_name="retrieval-notes.md",
            version_id="version-1",
            content_hash="hash-1",
            chunks=[
                PaperCardChunk(
                    id="chunk-1",
                    heading_path=["Retrieval Notes", "Problem"],
                    text="Long-form documents become unreliable when notes are disconnected from evidence.",
                    start_line=1,
                    end_line=2,
                ),
                PaperCardChunk(
                    id="chunk-2",
                    heading_path=["Retrieval Notes", "Method"],
                    text="The workspace normalises text, chunks sections, retrieves evidence, and validates citations.",
                    start_line=3,
                    end_line=4,
                ),
                PaperCardChunk(
                    id="chunk-3",
                    heading_path=["Retrieval Notes", "Evaluation"],
                    text="The evaluation checks retrieval hit rate, citation validity, and refusal behaviour.",
                    start_line=5,
                    end_line=6,
                ),
            ],
        )

        self.assertEqual(
            card.extracted_fields["problem"],
            "Long-form documents become unreliable when notes are disconnected from evidence.",
        )
        self.assertEqual(
            card.extracted_fields["method"],
            "The workspace normalises text, chunks sections, retrieves evidence, and validates citations.",
        )
        self.assertEqual(
            card.extracted_fields["metric"],
            "The evaluation checks retrieval hit rate, citation validity, and refusal behaviour.",
        )
        self.assertNotIn("Needs review.", card.markdown)


if __name__ == "__main__":
    unittest.main()
