import json
import unittest

from diw.web_concepts import CONCEPTS, concept_payload, glossary_items, render_concept_text


class WebConceptTests(unittest.TestCase):
    def test_known_concept_marker_becomes_link(self):
        rendered = render_concept_text("Uses [[rrf|rank fusion]].")

        self.assertEqual(
            rendered,
            'Uses <a class="concept" href="#concept-rrf" '
            'data-concept="rrf">rank fusion</a>.',
        )

    def test_unknown_concept_is_plain_escaped_text(self):
        rendered = render_concept_text("Uses [[not-defined|<unsafe>]].")

        self.assertEqual(rendered, "Uses &lt;unsafe&gt;.")
        self.assertNotIn("data-concept", rendered)

    def test_concept_payload_is_valid_and_contains_rendered_nested_links(self):
        payload = json.loads(concept_payload())

        self.assertEqual(set(payload), set(CONCEPTS))
        self.assertIn('data-concept="lexical-score"', payload["hybrid-retrieval"]["body"][0])

    def test_glossary_is_sorted_by_visible_title(self):
        items = glossary_items()
        titles = [title for _, title in items]

        self.assertEqual(titles, sorted(titles, key=str.lower))


if __name__ == "__main__":
    unittest.main()
