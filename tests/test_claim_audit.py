import unittest
from dataclasses import replace

from diw.core.claim_audit import (
    apply_claim_verification_gate,
    apply_evidence_repair,
    assess_claims,
    cohen_kappa,
    extract_atomic_claims,
    retrieval_gold_metrics,
    summarise_claim_audit,
)
from diw.core.qa import compose_source_cited_answer
from diw.core.retrieval import RetrievalResult


def result(text: str) -> RetrievalResult:
    return RetrievalResult(chunk_id="p_1", document_id="paper_1", version_id="v1", chunk_index=0, heading_path=["Method"], text=text, lexical_score=1.0, vector_score=1.0, score=1.0)


class ClaimAuditTests(unittest.TestCase):
    def test_claim_is_traceable_to_quote_and_assessed(self):
        evidence = result("Hybrid retrieval combines lexical matching and vector similarity over chunks.")
        answer = compose_source_cited_answer("How does hybrid retrieval combine lexical matching?", [evidence])
        claims = extract_atomic_claims(answer, question_id="q_001")
        assessments = assess_claims(claims, answer.citations, [evidence])
        self.assertEqual(claims[0].claim_id, "q_001_c1")
        self.assertTrue(assessments[0].source_exists)
        self.assertEqual(assessments[0].support_label, "fully_supported")

    def test_uncited_claim_is_unsupported(self):
        evidence = result("A method is described here.")
        answer = compose_source_cited_answer("method", [evidence])
        answer.answer = "A separate unsupported assertion."
        claims = extract_atomic_claims(answer, question_id="q_002")
        assessments = assess_claims(claims, answer.citations, [evidence])
        self.assertEqual(assessments[0].support_label, "unsupported")
        self.assertEqual(
            summarise_claim_audit(assessments)["automated_overlap_unsupported_rate"],
            1.0,
        )

    def test_cohen_kappa_handles_perfect_and_nonmatching_labels(self):
        self.assertEqual(cohen_kappa(["fully_supported", "unsupported"], ["fully_supported", "unsupported"]), 1.0)
        self.assertIsNotNone(cohen_kappa(["fully_supported", "unsupported"], ["unsupported", "unsupported"]))

    def test_verification_gate_removes_claims_without_full_support(self):
        evidence = result("Hybrid retrieval combines lexical matching and vector similarity over chunks.")
        answer = compose_source_cited_answer("How does hybrid retrieval combine lexical matching?", [evidence])
        claims = extract_atomic_claims(answer, question_id="q_003")
        assessments = assess_claims(claims, answer.citations, [evidence])
        gated = apply_claim_verification_gate(answer, claims, assessments)
        self.assertFalse(gated.insufficient_evidence)
        self.assertEqual(gated.answer, answer.answer)

        assessments[0].support_label = "unsupported"
        refused = apply_claim_verification_gate(answer, claims, assessments)
        self.assertTrue(refused.insufficient_evidence)

    def test_supported_gate_keeps_partially_supported_claims(self):
        evidence = result("Hybrid retrieval combines lexical matching and vector similarity over chunks.")
        answer = compose_source_cited_answer("How does hybrid retrieval combine lexical matching?", [evidence])
        claims = extract_atomic_claims(answer, question_id="q_004")
        assessments = assess_claims(claims, answer.citations, [evidence])
        assessments[0].support_label = "partially_supported"

        gated = apply_claim_verification_gate(answer, claims, assessments, policy="supported")

        self.assertFalse(gated.insufficient_evidence)
        self.assertEqual(gated.answer, answer.answer)

    def test_evidence_repair_rewrites_partial_claim_to_exact_span(self):
        evidence = result("Hybrid retrieval combines lexical matching and vector similarity over chunks.")
        answer = compose_source_cited_answer("How does hybrid retrieval combine matching?", [evidence])
        claims = extract_atomic_claims(answer, question_id="q_005")
        assessments = assess_claims(claims, answer.citations, [evidence])
        assessments[0].support_label = "partially_supported"

        repaired, actions = apply_evidence_repair(answer, claims, assessments)

        self.assertFalse(repaired.insufficient_evidence)
        self.assertIn(answer.citations[0].quote, repaired.answer)
        self.assertEqual(actions[0]["action"], "rewritten_to_evidence")

    def test_gold_chunk_metrics_measure_recall_and_first_rank(self):
        first = result("first")
        second = replace(result("second"), chunk_id="p_2")

        recall, mrr = retrieval_gold_metrics([first, second], ["p_2", "missing"])

        self.assertEqual(recall, 0.5)
        self.assertEqual(mrr, 0.5)


if __name__ == "__main__":
    unittest.main()
