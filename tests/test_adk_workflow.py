import unittest

from google.adk.tools import AgentTool

from diw.adk_workflow import (
    ADKEconomicsRecorder,
    build_adk_research_system,
    estimate_vertex_cost_usd,
    exact_quote_verifier,
)


class ADKWorkflowTests(unittest.TestCase):
    def test_coordinator_uses_two_real_adk_agent_tools(self):
        def search(query: str, top_k: int) -> dict:
            return {"query": query, "top_k": top_k, "chunks": []}

        coordinator, _ = build_adk_research_system(search_documents=search)

        self.assertEqual(coordinator.name, "research_coordinator")
        self.assertEqual(len(coordinator.tools), 2)
        self.assertTrue(all(isinstance(tool, AgentTool) for tool in coordinator.tools))
        self.assertEqual(
            {tool.agent.name for tool in coordinator.tools},
            {"retrieval_specialist", "citation_verification_specialist"},
        )

    def test_exact_quote_verifier_rejects_non_exact_citation(self):
        evidence = "Hybrid retrieval combines lexical and vector evidence."

        accepted = exact_quote_verifier(
            "The method is hybrid.",
            "Hybrid retrieval combines lexical and vector evidence.",
            evidence,
        )
        rejected = exact_quote_verifier(
            "The method is hybrid.",
            "Hybrid retrieval combines semantic and lexical evidence.",
            evidence,
        )

        self.assertTrue(accepted["exact_quote_valid"])
        self.assertFalse(rejected["exact_quote_valid"])

    def test_vertex_cost_uses_pinned_gemini_25_flash_rates(self):
        cost = estimate_vertex_cost_usd(
            model="gemini-2.5-flash",
            input_tokens=1_000,
            cached_input_tokens=200,
            output_tokens=300,
            thinking_tokens=100,
        )

        self.assertEqual(cost, 0.0005975)
        self.assertIsNone(
            estimate_vertex_cost_usd(
                model="unknown-model",
                input_tokens=1,
                cached_input_tokens=0,
                output_tokens=1,
            )
        )

    def test_empty_economics_summary_does_not_invent_cost_or_throughput(self):
        summary = ADKEconomicsRecorder().summary(total_latency_ms=12.5)

        self.assertEqual(summary["model_call_count"], 0)
        self.assertIsNone(summary["estimated_cost_usd"])
        self.assertIsNone(summary["aggregate_output_tokens_per_second"])


if __name__ == "__main__":
    unittest.main()
