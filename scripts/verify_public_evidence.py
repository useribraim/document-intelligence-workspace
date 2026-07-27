from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "results/evidence"


def load_json(name: str) -> dict:
    return json.loads((EVIDENCE / name).read_text(encoding="utf-8"))


def main() -> int:
    trace = load_json("retrieval-comparison.trace.json")
    assert trace["summary"]["questions"] == 40
    assert trace["summary"]["gold_eval_questions"] == 23
    assert len(trace["traces"]) == 40

    vertex = load_json("vertex-cloud-run-smoke.json")
    assert vertex["project"] == "<redacted-project-id>"
    assert vertex["errors"] == []
    assert vertex["providers"]["embedding"] == {
        "provider": "vertex",
        "model": "gemini-embedding-001",
        "dimensions": 768,
        "document_embedding_vectors_created": 8,
        "query_embedding_calls": 2,
        "successful": True,
    }
    assert vertex["providers"]["generation"] == {
        "provider": "vertex",
        "model": "gemini-2.5-flash",
        "successful_calls": 2,
    }
    cases = {case["case"]: case for case in vertex["cases"]}
    assert cases["supported"]["answer"]["insufficient_evidence"] is False
    assert cases["supported"]["answer"]["citations"]
    assert cases["supported"]["citation_validation"]["valid"] is True
    assert cases["unsupported"]["answer"]["insufficient_evidence"] is True
    assert cases["unsupported"]["answer"]["citations"] == []

    mcp = load_json("mcp-stdio-validation.json")
    assert mcp["errors"] == []
    assert all(mcp["assertions"].values())
    assert [tool["name"] for tool in mcp["discovered_tools"]] == [
        "search_documents",
        "get_research_record",
    ]
    assert all(
        "tenant_id" not in tool["inputSchema"].get("properties", {})
        for tool in mcp["discovered_tools"]
    )

    screenshot = ROOT / "docs/assets/cited-demo.jpg"
    assert screenshot.read_bytes().startswith(b"\xff\xd8\xff")

    public_text = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for directory in (ROOT / "README.md", ROOT / "docs", EVIDENCE)
        for path in (
            [directory]
            if directory.is_file()
            else sorted(item for item in directory.rglob("*") if item.is_file())
        )
        if path.suffix not in {".jpg", ".png"}
    )
    forbidden_patterns = (
        r"\b[A-Za-z0-9._%+-]+@gmail\.com\b",
        r"\bgen-lang-client-\d+\b",
    )
    assert not any(re.search(pattern, public_text) for pattern in forbidden_patterns)

    print("public evidence: valid, redacted, and internally consistent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
