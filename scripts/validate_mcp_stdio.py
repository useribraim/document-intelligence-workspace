from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime
from importlib.metadata import version
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from sqlalchemy.orm import Session

from diw.core.embeddings import LocalHashingEmbeddingProvider
from diw.core.ingestion import ingest_file
from diw.db.models import Base
from diw.db.repository import (
    create_tenant,
    embed_missing_chunks,
    grant_tenant_document_access,
    save_ingested_document,
    save_research_record,
)
from diw.db.session import build_engine


def _payload(result) -> dict:
    if result.structuredContent is not None:
        return result.structuredContent
    for content in result.content:
        text = getattr(content, "text", None)
        if text:
            return json.loads(text)
    return {}


def _elapsed_ms(started: float) -> float:
    return round((perf_counter() - started) * 1000, 2)


def _seed_validation_database(root: Path) -> dict:
    database = root / "mcp-validation.db"
    database_url = f"sqlite+pysqlite:///{database}"
    alpha_source = root / "alpha.md"
    beta_source = root / "beta.md"
    alpha_source.write_text(
        "# Alpha evidence\n\n"
        "The Alpha tenant uses citation-preserving hybrid retrieval for research notes.\n",
        encoding="utf-8",
    )
    beta_source.write_text(
        "# Beta evidence\n\n"
        "The Beta tenant has a private launch code named ORCHID-BETA.\n",
        encoding="utf-8",
    )

    engine = build_engine(database_url)
    Base.metadata.create_all(engine)
    try:
        with Session(engine) as session:
            alpha_document = ingest_file(alpha_source, target_chars=400, overlap_chars=0)
            beta_document = ingest_file(beta_source, target_chars=400, overlap_chars=0)
            save_ingested_document(session, alpha_document)
            save_ingested_document(session, beta_document)
            embed_missing_chunks(session, LocalHashingEmbeddingProvider())

            alpha = create_tenant(session, slug="mcp-alpha", name="MCP Alpha")
            beta = create_tenant(session, slug="mcp-beta", name="MCP Beta")
            grant_tenant_document_access(
                session,
                tenant_id=alpha.id,
                document_id=alpha_document.document_id,
            )
            grant_tenant_document_access(
                session,
                tenant_id=beta.id,
                document_id=beta_document.document_id,
            )
            alpha_record = save_research_record(
                session,
                tenant_id=alpha.id,
                record_type="paper_note",
                title="Alpha retrieval note",
                payload={"status": "reviewed", "topic": "citation-preserving retrieval"},
            )
            beta_record = save_research_record(
                session,
                tenant_id=beta.id,
                record_type="private_note",
                title="Beta private note",
                payload={"launch_code": "ORCHID-BETA"},
            )
            session.commit()
            return {
                "database_url": database_url,
                "alpha_tenant_id": alpha.id,
                "beta_tenant_id": beta.id,
                "alpha_document_id": alpha_document.document_id,
                "beta_document_id": beta_document.document_id,
                "alpha_record_id": alpha_record.id,
                "beta_record_id": beta_record.id,
            }
    finally:
        engine.dispose()


async def validate_mcp_stdio(*, out: Path) -> dict:
    with TemporaryDirectory(prefix="diw-mcp-client-") as temporary_directory:
        seeded = _seed_validation_database(Path(temporary_directory))
        server_parameters = StdioServerParameters(
            command=sys.executable,
            args=[
                "-m",
                "diw.mcp_server",
                "--tenant-id",
                seeded["alpha_tenant_id"],
                "--database-url",
                seeded["database_url"],
            ],
        )
        transcript: list[dict] = []
        started = perf_counter()
        async with stdio_client(server_parameters) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as client:
                initialize_started = perf_counter()
                initialized = await client.initialize()
                transcript.append(
                    {
                        "operation": "initialize",
                        "status": "ok",
                        "server_name": initialized.serverInfo.name,
                        "server_version": initialized.serverInfo.version,
                        "protocol_version": initialized.protocolVersion,
                        "latency_ms": _elapsed_ms(initialize_started),
                    }
                )

                discovery_started = perf_counter()
                tools_response = await client.list_tools()
                discovered_tools = [
                    tool.model_dump(by_alias=True, exclude_none=True)
                    for tool in tools_response.tools
                ]
                transcript.append(
                    {
                        "operation": "tools/list",
                        "status": "ok",
                        "tools": discovered_tools,
                        "latency_ms": _elapsed_ms(discovery_started),
                    }
                )

                search_arguments = {
                    "query": "How does the Alpha tenant preserve citations?",
                    "top_k": 5,
                }
                search_started = perf_counter()
                search_result = await client.call_tool(
                    "search_documents",
                    arguments=search_arguments,
                )
                search_payload = _payload(search_result)
                transcript.append(
                    {
                        "operation": "tools/call",
                        "tool": "search_documents",
                        "arguments": search_arguments,
                        "is_error": search_result.isError,
                        "result": search_payload,
                        "latency_ms": _elapsed_ms(search_started),
                    }
                )

                record_arguments = {"record_id": seeded["alpha_record_id"]}
                record_started = perf_counter()
                record_result = await client.call_tool(
                    "get_research_record",
                    arguments=record_arguments,
                )
                record_payload = _payload(record_result)
                transcript.append(
                    {
                        "operation": "tools/call",
                        "tool": "get_research_record",
                        "arguments": record_arguments,
                        "is_error": record_result.isError,
                        "result": record_payload,
                        "latency_ms": _elapsed_ms(record_started),
                    }
                )

                cross_tenant_arguments = {"record_id": seeded["beta_record_id"]}
                cross_tenant_started = perf_counter()
                cross_tenant_result = await client.call_tool(
                    "get_research_record",
                    arguments=cross_tenant_arguments,
                )
                cross_tenant_payload = _payload(cross_tenant_result)
                transcript.append(
                    {
                        "operation": "tools/call",
                        "tool": "get_research_record",
                        "arguments": cross_tenant_arguments,
                        "expected": "not found because record belongs to another tenant",
                        "is_error": cross_tenant_result.isError,
                        "result": cross_tenant_payload,
                        "latency_ms": _elapsed_ms(cross_tenant_started),
                    }
                )

                override_arguments = {
                    "query": "ORCHID-BETA",
                    "top_k": 5,
                    "tenant_id": seeded["beta_tenant_id"],
                }
                override_started = perf_counter()
                override_result = await client.call_tool(
                    "search_documents",
                    arguments=override_arguments,
                )
                override_payload = _payload(override_result)
                transcript.append(
                    {
                        "operation": "tools/call",
                        "tool": "search_documents",
                        "arguments": override_arguments,
                        "expected": (
                            "tenant override rejected or ignored while results remain "
                            "pinned to Alpha"
                        ),
                        "is_error": override_result.isError,
                        "result": override_payload,
                        "latency_ms": _elapsed_ms(override_started),
                    }
                )

        tool_names = {tool["name"] for tool in discovered_tools}
        schemas_exclude_tenant = all(
            "tenant_id" not in tool.get("inputSchema", {}).get("properties", {})
            for tool in discovered_tools
        )
        search_document_ids = {
            item["document_id"] for item in search_payload.get("results", [])
        }
        override_document_ids = {
            item["document_id"] for item in override_payload.get("results", [])
        }
        override_safe = override_result.isError or (
            override_payload.get("tenant_id") == seeded["alpha_tenant_id"]
            and seeded["beta_document_id"] not in override_document_ids
        )
        assertions = {
            "discovered_exact_read_only_tools": tool_names
            == {"search_documents", "get_research_record"},
            "tool_schemas_exclude_tenant_id": schemas_exclude_tenant,
            "evidence_search_is_tenant_pinned": (
                search_document_ids == {seeded["alpha_document_id"]}
                and seeded["beta_document_id"] not in search_document_ids
            ),
            "owned_research_record_found": (
                not record_result.isError and record_payload.get("found") is True
            ),
            "cross_tenant_research_record_hidden": (
                not cross_tenant_result.isError
                and cross_tenant_payload == {"found": False}
            ),
            "model_supplied_tenant_override_failed_safely": override_safe,
        }
        if not all(assertions.values()):
            failed = [name for name, passed in assertions.items() if not passed]
            raise ValueError(f"MCP validation assertions failed: {', '.join(failed)}")

        payload = {
            "schema_version": "mcp-stdio-validation-v1",
            "recorded_at": datetime.now(UTC).isoformat(),
            "client": {
                "implementation": "official Python MCP SDK ClientSession",
                "package": "mcp",
                "package_version": version("mcp"),
                "transport": "stdio",
                "server_process": "python -m diw.mcp_server",
            },
            "server_boundary": {
                "configured_tenant_id": seeded["alpha_tenant_id"],
                "tenant_source": "server process configuration",
                "model_can_supply_tenant_id": False,
                "write_tools_exposed": False,
            },
            "discovered_tools": discovered_tools,
            "transcript": transcript,
            "assertions": assertions,
            "errors": [],
            "total_latency_ms": _elapsed_ms(started),
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate the DIW stdio MCP server from an external SDK client."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/evidence/mcp-stdio-validation.json"),
    )
    args = parser.parse_args()
    try:
        payload = asyncio.run(validate_mcp_stdio(out=args.out))
    except Exception as error:
        print(
            json.dumps(
                {"status": "failed", "type": type(error).__name__, "message": str(error)},
                ensure_ascii=False,
            )
        )
        return 1
    print(
        json.dumps(
            {
                "status": "passed",
                "out": str(args.out),
                "tools": [tool["name"] for tool in payload["discovered_tools"]],
                "assertions": payload["assertions"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
