from __future__ import annotations

import argparse
import importlib
import os

from sqlalchemy.orm import Session

from diw.core.embeddings import build_embedding_provider
from diw.core.retrieval import retrieval_results_as_dicts, retrieve_chunks
from diw.db.repository import get_research_record_for_tenant, list_tenant_document_ids
from diw.db.schema import create_schema
from diw.db.session import build_engine


def build_mcp_server(
    *,
    tenant_id: str,
    database_url: str | None = None,
    embedding_provider: str = "local",
    embedding_model: str | None = None,
    dimensions: int | None = None,
):
    """Build a read-only, tenant-pinned MCP server.

    The tenant is process configuration rather than a model-controlled tool
    argument, preventing cross-tenant IDs from being supplied in tool calls.
    """
    if not tenant_id:
        raise ValueError("tenant_id is required")
    try:
        fastmcp = importlib.import_module("mcp.server.fastmcp")
    except ImportError as error:
        raise RuntimeError(
            "MCP support requires the 'mcp' extra: pip install -e '.[mcp]'"
        ) from error

    provider = build_embedding_provider(
        embedding_provider,
        embedding_model=embedding_model,
        dimensions=dimensions,
    )
    engine = build_engine(database_url)
    create_schema(engine)
    server = fastmcp.FastMCP("Document Intelligence Workspace", json_response=True)

    @server.tool()
    def search_documents(query: str, top_k: int = 5) -> dict:
        """Search evidence in documents granted to this MCP server's tenant."""
        bounded_top_k = max(1, min(top_k, 10))
        with Session(engine) as session:
            document_ids = set(list_tenant_document_ids(session, tenant_id=tenant_id))
            results = retrieve_chunks(
                session,
                query,
                provider,
                top_k=bounded_top_k,
                mode="hybrid",
                reranker="rrf",
                document_ids=document_ids,
            )
        return {
            "tenant_id": tenant_id,
            "query": query,
            "results": retrieval_results_as_dicts(results),
        }

    @server.tool()
    def get_research_record(record_id: str) -> dict:
        """Read a research record only when it belongs to the configured tenant."""
        with Session(engine) as session:
            record = get_research_record_for_tenant(
                session,
                tenant_id=tenant_id,
                record_id=record_id,
            )
            if record is None:
                return {"found": False}
            return {
                "found": True,
                "record": {
                    "id": record.id,
                    "record_type": record.record_type,
                    "title": record.title,
                    "payload": record.payload,
                },
            }

    return server


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the tenant-pinned read-only DIW MCP server.")
    parser.add_argument("--tenant-id", default=os.getenv("DIW_MCP_TENANT_ID"))
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL"))
    parser.add_argument(
        "--embedding-provider",
        choices=["local", "openai", "vertex"],
        default=os.getenv("EMBEDDING_PROVIDER", "local"),
    )
    parser.add_argument("--embedding-model", default=os.getenv("EMBEDDING_MODEL"))
    parser.add_argument("--dimensions", type=int)
    args = parser.parse_args()
    if not args.tenant_id:
        parser.error("--tenant-id or DIW_MCP_TENANT_ID is required")
    server = build_mcp_server(
        tenant_id=args.tenant_id,
        database_url=args.database_url,
        embedding_provider=args.embedding_provider,
        embedding_model=args.embedding_model,
        dimensions=args.dimensions,
    )
    server.run(transport="stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
