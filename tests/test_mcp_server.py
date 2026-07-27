import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

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
from diw.mcp_server import build_mcp_server


class FakeFastMCP:
    def __init__(self, *args, **kwargs):
        self.tools: dict[str, object] = {}

    def tool(self):
        def register(function):
            self.tools[function.__name__] = function
            return function

        return register


class MCPServerTests(unittest.TestCase):
    def test_tools_are_read_only_and_tenant_pinned(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            database_url = f"sqlite+pysqlite:///{root / 'mcp.db'}"
            engine = build_engine(database_url)
            Base.metadata.create_all(engine)
            with Session(engine) as session:
                alpha_path = root / "alpha.md"
                beta_path = root / "beta.md"
                alpha_path.write_text("# Alpha\n\nAlpha-only retrieval evidence.", encoding="utf-8")
                beta_path.write_text("# Beta\n\nBeta-only retrieval evidence.", encoding="utf-8")
                alpha_document = ingest_file(alpha_path)
                beta_document = ingest_file(beta_path)
                save_ingested_document(session, alpha_document)
                save_ingested_document(session, beta_document)
                embed_missing_chunks(session, LocalHashingEmbeddingProvider())
                alpha = create_tenant(session, slug="alpha-mcp", name="Alpha")
                beta = create_tenant(session, slug="beta-mcp", name="Beta")
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
                    record_type="note",
                    title="Alpha note",
                    payload={"private": "alpha"},
                )
                beta_record = save_research_record(
                    session,
                    tenant_id=beta.id,
                    record_type="note",
                    title="Beta note",
                    payload={"private": "beta"},
                )
                session.commit()
                alpha_id = alpha.id
                alpha_record_id = alpha_record.id
                beta_record_id = beta_record.id
            engine.dispose()

            fake_module = SimpleNamespace(FastMCP=FakeFastMCP)
            with patch.dict(sys.modules, {"mcp.server.fastmcp": fake_module}):
                server = build_mcp_server(
                    tenant_id=alpha_id,
                    database_url=database_url,
                )

            search = server.tools["search_documents"]("evidence", top_k=10)
            self.assertEqual(
                {item["document_id"] for item in search["results"]},
                {alpha_document.document_id},
            )
            self.assertTrue(server.tools["get_research_record"](alpha_record_id)["found"])
            self.assertFalse(server.tools["get_research_record"](beta_record_id)["found"])
            self.assertEqual(
                set(server.tools),
                {"search_documents", "get_research_record"},
            )


if __name__ == "__main__":
    unittest.main()
