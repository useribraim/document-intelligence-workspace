from __future__ import annotations

from contextlib import asynccontextmanager
import html
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session, sessionmaker

from diw.core.embeddings import LocalHashingEmbeddingProvider
from diw.core.ingestion import ingest_file
from diw.core.llm import DeterministicStructuredProvider, OpenAIChatProvider, generate_structured_answer
from diw.core.qa import validate_citations
from diw.core.retrieval import retrieval_results_as_dicts, retrieve_chunks
from diw.db.models import AIRun, AISuggestion, Chunk, DocumentVersion, SourceDocument
from diw.db.repository import (
    count_ai_suggestions,
    count_chunks,
    count_documents,
    count_embeddings,
    count_versions,
    embed_missing_chunks,
    list_ai_runs,
    list_ai_suggestions,
    list_chunks_for_version,
    list_document_versions,
    list_source_documents,
    record_review_decision,
    save_ai_run,
    save_ai_suggestion,
    save_ingested_document,
)
from diw.db.schema import create_schema
from diw.db.session import build_engine


class IngestRequest(BaseModel):
    path: str
    target_chars: int = Field(default=1200, gt=0)
    overlap_chars: int = Field(default=160, ge=0)


class AskRequest(BaseModel):
    query: str
    mode: str = Field(default="hybrid", pattern="^(lexical|vector|hybrid)$")
    top_k: int = Field(default=5, gt=0)
    dimensions: int = Field(default=64, gt=0)
    llm_provider: str = Field(default="deterministic", pattern="^(deterministic|openai)$")
    llm_model: str = "gpt-4.1-mini"
    ensure_embeddings: bool = True
    create_suggestion: bool = True


class ReviewDecisionRequest(BaseModel):
    decision: str = Field(pattern="^(accept|reject|edit)$")
    reviewer: str = "local-user"
    note: str | None = None
    edited_payload: dict | None = None


def _isoformat(value) -> str | None:
    if value is None:
        return None
    return value.isoformat()


def _document_as_dict(document: SourceDocument) -> dict:
    return {
        "id": document.id,
        "source_path": document.source_path,
        "source_name": document.source_name,
        "source_type": document.source_type,
        "created_at": _isoformat(document.created_at),
    }


def _version_as_dict(version: DocumentVersion) -> dict:
    return {
        "id": version.id,
        "document_id": version.document_id,
        "content_hash": version.content_hash,
        "normalisation_report": version.normalisation_report,
        "ingested_at": _isoformat(version.ingested_at),
    }


def _chunk_as_dict(chunk: Chunk) -> dict:
    return {
        "id": chunk.id,
        "version_id": chunk.version_id,
        "chunk_index": chunk.chunk_index,
        "heading_path": chunk.heading_path,
        "text": chunk.text,
        "content_hash": chunk.content_hash,
        "start_line": chunk.start_line,
        "end_line": chunk.end_line,
    }


def _ai_run_as_dict(run: AIRun) -> dict:
    return {
        "id": run.id,
        "run_type": run.run_type,
        "query": run.query,
        "retrieval_mode": run.retrieval_mode,
        "embedding_model": run.embedding_model,
        "llm_provider": run.llm_provider,
        "llm_model": run.llm_model,
        "prompt_version": run.prompt_version,
        "retrieved_chunk_ids": run.retrieved_chunk_ids,
        "citation_valid": run.citation_valid,
        "insufficient_evidence": run.insufficient_evidence,
        "output": run.output,
        "metrics": run.metrics,
        "created_at": _isoformat(run.created_at),
    }


def _suggestion_as_dict(suggestion: AISuggestion) -> dict:
    return {
        "id": suggestion.id,
        "ai_run_id": suggestion.ai_run_id,
        "suggestion_type": suggestion.suggestion_type,
        "status": suggestion.status,
        "title": suggestion.title,
        "payload": suggestion.payload,
        "created_at": _isoformat(suggestion.created_at),
        "reviewed_at": _isoformat(suggestion.reviewed_at),
    }


def _build_llm_provider(provider: str, model: str):
    if provider == "deterministic":
        return DeterministicStructuredProvider()
    if provider == "openai":
        return OpenAIChatProvider(model=model)
    raise HTTPException(status_code=400, detail=f"unsupported LLM provider: {provider}")


def create_app(database_url: str | None = None) -> FastAPI:
    engine = build_engine(database_url)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        create_schema(engine)
        try:
            yield
        finally:
            engine.dispose()

    app = FastAPI(
        title="Document Intelligence Workspace",
        version="1.0.0",
        lifespan=lifespan,
    )

    def get_session():
        with SessionLocal() as session:
            yield session

    @app.get("/", response_class=HTMLResponse)
    def dashboard(session: Session = Depends(get_session)):
        pending = list_ai_suggestions(session, status="pending")
        runs = list_ai_runs(session, limit=5)
        documents = list_source_documents(session)
        body = f"""
        <!doctype html>
        <html>
          <head>
            <meta charset="utf-8">
            <title>Document Intelligence Workspace</title>
            <style>
              body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 32px; line-height: 1.45; }}
              h1 {{ margin-bottom: 8px; }}
              section {{ margin-top: 28px; }}
              table {{ border-collapse: collapse; width: 100%; }}
              th, td {{ border-bottom: 1px solid #ddd; padding: 8px; text-align: left; vertical-align: top; }}
              code {{ background: #f4f4f4; padding: 2px 4px; }}
              .metric {{ display: inline-block; margin-right: 18px; }}
            </style>
          </head>
          <body>
            <h1>Document Intelligence Workspace</h1>
            <p>Inspectable source-cited QA, AI-run provenance, and human review.</p>
            <section>
              <strong class="metric">Documents: {count_documents(session)}</strong>
              <strong class="metric">Versions: {count_versions(session)}</strong>
              <strong class="metric">Chunks: {count_chunks(session)}</strong>
              <strong class="metric">Embeddings: {count_embeddings(session)}</strong>
              <strong class="metric">Pending review: {len(pending)}</strong>
            </section>
            <section>
              <h2>Documents</h2>
              {_html_documents_table(documents)}
            </section>
            <section>
              <h2>Pending Review</h2>
              {_html_suggestions_table(pending)}
            </section>
            <section>
              <h2>Recent AI Runs</h2>
              {_html_runs_table(runs)}
            </section>
            <section>
              <h2>API</h2>
              <p>Use <code>/docs</code> for the interactive API console.</p>
            </section>
          </body>
        </html>
        """
        return HTMLResponse(body)

    @app.get("/health")
    def health(session: Session = Depends(get_session)):
        return {
            "status": "ok",
            "documents": count_documents(session),
            "versions": count_versions(session),
            "chunks": count_chunks(session),
            "embeddings": count_embeddings(session),
            "pending_suggestions": count_ai_suggestions(session),
        }

    @app.post("/documents/ingest")
    def ingest_document(request: IngestRequest, session: Session = Depends(get_session)):
        path = Path(request.path)
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"input file not found: {path}")
        document = ingest_file(
            path,
            target_chars=request.target_chars,
            overlap_chars=request.overlap_chars,
        )
        save_ingested_document(session, document)
        session.commit()
        return {
            "document_id": document.document_id,
            "version_id": document.version_id,
            "chunk_count": len(document.chunks),
            "content_hash": document.content_hash,
        }

    @app.get("/documents")
    def documents(session: Session = Depends(get_session)):
        return {"documents": [_document_as_dict(document) for document in list_source_documents(session)]}

    @app.get("/documents/{document_id}/versions")
    def document_versions(document_id: str, session: Session = Depends(get_session)):
        versions = list_document_versions(session, document_id)
        if not versions and session.get(SourceDocument, document_id) is None:
            raise HTTPException(status_code=404, detail=f"unknown document_id: {document_id}")
        return {"document_id": document_id, "versions": [_version_as_dict(version) for version in versions]}

    @app.get("/versions/{version_id}/chunks")
    def version_chunks(version_id: str, session: Session = Depends(get_session)):
        chunks = list_chunks_for_version(session, version_id)
        if not chunks and session.get(DocumentVersion, version_id) is None:
            raise HTTPException(status_code=404, detail=f"unknown version_id: {version_id}")
        return {"version_id": version_id, "chunks": [_chunk_as_dict(chunk) for chunk in chunks]}

    @app.post("/embeddings")
    def create_embeddings(session: Session = Depends(get_session), dimensions: int = 64):
        provider = LocalHashingEmbeddingProvider(dimensions=dimensions)
        created = embed_missing_chunks(session, provider)
        session.commit()
        return {
            "embedding_model": provider.model_name,
            "dimensions": provider.dimensions,
            "embeddings_created": created,
            "embeddings_total": count_embeddings(session),
        }

    @app.post("/ask")
    def ask(request: AskRequest, session: Session = Depends(get_session)):
        embedding_provider = LocalHashingEmbeddingProvider(dimensions=request.dimensions)
        if request.ensure_embeddings:
            embed_missing_chunks(session, embedding_provider)
            session.commit()

        results = retrieve_chunks(
            session,
            request.query,
            embedding_provider,
            top_k=request.top_k,
            mode=request.mode,
        )
        llm_provider = _build_llm_provider(request.llm_provider, request.llm_model)
        answer = generate_structured_answer(request.query, results, llm_provider)
        validation = validate_citations(answer, results)
        payload = {
            "query": request.query,
            "mode": request.mode,
            "embedding_model": embedding_provider.model_name,
            "llm_provider": llm_provider.provider,
            "llm_model": llm_provider.model,
            "answer": answer.model_dump(),
            "citation_validation": validation.model_dump(),
            "retrieved_chunks": retrieval_results_as_dicts(results),
        }
        run = save_ai_run(
            session,
            run_type="answer_llm",
            query=request.query,
            retrieval_mode=request.mode,
            embedding_model=embedding_provider.model_name,
            llm_provider=llm_provider.provider,
            llm_model=llm_provider.model,
            prompt_version=answer.prompt_version,
            retrieved_chunk_ids=[result.chunk_id for result in results],
            citation_valid=validation.valid,
            insufficient_evidence=answer.insufficient_evidence,
            output=payload,
            metrics={
                "retrieved_chunk_count": len(results),
                "citation_count": len(answer.citations),
            },
        )
        payload["ai_run_id"] = run.id
        if request.create_suggestion:
            suggestion = save_ai_suggestion(
                session,
                suggestion_type="source_cited_answer",
                title=request.query,
                ai_run_id=run.id,
                payload=payload,
            )
            payload["suggestion_id"] = suggestion.id
        session.commit()
        return payload

    @app.get("/ai-runs")
    def ai_runs(session: Session = Depends(get_session), limit: int = 50):
        return {"ai_runs": [_ai_run_as_dict(run) for run in list_ai_runs(session, limit=limit)]}

    @app.get("/review/suggestions")
    def review_suggestions(session: Session = Depends(get_session), status: str | None = None):
        return {
            "status": status,
            "suggestions": [
                _suggestion_as_dict(suggestion)
                for suggestion in list_ai_suggestions(session, status=status)
            ],
        }

    @app.post("/review/suggestions/{suggestion_id}/decision")
    def review_suggestion(
        suggestion_id: str,
        request: ReviewDecisionRequest,
        session: Session = Depends(get_session),
    ):
        try:
            decision = record_review_decision(
                session,
                suggestion_id=suggestion_id,
                decision=request.decision,
                reviewer=request.reviewer,
                note=request.note,
                edited_payload=request.edited_payload,
            )
        except ValueError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        session.commit()
        return {
            "suggestion_id": decision.suggestion_id,
            "decision": decision.decision,
            "reviewer": decision.reviewer,
            "note": decision.note,
            "created_at": _isoformat(decision.created_at),
        }

    return app


def _html_documents_table(documents: list[SourceDocument]) -> str:
    if not documents:
        return "<p>No documents loaded.</p>"
    rows = []
    for document in documents:
        rows.append(
            "<tr>"
            f"<td><code>{html.escape(document.id)}</code></td>"
            f"<td>{html.escape(document.source_name)}</td>"
            f"<td>{html.escape(document.source_type)}</td>"
            "</tr>"
        )
    return "<table><tr><th>ID</th><th>Name</th><th>Type</th></tr>" + "".join(rows) + "</table>"


def _html_suggestions_table(suggestions: list[AISuggestion]) -> str:
    if not suggestions:
        return "<p>No pending suggestions.</p>"
    rows = []
    for suggestion in suggestions:
        rows.append(
            "<tr>"
            f"<td><code>{html.escape(suggestion.id)}</code></td>"
            f"<td>{html.escape(suggestion.status)}</td>"
            f"<td>{html.escape(suggestion.title)}</td>"
            "</tr>"
        )
    return "<table><tr><th>ID</th><th>Status</th><th>Title</th></tr>" + "".join(rows) + "</table>"


def _html_runs_table(runs: list[AIRun]) -> str:
    if not runs:
        return "<p>No AI runs yet.</p>"
    rows = []
    for run in runs:
        rows.append(
            "<tr>"
            f"<td><code>{html.escape(run.id)}</code></td>"
            f"<td>{html.escape(run.run_type)}</td>"
            f"<td>{html.escape(run.query or '')}</td>"
            f"<td>{html.escape(str(run.citation_valid))}</td>"
            "</tr>"
        )
    return "<table><tr><th>ID</th><th>Type</th><th>Query</th><th>Citations valid</th></tr>" + "".join(rows) + "</table>"


app = create_app()
