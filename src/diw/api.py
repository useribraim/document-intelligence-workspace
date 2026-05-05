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
        version="1.1.0",
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

    @app.get("/workspace", response_class=HTMLResponse)
    def workspace():
        return HTMLResponse(_workspace_html())

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


def _workspace_html() -> str:
    return """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Document Intelligence Workspace</title>
    <style>
      :root {
        --bg: #f6f7f9;
        --panel: #ffffff;
        --panel-subtle: #f1f3f5;
        --text: #17191c;
        --muted: #636b74;
        --border: #d8dde3;
        --accent: #216869;
        --accent-dark: #174d4f;
        --danger: #9f2f2f;
        --warning: #8a5a00;
        --ok: #216e39;
        --code: #20242a;
      }

      * {
        box-sizing: border-box;
      }

      body {
        margin: 0;
        min-height: 100vh;
        background: var(--bg);
        color: var(--text);
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        line-height: 1.45;
      }

      button,
      input,
      select,
      textarea {
        font: inherit;
      }

      .app-shell {
        min-height: 100vh;
        display: grid;
        grid-template-rows: auto 1fr auto;
      }

      .topbar {
        height: 58px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
        padding: 0 20px;
        background: var(--panel);
        border-bottom: 1px solid var(--border);
      }

      .brand {
        display: flex;
        flex-direction: column;
        gap: 1px;
      }

      .brand strong {
        font-size: 15px;
        line-height: 1.1;
      }

      .brand span {
        color: var(--muted);
        font-size: 12px;
      }

      .top-actions {
        display: flex;
        align-items: center;
        gap: 8px;
      }

      .layout {
        min-height: 0;
        display: grid;
        grid-template-columns: minmax(420px, 1fr) minmax(360px, 42vw);
      }

      .thread {
        min-width: 0;
        display: flex;
        flex-direction: column;
        border-right: 1px solid var(--border);
        background: #fbfcfd;
      }

      .thread-scroll {
        min-height: 0;
        flex: 1;
        overflow: auto;
        padding: 20px;
      }

      .message {
        max-width: 920px;
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 14px;
      }

      .message.user {
        border-left: 4px solid var(--accent);
      }

      .message.assistant {
        border-left: 4px solid #5d6775;
      }

      .message.system {
        background: var(--panel-subtle);
      }

      .message-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        margin-bottom: 8px;
        color: var(--muted);
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.04em;
      }

      .answer-text {
        margin: 0 0 12px;
        white-space: pre-wrap;
      }

      .field-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 8px;
        margin-top: 10px;
      }

      .field {
        padding: 10px;
        background: var(--panel-subtle);
        border-radius: 6px;
        border: 1px solid var(--border);
      }

      .field label {
        display: block;
        color: var(--muted);
        font-size: 12px;
        margin-bottom: 4px;
      }

      .field div {
        overflow-wrap: anywhere;
      }

      .composer {
        display: grid;
        grid-template-columns: minmax(0, 1fr) auto;
        gap: 10px;
        padding: 14px;
        background: var(--panel);
        border-top: 1px solid var(--border);
      }

      .composer textarea {
        min-height: 72px;
        max-height: 180px;
        resize: vertical;
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 10px 12px;
        color: var(--text);
      }

      .controls {
        display: grid;
        gap: 8px;
        min-width: 170px;
      }

      .inline-controls {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
      }

      select,
      input {
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 8px 9px;
        background: white;
      }

      button {
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 8px 11px;
        background: white;
        color: var(--text);
        cursor: pointer;
        white-space: nowrap;
      }

      button.primary {
        background: var(--accent);
        border-color: var(--accent);
        color: white;
      }

      button.primary:hover {
        background: var(--accent-dark);
      }

      button.danger {
        color: var(--danger);
      }

      button:disabled {
        opacity: 0.55;
        cursor: not-allowed;
      }

      .inspector {
        min-width: 0;
        display: grid;
        grid-template-rows: auto 1fr;
        background: var(--panel);
      }

      .tabs {
        display: flex;
        gap: 2px;
        padding: 10px 12px 0;
        border-bottom: 1px solid var(--border);
        background: var(--panel);
      }

      .tab {
        border-bottom-left-radius: 0;
        border-bottom-right-radius: 0;
        border-bottom-color: transparent;
      }

      .tab.active {
        background: var(--panel-subtle);
        border-color: var(--border);
        border-bottom-color: var(--panel-subtle);
      }

      .inspector-scroll {
        min-height: 0;
        overflow: auto;
        padding: 16px;
        background: var(--panel-subtle);
      }

      .panel {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 12px;
      }

      .panel h2,
      .panel h3 {
        margin: 0 0 10px;
        font-size: 15px;
      }

      .meta-grid {
        display: grid;
        grid-template-columns: 120px minmax(0, 1fr);
        gap: 7px 10px;
        font-size: 13px;
      }

      .meta-grid span:nth-child(odd) {
        color: var(--muted);
      }

      code,
      pre {
        font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      }

      code {
        overflow-wrap: anywhere;
      }

      pre {
        margin: 10px 0 0;
        white-space: pre-wrap;
        overflow-wrap: anywhere;
        color: var(--code);
        background: #f8fafc;
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 10px;
        font-size: 12px;
      }

      .chunk {
        border: 1px solid var(--border);
        border-radius: 8px;
        background: white;
        padding: 12px;
        margin-bottom: 10px;
      }

      .chunk-header {
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 10px;
        color: var(--muted);
        font-size: 12px;
        margin-bottom: 8px;
      }

      .chunk p {
        margin: 0;
        white-space: pre-wrap;
      }

      .pill {
        display: inline-flex;
        align-items: center;
        border: 1px solid var(--border);
        border-radius: 999px;
        padding: 2px 8px;
        font-size: 12px;
        color: var(--muted);
        background: white;
      }

      .pill.ok {
        color: var(--ok);
        border-color: #b8d7c0;
        background: #f0faf2;
      }

      .pill.warn {
        color: var(--warning);
        border-color: #dec58f;
        background: #fff8e8;
      }

      .pill.danger {
        color: var(--danger);
        border-color: #e4b6b6;
        background: #fff2f2;
      }

      .empty {
        color: var(--muted);
        margin: 0;
      }

      .review-card {
        display: grid;
        gap: 10px;
      }

      .review-actions {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
      }

      @media (max-width: 920px) {
        .layout {
          grid-template-columns: 1fr;
        }

        .thread {
          border-right: 0;
        }

        .inspector {
          min-height: 520px;
          border-top: 1px solid var(--border);
        }

        .composer {
          grid-template-columns: 1fr;
        }

        .controls {
          min-width: 0;
        }
      }
    </style>
  </head>
  <body>
    <div class="app-shell">
      <header class="topbar">
        <div class="brand">
          <strong>Document Intelligence Workspace</strong>
          <span>Source-cited QA with validation and review</span>
        </div>
        <div class="top-actions">
          <a href="/"><button type="button">Dashboard</button></a>
          <a href="/docs"><button type="button">API</button></a>
          <button id="refreshButton" type="button">Refresh</button>
        </div>
      </header>

      <main class="layout">
        <section class="thread" aria-label="Answer thread">
          <div id="threadScroll" class="thread-scroll">
            <article class="message system">
              <div class="message-header">
                <span>Workspace</span>
                <span id="healthStatus">Checking database...</span>
              </div>
              <p class="answer-text">Ask a question or run a structured extraction against the loaded document corpus. The answer, evidence, validation result, AI-run metadata, and review status will stay visible in one workspace.</p>
            </article>
            <div id="messages"></div>
          </div>

          <form id="askForm" class="composer">
            <textarea id="queryInput" aria-label="Question or extraction instruction">Extract the method, dataset, metric, and limitation from the cited paper section.</textarea>
            <div class="controls">
              <div class="inline-controls">
                <select id="modeSelect" aria-label="Retrieval mode">
                  <option value="hybrid">hybrid</option>
                  <option value="vector">vector</option>
                  <option value="lexical">lexical</option>
                </select>
                <input id="topKInput" aria-label="Top K" type="number" min="1" max="12" value="3">
              </div>
              <button id="askButton" class="primary" type="submit">Run</button>
            </div>
          </form>
        </section>

        <aside class="inspector" aria-label="Evidence and review inspector">
          <nav class="tabs" aria-label="Inspector tabs">
            <button type="button" class="tab active" data-tab="evidence">Evidence</button>
            <button type="button" class="tab" data-tab="run">Run</button>
            <button type="button" class="tab" data-tab="review">Review</button>
          </nav>
          <div class="inspector-scroll">
            <section id="evidencePanel" class="tab-panel"></section>
            <section id="runPanel" class="tab-panel" hidden></section>
            <section id="reviewPanel" class="tab-panel" hidden></section>
          </div>
        </aside>
      </main>
    </div>

    <script>
      const state = {
        answer: null,
        suggestions: [],
        activeTab: "evidence"
      };

      const messages = document.getElementById("messages");
      const evidencePanel = document.getElementById("evidencePanel");
      const runPanel = document.getElementById("runPanel");
      const reviewPanel = document.getElementById("reviewPanel");
      const askForm = document.getElementById("askForm");
      const askButton = document.getElementById("askButton");
      const queryInput = document.getElementById("queryInput");
      const modeSelect = document.getElementById("modeSelect");
      const topKInput = document.getElementById("topKInput");
      const healthStatus = document.getElementById("healthStatus");

      function escapeHtml(value) {
        return String(value ?? "")
          .replaceAll("&", "&amp;")
          .replaceAll("<", "&lt;")
          .replaceAll(">", "&gt;")
          .replaceAll('"', "&quot;")
          .replaceAll("'", "&#039;");
      }

      function statusPill(label, status) {
        return `<span class="pill ${status}">${escapeHtml(label)}</span>`;
      }

      function renderAnswerBody(value) {
        const lines = String(value || "").split("\\n");
        return lines.map((line) => {
          if (line.startsWith("## ")) {
            return escapeHtml(line.slice(3));
          }
          if (line.startsWith("# ")) {
            return escapeHtml(line.slice(2));
          }
          return escapeHtml(line);
        }).join("<br>");
      }

      function shortId(value) {
        if (!value) return "";
        return String(value).slice(0, 8);
      }

      function renderThread() {
        if (!state.answer) {
          return;
        }
        const answer = state.answer.answer || {};
        const validation = state.answer.citation_validation || {};
        const fields = answer.extracted_fields || {};
        const fieldHtml = Object.entries(fields).map(([key, value]) => `
          <div class="field">
            <label>${escapeHtml(key)}</label>
            <div>${escapeHtml(value)}</div>
          </div>
        `).join("");

        messages.innerHTML = `
          <article class="message user">
            <div class="message-header">
              <span>User</span>
              <span>${escapeHtml(state.answer.mode)} retrieval</span>
            </div>
            <p class="answer-text">${escapeHtml(state.answer.query)}</p>
          </article>
          <article class="message assistant">
            <div class="message-header">
              <span>Assistant</span>
              <span>${validation.valid ? "citations valid" : "citation issue"}</span>
            </div>
            <p class="answer-text">${renderAnswerBody(answer.answer || "")}</p>
            ${fieldHtml ? `<div class="field-grid">${fieldHtml}</div>` : ""}
          </article>
        `;
        document.getElementById("threadScroll").scrollTop = document.getElementById("threadScroll").scrollHeight;
      }

      function renderEvidence() {
        if (!state.answer) {
          evidencePanel.innerHTML = `
            <div class="panel">
              <h2>Evidence</h2>
              <p class="empty">Run a question to inspect retrieved chunks, scores, and citation validation.</p>
            </div>
          `;
          return;
        }

        const validation = state.answer.citation_validation || {};
        const chunks = state.answer.retrieved_chunks || [];
        const validationClass = validation.valid ? "ok" : "danger";
        evidencePanel.innerHTML = `
          <div class="panel">
            <h2>Citation validation</h2>
            ${statusPill(validation.valid ? "valid" : "invalid", validationClass)}
            <pre>${escapeHtml(JSON.stringify(validation, null, 2))}</pre>
          </div>
          <div class="panel">
            <h2>Retrieved chunks</h2>
            ${chunks.length ? chunks.map(renderChunk).join("") : `<p class="empty">No chunks retrieved.</p>`}
          </div>
        `;
      }

      function renderChunk(chunk) {
        const heading = (chunk.heading_path || []).join(" / ") || "Untitled section";
        return `
          <article class="chunk">
            <div class="chunk-header">
              <span>${escapeHtml(heading)}</span>
              <span>score ${escapeHtml(chunk.score)} | lex ${escapeHtml(chunk.lexical_score)} | vec ${escapeHtml(chunk.vector_score)}</span>
            </div>
            <p>${escapeHtml(chunk.text)}</p>
            <pre>chunk_id: ${escapeHtml(chunk.chunk_id)}
document_id: ${escapeHtml(chunk.document_id)}
version_id: ${escapeHtml(chunk.version_id)}</pre>
          </article>
        `;
      }

      function renderRun() {
        if (!state.answer) {
          runPanel.innerHTML = `
            <div class="panel">
              <h2>AI run</h2>
              <p class="empty">No AI run selected.</p>
            </div>
          `;
          return;
        }

        runPanel.innerHTML = `
          <div class="panel">
            <h2>Run metadata</h2>
            <div class="meta-grid">
              <span>Run ID</span><code>${escapeHtml(state.answer.ai_run_id)}</code>
              <span>Suggestion</span><code>${escapeHtml(state.answer.suggestion_id)}</code>
              <span>Provider</span><span>${escapeHtml(state.answer.llm_provider)}</span>
              <span>Model</span><span>${escapeHtml(state.answer.llm_model)}</span>
              <span>Embedding</span><span>${escapeHtml(state.answer.embedding_model)}</span>
              <span>Mode</span><span>${escapeHtml(state.answer.mode)}</span>
              <span>Evidence</span><span>${escapeHtml((state.answer.retrieved_chunks || []).length)} chunks</span>
            </div>
          </div>
          <div class="panel">
            <h2>Raw output</h2>
            <pre>${escapeHtml(JSON.stringify(state.answer.answer, null, 2))}</pre>
          </div>
        `;
      }

      function renderReview() {
        const currentSuggestionId = state.answer?.suggestion_id;
        const current = state.suggestions.find((item) => item.id === currentSuggestionId);
        if (!state.answer) {
          reviewPanel.innerHTML = `
            <div class="panel">
              <h2>Review</h2>
              <p class="empty">Run a question to create a pending review suggestion.</p>
            </div>
          `;
          return;
        }

        if (!current) {
          reviewPanel.innerHTML = `
            <div class="panel">
              <h2>Review</h2>
              <p class="empty">Suggestion ${escapeHtml(shortId(currentSuggestionId))} is no longer pending.</p>
            </div>
          `;
          return;
        }

        reviewPanel.innerHTML = `
          <div class="panel review-card">
            <h2>Pending suggestion</h2>
            <div class="meta-grid">
              <span>Status</span><span>${statusPill(current.status, current.status === "pending" ? "warn" : "ok")}</span>
              <span>ID</span><code>${escapeHtml(current.id)}</code>
              <span>AI run</span><code>${escapeHtml(current.ai_run_id)}</code>
              <span>Type</span><span>${escapeHtml(current.suggestion_type)}</span>
            </div>
            <div class="review-actions">
              <button type="button" class="primary" data-review="accept">Accept</button>
              <button type="button" data-review="edit">Mark edited</button>
              <button type="button" class="danger" data-review="reject">Reject</button>
            </div>
          </div>
        `;
      }

      function renderPanels() {
        renderEvidence();
        renderRun();
        renderReview();
      }

      async function requestJson(url, options = {}) {
        const response = await fetch(url, {
          headers: { "Content-Type": "application/json", ...(options.headers || {}) },
          ...options
        });
        if (!response.ok) {
          const detail = await response.text();
          throw new Error(`${response.status} ${detail}`);
        }
        return response.json();
      }

      async function refreshHealth() {
        const health = await requestJson("/health");
        healthStatus.textContent = `${health.documents} docs | ${health.chunks} chunks | ${health.pending_suggestions} pending`;
      }

      async function refreshSuggestions() {
        const payload = await requestJson("/review/suggestions?status=pending");
        state.suggestions = payload.suggestions || [];
      }

      async function runAsk(event) {
        event.preventDefault();
        askButton.disabled = true;
        askButton.textContent = "Running...";
        try {
          const query = queryInput.value.trim();
          if (!query) {
            throw new Error("Question is required.");
          }
          state.answer = await requestJson("/ask", {
            method: "POST",
            body: JSON.stringify({
              query,
              mode: modeSelect.value,
              top_k: Number(topKInput.value || 3),
              ensure_embeddings: true,
              create_suggestion: true
            })
          });
          await refreshSuggestions();
          renderThread();
          renderPanels();
          switchTab("evidence");
          await refreshHealth();
        } catch (error) {
          messages.innerHTML += `
            <article class="message system">
              <div class="message-header"><span>Error</span><span>request failed</span></div>
              <p class="answer-text">${escapeHtml(error.message)}</p>
            </article>
          `;
        } finally {
          askButton.disabled = false;
          askButton.textContent = "Run";
        }
      }

      async function submitReview(decision) {
        if (!state.answer?.suggestion_id) return;
        await requestJson(`/review/suggestions/${state.answer.suggestion_id}/decision`, {
          method: "POST",
          body: JSON.stringify({
            decision,
            reviewer: "local-user",
            note: decision === "accept" ? "Accepted from workspace." : `Marked ${decision} from workspace.`
          })
        });
        await refreshSuggestions();
        renderPanels();
        await refreshHealth();
      }

      function switchTab(tabName) {
        state.activeTab = tabName;
        document.querySelectorAll(".tab").forEach((button) => {
          button.classList.toggle("active", button.dataset.tab === tabName);
        });
        document.querySelectorAll(".tab-panel").forEach((panel) => {
          panel.hidden = panel.id !== `${tabName}Panel`;
        });
      }

      askForm.addEventListener("submit", runAsk);
      document.getElementById("refreshButton").addEventListener("click", async () => {
        await refreshHealth();
        await refreshSuggestions();
        renderPanels();
      });
      document.querySelectorAll(".tab").forEach((button) => {
        button.addEventListener("click", () => switchTab(button.dataset.tab));
      });
      reviewPanel.addEventListener("click", async (event) => {
        const button = event.target.closest("[data-review]");
        if (!button) return;
        button.disabled = true;
        try {
          await submitReview(button.dataset.review);
        } finally {
          button.disabled = false;
        }
      });

      refreshHealth()
        .then(refreshSuggestions)
        .then(renderPanels)
        .catch((error) => {
          healthStatus.textContent = "Database unavailable";
          messages.innerHTML = `
            <article class="message system">
              <div class="message-header"><span>Error</span><span>startup check failed</span></div>
              <p class="answer-text">${escapeHtml(error.message)}</p>
            </article>
          `;
          renderPanels();
        });
    </script>
  </body>
</html>
"""


app = create_app()
