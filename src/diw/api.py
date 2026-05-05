from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime
import html
import json
from pathlib import Path
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session, sessionmaker

from diw.core.embeddings import LocalHashingEmbeddingProvider
from diw.core.ingestion import ingest_file
from diw.core.llm import DeterministicStructuredProvider, OpenAIChatProvider, generate_structured_answer
from diw.core.paper_card import PaperCardChunk, build_paper_card
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


class RetrievalPreviewRequest(BaseModel):
    query: str
    mode: str = Field(default="hybrid", pattern="^(lexical|vector|hybrid)$")
    top_k: int = Field(default=5, gt=0)
    dimensions: int = Field(default=64, gt=0)
    ensure_embeddings: bool = True


class ReviewDecisionRequest(BaseModel):
    decision: str = Field(pattern="^(accept|reject|edit)$")
    reviewer: str = "local-user"
    note: str | None = None
    edited_payload: dict | None = None


class EvaluationCaseRequest(BaseModel):
    source: str = "review"
    query: str
    task: str = "structured_extraction"
    expected_behavior: str = "correct_structured_answer"
    expected_fields: dict[str, str] | list[str] | None = None
    review_note: str | None = None
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    ai_run_id: str | None = None
    suggestion_id: str | None = None


class PaperCardDraftRequest(BaseModel):
    version_id: str
    title: str | None = None
    create_suggestion: bool = True


class PaperCardSaveRequest(BaseModel):
    title: str
    markdown: str
    suggestion_id: str | None = None


EVAL_CASES_PATH = Path("data/demo/evals/review_cases.jsonl")
PAPER_CARDS_DIR = Path("data/demo/wiki/paper_cards")


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


def _append_evaluation_case(request: EvaluationCaseRequest, path: Path = EVAL_CASES_PATH) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "id": f"review-{uuid4()}",
        "source": request.source,
        "task": request.task,
        "question": request.query,
        "expected_behavior": request.expected_behavior,
        "expected_fields": request.expected_fields or {},
        "review_note": request.review_note,
        "retrieved_chunk_ids": request.retrieved_chunk_ids,
        "ai_run_id": request.ai_run_id,
        "suggestion_id": request.suggestion_id,
    }
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return payload


def _load_evaluation_cases(limit: int = 20, path: Path = EVAL_CASES_PATH) -> list[dict]:
    if not path.exists():
        return []
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            cases.append(json.loads(line))
    return list(reversed(cases))[:limit]


def _paper_card_path(title: str, directory: Path = PAPER_CARDS_DIR) -> Path:
    slug = "-".join(token for token in title.lower().split() if token)
    slug = "".join(char for char in slug if char.isalnum() or char in {"-", "_"})
    return directory / f"{slug or 'paper-card'}.md"


def _save_paper_card(request: PaperCardSaveRequest, directory: Path = PAPER_CARDS_DIR) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = _paper_card_path(request.title, directory=directory)
    path.write_text(request.markdown, encoding="utf-8")
    return path


def _list_paper_cards(directory: Path = PAPER_CARDS_DIR) -> list[dict]:
    if not directory.exists():
        return []
    cards = []
    for path in sorted(directory.glob("*.md")):
        cards.append(
            {
                "path": str(path),
                "title": path.stem.replace("-", " ").title(),
                "size_bytes": path.stat().st_size,
            }
        )
    return cards


def create_app(
    database_url: str | None = None,
    evaluation_cases_path: Path = EVAL_CASES_PATH,
    paper_cards_dir: Path = PAPER_CARDS_DIR,
) -> FastAPI:
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
        version="1.7.0",
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

    @app.post("/retrieval-preview")
    def retrieval_preview(request: RetrievalPreviewRequest, session: Session = Depends(get_session)):
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
        return {
            "query": request.query,
            "mode": request.mode,
            "embedding_model": embedding_provider.model_name,
            "retrieved_chunks": retrieval_results_as_dicts(results),
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

    @app.post("/evaluation-cases")
    def create_evaluation_case(request: EvaluationCaseRequest):
        case = _append_evaluation_case(request, path=evaluation_cases_path)
        return {
            "saved": True,
            "path": str(evaluation_cases_path),
            "case": case,
        }

    @app.get("/evaluation-cases")
    def evaluation_cases(limit: int = 20):
        cases = _load_evaluation_cases(limit=limit, path=evaluation_cases_path)
        return {
            "path": str(evaluation_cases_path),
            "count": len(cases),
            "cases": cases,
        }

    @app.post("/paper-cards/draft")
    def draft_paper_card(request: PaperCardDraftRequest, session: Session = Depends(get_session)):
        version = session.get(DocumentVersion, request.version_id)
        if version is None:
            raise HTTPException(status_code=404, detail=f"unknown version_id: {request.version_id}")
        document = session.get(SourceDocument, version.document_id)
        chunks = list_chunks_for_version(session, version.id)
        title = request.title or (document.source_name if document is not None else "Paper Card")
        card = build_paper_card(
            title=title,
            source_name=document.source_name if document is not None else version.document_id,
            version_id=version.id,
            content_hash=version.content_hash,
            chunks=[
                PaperCardChunk(
                    id=chunk.id,
                    heading_path=list(chunk.heading_path),
                    text=chunk.text,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                )
                for chunk in chunks
            ],
        )
        payload = {
            "title": card.title,
            "markdown": card.markdown,
            "schema_version": card.schema_version,
            "extracted_fields": card.extracted_fields,
            "source_chunk_ids": card.source_chunk_ids,
            "version_id": version.id,
            "document_id": version.document_id,
        }
        run = save_ai_run(
            session,
            run_type="paper_card",
            query=f"Draft paper card for {title}",
            retrieved_chunk_ids=card.source_chunk_ids,
            output=payload,
            metrics={"source_chunk_count": len(card.source_chunk_ids)},
        )
        payload["ai_run_id"] = run.id
        if request.create_suggestion:
            suggestion = save_ai_suggestion(
                session,
                suggestion_type="markdown_paper_card",
                title=title,
                ai_run_id=run.id,
                payload=payload,
            )
            payload["suggestion_id"] = suggestion.id
        session.commit()
        return payload

    @app.post("/paper-cards/save")
    def save_paper_card(request: PaperCardSaveRequest):
        path = _save_paper_card(request, directory=paper_cards_dir)
        return {
            "saved": True,
            "path": str(path),
            "suggestion_id": request.suggestion_id,
            "saved_at": _isoformat(datetime.now(UTC)),
        }

    @app.get("/paper-cards")
    def paper_cards():
        cards = _list_paper_cards(directory=paper_cards_dir)
        return {
            "directory": str(paper_cards_dir),
            "count": len(cards),
            "cards": cards,
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
        min-width: 0;
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
        flex-shrink: 0;
      }

      .status-chip {
        color: var(--muted);
        font-size: 12px;
        white-space: nowrap;
      }

      .layout {
        min-height: 0;
        display: grid;
        grid-template-columns: minmax(560px, 1fr) minmax(320px, 34vw);
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
        padding: 16px;
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

      .paper-workspace {
        max-width: none;
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 8px;
        margin-bottom: 14px;
        overflow: hidden;
      }

      .paper-workspace-header {
        display: flex;
        justify-content: space-between;
        gap: 14px;
        padding: 14px;
        border-bottom: 1px solid var(--border);
        background: #f8fafc;
      }

      .paper-workspace-title {
        min-width: 0;
      }

      .paper-workspace-title h2 {
        margin: 0 0 4px;
        font-size: 17px;
      }

      .paper-workspace-title p {
        margin: 0;
        color: var(--muted);
        font-size: 13px;
      }

      .paper-workspace-actions {
        display: flex;
        flex-wrap: wrap;
        align-items: flex-start;
        justify-content: flex-end;
        gap: 8px;
      }

      .paper-workspace-grid {
        display: grid;
        grid-template-columns: minmax(0, 0.95fr) minmax(0, 1.15fr);
        gap: 0;
      }

      .paper-workspace-column {
        min-width: 0;
        padding: 14px;
      }

      .paper-workspace-column + .paper-workspace-column {
        border-left: 1px solid var(--border);
      }

      .paper-workspace-column h3 {
        margin: 0 0 10px;
        font-size: 14px;
      }

      .paper-summary {
        display: grid;
        grid-template-columns: 92px minmax(0, 1fr);
        gap: 7px 10px;
        margin-bottom: 12px;
        font-size: 13px;
      }

      .paper-summary span:nth-child(odd) {
        color: var(--muted);
      }

      .paper-chunk-list {
        display: grid;
        gap: 8px;
      }

      .paper-chunk {
        text-align: left;
        white-space: normal;
        overflow-wrap: anywhere;
      }

      .paper-chunk.active {
        border-color: var(--accent);
        background: #eef8f7;
      }

      .paper-artifact-preview {
        max-height: 420px;
        overflow: auto;
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
        grid-template-columns: minmax(0, 1fr) minmax(260px, 320px);
        gap: 10px;
        padding: 14px;
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 8px;
        margin-bottom: 14px;
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
        min-width: 0;
      }

      .inline-controls {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
      }

      .technical-actions {
        display: none;
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

      button.secondary {
        background: #e9eef2;
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
        overflow-x: auto;
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

      .review-form {
        display: grid;
        gap: 8px;
      }

      .review-form textarea {
        min-height: 72px;
        resize: vertical;
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 9px;
      }

      .history-item {
        width: 100%;
        display: block;
        text-align: left;
        margin-bottom: 8px;
        overflow-wrap: anywhere;
        white-space: normal;
      }

      .history-item small {
        display: block;
        color: var(--muted);
        margin-top: 4px;
      }

      .corpus-grid {
        display: grid;
        grid-template-columns: minmax(0, 1fr) minmax(0, 1.2fr);
        gap: 10px;
      }

      .corpus-list {
        display: grid;
        gap: 8px;
      }

      .corpus-item {
        width: 100%;
        text-align: left;
        white-space: normal;
        overflow-wrap: anywhere;
      }

      .corpus-item.active {
        border-color: var(--accent);
        background: #eef8f7;
      }

      .chunk-detail {
        min-height: 180px;
      }

      @media (max-width: 1120px) {
        .corpus-grid {
          grid-template-columns: 1fr;
        }
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

        .paper-workspace-header,
        .paper-workspace-grid {
          display: block;
        }

        .paper-workspace-actions {
          justify-content: flex-start;
          margin-top: 12px;
        }

        .paper-workspace-column + .paper-workspace-column {
          border-left: 0;
          border-top: 1px solid var(--border);
        }
      }

      @media (max-width: 760px) {
        .topbar {
          padding: 0 12px;
          gap: 10px;
        }

        .brand span,
        .status-chip {
          display: none;
        }

        .brand strong {
          font-size: 14px;
        }

        .top-actions {
          gap: 6px;
        }

        .top-actions button {
          padding: 7px 9px;
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
          <span id="healthStatus" class="status-chip">Checking database...</span>
          <a href="/"><button type="button">Dashboard</button></a>
          <a href="/docs"><button type="button">API</button></a>
          <button id="refreshButton" type="button">Refresh</button>
        </div>
      </header>

      <main class="layout">
        <section class="thread" aria-label="Paper workflow">
          <div id="threadScroll" class="thread-scroll">
            <form id="workflowForm" class="composer">
            <textarea id="queryInput" aria-label="Question or extraction instruction">Extract the method, dataset, metric, and limitation from the cited paper section.</textarea>
            <div class="controls">
              <select id="taskModeSelect" aria-label="Task mode">
                <option value="structured_extraction">extract fields</option>
                <option value="question_answering">ask question</option>
                <option value="comparison">compare documents</option>
                <option value="contradiction_check">find contradictions</option>
                <option value="study_note">study note</option>
              </select>
              <div class="inline-controls">
                <select id="modeSelect" aria-label="Retrieval mode">
                  <option value="hybrid">hybrid</option>
                  <option value="vector">vector</option>
                  <option value="lexical">lexical</option>
                </select>
                <input id="topKInput" aria-label="Top K" type="number" min="1" max="12" value="3">
              </div>
              <div class="technical-actions">
                <button id="previewButton" class="secondary" type="button">Preview evidence</button>
                <button id="generateButton" class="primary" type="submit" disabled>Generate from evidence</button>
              </div>
            </div>
            </form>
            <section id="paperWorkspace" class="paper-workspace" aria-label="Paper workspace"></section>
            <div id="messages"></div>
          </div>
        </section>

        <aside class="inspector" aria-label="Evidence and review inspector">
          <nav class="tabs" aria-label="Inspector tabs">
            <button type="button" class="tab active" data-tab="evidence">Evidence</button>
            <button type="button" class="tab" data-tab="review">Review</button>
            <button type="button" class="tab" data-tab="runs">Runs</button>
            <button type="button" class="tab" data-tab="eval">Eval</button>
            <button type="button" class="tab" data-tab="corpus">Corpus</button>
            <button type="button" class="tab" data-tab="paperCard">Card</button>
            <button type="button" class="tab" data-tab="run">Run</button>
          </nav>
          <div class="inspector-scroll">
            <section id="evidencePanel" class="tab-panel"></section>
            <section id="corpusPanel" class="tab-panel" hidden></section>
            <section id="paperCardPanel" class="tab-panel" hidden></section>
            <section id="runPanel" class="tab-panel" hidden></section>
            <section id="reviewPanel" class="tab-panel" hidden></section>
            <section id="runsPanel" class="tab-panel" hidden></section>
            <section id="evalPanel" class="tab-panel" hidden></section>
          </div>
        </aside>
      </main>
    </div>

    <script>
      const state = {
        answer: null,
        preview: null,
        documents: [],
        versions: [],
        chunks: [],
        selectedDocumentId: null,
        selectedVersionId: null,
        selectedChunkId: null,
        paperCard: null,
        paperCards: [],
        paperCardsDir: "",
        reviewDecision: null,
        workflowStatus: "Ready",
        savedArtifact: null,
        lastAnswerKey: "",
        evalCases: [],
        evalCasesPath: "",
        runs: [],
        suggestions: [],
        activeTab: "evidence"
      };

      const messages = document.getElementById("messages");
      const paperWorkspace = document.getElementById("paperWorkspace");
      const evidencePanel = document.getElementById("evidencePanel");
      const corpusPanel = document.getElementById("corpusPanel");
      const paperCardPanel = document.getElementById("paperCardPanel");
      const runPanel = document.getElementById("runPanel");
      const reviewPanel = document.getElementById("reviewPanel");
      const runsPanel = document.getElementById("runsPanel");
      const evalPanel = document.getElementById("evalPanel");
      const workflowForm = document.getElementById("workflowForm");
      const previewButton = document.getElementById("previewButton");
      const generateButton = document.getElementById("generateButton");
      const queryInput = document.getElementById("queryInput");
      const taskModeSelect = document.getElementById("taskModeSelect");
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
        messages.innerHTML = "";
      }

      function selectedDocument() {
        return state.documents.find((document) => document.id === state.selectedDocumentId) || null;
      }

      function selectedVersion() {
        return state.versions.find((version) => version.id === state.selectedVersionId) || null;
      }

      function selectedChunk() {
        return state.chunks.find((chunk) => chunk.id === state.selectedChunkId) || null;
      }

      function currentRequestKey() {
        return [
          queryInput.value.trim(),
          taskModeSelect.value,
          modeSelect.value,
          topKInput.value || "3"
        ].join("|");
      }

      function renderPaperWorkspace() {
        const documentRecord = selectedDocument();
        const version = selectedVersion();
        const chunk = selectedChunk();
        const latestEvidence = state.answer || state.preview;
        const extractedFields = state.answer?.answer?.extracted_fields || {};
        const extractedFieldCount = Object.keys(extractedFields).length;
        const currentSuggestion = state.answer?.suggestion_id
          ? state.suggestions.find((item) => item.id === state.answer.suggestion_id)
          : null;
        const cardState = state.paperCard ? "draft ready" : "not drafted";
        const reviewState = currentSuggestion ? "pending review" : state.reviewDecision?.decision || "not reviewed";

        paperWorkspace.innerHTML = `
          <div class="paper-workspace-header">
            <div class="paper-workspace-title">
              <h2>${escapeHtml(documentRecord?.source_name || "No paper selected")}</h2>
              <p>${escapeHtml(version ? `Version ${shortId(version.id)} | ${state.chunks.length} chunks | card ${cardState}` : "Load a document before using the paper workspace.")}</p>
              <p>${escapeHtml(state.workflowStatus)}</p>
            </div>
            <div class="paper-workspace-actions">
              <button type="button" data-workspace-action="preview">Extract claims</button>
              <button type="button" class="primary" data-workspace-action="generate" ${latestEvidence ? "" : "disabled"}>Generate study answer</button>
              <button type="button" data-workspace-action="draft-card" ${state.selectedVersionId ? "" : "disabled"}>Draft paper card</button>
              <button type="button" data-workspace-action="save-card" ${state.paperCard ? "" : "disabled"}>Save to wiki</button>
              <button type="button" data-workspace-action="create-eval" ${state.answer && state.reviewDecision ? "" : "disabled"}>Create eval case</button>
            </div>
          </div>
          <div class="paper-workspace-grid">
            <div class="paper-workspace-column">
              <h3>Source and evidence</h3>
              <div class="paper-summary">
                <span>Document</span><span>${escapeHtml(documentRecord?.source_name || "none")}</span>
                <span>Version</span><code>${escapeHtml(version?.id || "none")}</code>
                <span>Selected</span><span>${escapeHtml(chunk ? ((chunk.heading_path || []).join(" / ") || `chunk ${chunk.chunk_index}`) : "none")}</span>
                <span>Evidence</span><span>${escapeHtml(latestEvidence ? `${(latestEvidence.retrieved_chunks || []).length} retrieved chunks` : "not previewed")}</span>
                <span>Review</span><span>${escapeHtml(reviewState)}</span>
              </div>
              <div class="paper-chunk-list">
                ${state.chunks.length ? state.chunks.slice(0, 6).map((item) => `
                  <button type="button" class="paper-chunk ${item.id === state.selectedChunkId ? "active" : ""}" data-workspace-chunk-id="${escapeHtml(item.id)}">
                    ${escapeHtml((item.heading_path || []).join(" / ") || `Chunk ${item.chunk_index}`)}
                    <small>lines ${escapeHtml(item.start_line)}-${escapeHtml(item.end_line)} | ${escapeHtml(shortId(item.content_hash))}</small>
                  </button>
                `).join("") : `<p class="empty">No chunks loaded for the selected version.</p>`}
              </div>
              ${chunk ? `<pre>${escapeHtml(chunk.text)}</pre>` : ""}
            </div>
            <div class="paper-workspace-column">
              <h3>Study artifact</h3>
              <div class="paper-summary">
                <span>Answer</span><span>${escapeHtml(state.answer ? (extractedFieldCount ? `${extractedFieldCount} extracted fields` : "generated answer") : "not generated")}</span>
                <span>Citations</span><span>${escapeHtml(state.answer?.citation_validation?.valid === true ? "valid" : state.answer ? "needs attention" : "not checked")}</span>
                <span>Paper card</span><span>${escapeHtml(cardState)}</span>
                <span>Saved cards</span><span>${escapeHtml(state.paperCards.length)}</span>
                ${state.savedArtifact ? `
                  <span>Saved path</span><code>${escapeHtml(state.savedArtifact.path)}</code>
                  <span>Saved at</span><span>${escapeHtml(state.savedArtifact.saved_at)}</span>
                ` : ""}
              </div>
              ${state.paperCard ? `
                <pre class="paper-artifact-preview">${escapeHtml(state.paperCard.markdown)}</pre>
              ` : state.answer ? `
                <div class="field">
                  <label>Generated answer</label>
                  <div>${renderAnswerBody(state.answer.answer?.answer || "")}</div>
                </div>
                ${extractedFieldCount ? `
                  <div class="field-grid">
                    ${Object.entries(extractedFields).map(([key, value]) => `
                      <div class="field">
                        <label>${escapeHtml(key)}</label>
                        <div>${escapeHtml(value)}</div>
                      </div>
                    `).join("")}
                  </div>
                ` : ""}
              ` : `
                <p class="empty">Preview evidence, generate an answer, then draft a paper card from the selected version.</p>
              `}
            </div>
          </div>
        `;
      }

      function renderEvidence() {
        const evidenceSource = state.answer || state.preview;
        if (!evidenceSource) {
          evidencePanel.innerHTML = `
            <div class="panel">
              <h2>Evidence</h2>
              <p class="empty">Preview evidence before generating an answer.</p>
            </div>
          `;
          return;
        }

        const validation = state.answer?.citation_validation;
        const chunks = evidenceSource.retrieved_chunks || [];
        const validationClass = validation?.valid ? "ok" : "danger";
        evidencePanel.innerHTML = `
          <div class="panel">
            <h2>Evidence workflow</h2>
            <div class="meta-grid">
              <span>Task</span><span>${escapeHtml(taskModeSelect.value)}</span>
              <span>Query</span><span>${escapeHtml(evidenceSource.query)}</span>
              <span>Mode</span><span>${escapeHtml(evidenceSource.mode)}</span>
              <span>Chunks</span><span>${escapeHtml(chunks.length)}</span>
            </div>
          </div>
          ${validation ? `
            <div class="panel">
              <h2>Citation validation</h2>
              ${statusPill(validation.valid ? "valid" : "invalid", validationClass)}
              <pre>${escapeHtml(JSON.stringify(validation, null, 2))}</pre>
            </div>
          ` : ""}
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
              <p class="empty">Preview evidence first, then generate to create an AI run.</p>
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

      function renderRuns() {
        if (!state.runs.length) {
          runsPanel.innerHTML = `
            <div class="panel">
              <h2>Run history</h2>
              <p class="empty">No previous runs loaded.</p>
            </div>
          `;
          return;
        }
        runsPanel.innerHTML = `
          <div class="panel">
            <h2>Run history</h2>
            ${state.runs.map((run) => `
              <button type="button" class="history-item" data-run-id="${escapeHtml(run.id)}">
                ${escapeHtml(run.query || run.run_type)}
                <small>${escapeHtml(run.run_type)} | ${escapeHtml(run.retrieval_mode || "")} | citations ${escapeHtml(String(run.citation_valid))}</small>
              </button>
            `).join("")}
          </div>
        `;
      }

      function renderCorpus() {
        const selectedChunk = state.chunks.find((chunk) => chunk.id === state.selectedChunkId);
        corpusPanel.innerHTML = `
          <div class="panel">
            <h2>Corpus browser</h2>
            <div class="meta-grid">
              <span>Documents</span><span>${escapeHtml(state.documents.length)}</span>
              <span>Versions</span><span>${escapeHtml(state.versions.length)}</span>
              <span>Chunks</span><span>${escapeHtml(state.chunks.length)}</span>
            </div>
          </div>
          <div class="corpus-grid">
            <div class="panel">
              <h3>Documents</h3>
              <div class="corpus-list">
                ${state.documents.length ? state.documents.map((document) => `
                  <button type="button" class="corpus-item ${document.id === state.selectedDocumentId ? "active" : ""}" data-document-id="${escapeHtml(document.id)}">
                    ${escapeHtml(document.source_name)}
                    <small>${escapeHtml(document.source_type)} | ${escapeHtml(shortId(document.id))}</small>
                  </button>
                `).join("") : `<p class="empty">No documents loaded.</p>`}
              </div>
            </div>
            <div class="panel">
              <h3>Versions</h3>
              <div class="corpus-list">
                ${state.versions.length ? state.versions.map((version) => `
                  <button type="button" class="corpus-item ${version.id === state.selectedVersionId ? "active" : ""}" data-version-id="${escapeHtml(version.id)}">
                    ${escapeHtml(shortId(version.id))}
                    <small>hash ${escapeHtml(shortId(version.content_hash))} | ${escapeHtml(version.ingested_at || "")}</small>
                  </button>
                `).join("") : `<p class="empty">Select a document to inspect versions.</p>`}
              </div>
            </div>
          </div>
          <div class="panel">
            <h3>Chunks</h3>
            ${state.chunks.length ? state.chunks.map((chunk) => `
              <button type="button" class="corpus-item ${chunk.id === state.selectedChunkId ? "active" : ""}" data-chunk-id="${escapeHtml(chunk.id)}">
                ${escapeHtml((chunk.heading_path || []).join(" / ") || "Untitled section")}
                <small>chunk ${escapeHtml(chunk.chunk_index)} | lines ${escapeHtml(chunk.start_line)}-${escapeHtml(chunk.end_line)}</small>
              </button>
            `).join("") : `<p class="empty">Select a version to inspect chunks.</p>`}
          </div>
          <div class="panel chunk-detail">
            <h3>Chunk detail</h3>
            ${selectedChunk ? `
              <div class="meta-grid">
                <span>Chunk ID</span><code>${escapeHtml(selectedChunk.id)}</code>
                <span>Version</span><code>${escapeHtml(selectedChunk.version_id)}</code>
                <span>Hash</span><code>${escapeHtml(selectedChunk.content_hash)}</code>
                <span>Lines</span><span>${escapeHtml(selectedChunk.start_line)}-${escapeHtml(selectedChunk.end_line)}</span>
              </div>
              <pre>${escapeHtml(selectedChunk.text)}</pre>
            ` : `<p class="empty">Select a chunk to inspect source text and provenance.</p>`}
          </div>
        `;
      }

      function renderPaperCard() {
        paperCardPanel.innerHTML = `
          <div class="panel">
            <h2>Paper card compiler</h2>
            <div class="meta-grid">
              <span>Version</span><code>${escapeHtml(state.selectedVersionId || "none selected")}</code>
              <span>Saved cards</span><span>${escapeHtml(state.paperCards.length)}</span>
              <span>Directory</span><code>${escapeHtml(state.paperCardsDir || "data/demo/wiki/paper_cards")}</code>
            </div>
            <div class="review-actions" style="margin-top: 12px;">
              <button type="button" class="primary" data-paper-card-action="draft" ${state.selectedVersionId ? "" : "disabled"}>Draft paper card</button>
              <button type="button" data-paper-card-action="save" ${state.paperCard ? "" : "disabled"}>Save Markdown artifact</button>
            </div>
          </div>
          <div class="panel">
            <h2>Draft</h2>
            ${state.paperCard ? `
              <div class="meta-grid">
                <span>Title</span><span>${escapeHtml(state.paperCard.title)}</span>
                <span>Suggestion</span><code>${escapeHtml(state.paperCard.suggestion_id)}</code>
                <span>Chunks</span><span>${escapeHtml((state.paperCard.source_chunk_ids || []).length)}</span>
              </div>
              <pre>${escapeHtml(state.paperCard.markdown)}</pre>
            ` : `<p class="empty">Select a document version in Corpus, then draft a paper card.</p>`}
          </div>
          <div class="panel">
            <h2>Saved artifacts</h2>
            ${state.paperCards.length ? state.paperCards.map((card) => `
              <article class="chunk">
                <div class="chunk-header">
                  <span>${escapeHtml(card.title)}</span>
                  <span>${escapeHtml(card.size_bytes)} bytes</span>
                </div>
                <code>${escapeHtml(card.path)}</code>
              </article>
            `).join("") : `<p class="empty">No paper cards saved yet.</p>`}
          </div>
        `;
      }

      function renderEvalCases() {
        const latestCases = state.evalCases || [];
        evalPanel.innerHTML = `
          <div class="panel">
            <h2>Review evaluation cases</h2>
            <div class="meta-grid">
              <span>Saved cases</span><span>${escapeHtml(latestCases.length)}</span>
              <span>Path</span><code>${escapeHtml(state.evalCasesPath || "data/demo/evals/review_cases.jsonl")}</code>
            </div>
          </div>
          <div class="panel">
            <h2>Latest cases</h2>
            ${latestCases.length ? latestCases.map((testCase) => `
              <article class="chunk">
                <div class="chunk-header">
                  <span>${escapeHtml(testCase.task)}</span>
                  <span>${escapeHtml(testCase.expected_behavior)}</span>
                </div>
                <p>${escapeHtml(testCase.question)}</p>
                <pre>${escapeHtml(JSON.stringify({
                  id: testCase.id,
                  suggestion_id: testCase.suggestion_id,
                  retrieved_chunk_ids: testCase.retrieved_chunk_ids,
                  review_note: testCase.review_note
                }, null, 2))}</pre>
              </article>
            `).join("") : `<p class="empty">No review-derived eval cases saved yet.</p>`}
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
              ${state.reviewDecision && ["reject", "edit"].includes(state.reviewDecision.decision) ? `
                <div class="review-actions">
                  <button type="button" class="primary" data-create-eval-case="true">Create eval case</button>
                </div>
              ` : ""}
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
            <div class="review-form">
              <select id="citationQualitySelect" aria-label="Citation quality">
                <option value="citations_supported">citations supported</option>
                <option value="citation_issue">citation issue</option>
                <option value="not_checked">not checked</option>
              </select>
              <select id="completenessSelect" aria-label="Completeness">
                <option value="complete">complete</option>
                <option value="incomplete">incomplete</option>
                <option value="too_broad">too broad</option>
              </select>
              <select id="fieldCorrectnessSelect" aria-label="Field correctness">
                <option value="fields_correct">fields correct</option>
                <option value="field_issue">field issue</option>
                <option value="not_applicable">not applicable</option>
              </select>
              <textarea id="reviewNoteInput" aria-label="Review note" placeholder="Evidence-based review note"></textarea>
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
        renderPaperWorkspace();
        renderEvidence();
        renderCorpus();
        renderPaperCard();
        renderRun();
        renderReview();
        renderRuns();
        renderEvalCases();
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

      async function refreshRuns() {
        const payload = await requestJson("/ai-runs?limit=10");
        state.runs = payload.ai_runs || [];
      }

      async function refreshEvalCases() {
        const payload = await requestJson("/evaluation-cases?limit=10");
        state.evalCases = payload.cases || [];
        state.evalCasesPath = payload.path || "";
      }

      async function refreshCorpus() {
        const payload = await requestJson("/documents");
        state.documents = payload.documents || [];
        if (!state.selectedDocumentId && state.documents.length) {
          state.selectedDocumentId = state.documents[0].id;
          await loadDocumentVersions(state.selectedDocumentId);
        }
      }

      async function loadDocumentVersions(documentId) {
        state.selectedDocumentId = documentId;
        const payload = await requestJson(`/documents/${documentId}/versions`);
        state.versions = payload.versions || [];
        state.selectedVersionId = state.versions.length ? state.versions[0].id : null;
        state.chunks = [];
        state.selectedChunkId = null;
        if (state.selectedVersionId) {
          await loadVersionChunks(state.selectedVersionId);
        }
      }

      async function loadVersionChunks(versionId) {
        state.selectedVersionId = versionId;
        const payload = await requestJson(`/versions/${versionId}/chunks`);
        state.chunks = payload.chunks || [];
        state.selectedChunkId = state.chunks.length ? state.chunks[0].id : null;
      }

      async function refreshPaperCards() {
        const payload = await requestJson("/paper-cards");
        state.paperCards = payload.cards || [];
        state.paperCardsDir = payload.directory || "";
      }

      async function draftPaperCard() {
        if (!state.selectedVersionId) return;
        state.workflowStatus = "Drafting paper card...";
        renderPanels();
        state.paperCard = await requestJson("/paper-cards/draft", {
          method: "POST",
          body: JSON.stringify({
            version_id: state.selectedVersionId,
            create_suggestion: false
          })
        });
        state.workflowStatus = "Paper card drafted";
        await refreshSuggestions();
        await refreshRuns();
        renderPanels();
      }

      async function savePaperCard() {
        if (!state.paperCard) return;
        state.workflowStatus = "Saving paper card...";
        renderPanels();
        const savedArtifact = await requestJson("/paper-cards/save", {
          method: "POST",
          body: JSON.stringify({
            title: state.paperCard.title,
            markdown: state.paperCard.markdown,
            suggestion_id: state.paperCard.suggestion_id
          })
        });
        state.savedArtifact = savedArtifact;
        state.workflowStatus = `Saved to ${savedArtifact.path}`;
        await refreshPaperCards();
        renderPanels();
      }

      async function previewEvidence() {
        previewButton.disabled = true;
        previewButton.textContent = "Previewing...";
        try {
          const query = queryInput.value.trim();
          if (!query) {
            throw new Error("Question is required.");
          }
          state.answer = null;
          state.workflowStatus = "Extracting evidence...";
          state.preview = await requestJson("/retrieval-preview", {
            method: "POST",
            body: JSON.stringify({
              query,
              mode: modeSelect.value,
              top_k: Number(topKInput.value || 3),
              ensure_embeddings: true
            })
          });
          state.workflowStatus = `Evidence extracted: ${(state.preview.retrieved_chunks || []).length} chunks`;
          generateButton.disabled = false;
          renderThread();
          renderPanels();
          switchTab("evidence");
          await refreshHealth();
        } catch (error) {
          messages.innerHTML += `
            <article class="message system">
              <div class="message-header"><span>Error</span><span>preview failed</span></div>
              <p class="answer-text">${escapeHtml(error.message)}</p>
            </article>
          `;
        } finally {
          previewButton.disabled = false;
          previewButton.textContent = "Preview evidence";
        }
      }

      async function runAsk(event) {
        event.preventDefault();
        generateButton.disabled = true;
        generateButton.textContent = "Generating...";
        try {
          const query = queryInput.value.trim();
          if (!query) {
            throw new Error("Question is required.");
          }
          const requestKey = currentRequestKey();
          if (state.answer && state.lastAnswerKey === requestKey) {
            state.workflowStatus = "Answer already generated for this request";
            renderPanels();
            return;
          }
          state.workflowStatus = "Generating study answer...";
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
          state.workflowStatus = "Study answer generated";
          state.lastAnswerKey = requestKey;
          state.preview = null;
          state.reviewDecision = null;
          await refreshSuggestions();
          await refreshRuns();
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
          generateButton.disabled = false;
          generateButton.textContent = "Generate from evidence";
        }
      }

      async function submitReview(decision) {
        if (!state.answer?.suggestion_id) return;
        const citationQuality = document.getElementById("citationQualitySelect")?.value || "not_checked";
        const completeness = document.getElementById("completenessSelect")?.value || "not_checked";
        const fieldCorrectness = document.getElementById("fieldCorrectnessSelect")?.value || "not_checked";
        const reviewNote = document.getElementById("reviewNoteInput")?.value || "";
        const decisionPayload = await requestJson(`/review/suggestions/${state.answer.suggestion_id}/decision`, {
          method: "POST",
          body: JSON.stringify({
            decision,
            reviewer: "local-user",
            note: [
              `citation_quality=${citationQuality}`,
              `completeness=${completeness}`,
              `field_correctness=${fieldCorrectness}`,
              reviewNote.trim()
            ].filter(Boolean).join("; ")
          })
        });
        state.reviewDecision = decisionPayload;
        state.workflowStatus = `Review recorded: ${decision}`;
        await refreshSuggestions();
        renderPanels();
        await refreshHealth();
      }

      async function createEvalCaseFromReview() {
        if (!state.answer || !state.reviewDecision) return;
        const answer = state.answer.answer || {};
        const retrievedChunks = state.answer.retrieved_chunks || [];
        const expectedFields = answer.extracted_fields || {};
        await requestJson("/evaluation-cases", {
          method: "POST",
          body: JSON.stringify({
            source: "review",
            query: state.answer.query,
            task: taskModeSelect.value,
            expected_behavior: state.reviewDecision.decision === "reject" ? "failure_regression" : "corrected_output",
            expected_fields: expectedFields,
            review_note: state.reviewDecision.note,
            retrieved_chunk_ids: retrievedChunks.map((chunk) => chunk.chunk_id),
            ai_run_id: state.answer.ai_run_id,
            suggestion_id: state.answer.suggestion_id
          })
        });
        state.workflowStatus = "Evaluation case created";
        await refreshEvalCases();
        renderPanels();
        switchTab("eval");
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

      workflowForm.addEventListener("submit", runAsk);
      previewButton.addEventListener("click", previewEvidence);
      paperWorkspace.addEventListener("click", async (event) => {
        const chunkButton = event.target.closest("[data-workspace-chunk-id]");
        if (chunkButton) {
          state.selectedChunkId = chunkButton.dataset.workspaceChunkId;
          renderPanels();
          return;
        }
        const button = event.target.closest("[data-workspace-action]");
        if (!button) return;
        button.disabled = true;
        try {
          if (button.dataset.workspaceAction === "preview") {
            await previewEvidence();
          }
          if (button.dataset.workspaceAction === "generate") {
            await runAsk(new Event("submit"));
          }
          if (button.dataset.workspaceAction === "draft-card") {
            await draftPaperCard();
            switchTab("paperCard");
          }
          if (button.dataset.workspaceAction === "save-card") {
            await savePaperCard();
          }
          if (button.dataset.workspaceAction === "create-eval") {
            await createEvalCaseFromReview();
          }
        } finally {
          button.disabled = false;
        }
      });
      document.getElementById("refreshButton").addEventListener("click", async () => {
        await refreshHealth();
        await refreshSuggestions();
        await refreshCorpus();
        await refreshPaperCards();
        renderPanels();
      });
      document.querySelectorAll(".tab").forEach((button) => {
        button.addEventListener("click", () => switchTab(button.dataset.tab));
      });
      reviewPanel.addEventListener("click", async (event) => {
        const evalButton = event.target.closest("[data-create-eval-case]");
        if (evalButton) {
          evalButton.disabled = true;
          try {
            await createEvalCaseFromReview();
          } finally {
            evalButton.disabled = false;
          }
          return;
        }
        const button = event.target.closest("[data-review]");
        if (!button) return;
        button.disabled = true;
        try {
          await submitReview(button.dataset.review);
        } finally {
          button.disabled = false;
        }
      });
      runsPanel.addEventListener("click", (event) => {
        const button = event.target.closest("[data-run-id]");
        if (!button) return;
        const run = state.runs.find((item) => item.id === button.dataset.runId);
        if (!run) return;
        state.answer = run.output;
        state.answer.ai_run_id = run.id;
        state.preview = null;
        renderThread();
        renderPanels();
        switchTab("evidence");
      });
      corpusPanel.addEventListener("click", async (event) => {
        const documentButton = event.target.closest("[data-document-id]");
        if (documentButton) {
          await loadDocumentVersions(documentButton.dataset.documentId);
          renderPanels();
          return;
        }
        const versionButton = event.target.closest("[data-version-id]");
        if (versionButton) {
          await loadVersionChunks(versionButton.dataset.versionId);
          renderPanels();
          return;
        }
        const chunkButton = event.target.closest("[data-chunk-id]");
        if (chunkButton) {
          state.selectedChunkId = chunkButton.dataset.chunkId;
          renderPanels();
        }
      });
      paperCardPanel.addEventListener("click", async (event) => {
        const button = event.target.closest("[data-paper-card-action]");
        if (!button) return;
        button.disabled = true;
        try {
          if (button.dataset.paperCardAction === "draft") {
            await draftPaperCard();
          }
          if (button.dataset.paperCardAction === "save") {
            await savePaperCard();
          }
        } finally {
          button.disabled = false;
        }
      });

      refreshHealth()
        .then(refreshSuggestions)
        .then(refreshRuns)
        .then(refreshEvalCases)
        .then(refreshCorpus)
        .then(refreshPaperCards)
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
