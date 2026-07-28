from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from diw.auth import (
    AuthenticatedPrincipal,
    GoogleOIDCAuthenticator,
    OIDCAuthenticator,
    TokenAuthenticator,
)
from diw.core.agent import run_bounded_agent
from diw.core.embeddings import (
    LocalHashingEmbeddingProvider,
    build_embedding_provider,
    cosine_similarity,
    embed_document,
    embed_query,
)
from diw.core.ingestion import ingest_file
from diw.core.llm import (
    DeterministicStructuredProvider,
    OpenAIChatProvider,
    VertexAIGeminiProvider,
    generate_structured_answer,
)
from diw.core.paper_card import PaperCardChunk, build_paper_card
from diw.core.qa import compose_source_cited_answer, validate_citations
from diw.core.retrieval import (
    RetrievalResult,
    _rank_results,
    bm25_scores,
    retrieval_results_as_dicts,
    retrieve_chunks,
)
from diw.db.models import (
    AgentRun,
    AgentRunStep,
    AIRun,
    AISuggestion,
    Chunk,
    DocumentVersion,
    SourceDocument,
    WorkspaceUser,
)
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
from diw.web_views import (
    _dashboard_html,
    _google_signin_html,
    _public_demo_html,
    _public_evidence_html,
    _public_landing_html,
    _workspace_html,
)


class IngestRequest(BaseModel):
    path: str
    target_chars: int = Field(default=1200, gt=0)
    overlap_chars: int = Field(default=160, ge=0)


class AskRequest(BaseModel):
    query: str
    mode: str = Field(default="hybrid", pattern="^(lexical|vector|hybrid)$")
    top_k: int = Field(default=5, gt=0)
    dimensions: int | None = Field(default=None, gt=0)
    embedding_provider: str = Field(default="local", pattern="^(local|openai|vertex)$")
    embedding_model: str | None = None
    reranker: str = Field(default="weighted", pattern="^(weighted|rrf)$")
    llm_provider: str = Field(default="deterministic", pattern="^(deterministic|openai|vertex)$")
    llm_model: str | None = None
    ensure_embeddings: bool = True
    create_suggestion: bool = True


class RetrievalPreviewRequest(BaseModel):
    query: str
    mode: str = Field(default="hybrid", pattern="^(lexical|vector|hybrid)$")
    top_k: int = Field(default=5, gt=0)
    dimensions: int | None = Field(default=None, gt=0)
    embedding_provider: str = Field(default="local", pattern="^(local|openai|vertex)$")
    embedding_model: str | None = None
    reranker: str = Field(default="weighted", pattern="^(weighted|rrf)$")
    ensure_embeddings: bool = True


class AgentRunRequest(BaseModel):
    tenant_id: str = Field(min_length=1, max_length=36)
    actor_user_id: str = Field(min_length=1, max_length=36)
    query: str = Field(min_length=1, max_length=2_000)
    max_steps: int = Field(default=5, ge=1, le=5)
    dimensions: int | None = Field(default=None, gt=0)
    embedding_provider: str = Field(default="local", pattern="^(local|openai|vertex)$")
    embedding_model: str | None = None
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


class PublicDemoAskRequest(BaseModel):
    query: str = Field(min_length=2, max_length=500)


EVAL_CASES_PATH = Path("data/demo/evals/review_cases.jsonl")
PAPER_CARDS_DIR = Path("data/demo/wiki/paper_cards")
UPLOADED_DOCUMENTS_DIR = Path("data/demo/raw/uploads")
PUBLIC_DEMO_CORPUS_DIR = Path("data/demo/raw")
SUPPORTED_IMPORT_SUFFIXES = {".md", ".txt"}


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


def _build_llm_provider(provider: str, model: str | None):
    if provider == "deterministic":
        return DeterministicStructuredProvider()
    if provider == "openai":
        return OpenAIChatProvider(model=model or "gpt-5-mini-2025-08-07")
    if provider == "vertex":
        return VertexAIGeminiProvider(model=model)
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


def _safe_upload_name(filename: str) -> str:
    source_name = Path(filename or "document.md").name
    safe_name = "".join(char if char.isalnum() or char in {"-", "_", "."} else "-" for char in source_name)
    safe_name = safe_name.strip(".-")
    return safe_name or "document.md"


def _write_uploaded_document(
    filename: str,
    content: bytes,
    directory: Path = UPLOADED_DOCUMENTS_DIR,
) -> Path:
    safe_name = _safe_upload_name(filename)
    suffix = Path(safe_name).suffix.lower()
    if suffix not in SUPPORTED_IMPORT_SUFFIXES:
        raise HTTPException(status_code=400, detail="only .md and .txt uploads are supported")
    if not content.strip():
        raise HTTPException(status_code=400, detail="uploaded document is empty")
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / safe_name
    if destination.exists():
        stem = destination.stem
        suffix = destination.suffix
        timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
        destination = directory / f"{stem}-{timestamp}{suffix}"
    destination.write_bytes(content)
    return destination


def _public_demo_signature(corpus_dir: Path) -> tuple[tuple[str, int, int], ...]:
    return tuple(
        (path.name, path.stat().st_mtime_ns, path.stat().st_size)
        for path in sorted(corpus_dir.glob("*.md"))
    )


@lru_cache(maxsize=8)
def _load_public_demo_index(
    corpus_dir_value: str, signature: tuple[tuple[str, int, int], ...]
) -> tuple[tuple[tuple[str, str, str, int, tuple[str, ...], str, tuple[float, ...]], ...], int]:
    """Ingest and embed the public corpus once per immutable file signature."""

    del signature  # It is intentionally part of the cache key for invalidation.
    corpus_dir = Path(corpus_dir_value)
    provider = LocalHashingEmbeddingProvider(dimensions=256)
    chunks = []
    for path in sorted(corpus_dir.glob("*.md")):
        document = ingest_file(path, target_chars=650, overlap_chars=80)
        for chunk in document.chunks:
            text = str(chunk["text"])
            heading_path = tuple(str(value) for value in chunk["heading_path"])
            chunks.append(
                (
                    f"demo:{path.stem}:{int(chunk['chunk_index'])}",
                    document.document_id,
                    document.version_id,
                    int(chunk["chunk_index"]),
                    heading_path,
                    text,
                    tuple(embed_document(provider, " ".join([*heading_path, text]))),
                )
            )
    return tuple(chunks), len({chunk[1] for chunk in chunks})


def _retrieve_public_demo_chunks(
    query: str,
    *,
    corpus_dir: Path = PUBLIC_DEMO_CORPUS_DIR,
    top_k: int = 5,
) -> tuple[list[RetrievalResult], int]:
    """Search the bundled demo corpus without a database or external service."""
    provider = LocalHashingEmbeddingProvider(dimensions=256)
    query_vector = embed_query(provider, query)
    cached_chunks, document_count = _load_public_demo_index(
        str(corpus_dir.resolve()), _public_demo_signature(corpus_dir)
    )
    results: list[RetrievalResult] = []
    lexical_inputs = [
        SimpleNamespace(heading_path=list(heading_path), text=text)
        for _, _, _, _, heading_path, text, _ in cached_chunks
    ]
    lexical_scores = bm25_scores(query, lexical_inputs)
    for (
        chunk_id,
        document_id,
        version_id,
        chunk_index,
        heading_path,
        text,
        embedding,
    ), lexical in zip(cached_chunks, lexical_scores, strict=True):
        vector = cosine_similarity(query_vector, list(embedding))
        weighted = (0.55 * lexical) + (0.45 * max(vector, 0.0))
        results.append(
            RetrievalResult(
                chunk_id=chunk_id,
                document_id=document_id,
                version_id=version_id,
                chunk_index=chunk_index,
                heading_path=list(heading_path),
                text=text,
                lexical_score=round(lexical, 6),
                vector_score=round(vector, 6),
                score=round(weighted, 6),
            )
        )

    ranked = _rank_results(results, top_k=top_k, mode="hybrid", reranker="rrf")
    return ranked, document_count


def _public_demo_answer(
    query: str,
    *,
    corpus_dir: Path = PUBLIC_DEMO_CORPUS_DIR,
) -> dict:
    clean_query = query.strip()
    if len(clean_query) < 2:
        raise HTTPException(status_code=422, detail="query must contain at least two characters")

    started = perf_counter()
    results, document_count = _retrieve_public_demo_chunks(
        clean_query,
        corpus_dir=corpus_dir,
    )
    answer = compose_source_cited_answer(
        clean_query,
        results,
        min_lexical_score=0.2,
        max_citations=1,
    )
    validation = validate_citations(answer, results)
    elapsed_ms = round((perf_counter() - started) * 1000, 2)
    return {
        "query": clean_query,
        "answer": answer.model_dump(),
        "citation_validation": validation.model_dump(),
        "retrieved_chunks": retrieval_results_as_dicts(results),
        "trace": {
            "access": "public_read_only",
            "corpus": "bundled synthetic ML-paper excerpts",
            "corpus_documents": document_count,
            "retrieval": "hybrid lexical + local hashing vector",
            "reranker": "reciprocal_rank_fusion",
            "generation": "deterministic extractive response",
            "external_model_request": False,
            "write_tools_available": False,
            "writes_performed": 0,
            "latency_ms": elapsed_ms,
        },
    }


def create_app(
    database_url: str | None = None,
    evaluation_cases_path: Path = EVAL_CASES_PATH,
    paper_cards_dir: Path = PAPER_CARDS_DIR,
    uploaded_documents_dir: Path = UPLOADED_DOCUMENTS_DIR,
    public_demo_corpus_dir: Path = PUBLIC_DEMO_CORPUS_DIR,
    authenticator: TokenAuthenticator | None = None,
) -> FastAPI:
    if authenticator is None and os.getenv("AUTH_MODE", "off").lower() == "oidc":
        authenticator = OIDCAuthenticator.from_env()
    if authenticator is None and os.getenv("AUTH_MODE", "off").lower() == "google":
        authenticator = GoogleOIDCAuthenticator.from_env()
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
        version="1.8.2",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def production_route_boundary(request: Request, call_next):
        if authenticator is not None:
            allowed = (
                request.url.path
                in {
                    "/",
                    "/demo",
                    "/evidence",
                    "/health",
                    "/startup",
                    "/openapi.json",
                    "/docs",
                    "/redoc",
                    "/signin",
                }
                or request.url.path.startswith("/demo/")
                or request.url.path.startswith("/auth/")
                or request.url.path.startswith("/agent-runs")
            )
            if not allowed:
                return JSONResponse(
                    status_code=403,
                    content={
                        "detail": (
                            "route disabled in OIDC mode until it has tenant-scoped authorization"
                        )
                    },
                )
        return await call_next(request)

    def get_session():
        with SessionLocal() as session:
            yield session

    def get_principal(request: Request) -> AuthenticatedPrincipal | None:
        if authenticator is None:
            return None
        authorization = request.headers.get("authorization", "")
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() != "bearer" or not token:
            raise HTTPException(status_code=401, detail="Bearer token required")
        try:
            return authenticator.authenticate(token)
        except Exception as error:
            raise HTTPException(status_code=401, detail="Invalid bearer token") from error

    def enforce_agent_identity(
        *,
        principal: AuthenticatedPrincipal | None,
        tenant_id: str,
        actor_user_id: str | None,
        session: Session,
    ) -> None:
        if principal is None:
            return
        if principal.tenant_id is not None and principal.tenant_id != tenant_id:
            raise HTTPException(status_code=403, detail="tenant claim does not match request")
        user = session.scalar(
            select(WorkspaceUser).where(
                WorkspaceUser.tenant_id == tenant_id,
                WorkspaceUser.subject == principal.subject,
            )
        )
        if user is None:
            raise HTTPException(status_code=403, detail="token subject has no tenant membership")
        if actor_user_id is not None and user.id != actor_user_id:
            raise HTTPException(status_code=403, detail="token subject does not match actor")

    @app.get("/", response_class=HTMLResponse)
    def public_landing():
        return HTMLResponse(_public_landing_html())

    @app.get("/demo", response_class=HTMLResponse)
    def public_demo():
        return HTMLResponse(_public_demo_html())

    @app.post("/demo/ask")
    def public_demo_ask(request: PublicDemoAskRequest):
        return _public_demo_answer(request.query, corpus_dir=public_demo_corpus_dir)

    @app.get("/evidence", response_class=HTMLResponse)
    def public_evidence():
        return HTMLResponse(_public_evidence_html())

    @app.get("/dashboard", response_class=HTMLResponse)
    def dashboard(session: Session = Depends(get_session)):
        pending = list_ai_suggestions(session, status="pending")
        runs = list_ai_runs(session, limit=5)
        documents = list_source_documents(session)
        return HTMLResponse(
            _dashboard_html(
                documents=documents,
                pending=pending,
                runs=runs,
                document_count=count_documents(session),
                version_count=count_versions(session),
                chunk_count=count_chunks(session),
                embedding_count=count_embeddings(session),
            )
        )

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

    @app.get("/startup")
    def startup():
        return {"status": "ready"}

    @app.get("/signin", response_class=HTMLResponse)
    def signin():
        client_id = os.getenv("GOOGLE_OAUTH_CLIENT_ID", "")
        if not client_id:
            raise HTTPException(status_code=503, detail="Google OAuth client is not configured")
        return HTMLResponse(_google_signin_html(client_id))

    @app.get("/auth/whoami")
    def whoami(principal: AuthenticatedPrincipal | None = Depends(get_principal)):
        if principal is None:
            raise HTTPException(status_code=401, detail="authentication is disabled")
        return {
            "subject": principal.subject,
            "email": principal.claims.get("email"),
            "email_verified": principal.claims.get("email_verified"),
            "tenant_id": principal.tenant_id,
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

    @app.post("/documents/import")
    async def import_document(
        request: Request,
        filename: str = Query(...),
        target_chars: int = Query(1200, gt=0),
        overlap_chars: int = Query(160, ge=0),
        session: Session = Depends(get_session),
    ):
        content = await request.body()
        path = _write_uploaded_document(filename, content, directory=uploaded_documents_dir)
        document = ingest_file(
            path,
            target_chars=target_chars,
            overlap_chars=overlap_chars,
        )
        save_ingested_document(session, document)
        session.commit()
        return {
            "document_id": document.document_id,
            "version_id": document.version_id,
            "chunk_count": len(document.chunks),
            "content_hash": document.content_hash,
            "source_name": document.source_name,
            "source_path": str(path),
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
    def create_embeddings(
        session: Session = Depends(get_session),
        dimensions: int | None = None,
        embedding_provider: str = Query(default="local", pattern="^(local|openai|vertex)$"),
        embedding_model: str | None = None,
    ):
        provider = build_embedding_provider(
            embedding_provider,
            dimensions=dimensions,
            embedding_model=embedding_model,
        )
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
        embedding_provider = build_embedding_provider(
            request.embedding_provider,
            dimensions=request.dimensions,
            embedding_model=request.embedding_model,
        )
        if request.ensure_embeddings:
            embed_missing_chunks(session, embedding_provider)
            session.commit()

        results = retrieve_chunks(
            session,
            request.query,
            embedding_provider,
            top_k=request.top_k,
            mode=request.mode,
            reranker=request.reranker,
        )
        return {
            "query": request.query,
            "mode": request.mode,
            "embedding_model": embedding_provider.model_name,
            "retrieved_chunks": retrieval_results_as_dicts(results),
        }

    @app.post("/agent-runs")
    def agent_run(
        request: AgentRunRequest,
        session: Session = Depends(get_session),
        principal: AuthenticatedPrincipal | None = Depends(get_principal),
    ):
        enforce_agent_identity(
            principal=principal,
            tenant_id=request.tenant_id,
            actor_user_id=request.actor_user_id,
            session=session,
        )
        embedding_provider = build_embedding_provider(
            request.embedding_provider,
            dimensions=request.dimensions,
            embedding_model=request.embedding_model,
        )
        if request.ensure_embeddings:
            embed_missing_chunks(session, embedding_provider)
            session.flush()

        payload = run_bounded_agent(
            session,
            tenant_id=request.tenant_id,
            actor_user_id=request.actor_user_id,
            query=request.query,
            embedding_provider=embedding_provider,
            max_steps=request.max_steps,
        )
        session.commit()
        return payload

    @app.get("/agent-runs/{agent_run_id}")
    def agent_run_details(
        agent_run_id: str,
        tenant_id: str = Query(..., min_length=1, max_length=36),
        session: Session = Depends(get_session),
        principal: AuthenticatedPrincipal | None = Depends(get_principal),
    ):
        enforce_agent_identity(
            principal=principal,
            tenant_id=tenant_id,
            actor_user_id=None,
            session=session,
        )
        run = session.get(AgentRun, agent_run_id)
        if run is None or run.tenant_id != tenant_id:
            raise HTTPException(status_code=404, detail="agent run not found")
        steps = session.scalars(
            select(AgentRunStep)
            .where(AgentRunStep.agent_run_id == run.id)
            .order_by(AgentRunStep.sequence)
        ).all()
        return {
            "agent_run_id": run.id,
            "tenant_id": run.tenant_id,
            "status": run.status,
            "query": run.query,
            "trace_id": run.trace_id,
            "tool_policy_version": run.tool_policy_version,
            "max_steps": run.max_steps,
            "output": run.output,
            "metrics": run.metrics,
            "steps": [
                {
                    "id": step.id,
                    "sequence": step.sequence,
                    "tool_name": step.tool_name,
                    "tool_args": step.tool_args,
                    "observation": step.observation,
                    "status": step.status,
                    "error_code": step.error_code,
                    "latency_ms": step.latency_ms,
                }
                for step in steps
            ],
        }

    @app.post("/ask")
    def ask(request: AskRequest, session: Session = Depends(get_session)):
        embedding_provider = build_embedding_provider(
            request.embedding_provider,
            dimensions=request.dimensions,
            embedding_model=request.embedding_model,
        )
        if request.ensure_embeddings:
            embed_missing_chunks(session, embedding_provider)
            session.commit()

        results = retrieve_chunks(
            session,
            request.query,
            embedding_provider,
            top_k=request.top_k,
            mode=request.mode,
            reranker=request.reranker,
        )
        llm_provider = _build_llm_provider(request.llm_provider, request.llm_model)
        answer = generate_structured_answer(request.query, results, llm_provider)
        validation = validate_citations(answer, results)
        payload = {
            "query": request.query,
            "mode": request.mode,
            "reranker": request.reranker,
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


app = create_app()
