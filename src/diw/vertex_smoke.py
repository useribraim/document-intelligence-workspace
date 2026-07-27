from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Protocol

from sqlalchemy.orm import Session

from diw.core.embeddings import EmbeddingProvider, VertexAIEmbeddingProvider
from diw.core.ingestion import ingest_file
from diw.core.llm import LLMProvider, VertexAIGeminiProvider, generate_structured_answer
from diw.core.qa import validate_citations
from diw.core.retrieval import retrieval_results_as_dicts, retrieve_chunks
from diw.db.repository import (
    embed_missing_chunks,
    save_ai_run,
    save_ingested_document,
)
from diw.db.schema import create_schema
from diw.db.session import build_engine

DEFAULT_SOURCES = (
    Path("data/demo/raw/rag-systems-paper.md"),
    Path("data/demo/raw/evaluation-harness-paper.md"),
    Path("data/demo/raw/ml-paper-excerpt.md"),
)
DEFAULT_SUPPORTED_QUERY = (
    "What should answer generation do when retrieved evidence does not support the request?"
)
DEFAULT_REFUSAL_QUERY = (
    "What binding legal determination does this corpus make about commercial use worldwide?"
)
RESULT_PREFIX = "DIW_VERTEX_SMOKE_RESULT="


class NamedEmbeddingProvider(EmbeddingProvider, Protocol):
    model_name: str
    dimensions: int


def _elapsed_ms(started: float) -> float:
    return round((perf_counter() - started) * 1000, 2)


def run_vertex_smoke(
    *,
    project: str,
    location: str,
    embedding_model: str,
    generation_model: str,
    dimensions: int,
    source_paths: tuple[Path, ...] = DEFAULT_SOURCES,
    supported_query: str = DEFAULT_SUPPORTED_QUERY,
    refusal_query: str = DEFAULT_REFUSAL_QUERY,
    database_url: str | None = None,
    embedding_provider: NamedEmbeddingProvider | None = None,
    llm_provider: LLMProvider | None = None,
) -> dict:
    """Run a credential-free-at-rest Vertex RAG smoke workflow.

    Credentials are resolved by Application Default Credentials. The returned artifact contains
    provider/model provenance, evidence, citations, usage, and a compact redacted trace, but never
    access tokens, request headers, vectors, or credential paths.
    """
    if not project:
        raise ValueError("project is required")
    if not location:
        raise ValueError("location is required")
    if not source_paths:
        raise ValueError("at least one source path is required")
    missing_sources = [str(path) for path in source_paths if not path.is_file()]
    if missing_sources:
        raise ValueError(f"missing smoke sources: {', '.join(missing_sources)}")

    effective_embedding_provider = embedding_provider or VertexAIEmbeddingProvider(
        model_name=embedding_model,
        dimensions=dimensions,
        project=project,
        location=location,
    )
    effective_llm_provider = llm_provider or VertexAIGeminiProvider(
        model=generation_model,
        project=project,
        location=location,
    )

    with TemporaryDirectory(prefix="diw-vertex-smoke-") as temporary_directory:
        effective_database_url = database_url or (
            f"sqlite+pysqlite:///{Path(temporary_directory) / 'vertex-smoke.db'}"
        )
        engine = build_engine(effective_database_url)
        trace: list[dict] = []
        workflow_started = perf_counter()
        try:
            create_schema(engine)
            with Session(engine, autoflush=False, expire_on_commit=False) as session:
                ingest_started = perf_counter()
                documents = [
                    ingest_file(path, target_chars=650, overlap_chars=80)
                    for path in source_paths
                ]
                for document in documents:
                    save_ingested_document(session, document)
                session.flush()
                trace.append(
                    {
                        "step": "ingest_bundled_sources",
                        "status": "ok",
                        "documents": len(documents),
                        "chunks": sum(len(document.chunks) for document in documents),
                        "latency_ms": _elapsed_ms(ingest_started),
                    }
                )

                embedding_started = perf_counter()
                vectors_created = embed_missing_chunks(session, effective_embedding_provider)
                session.flush()
                trace.append(
                    {
                        "step": "vertex_document_embeddings",
                        "status": "ok",
                        "provider": "vertex",
                        "model": effective_embedding_provider.model_name,
                        "dimensions": effective_embedding_provider.dimensions,
                        "vectors_created": vectors_created,
                        "latency_ms": _elapsed_ms(embedding_started),
                    }
                )

                cases = []
                for case_name, query, expected_refusal in (
                    ("supported", supported_query, False),
                    ("unsupported", refusal_query, True),
                ):
                    retrieval_started = perf_counter()
                    results = retrieve_chunks(
                        session,
                        query,
                        effective_embedding_provider,
                        top_k=5,
                        mode="hybrid",
                        reranker="rrf",
                    )
                    trace.append(
                        {
                            "step": f"{case_name}_vertex_query_embedding_and_retrieval",
                            "status": "ok",
                            "provider": "vertex",
                            "model": effective_embedding_provider.model_name,
                            "retrieved_chunks": len(results),
                            "latency_ms": _elapsed_ms(retrieval_started),
                        }
                    )

                    generation_started = perf_counter()
                    answer = generate_structured_answer(
                        query,
                        results,
                        effective_llm_provider,
                    )
                    validation = validate_citations(answer, results)
                    if not validation.valid:
                        raise ValueError(
                            f"{case_name} citation validation failed: {validation.errors}"
                        )
                    if answer.insufficient_evidence != expected_refusal:
                        raise ValueError(
                            f"{case_name} refusal expectation failed: "
                            f"expected {expected_refusal}, got {answer.insufficient_evidence}"
                        )

                    case_payload = {
                        "case": case_name,
                        "query": query,
                        "expected_refusal": expected_refusal,
                        "answer": answer.model_dump(),
                        "citation_validation": validation.model_dump(),
                        "retrieved_chunks": retrieval_results_as_dicts(results),
                    }
                    case_run = save_ai_run(
                        session,
                        run_type="vertex_smoke_case",
                        query=query,
                        retrieval_mode="hybrid_rrf",
                        embedding_model=effective_embedding_provider.model_name,
                        llm_provider=effective_llm_provider.provider,
                        llm_model=effective_llm_provider.model,
                        prompt_version=answer.prompt_version,
                        retrieved_chunk_ids=[result.chunk_id for result in results],
                        citation_valid=validation.valid,
                        insufficient_evidence=answer.insufficient_evidence,
                        output=case_payload,
                        metrics={
                            "input_tokens": answer.input_tokens,
                            "cached_input_tokens": answer.cached_input_tokens,
                            "output_tokens": answer.output_tokens,
                            "completion_attempts": answer.completion_attempts,
                            "retrieved_chunk_count": len(results),
                            "citation_count": len(answer.citations),
                        },
                    )
                    session.flush()
                    case_payload["ai_run_id"] = case_run.id
                    case_run.output = case_payload
                    cases.append(case_payload)
                    trace.append(
                        {
                            "step": f"{case_name}_vertex_gemini_generation",
                            "status": "ok",
                            "provider": effective_llm_provider.provider,
                            "model": effective_llm_provider.model,
                            "input_tokens": answer.input_tokens,
                            "cached_input_tokens": answer.cached_input_tokens,
                            "output_tokens": answer.output_tokens,
                            "completion_attempts": answer.completion_attempts,
                            "citation_valid": validation.valid,
                            "refused": answer.insufficient_evidence,
                            "latency_ms": _elapsed_ms(generation_started),
                        }
                    )

                payload = {
                    "schema_version": "vertex-smoke-v1",
                    "recorded_at": datetime.now(UTC).isoformat(),
                    "platform": {
                        "cloud_run_job": os.getenv("CLOUD_RUN_JOB"),
                        "cloud_run_execution": os.getenv("CLOUD_RUN_EXECUTION"),
                        "cloud_run_task_index": os.getenv("CLOUD_RUN_TASK_INDEX"),
                        "cloud_run_task_attempt": os.getenv("CLOUD_RUN_TASK_ATTEMPT"),
                    },
                    "project": project,
                    "location": location,
                    "providers": {
                        "embedding": {
                            "provider": "vertex",
                            "model": effective_embedding_provider.model_name,
                            "dimensions": effective_embedding_provider.dimensions,
                            "document_embedding_vectors_created": vectors_created,
                            "query_embedding_calls": len(cases),
                            "successful": True,
                        },
                        "generation": {
                            "provider": effective_llm_provider.provider,
                            "model": effective_llm_provider.model,
                            "successful_calls": len(cases),
                        },
                    },
                    "prompt_version": cases[0]["answer"]["prompt_version"],
                    "cases": cases,
                    "trace": trace,
                    "errors": [],
                    "total_latency_ms": _elapsed_ms(workflow_started),
                }
                workflow_run = save_ai_run(
                    session,
                    run_type="vertex_smoke_workflow",
                    retrieval_mode="hybrid_rrf",
                    embedding_model=effective_embedding_provider.model_name,
                    llm_provider=effective_llm_provider.provider,
                    llm_model=effective_llm_provider.model,
                    prompt_version=payload["prompt_version"],
                    retrieved_chunk_ids=sorted(
                        {
                            chunk["chunk_id"]
                            for case in cases
                            for chunk in case["retrieved_chunks"]
                        }
                    ),
                    citation_valid=all(
                        case["citation_validation"]["valid"] for case in cases
                    ),
                    output=payload,
                    metrics={
                        "cases": len(cases),
                        "document_embedding_vectors_created": vectors_created,
                        "input_tokens": sum(
                            case["answer"]["input_tokens"] for case in cases
                        ),
                        "output_tokens": sum(
                            case["answer"]["output_tokens"] for case in cases
                        ),
                    },
                )
                session.flush()
                payload["workflow_run_id"] = workflow_run.id
                workflow_run.output = payload
                session.commit()
                return payload
        finally:
            engine.dispose()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an end-to-end Vertex embedding and Gemini RAG smoke workflow."
    )
    parser.add_argument(
        "--project",
        default=os.getenv("GOOGLE_CLOUD_PROJECT"),
        required=not bool(os.getenv("GOOGLE_CLOUD_PROJECT")),
    )
    parser.add_argument(
        "--location",
        default=os.getenv("GOOGLE_CLOUD_LOCATION", "global"),
    )
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("VERTEX_EMBEDDING_MODEL", "gemini-embedding-001"),
    )
    parser.add_argument(
        "--generation-model",
        default=os.getenv("VERTEX_CHAT_MODEL", "gemini-2.5-flash"),
    )
    parser.add_argument("--dimensions", type=int, default=768)
    parser.add_argument("--out")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        payload = run_vertex_smoke(
            project=args.project,
            location=args.location,
            embedding_model=args.embedding_model,
            generation_model=args.generation_model,
            dimensions=args.dimensions,
        )
    except Exception as error:  # noqa: BLE001 - CLI must serialize provider failures.
        print(
            "DIW_VERTEX_SMOKE_ERROR="
            + json.dumps(
                {"type": type(error).__name__, "message": str(error)},
                ensure_ascii=False,
            )
        )
        return 1

    rendered = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(RESULT_PREFIX + rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
