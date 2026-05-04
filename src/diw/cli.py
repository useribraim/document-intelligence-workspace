from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from diw.core.embeddings import LocalHashingEmbeddingProvider
from diw.core.evaluation import (
    load_golden_cases,
    render_markdown_report,
    score_case,
    summarise_eval,
)
from diw.core.ingestion import ingest_file
from diw.core.llm import DeterministicStructuredProvider, OpenAIChatProvider, generate_structured_answer
from diw.core.normalisation import normalise_text_with_report
from diw.core.qa import compose_source_cited_answer, validate_citations
from diw.core.retrieval import retrieval_results_as_dicts, retrieve_chunks
from diw.db.repository import (
    count_chunks,
    count_documents,
    count_embeddings,
    count_versions,
    embed_missing_chunks,
    list_ai_suggestions,
    record_review_decision,
    save_ai_run,
    save_ai_suggestion,
    save_ingested_document,
)
from diw.db.schema import create_schema
from diw.db.session import build_engine
from sqlalchemy.orm import Session


def _print_report(report) -> None:
    print("NormalisationReport:")
    print(f"  original_line_count: {report.original_line_count}")
    print(f"  normalised_line_count: {report.normalised_line_count}")
    print(f"  trailing_whitespace_lines: {report.trailing_whitespace_lines}")
    print(f"  collapsed_blank_lines: {report.collapsed_blank_lines}")


def normalise_command(args: argparse.Namespace) -> int:
    path = Path(args.input)
    result = normalise_text_with_report(path.read_text(encoding="utf-8"))

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(result.text, encoding="utf-8")
        print(f"saved: {out}")
    else:
        print(result.text)

    if args.report:
        if not args.out:
            print()
        _print_report(result.report)

    return 0


def chunk_command(args: argparse.Namespace) -> int:
    document = ingest_file(
        Path(args.input),
        target_chars=args.target_chars,
        overlap_chars=args.overlap_chars,
    )

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(document.to_json(), encoding="utf-8")
        print(f"saved: {out}")
        print(f"chunks: {len(document.chunks)}")
        print(f"document_id: {document.document_id}")
        print(f"version_id: {document.version_id}")
    else:
        print(document.to_json())

    return 0


def init_db_command(args: argparse.Namespace) -> int:
    engine = build_engine(args.database_url)
    try:
        create_schema(engine)
    finally:
        engine.dispose()
    print("database schema ready")
    return 0


def load_command(args: argparse.Namespace) -> int:
    engine = build_engine(args.database_url)
    try:
        create_schema(engine)
        document = ingest_file(
            Path(args.input),
            target_chars=args.target_chars,
            overlap_chars=args.overlap_chars,
        )

        with Session(engine, autoflush=False, expire_on_commit=False) as session:
            save_ingested_document(session, document)
            session.commit()
            documents = count_documents(session)
            versions = count_versions(session)
            chunks = count_chunks(session)
    finally:
        engine.dispose()

    print(f"loaded: {document.source_path}")
    print(f"document_id: {document.document_id}")
    print(f"version_id: {document.version_id}")
    print(f"chunks_loaded: {len(document.chunks)}")
    print(f"database_totals: documents={documents} versions={versions} chunks={chunks}")
    return 0


def embed_command(args: argparse.Namespace) -> int:
    provider = LocalHashingEmbeddingProvider(dimensions=args.dimensions)
    engine = build_engine(args.database_url)
    try:
        create_schema(engine)
        with Session(engine, autoflush=False, expire_on_commit=False) as session:
            created = embed_missing_chunks(session, provider)
            session.commit()
            total = count_embeddings(session)
    finally:
        engine.dispose()

    print(f"embedding_model: {provider.model_name}")
    print(f"dimensions: {provider.dimensions}")
    print(f"embeddings_created: {created}")
    print(f"embeddings_total: {total}")
    return 0


def retrieve_command(args: argparse.Namespace) -> int:
    provider = LocalHashingEmbeddingProvider(dimensions=args.dimensions)
    engine = build_engine(args.database_url)
    try:
        create_schema(engine)
        with Session(engine, autoflush=False, expire_on_commit=False) as session:
            results = retrieve_chunks(
                session,
                args.query,
                provider,
                top_k=args.top_k,
                mode=args.mode,
            )
    finally:
        engine.dispose()

    payload = {
        "query": args.query,
        "mode": args.mode,
        "embedding_model": provider.model_name,
        "results": retrieval_results_as_dicts(results),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def answer_command(args: argparse.Namespace) -> int:
    provider = LocalHashingEmbeddingProvider(dimensions=args.dimensions)
    engine = build_engine(args.database_url)
    try:
        create_schema(engine)
        with Session(engine, autoflush=False, expire_on_commit=False) as session:
            if args.ensure_embeddings:
                embed_missing_chunks(session, provider)
                session.commit()

            results = retrieve_chunks(
                session,
                args.query,
                provider,
                top_k=args.top_k,
                mode=args.mode,
            )
    finally:
        engine.dispose()

    answer = compose_source_cited_answer(
        args.query,
        results,
        min_score=args.min_score,
        max_citations=args.max_citations,
    )
    validation = validate_citations(answer, results)
    payload = {
        "query": args.query,
        "mode": args.mode,
        "embedding_model": provider.model_name,
        "answer": answer.model_dump(),
        "citation_validation": validation.model_dump(),
        "retrieved_chunks": retrieval_results_as_dicts(results),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def _build_llm_provider(args: argparse.Namespace):
    if args.llm_provider == "deterministic":
        return DeterministicStructuredProvider()
    if args.llm_provider == "openai":
        return OpenAIChatProvider(model=args.llm_model)
    raise ValueError(f"unsupported LLM provider: {args.llm_provider}")


def _isoformat(value) -> str | None:
    if value is None:
        return None
    return value.isoformat()


def _suggestion_as_dict(suggestion) -> dict:
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


def _retrieve_for_query(args: argparse.Namespace, query: str):
    provider = LocalHashingEmbeddingProvider(dimensions=args.dimensions)
    engine = build_engine(args.database_url)
    try:
        create_schema(engine)
        with Session(engine, autoflush=False, expire_on_commit=False) as session:
            if args.ensure_embeddings:
                embed_missing_chunks(session, provider)
                session.commit()

            results = retrieve_chunks(
                session,
                query,
                provider,
                top_k=args.top_k,
                mode=args.mode,
            )
    finally:
        engine.dispose()
    return provider, results


def answer_llm_command(args: argparse.Namespace) -> int:
    embedding_provider, results = _retrieve_for_query(args, args.query)
    llm_provider = _build_llm_provider(args)
    answer = generate_structured_answer(args.query, results, llm_provider)
    validation = validate_citations(answer, results)
    payload = {
        "query": args.query,
        "mode": args.mode,
        "embedding_model": embedding_provider.model_name,
        "llm_provider": llm_provider.provider,
        "llm_model": llm_provider.model,
        "answer": answer.model_dump(),
        "citation_validation": validation.model_dump(),
        "retrieved_chunks": retrieval_results_as_dicts(results),
    }
    if args.log_run or args.create_suggestion:
        engine = build_engine(args.database_url)
        try:
            create_schema(engine)
            with Session(engine, autoflush=False, expire_on_commit=False) as session:
                run = None
                if args.log_run:
                    run = save_ai_run(
                        session,
                        run_type="answer_llm",
                        query=args.query,
                        retrieval_mode=args.mode,
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
                if args.create_suggestion:
                    suggestion = save_ai_suggestion(
                        session,
                        suggestion_type="source_cited_answer",
                        title=args.query,
                        ai_run_id=run.id if run is not None else None,
                        payload=payload,
                    )
                    payload["suggestion_id"] = suggestion.id
                session.commit()
        finally:
            engine.dispose()
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def review_list_command(args: argparse.Namespace) -> int:
    engine = build_engine(args.database_url)
    try:
        create_schema(engine)
        with Session(engine, autoflush=False, expire_on_commit=False) as session:
            suggestions = list_ai_suggestions(session, status=args.status)
    finally:
        engine.dispose()
    payload = {
        "status": args.status,
        "suggestions": [_suggestion_as_dict(suggestion) for suggestion in suggestions],
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def review_decide_command(args: argparse.Namespace) -> int:
    edited_payload = None
    if args.edited_json:
        edited_payload = json.loads(Path(args.edited_json).read_text(encoding="utf-8"))

    engine = build_engine(args.database_url)
    try:
        create_schema(engine)
        with Session(engine, autoflush=False, expire_on_commit=False) as session:
            review = record_review_decision(
                session,
                suggestion_id=args.suggestion_id,
                decision=args.decision,
                reviewer=args.reviewer,
                note=args.note,
                edited_payload=edited_payload,
            )
            session.commit()
            payload = {
                "suggestion_id": review.suggestion_id,
                "decision": review.decision,
                "reviewer": review.reviewer,
                "note": review.note,
                "created_at": _isoformat(review.created_at),
            }
    finally:
        engine.dispose()
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def eval_command(args: argparse.Namespace) -> int:
    llm_provider = _build_llm_provider(args)
    cases = load_golden_cases(Path(args.cases))
    case_results = []
    for case in cases:
        _, results = _retrieve_for_query(args, case.question)
        case_results.append(score_case(case, results, llm_provider))

    payload = {
        "llm_provider": llm_provider.provider,
        "llm_model": llm_provider.model,
        "summary": summarise_eval(case_results),
        "cases": case_results,
    }
    if args.log_run:
        engine = build_engine(args.database_url)
        try:
            create_schema(engine)
            with Session(engine, autoflush=False, expire_on_commit=False) as session:
                run = save_ai_run(
                    session,
                    run_type="eval",
                    retrieval_mode=args.mode,
                    llm_provider=llm_provider.provider,
                    llm_model=llm_provider.model,
                    output=payload,
                    metrics=payload["summary"],
                )
                session.commit()
                payload["ai_run_id"] = run.id
        finally:
            engine.dispose()
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"saved: {out}")
        return 0
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def eval_report_command(args: argparse.Namespace) -> int:
    report_path = Path(args.report)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    markdown = render_markdown_report(report)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(markdown, encoding="utf-8")
        print(f"saved: {out}")
    else:
        print(markdown)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="diw",
        description="Document Intelligence Workspace command-line tools.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    normalise = subparsers.add_parser(
        "normalise",
        help="Normalise Markdown/plain text without changing document meaning.",
    )
    normalise.add_argument("input", help="Input Markdown/plain-text file.")
    normalise.add_argument("--out", help="Optional output file for normalised text.")
    normalise.add_argument("--report", action="store_true", help="Print normalisation report.")
    normalise.set_defaults(func=normalise_command)

    chunk = subparsers.add_parser(
        "chunk",
        help="Normalise and heading-aware chunk a Markdown/plain-text file.",
    )
    chunk.add_argument("input", help="Input Markdown/plain-text file.")
    chunk.add_argument("--out", help="Optional output file for chunk JSON.")
    chunk.add_argument("--target-chars", type=int, default=1200, help="Target chunk size.")
    chunk.add_argument("--overlap-chars", type=int, default=160, help="Character overlap.")
    chunk.set_defaults(func=chunk_command)

    init_db = subparsers.add_parser(
        "init-db",
        help="Create local database tables for source documents, versions, and chunks.",
    )
    init_db.add_argument("--database-url", help="Database URL. Defaults to DATABASE_URL or local SQLite.")
    init_db.set_defaults(func=init_db_command)

    load = subparsers.add_parser(
        "load",
        help="Ingest, chunk, and persist a Markdown/plain-text file.",
    )
    load.add_argument("input", help="Input Markdown/plain-text file.")
    load.add_argument("--database-url", help="Database URL. Defaults to DATABASE_URL or local SQLite.")
    load.add_argument("--target-chars", type=int, default=1200, help="Target chunk size.")
    load.add_argument("--overlap-chars", type=int, default=160, help="Character overlap.")
    load.set_defaults(func=load_command)

    embed = subparsers.add_parser(
        "embed",
        help="Create deterministic local embeddings for stored chunks.",
    )
    embed.add_argument("--database-url", help="Database URL. Defaults to DATABASE_URL or local SQLite.")
    embed.add_argument("--dimensions", type=int, default=64, help="Embedding dimensions.")
    embed.set_defaults(func=embed_command)

    retrieve = subparsers.add_parser(
        "retrieve",
        help="Retrieve stored chunks using lexical, vector, or hybrid scoring.",
    )
    retrieve.add_argument("query", help="Search query.")
    retrieve.add_argument("--database-url", help="Database URL. Defaults to DATABASE_URL or local SQLite.")
    retrieve.add_argument("--mode", choices=["lexical", "vector", "hybrid"], default="hybrid")
    retrieve.add_argument("--top-k", type=int, default=5, help="Number of chunks to return.")
    retrieve.add_argument("--dimensions", type=int, default=64, help="Embedding dimensions.")
    retrieve.set_defaults(func=retrieve_command)

    answer = subparsers.add_parser(
        "answer",
        help="Retrieve evidence and produce a source-cited answer with citation validation.",
    )
    answer.add_argument("query", help="Question to answer from stored chunks.")
    answer.add_argument("--database-url", help="Database URL. Defaults to DATABASE_URL or local SQLite.")
    answer.add_argument("--mode", choices=["lexical", "vector", "hybrid"], default="hybrid")
    answer.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve.")
    answer.add_argument("--dimensions", type=int, default=64, help="Embedding dimensions.")
    answer.add_argument(
        "--min-score",
        type=float,
        default=0.12,
        help="Minimum retrieval score required before answering.",
    )
    answer.add_argument("--max-citations", type=int, default=3, help="Maximum citations to use.")
    answer.add_argument(
        "--no-ensure-embeddings",
        action="store_false",
        dest="ensure_embeddings",
        help="Do not create missing local embeddings before retrieval.",
    )
    answer.set_defaults(func=answer_command, ensure_embeddings=True)

    answer_llm = subparsers.add_parser(
        "answer-llm",
        help="Retrieve evidence and generate a structured source-cited answer with an LLM provider.",
    )
    answer_llm.add_argument("query", help="Question to answer from stored chunks.")
    answer_llm.add_argument("--database-url", help="Database URL. Defaults to DATABASE_URL or local SQLite.")
    answer_llm.add_argument("--mode", choices=["lexical", "vector", "hybrid"], default="hybrid")
    answer_llm.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve.")
    answer_llm.add_argument("--dimensions", type=int, default=64, help="Embedding dimensions.")
    answer_llm.add_argument(
        "--llm-provider",
        choices=["deterministic", "openai"],
        default="deterministic",
        help="Structured answer provider.",
    )
    answer_llm.add_argument("--llm-model", default="gpt-4.1-mini", help="OpenAI model name.")
    answer_llm.add_argument(
        "--no-ensure-embeddings",
        action="store_false",
        dest="ensure_embeddings",
        help="Do not create missing local embeddings before retrieval.",
    )
    answer_llm.add_argument(
        "--no-log-run",
        action="store_false",
        dest="log_run",
        help="Do not persist an AI run record.",
    )
    answer_llm.add_argument(
        "--no-create-suggestion",
        action="store_false",
        dest="create_suggestion",
        help="Do not add the generated answer to the human review queue.",
    )
    answer_llm.set_defaults(
        func=answer_llm_command,
        ensure_embeddings=True,
        log_run=True,
        create_suggestion=True,
    )

    eval_parser = subparsers.add_parser(
        "eval",
        help="Run golden-case evaluation for structured QA and citation validation.",
    )
    eval_parser.add_argument(
        "--cases",
        default="data/demo/evals/golden_cases.jsonl",
        help="JSONL file containing golden evaluation cases.",
    )
    eval_parser.add_argument("--database-url", help="Database URL. Defaults to DATABASE_URL or local SQLite.")
    eval_parser.add_argument("--mode", choices=["lexical", "vector", "hybrid"], default="hybrid")
    eval_parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve.")
    eval_parser.add_argument("--dimensions", type=int, default=64, help="Embedding dimensions.")
    eval_parser.add_argument(
        "--llm-provider",
        choices=["deterministic", "openai"],
        default="deterministic",
        help="Structured answer provider.",
    )
    eval_parser.add_argument("--llm-model", default="gpt-4.1-mini", help="OpenAI model name.")
    eval_parser.add_argument("--out", help="Optional path to write the JSON evaluation report.")
    eval_parser.add_argument(
        "--no-ensure-embeddings",
        action="store_false",
        dest="ensure_embeddings",
        help="Do not create missing local embeddings before retrieval.",
    )
    eval_parser.add_argument(
        "--no-log-run",
        action="store_false",
        dest="log_run",
        help="Do not persist an AI run record.",
    )
    eval_parser.set_defaults(func=eval_command, ensure_embeddings=True, log_run=True)

    eval_report = subparsers.add_parser(
        "eval-report",
        help="Render a Markdown summary from a JSON evaluation report.",
    )
    eval_report.add_argument("report", help="JSON evaluation report path.")
    eval_report.add_argument("--out", help="Optional Markdown output path.")
    eval_report.set_defaults(func=eval_report_command)

    review_list = subparsers.add_parser(
        "review-list",
        help="List AI suggestions awaiting or already given human review.",
    )
    review_list.add_argument("--database-url", help="Database URL. Defaults to DATABASE_URL or local SQLite.")
    review_list.add_argument(
        "--status",
        choices=["pending", "accepted", "rejected", "edited"],
        help="Optional suggestion status filter.",
    )
    review_list.set_defaults(func=review_list_command)

    review_decide = subparsers.add_parser(
        "review-decide",
        help="Record a human review decision for an AI suggestion.",
    )
    review_decide.add_argument("suggestion_id", help="Suggestion ID from answer-llm or review-list.")
    review_decide.add_argument("--database-url", help="Database URL. Defaults to DATABASE_URL or local SQLite.")
    review_decide.add_argument("--decision", choices=["accept", "reject", "edit"], required=True)
    review_decide.add_argument("--reviewer", default="local-user", help="Reviewer name or identifier.")
    review_decide.add_argument("--note", help="Optional review note.")
    review_decide.add_argument("--edited-json", help="Optional JSON payload for edited suggestions.")
    review_decide.set_defaults(func=review_decide_command)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
