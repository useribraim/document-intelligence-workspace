from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys
from time import perf_counter

from dotenv import load_dotenv
from diw.core.embeddings import build_embedding_provider
from diw.core.claim_audit import (
    AUDIT_PROMPT_VERSION,
    ClaimCitationAssessment,
    apply_evidence_repair,
    apply_claim_verification_gate,
    assess_claims,
    cohen_kappa,
    config_hash,
    extract_atomic_claims,
    retrieval_gold_metrics,
    summarise_claim_audit,
)
from diw.core.evaluation import (
    load_golden_cases,
    render_markdown_report,
    score_case,
    summarise_eval,
)
from diw.core.ingestion import ingest_file
from diw.core.llm import (
    DeterministicStructuredProvider,
    OpenAIChatProvider,
    VertexAIGeminiProvider,
    estimate_openai_cost_usd,
    generate_structured_answer,
)
from diw.core.normalisation import normalise_text_with_report
from diw.core.qa import EvidenceCitation, SourceCitedAnswer, compose_source_cited_answer, validate_citations
from diw.core.retrieval import RetrievalResult, retrieval_results_as_dicts, retrieve_chunks
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
from sqlalchemy import select
from diw.db.models import SourceDocument


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


def _build_embedding_provider(args: argparse.Namespace):
    return build_embedding_provider(
        getattr(args, "embedding_provider", "local"),
        dimensions=args.dimensions,
        embedding_model=getattr(args, "embedding_model", None),
    )


def _add_embedding_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dimensions",
        type=int,
        default=None,
        help="Embedding dimensions. Defaults to 64 locally, 768 on Vertex, or the OpenAI model default.",
    )
    parser.add_argument(
        "--embedding-provider",
        choices=["local", "openai", "vertex"],
        default="local",
        help="Embedding provider. Cloud providers require their normal credentials.",
    )
    parser.add_argument(
        "--embedding-model",
        help="Provider embedding model name.",
    )


def _add_reranker_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--reranker",
        choices=["weighted", "rrf"],
        default="weighted",
        help="Hybrid fusion strategy. RRF is opt-in until benchmarked against the frozen baseline.",
    )


def embed_command(args: argparse.Namespace) -> int:
    provider = _build_embedding_provider(args)
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
    provider = _build_embedding_provider(args)
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
                reranker=args.reranker,
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
    provider = _build_embedding_provider(args)
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
                reranker=args.reranker,
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
        return OpenAIChatProvider(
            model=args.llm_model or "gpt-5-mini-2025-08-07",
            max_output_tokens=getattr(args, "max_output_tokens", 1200),
        )
    if args.llm_provider == "vertex":
        return VertexAIGeminiProvider(
            model=args.llm_model,
            max_output_tokens=getattr(args, "max_output_tokens", 1200),
        )
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
    provider = _build_embedding_provider(args)
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
                reranker=args.reranker,
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


def claim_audit_command(args: argparse.Namespace) -> int:
    """Run the reproducible measurement path; annotations are deliberately separate."""
    question_path = Path(args.questions)
    questions = [json.loads(line) for line in question_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    manifest = {
        item["document_id"]: item
        for item in (
            json.loads(line)
            for line in Path(args.manifest).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    embedding_provider = _build_embedding_provider(args)
    llm_provider = _build_llm_provider(args)
    run_id = args.run_id or f"{args.mode}-deterministic-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    config = {
        "retrieval_mode": args.mode,
        "reranker": args.reranker,
        "top_k": args.top_k,
        "embedding_model": embedding_provider.model_name,
        "generator_provider": llm_provider.provider,
        "generator_model": llm_provider.model,
        "max_output_tokens": args.max_output_tokens,
        "prompt_version": AUDIT_PROMPT_VERSION,
        "verification_gate": args.verification_gate,
        "verification_gate_policy": args.verification_gate_policy if args.verification_gate else None,
        "evidence_repair": args.evidence_repair,
        "require_gold_evidence": args.require_gold_evidence,
    }
    out = Path(args.out)
    checkpoint = out.with_name(f"{out.stem}.partial.json")
    records: list[dict] = []
    if args.resume and checkpoint.is_file():
        saved = json.loads(checkpoint.read_text(encoding="utf-8"))
        if saved.get("run_id") != run_id or saved.get("config") != config:
            raise ValueError(f"checkpoint does not match this run: {checkpoint}")
        records = saved.get("runs", [])
    completed_question_ids = {record["question_id"] for record in records}
    engine = build_engine(args.database_url)
    try:
        create_schema(engine)
        with Session(engine, autoflush=False, expire_on_commit=False) as session:
            path_to_database_id = {
                document.source_path: document.id
                for document in session.scalars(select(SourceDocument)).all()
            }
            if args.ensure_embeddings:
                embed_missing_chunks(session, embedding_provider)
                session.commit()
            for question in questions:
                if question["question_id"] in completed_question_ids:
                    continue
                started = datetime.now(timezone.utc)
                requested_documents = question.get("source_documents", [])
                missing_documents = [item for item in requested_documents if item not in manifest]
                if missing_documents:
                    raise ValueError(
                        f"question {question['question_id']} references unknown manifest IDs: "
                        f"{', '.join(missing_documents)}"
                    )
                document_ids = {
                    path_to_database_id[manifest[item]["text_path"]]
                    for item in requested_documents
                    if manifest[item]["text_path"] in path_to_database_id
                }
                if requested_documents and len(document_ids) != len(requested_documents):
                    raise ValueError(
                        f"ingest all manifest texts before auditing question "
                        f"{question['question_id']}"
                    )
                gold_chunk_ids = list(question.get("gold_evidence_chunk_ids", []))
                answerable = question.get("expected_evidence_status") != "insufficient"
                if args.require_gold_evidence and answerable and not gold_chunk_ids:
                    raise ValueError(
                        f"question {question['question_id']} is answerable but has no gold evidence chunks"
                    )
                results = retrieve_chunks(
                    session,
                    question["question"],
                    embedding_provider,
                    top_k=args.top_k,
                    mode=args.mode,
                    reranker=args.reranker,
                    document_ids=document_ids or None,
                )
                try:
                    answer = generate_structured_answer(question["question"], results, llm_provider)
                except ValueError as error:
                    raise ValueError(f"question {question['question_id']}: {error}") from error
                claims = extract_atomic_claims(answer, question_id=question["question_id"])
                assessments = assess_claims(
                    claims,
                    answer.citations,
                    results,
                    expected_evidence_status=question.get("expected_evidence_status"),
                )
                draft_answer = answer.answer
                repair_actions: list[dict[str, str]] = []
                if args.verification_gate:
                    answer = apply_claim_verification_gate(
                        answer,
                        claims,
                        assessments,
                        policy=args.verification_gate_policy,
                    )
                    claims = extract_atomic_claims(answer, question_id=question["question_id"])
                    assessments = assess_claims(
                        claims,
                        answer.citations,
                        results,
                        expected_evidence_status=question.get("expected_evidence_status"),
                    )
                if args.evidence_repair:
                    answer, repair_actions = apply_evidence_repair(answer, claims, assessments)
                    claims = extract_atomic_claims(answer, question_id=question["question_id"])
                    assessments = assess_claims(
                        claims,
                        answer.citations,
                        results,
                        expected_evidence_status=question.get("expected_evidence_status"),
                    )
                retrieval_recall_at_k, retrieval_mrr = retrieval_gold_metrics(results, gold_chunk_ids)
                expected_min_claim_count = _expected_min_claim_count(question)
                cited_gold_chunk_ids = sorted(
                    {citation.chunk_id for citation in answer.citations} & set(gold_chunk_ids)
                )
                gold_citation_recall = (
                    round(len(cited_gold_chunk_ids) / len(set(gold_chunk_ids)), 4)
                    if gold_chunk_ids
                    else None
                )
                structural_complete = _is_structurally_complete(
                    expected_evidence_status=question.get("expected_evidence_status"),
                    insufficient_evidence=answer.insufficient_evidence,
                    claim_count=len(claims),
                    expected_min_claim_count=expected_min_claim_count,
                )
                records.append(
                    {
                        "run_id": run_id,
                        "question_id": question["question_id"],
                        "expected_evidence_status": question.get("expected_evidence_status"),
                        "gold_evidence_chunk_ids": gold_chunk_ids,
                        "retrieval_recall_at_k": retrieval_recall_at_k,
                        "retrieval_mrr": retrieval_mrr,
                        "cited_gold_chunk_ids": cited_gold_chunk_ids,
                        "gold_citation_recall": gold_citation_recall,
                        "expected_min_claim_count": expected_min_claim_count,
                        "structural_complete": structural_complete,
                        "retrieval_config_hash": config_hash(config),
                        "prompt_hash": hashlib.sha256(AUDIT_PROMPT_VERSION.encode()).hexdigest(),
                        "model_id": llm_provider.model,
                        "model_version": answer.prompt_version,
                        "temperature": getattr(llm_provider, "temperature", None),
                        "retrieved_passage_ids": [item.chunk_id for item in results],
                        "retrieved_passages": [
                            {
                                "rank": rank,
                                **item,
                                "text_sha256": hashlib.sha256(item["text"].encode("utf-8")).hexdigest(),
                            }
                            for rank, item in enumerate(retrieval_results_as_dicts(results), start=1)
                        ],
                        "draft_answer": draft_answer if (args.verification_gate or args.evidence_repair) else None,
                        "evidence_repair_actions": repair_actions,
                        "answer": answer.answer,
                        "answer_sha256": hashlib.sha256(answer.answer.encode("utf-8")).hexdigest(),
                        "insufficient_evidence": answer.insufficient_evidence,
                        "latency_ms": int((datetime.now(timezone.utc) - started).total_seconds() * 1000),
                        "input_tokens": answer.input_tokens,
                        "cached_input_tokens": answer.cached_input_tokens,
                        "output_tokens": answer.output_tokens,
                        "completion_attempts": answer.completion_attempts,
                        "estimated_cost_usd": estimate_openai_cost_usd(
                            model=llm_provider.model,
                            input_tokens=answer.input_tokens,
                            cached_input_tokens=answer.cached_input_tokens,
                            output_tokens=answer.output_tokens,
                        ) if llm_provider.provider == "openai" else 0.0,
                        "claims": [claim.model_dump() for claim in claims],
                        "annotations_pending": [assessment.model_dump() for assessment in assessments],
                    }
                )
                _write_claim_audit_checkpoint(
                    checkpoint,
                    run_id=run_id,
                    config=config,
                    records=records,
                )
    finally:
        engine.dispose()
    summary = _summarise_claim_audit_records(records)
    payload = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "question_count": len(records),
        "summary": summary,
        "runs": records,
        "note": "Deterministic labels are automated intervention diagnostics, not human ground truth.",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    checkpoint.unlink(missing_ok=True)
    print(f"saved: {out}")
    print(json.dumps(summary, indent=2))
    return 0


def _summarise_claim_audit_records(records: list[dict]) -> dict[str, float | int | None]:
    all_assessments = [assessment for record in records for assessment in record["annotations_pending"]]
    summary = summarise_claim_audit(
        [ClaimCitationAssessment.model_validate(item) for item in all_assessments]
    )
    refusal_expected = [
        record for record in records if record["expected_evidence_status"] == "insufficient"
    ]
    refusals = [record for record in records if record["insufficient_evidence"]]
    appropriate_refusals = [record for record in refusal_expected if record["insufficient_evidence"]]
    summary.update(
        {
            "answer_claim_count": sum(len(record["claims"]) for record in records),
            "refusal_rate": round(len(refusals) / len(records), 4) if records else 0.0,
            "appropriate_refusal_recall": (
                round(len(appropriate_refusals) / len(refusal_expected), 4)
                if refusal_expected
                else 0.0
            ),
            "refusal_precision": (
                round(len(appropriate_refusals) / len(refusals), 4) if refusals else 0.0
            ),
            "mean_latency_ms": round(
                sum(record["latency_ms"] for record in records) / len(records), 2
            ) if records else 0.0,
            "input_tokens": sum(record["input_tokens"] for record in records),
            "cached_input_tokens": sum(record["cached_input_tokens"] for record in records),
            "output_tokens": sum(record["output_tokens"] for record in records),
            "estimated_cost_usd": round(
                sum(record["estimated_cost_usd"] or 0.0 for record in records), 8
            ),
            "retrieval_eval_cases": sum(
                record["retrieval_recall_at_k"] is not None for record in records
            ),
            "retrieval_recall_at_k": _average_optional(records, "retrieval_recall_at_k"),
            "retrieval_mrr": _average_optional(records, "retrieval_mrr"),
            "gold_citation_recall": _average_optional(records, "gold_citation_recall"),
            "structural_completeness_rate": round(
                sum(record["structural_complete"] for record in records) / len(records), 4
            ) if records else 0.0,
        }
    )
    return summary


def evidence_repair_run_command(args: argparse.Namespace) -> int:
    """Repair a saved baseline in place logically, without a second model call.

    This preserves the original generated answers and retrieved passages, making
    the repair comparison causal rather than a second stochastic generation run.
    """
    baseline = json.loads(Path(args.run).read_text(encoding="utf-8"))
    if baseline.get("config", {}).get("evidence_repair"):
        raise ValueError("input run already has evidence repair enabled")
    run_id = args.run_id or f"{baseline['run_id']}-repair"
    config = {
        **baseline.get("config", {}),
        "evidence_repair": True,
        "repair_parent_run_id": baseline["run_id"],
        "repair_algorithm": "exact-span-v1",
    }
    records: list[dict] = []
    for baseline_record in baseline["runs"]:
        repair_started = perf_counter()
        results = [
            RetrievalResult(
                chunk_id=item["chunk_id"], document_id=item["document_id"],
                version_id=item["version_id"], chunk_index=item["chunk_index"],
                heading_path=item["heading_path"], text=item["text"],
                lexical_score=item["lexical_score"], vector_score=item["vector_score"],
                score=item["score"],
            )
            for item in baseline_record["retrieved_passages"]
        ]
        citations: dict[str, EvidenceCitation] = {}
        for assessment in baseline_record["annotations_pending"]:
            label = assessment.get("citation_id")
            if not label or label in citations or not label[1:].isdigit():
                continue
            rank = int(label[1:])
            if not 1 <= rank <= len(results):
                continue
            source = results[rank - 1]
            citations[label] = EvidenceCitation(
                label=label, chunk_id=source.chunk_id, document_id=source.document_id,
                version_id=source.version_id, heading_path=source.heading_path,
                quote=assessment.get("evidence_span") or "", score=source.score,
                quote_alignment=assessment.get("quote_alignment") or "exact",
            )
        answer = SourceCitedAnswer(
            query=baseline_record["question_id"], answer=baseline_record["answer"],
            insufficient_evidence=baseline_record["insufficient_evidence"],
            citations=list(citations.values()), model=baseline_record.get("model_id"),
            provider=baseline.get("config", {}).get("generator_provider"),
            prompt_version=baseline_record.get("model_version"),
            input_tokens=baseline_record.get("input_tokens", 0),
            cached_input_tokens=baseline_record.get("cached_input_tokens", 0),
            output_tokens=baseline_record.get("output_tokens", 0),
            completion_attempts=baseline_record.get("completion_attempts", 0),
        )
        claims = extract_atomic_claims(answer, question_id=baseline_record["question_id"])
        assessments = [
            ClaimCitationAssessment.model_validate(item)
            for item in baseline_record["annotations_pending"]
        ]
        repaired, actions = apply_evidence_repair(answer, claims, assessments)
        repaired_claims = extract_atomic_claims(repaired, question_id=baseline_record["question_id"])
        repaired_assessments = assess_claims(
            repaired_claims, repaired.citations, results,
            expected_evidence_status=baseline_record.get("expected_evidence_status"),
        )
        gold_chunk_ids = baseline_record.get("gold_evidence_chunk_ids", [])
        cited_gold_chunk_ids = sorted(
            {citation.chunk_id for citation in repaired.citations} & set(gold_chunk_ids)
        )
        record = dict(baseline_record)
        record.update(
            {
                "run_id": run_id,
                "retrieval_config_hash": config_hash(config),
                "draft_answer": baseline_record["answer"],
                "evidence_repair_actions": actions,
                "answer": repaired.answer,
                "answer_sha256": hashlib.sha256(repaired.answer.encode("utf-8")).hexdigest(),
                "insufficient_evidence": repaired.insufficient_evidence,
                "cited_gold_chunk_ids": cited_gold_chunk_ids,
                "gold_citation_recall": (
                    round(len(cited_gold_chunk_ids) / len(set(gold_chunk_ids)), 4)
                    if gold_chunk_ids else None
                ),
                "structural_complete": _is_structurally_complete(
                    expected_evidence_status=baseline_record.get("expected_evidence_status"),
                    insufficient_evidence=repaired.insufficient_evidence,
                    claim_count=len(repaired_claims),
                    expected_min_claim_count=baseline_record.get("expected_min_claim_count", 1),
                ),
                "repair_latency_ms": round((perf_counter() - repair_started) * 1000, 3),
                "repair_incremental_cost_usd": 0.0,
                "claims": [claim.model_dump() for claim in repaired_claims],
                "annotations_pending": [item.model_dump() for item in repaired_assessments],
            }
        )
        records.append(record)
    summary = _summarise_claim_audit_records(records)
    summary["mean_repair_latency_ms"] = round(
        sum(record["repair_latency_ms"] for record in records) / len(records), 3
    ) if records else 0.0
    summary["repair_incremental_cost_usd"] = 0.0
    payload = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "parent_run_id": baseline["run_id"],
        "config": config,
        "question_count": len(records),
        "summary": summary,
        "runs": records,
        "note": "Offline deterministic evidence repair; labels remain automated diagnostics, not human ground truth.",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"saved: {out}")
    print(json.dumps(summary, indent=2))
    return 0


def _average_optional(records: list[dict], key: str) -> float | None:
    values = [float(record[key]) for record in records if record.get(key) is not None]
    return round(sum(values) / len(values), 4) if values else None


def _expected_min_claim_count(question: dict) -> int:
    """Return a preregistered, category-level completeness threshold.

    This is a structural proxy only: it prevents a one-sentence response from
    passing a synthesis or multi-claim question. Semantic completeness remains
    a human-review field in the annotation packet.
    """
    if question.get("expected_evidence_status") == "insufficient":
        return 0
    return {
        "direct_extraction": 1,
        "synthesis": 2,
        "multi_claim": 3,
        "conflicting_evidence": 1,
    }.get(question.get("category"), 1)


def _is_structurally_complete(
    *,
    expected_evidence_status: str | None,
    insufficient_evidence: bool,
    claim_count: int,
    expected_min_claim_count: int,
) -> bool:
    """Measure preregistered response structure without pretending it is a human label."""
    if expected_evidence_status == "insufficient":
        return insufficient_evidence
    return not insufficient_evidence and claim_count >= expected_min_claim_count


def _write_claim_audit_checkpoint(
    checkpoint: Path,
    *,
    run_id: str,
    config: dict,
    records: list[dict],
) -> None:
    """Persist completed questions so a transient provider failure can be resumed safely."""
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "config": config,
                "question_count": len(records),
                "status": "in_progress",
                "runs": records,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def annotation_template_command(args: argparse.Namespace) -> int:
    """Create human-owned annotation records from a baseline artifact."""
    run = json.loads(Path(args.run).read_text(encoding="utf-8"))
    questions = {
        item["question_id"]: item
        for item in (
            json.loads(line)
            for line in Path(args.questions).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    lines = []
    for question_run in run["runs"]:
        question = questions.get(question_run["question_id"])
        if question is None:
            raise ValueError(f"question packet is missing {question_run['question_id']}")
        lines.append(
            {
                "review_type": "answer_level",
                "run_id": run["run_id"],
                "question_id": question_run["question_id"],
                "question": question["question"],
                "claim_id": None,
                "claim_text": None,
                "citation_id": None,
                "evidence_span": None,
                "quote_alignment": None,
                "automation_prefill": {
                    "expected_evidence_status": question_run["expected_evidence_status"],
                    "answer": question_run["answer"],
                    "insufficient_evidence": question_run["insufficient_evidence"],
                },
                "source_exists": None,
                "citation_relevant": None,
                "support_label": None,
                "support_rationale": None,
                "answer_completeness": None,
                "refusal_appropriate": None,
                "rubric_version": args.rubric_version,
                "annotator_id": args.annotator_id,
                "annotation_status": "pending_human_confirmation",
            }
        )
        for prefill in question_run["annotations_pending"]:
            lines.append(
                {
                    "review_type": "claim_citation",
                    "run_id": run["run_id"],
                    "question_id": question_run["question_id"],
                    "question": question["question"],
                    "claim_id": prefill["claim_id"],
                    "claim_text": prefill["claim_text"],
                    "citation_id": prefill["citation_id"],
                    "evidence_span": prefill["evidence_span"],
                    "quote_alignment": prefill.get("quote_alignment"),
                    "automation_prefill": prefill,
                    "source_exists": None,
                    "citation_relevant": None,
                    "support_label": None,
                    "support_rationale": None,
                    "failure_mode": None,
                    "answer_completeness": None,
                    "refusal_appropriate": None,
                    "rubric_version": args.rubric_version,
                    "annotator_id": args.annotator_id,
                    "annotation_status": "pending_human_confirmation",
                }
            )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "".join(json.dumps(line, ensure_ascii=False) + "\n" for line in lines),
        encoding="utf-8",
    )
    print(f"saved: {out}")
    print(f"pending_pairs: {len(lines)}")
    return 0


def annotation_blind_sample_command(args: argparse.Namespace) -> int:
    """Create a label-blind answer/claim review packet for a reliability check."""
    run = json.loads(Path(args.run).read_text(encoding="utf-8"))
    questions = {
        item["question_id"]: item
        for item in (
            json.loads(line)
            for line in Path(args.questions).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    by_question_id = {item["question_id"]: item for item in run["runs"]}
    if args.question_ids:
        selected_question_ids = [item.strip() for item in args.question_ids.split(",") if item.strip()]
    else:
        if args.question_count > len(by_question_id):
            raise ValueError("question-count exceeds completed questions in the run")
        selected_question_ids = sorted(
            random.Random(args.seed).sample(sorted(by_question_id), args.question_count)
        )
    unknown = sorted(set(selected_question_ids) - set(by_question_id))
    if unknown:
        raise ValueError(f"unknown question IDs in run: {', '.join(unknown)}")

    lines = []
    for question_id in selected_question_ids:
        question_run = by_question_id[question_id]
        question = questions.get(question_id)
        if question is None:
            raise ValueError(f"question packet is missing {question_id}")
        lines.append(
            {
                "review_type": "answer_level",
                "source_run_id": run["run_id"],
                "question_id": question_id,
                "question": question["question"],
                "claim_id": None,
                "claim_text": None,
                "citation_id": None,
                "evidence_span": None,
                "quote_alignment": None,
                "review_context": {
                    "answer": question_run["answer"],
                    "insufficient_evidence": question_run["insufficient_evidence"],
                },
                "source_exists": None,
                "citation_relevant": None,
                "support_label": None,
                "support_rationale": None,
                "failure_mode": None,
                "answer_completeness": None,
                "refusal_appropriate": None,
                "rubric_version": args.rubric_version,
                "annotator_id": args.annotator_id,
                "annotation_status": "pending_human_confirmation",
                "blinded": True,
            }
        )
        for prefill in question_run["annotations_pending"]:
            lines.append(
                {
                    "review_type": "claim_citation",
                    "source_run_id": run["run_id"],
                    "question_id": question_id,
                    "question": question["question"],
                    "claim_id": prefill["claim_id"],
                    "claim_text": prefill["claim_text"],
                    "citation_id": prefill["citation_id"],
                    "evidence_span": prefill["evidence_span"],
                    "quote_alignment": prefill.get("quote_alignment"),
                    "source_exists": None,
                    "citation_relevant": None,
                    "support_label": None,
                    "support_rationale": None,
                    "failure_mode": None,
                    "answer_completeness": None,
                    "refusal_appropriate": None,
                    "rubric_version": args.rubric_version,
                    "annotator_id": args.annotator_id,
                    "annotation_status": "pending_human_confirmation",
                    "blinded": True,
                }
            )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "".join(json.dumps(line, ensure_ascii=False) + "\n" for line in lines),
        encoding="utf-8",
    )
    print(f"saved: {out}")
    print(json.dumps({"selected_question_ids": selected_question_ids, "records": len(lines)}))
    return 0


def annotation_apply_decisions_command(args: argparse.Namespace) -> int:
    """Apply an auditable decision map to a review packet without overwriting it."""
    records = [
        json.loads(line)
        for line in Path(args.annotations).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    decisions = json.loads(Path(args.decisions).read_text(encoding="utf-8"))
    applied: set[tuple[str, str]] = set()
    for record in records:
        review_type = record["review_type"]
        key = record["question_id"] if review_type == "answer_level" else record["claim_id"]
        decision = decisions.get(review_type, {}).get(key)
        if decision is None:
            raise ValueError(f"missing {review_type} decision for {key}")
        record.update(decision)
        record["annotator_id"] = args.annotator_id
        record["annotation_method"] = args.annotation_method
        record["annotation_status"] = "completed"
        applied.add((review_type, key))
    expected = {
        (review_type, key)
        for review_type, values in decisions.items()
        for key in values
    }
    unused = expected - applied
    if unused:
        raise ValueError(f"decision map has unused keys: {sorted(unused)}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    print(f"saved: {out}")
    print(json.dumps({"completed_records": len(records), "annotation_method": args.annotation_method}))
    return 0


def annotation_prefill_reference_command(args: argparse.Namespace) -> int:
    """Materialize saved automated prefill as a separate, non-human reference packet."""
    records = [
        json.loads(line)
        for line in Path(args.annotations).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    materialized = []
    for record in records:
        item = dict(record)
        prefill = item.get("automation_prefill", {})
        if item["review_type"] == "claim_citation":
            for field in (
                "source_exists",
                "citation_relevant",
                "support_label",
                "support_rationale",
            ):
                item[field] = prefill.get(field)
        item["annotator_id"] = args.annotator_id
        item["annotation_method"] = args.annotation_method
        item["annotation_status"] = "automated_reference"
        materialized.append(item)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in materialized),
        encoding="utf-8",
    )
    print(f"saved: {out}")
    print(json.dumps({"records": len(materialized), "annotation_method": args.annotation_method}))
    return 0


def annotation_summary_command(args: argparse.Namespace) -> int:
    """Summarize completed annotation labels while preserving their stated method."""
    records = [
        json.loads(line)
        for line in Path(args.annotations).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    claims = [item for item in records if item["review_type"] == "claim_citation"]
    answers = [item for item in records if item["review_type"] == "answer_level"]
    labels = Counter(item["support_label"] for item in claims if item.get("support_label"))
    failure_modes = Counter(
        item["failure_mode"]
        for item in claims
        if item.get("support_label") not in {None, "fully_supported", "not_applicable"}
        and item.get("failure_mode")
    )
    refusal_labels = [item["refusal_appropriate"] for item in answers if item.get("refusal_appropriate") is not None]
    total_claims = len(claims)
    completed_records = sum(
        item.get("annotation_status") in {"completed", "completed_human"} for item in records
    )
    payload = {
        "annotation_methods": sorted({item.get("annotation_method") for item in records if item.get("annotation_method")} ),
        "records": len(records),
        "completed_records": completed_records,
        "pending_records": len(records) - completed_records,
        "answer_level_records": len(answers),
        "claim_citation_records": total_claims,
        "support_counts": dict(labels),
        "support_rates": {
            label: round(labels[label] / total_claims, 4) if total_claims else 0.0
            for label in ("fully_supported", "partially_supported", "unsupported", "contradicted", "not_applicable")
        },
        "failure_mode_counts": dict(failure_modes),
        "appropriate_refusal_rate": round(sum(refusal_labels) / len(refusal_labels), 4)
        if refusal_labels
        else None,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def corpus_verify_command(args: argparse.Namespace) -> int:
    """Verify corpus provenance and, when available, exact local text hashes."""
    failures = []
    missing = []
    records = [
        json.loads(line)
        for line in Path(args.manifest).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for record in records:
        required = {
            "document_id",
            "canonical_url",
            "license_url",
            "redistributed",
            "version_identifier",
            "text_path",
            "sha256",
        }
        absent_fields = sorted(required - set(record))
        if absent_fields:
            failures.append(
                {
                    "document_id": record.get("document_id"),
                    "error": "missing manifest fields",
                    "fields": absent_fields,
                }
            )
            continue
        if record["redistributed"] is not False:
            failures.append(
                {
                    "document_id": record["document_id"],
                    "error": "audit paper text must not be marked as redistributed",
                }
            )
            continue
        path = Path(record["text_path"])
        if not path.is_file():
            missing.append(record["document_id"])
            if not args.allow_missing:
                failures.append(
                    {
                        "document_id": record["document_id"],
                        "error": "local corpus text is missing",
                        "path": str(path),
                    }
                )
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != record["sha256"]:
            failures.append(
                {
                    "document_id": record["document_id"],
                    "error": "SHA-256 mismatch",
                    "expected": record["sha256"],
                    "actual": actual,
                }
            )
    payload = {
        "manifest_documents": len(records),
        "local_documents": len(records) - len(missing),
        "missing_documents": missing,
        "allow_missing": args.allow_missing,
        "valid": not failures,
        "failures": failures,
    }
    print(json.dumps(payload, indent=2))
    return 0 if not failures else 1


def evaluation_freeze_command(args: argparse.Namespace) -> int:
    """Hash the exact inputs for an annotation/evaluation pass.

    This is a content-addressed freeze record, not a filesystem lock. Later
    changes require a new manifest and an explicit versioned evaluation pass.
    """
    artifacts = []
    for supplied_path in args.artifact:
        path = Path(supplied_path)
        if not path.is_file():
            raise ValueError(f"cannot freeze missing artifact: {path}")
        artifacts.append(
            {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
            }
        )
    payload = {
        "freeze_version": args.freeze_version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "artifacts": artifacts,
        "rule": (
            "Do not overwrite frozen artifacts, labels, thresholds, prompts, or questions. "
            "Create a new versioned manifest and document the change instead."
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"saved: {out}")
    print(json.dumps({"freeze_version": args.freeze_version, "artifacts": len(artifacts)}))
    return 0


def annotation_agreement_command(args: argparse.Namespace) -> int:
    def load(path: str) -> dict[tuple[str, str, str | None], dict]:
        return {
            (item["question_id"], item["claim_id"], item.get("citation_id")): item
            for item in (
                json.loads(line)
                for line in Path(path).read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
            if item.get("review_type", "claim_citation") == "claim_citation"
        }

    first, second = load(args.first), load(args.second)
    shared = sorted(
        set(first) & set(second),
        key=lambda item: (item[0], item[1] or "", item[2] or ""),
    )
    first_labels = [first[key].get("support_label") for key in shared]
    second_labels = [second[key].get("support_label") for key in shared]
    valid = [
        (first_label, second_label)
        for first_label, second_label in zip(first_labels, second_labels)
        if first_label and second_label
    ]
    labels = sorted({label for pair in valid for label in pair})
    first_annotators = sorted(
        {
            first[key].get("annotator_id")
            for key in shared
            if first[key].get("support_label") and first[key].get("annotator_id")
        }
    )
    second_annotators = sorted(
        {
            second[key].get("annotator_id")
            for key in shared
            if second[key].get("support_label") and second[key].get("annotator_id")
        }
    )
    confusion_matrix = {
        first_label: {
            second_label: sum(
                observed_first == first_label and observed_second == second_label
                for observed_first, observed_second in valid
            )
            for second_label in labels
        }
        for first_label in labels
    }
    errors = []
    if args.require_complete and len(valid) != len(shared):
        errors.append(
            f"only {len(valid)} of {len(shared)} shared claim-citation pairs are labeled"
        )
    if len(valid) < args.minimum_pairs:
        errors.append(
            f"only {len(valid)} labeled pairs; minimum required is {args.minimum_pairs}"
        )
    if args.require_distinct_annotators and set(first_annotators) & set(second_annotators):
        errors.append("annotator IDs overlap between the first and second packets")

    payload = {
        "inputs": {
            "first": {
                "path": args.first,
                "sha256": hashlib.sha256(Path(args.first).read_bytes()).hexdigest(),
            },
            "second": {
                "path": args.second,
                "sha256": hashlib.sha256(Path(args.second).read_bytes()).hexdigest(),
            },
        },
        "shared_pairs": len(shared),
        "labeled_pairs": len(valid),
        "complete": len(valid) == len(shared) and bool(shared),
        "minimum_pairs": args.minimum_pairs,
        "first_annotators": first_annotators,
        "second_annotators": second_annotators,
        "raw_agreement": round(sum(a == b for a, b in valid) / len(valid), 4) if valid else None,
        "cohen_kappa": cohen_kappa([a for a, _ in valid], [b for _, b in valid]),
        "confusion_matrix": confusion_matrix,
        "disagreements": [
            {
                "question_id": key[0],
                "claim_id": key[1],
                "citation_id": key[2],
                "first_label": first[key].get("support_label"),
                "second_label": second[key].get("support_label"),
                "first_rationale": first[key].get("support_rationale"),
                "second_rationale": second[key].get("support_rationale"),
            }
            for key in shared
            if first[key].get("support_label") and second[key].get("support_label")
            and first[key].get("support_label") != second[key].get("support_label")
        ],
        "gate_passed": not errors,
        "errors": errors,
    }
    rendered = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if not errors else 1


def audit_comparison_command(args: argparse.Namespace) -> int:
    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    intervention = json.loads(Path(args.intervention).read_text(encoding="utf-8"))
    keys = [
        "retrieval_eval_cases",
        "retrieval_recall_at_k",
        "retrieval_mrr",
        "gold_citation_recall",
        "fully_supported_rate",
        "partially_supported_rate",
        "unsupported_rate",
        "structural_completeness_rate",
        "refusal_precision",
        "appropriate_refusal_recall",
        "answer_claim_count",
        "mean_latency_ms",
        "input_tokens",
        "output_tokens",
        "estimated_cost_usd",
    ]
    lines = [
        f"# {args.title}",
        "",
        "> These are deterministic diagnostic metrics, not human-annotation results.",
        "",
        f"- {args.left_label}: `{baseline['run_id']}`",
        f"- {args.right_label}: `{intervention['run_id']}`",
        "",
        f"| Metric | {args.left_label} | {args.right_label} |",
        "| --- | ---: | ---: |",
    ]
    for key in keys:
        lines.append(
            f"| {key} | {baseline['summary'].get(key, 'n/a')} | "
            f"{intervention['summary'].get(key, 'n/a')} |"
        )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved: {out}")
    return 0


def retrieval_trace_command(args: argparse.Namespace) -> int:
    """Write per-question rank traces for a controlled retrieval comparison."""
    left = json.loads(Path(args.left).read_text(encoding="utf-8"))
    right = json.loads(Path(args.right).read_text(encoding="utf-8"))
    left_by_question = {record["question_id"]: record for record in left["runs"]}
    right_by_question = {record["question_id"]: record for record in right["runs"]}
    if set(left_by_question) != set(right_by_question):
        raise ValueError("runs do not contain the same question IDs")

    def ranks(record: dict, gold: set[str]) -> list[int]:
        return [
            rank
            for rank, chunk_id in enumerate(record["retrieved_passage_ids"], start=1)
            if chunk_id in gold
        ]

    traces = []
    for question_id in sorted(left_by_question):
        left_record = left_by_question[question_id]
        right_record = right_by_question[question_id]
        gold = set(left_record.get("gold_evidence_chunk_ids", []))
        left_ids = left_record["retrieved_passage_ids"]
        right_ids = right_record["retrieved_passage_ids"]
        traces.append(
            {
                "question_id": question_id,
                "gold_evidence_chunk_ids": sorted(gold),
                "left_gold_ranks": ranks(left_record, gold),
                "right_gold_ranks": ranks(right_record, gold),
                "left_top_k": left_ids,
                "right_top_k": right_ids,
                "top_k_overlap": len(set(left_ids) & set(right_ids)),
            }
        )
    gold_traces = [trace for trace in traces if trace["gold_evidence_chunk_ids"]]
    payload = {
        "left_run_id": left["run_id"],
        "right_run_id": right["run_id"],
        "summary": {
            "questions": len(traces),
            "identical_top_k_questions": sum(
                trace["left_top_k"] == trace["right_top_k"] for trace in traces
            ),
            "mean_top_k_overlap": round(
                sum(trace["top_k_overlap"] for trace in traces) / len(traces), 4
            ) if traces else 0.0,
            "gold_eval_questions": len(gold_traces),
            "gold_hit_set_changed_questions": sum(
                {
                    trace["left_top_k"][rank - 1]
                    for rank in trace["left_gold_ranks"]
                }
                != {
                    trace["right_top_k"][rank - 1]
                    for rank in trace["right_gold_ranks"]
                }
                for trace in gold_traces
            ),
        },
        "traces": traces,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"saved: {out}")
    print(json.dumps(payload["summary"], indent=2))
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
        help="Create embeddings for stored chunks with the selected provider.",
    )
    embed.add_argument("--database-url", help="Database URL. Defaults to DATABASE_URL or local SQLite.")
    _add_embedding_arguments(embed)
    embed.set_defaults(func=embed_command)

    retrieve = subparsers.add_parser(
        "retrieve",
        help="Retrieve stored chunks using lexical, vector, or hybrid scoring.",
    )
    retrieve.add_argument("query", help="Search query.")
    retrieve.add_argument("--database-url", help="Database URL. Defaults to DATABASE_URL or local SQLite.")
    retrieve.add_argument("--mode", choices=["lexical", "vector", "hybrid"], default="hybrid")
    retrieve.add_argument("--top-k", type=int, default=5, help="Number of chunks to return.")
    _add_embedding_arguments(retrieve)
    _add_reranker_argument(retrieve)
    retrieve.set_defaults(func=retrieve_command)

    answer = subparsers.add_parser(
        "answer",
        help="Retrieve evidence and produce a source-cited answer with citation validation.",
    )
    answer.add_argument("query", help="Question to answer from stored chunks.")
    answer.add_argument("--database-url", help="Database URL. Defaults to DATABASE_URL or local SQLite.")
    answer.add_argument("--mode", choices=["lexical", "vector", "hybrid"], default="hybrid")
    answer.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve.")
    _add_embedding_arguments(answer)
    _add_reranker_argument(answer)
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
    _add_embedding_arguments(answer_llm)
    _add_reranker_argument(answer_llm)
    answer_llm.add_argument(
        "--llm-provider",
        choices=["deterministic", "openai", "vertex"],
        default="deterministic",
        help="Structured answer provider.",
    )
    answer_llm.add_argument("--llm-model", help="Provider model name.")
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
    _add_embedding_arguments(eval_parser)
    _add_reranker_argument(eval_parser)
    eval_parser.add_argument(
        "--llm-provider",
        choices=["deterministic", "openai", "vertex"],
        default="deterministic",
        help="Structured answer provider.",
    )
    eval_parser.add_argument("--llm-model", help="Provider model name.")
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

    audit = subparsers.add_parser("claim-audit", help="Run a traceable claim-to-evidence audit over a frozen question JSONL.")
    audit.add_argument("--questions", default="data/audit/questions/v1_40_gold.jsonl")
    audit.add_argument("--manifest", default="data/audit/corpus_manifest.jsonl")
    audit.add_argument("--out", default="results/runs/claim-audit.json")
    audit.add_argument("--run-id")
    audit.add_argument("--database-url")
    audit.add_argument("--mode", choices=["lexical", "hybrid"], default="hybrid")
    audit.add_argument("--top-k", type=int, default=5)
    _add_embedding_arguments(audit)
    _add_reranker_argument(audit)
    audit.add_argument(
        "--llm-provider",
        choices=["deterministic", "openai", "vertex"],
        default="deterministic",
        help="Pinned generation provider; OpenAI requires OPENAI_API_KEY.",
    )
    audit.add_argument("--llm-model")
    audit.add_argument(
        "--max-output-tokens",
        type=int,
        default=1200,
        help="Hard completion cap for each generated answer.",
    )
    audit.add_argument(
        "--verification-gate",
        action="store_true",
        help="Retain only claims that pass deterministic claim-to-evidence verification.",
    )
    audit.add_argument(
        "--verification-gate-policy",
        choices=["strict", "supported"],
        default="strict",
        help="Strict retains only fully-supported claims; supported also retains partially-supported claims.",
    )
    audit.add_argument(
        "--evidence-repair",
        action="store_true",
        help="Rewrite partially-supported claims to exact evidence spans and remove unsupported claims.",
    )
    audit.add_argument(
        "--require-gold-evidence",
        action="store_true",
        help="Require gold_evidence_chunk_ids for every answerable benchmark question.",
    )
    audit.add_argument("--no-ensure-embeddings", action="store_false", dest="ensure_embeddings")
    audit.add_argument(
        "--resume",
        action="store_true",
        help="Resume matching completed questions from the run's partial checkpoint.",
    )
    audit.set_defaults(func=claim_audit_command, ensure_embeddings=True, resume=False)

    repair = subparsers.add_parser(
        "audit-evidence-repair",
        help="Apply deterministic evidence repair to a saved claim-audit run without new generation.",
    )
    repair.add_argument("--run", required=True, help="Baseline claim-audit JSON artifact.")
    repair.add_argument("--out", required=True, help="Path for the repaired JSON artifact.")
    repair.add_argument("--run-id", help="Optional stable ID for the repaired artifact.")
    repair.set_defaults(func=evidence_repair_run_command)

    annotation_template = subparsers.add_parser(
        "annotation-template",
        help="Create human-review JSONL from a claim-audit baseline artifact.",
    )
    annotation_template.add_argument("--run", required=True)
    annotation_template.add_argument("--questions", required=True)
    annotation_template.add_argument("--out", required=True)
    annotation_template.add_argument("--annotator-id", default="a1")
    annotation_template.add_argument("--rubric-version", default="v1.0")
    annotation_template.set_defaults(func=annotation_template_command)

    blind_sample = subparsers.add_parser(
        "annotation-blind-sample",
        help="Create a label-blind answer/claim review packet for reliability annotation.",
    )
    blind_sample.add_argument("--run", required=True)
    blind_sample.add_argument("--questions", required=True)
    blind_sample.add_argument("--out", required=True)
    blind_sample.add_argument("--question-count", type=int, default=5)
    blind_sample.add_argument("--seed", type=int, default=20260723)
    blind_sample.add_argument(
        "--question-ids",
        help="Optional comma-separated fixed question IDs; takes precedence over --question-count and --seed.",
    )
    blind_sample.add_argument("--annotator-id", default="a1")
    blind_sample.add_argument("--rubric-version", default="v1.0")
    blind_sample.set_defaults(func=annotation_blind_sample_command)

    apply_decisions = subparsers.add_parser(
        "annotation-apply-decisions",
        help="Apply an auditable decision map to an annotation packet and write a new completed file.",
    )
    apply_decisions.add_argument("--annotations", required=True)
    apply_decisions.add_argument("--decisions", required=True)
    apply_decisions.add_argument("--out", required=True)
    apply_decisions.add_argument("--annotator-id", required=True)
    apply_decisions.add_argument("--annotation-method", required=True)
    apply_decisions.set_defaults(func=annotation_apply_decisions_command)

    prefill_reference = subparsers.add_parser(
        "annotation-prefill-reference",
        help="Materialize automated prefill as a separate non-human calibration reference.",
    )
    prefill_reference.add_argument("--annotations", required=True)
    prefill_reference.add_argument("--out", required=True)
    prefill_reference.add_argument("--annotator-id", default="deterministic-verifier")
    prefill_reference.add_argument(
        "--annotation-method", default="deterministic_verifier_prefill_v1"
    )
    prefill_reference.set_defaults(func=annotation_prefill_reference_command)

    annotation_summary = subparsers.add_parser(
        "annotation-summary",
        help="Summarize labels from a completed annotation packet.",
    )
    annotation_summary.add_argument("--annotations", required=True)
    annotation_summary.set_defaults(func=annotation_summary_command)

    corpus_verify = subparsers.add_parser(
        "corpus-verify",
        help="Verify audit-corpus provenance and local text SHA-256 hashes.",
    )
    corpus_verify.add_argument("--manifest", default="data/audit/corpus_manifest.jsonl")
    corpus_verify.add_argument(
        "--allow-missing",
        action="store_true",
        help="Validate the manifest and any present local files without requiring paper text.",
    )
    corpus_verify.set_defaults(func=corpus_verify_command)

    freeze = subparsers.add_parser(
        "evaluation-freeze",
        help="Write a content-hash manifest for the exact artifacts in an evaluation pass.",
    )
    freeze.add_argument("--freeze-version", required=True)
    freeze.add_argument("--artifact", action="append", required=True)
    freeze.add_argument("--out", required=True)
    freeze.set_defaults(func=evaluation_freeze_command)

    agreement = subparsers.add_parser(
        "annotation-agreement",
        help="Calculate raw agreement and Cohen's kappa for aligned human labels.",
    )
    agreement.add_argument("--first", required=True)
    agreement.add_argument("--second", required=True)
    agreement.add_argument("--out")
    agreement.add_argument("--minimum-pairs", type=int, default=1)
    agreement.add_argument("--require-complete", action="store_true")
    agreement.add_argument("--require-distinct-annotators", action="store_true")
    agreement.set_defaults(func=annotation_agreement_command)

    comparison = subparsers.add_parser(
        "audit-comparison",
        help="Render an automated baseline-versus-intervention comparison report.",
    )
    comparison.add_argument("--baseline", required=True)
    comparison.add_argument("--intervention", required=True)
    comparison.add_argument("--out", required=True)
    comparison.add_argument("--title", default="Automated Claim-Verification Gate Comparison")
    comparison.add_argument("--left-label", default="Baseline")
    comparison.add_argument("--right-label", default="Verification gate")
    comparison.set_defaults(func=audit_comparison_command)

    trace = subparsers.add_parser(
        "audit-retrieval-trace",
        help="Write per-question gold-rank and top-k-overlap traces for two saved runs.",
    )
    trace.add_argument("--left", required=True)
    trace.add_argument("--right", required=True)
    trace.add_argument("--out", required=True)
    trace.set_defaults(func=retrieval_trace_command)

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
    load_dotenv(dotenv_path=Path.cwd() / ".env.local", override=False)
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
