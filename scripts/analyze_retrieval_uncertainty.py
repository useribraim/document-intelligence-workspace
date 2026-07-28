#!/usr/bin/env python3
"""Compute paired bootstrap intervals from a published retrieval trace."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def _recall_at_k(gold_ranks: list[int], gold_count: int) -> float:
    return len(gold_ranks) / gold_count if gold_count else 0.0


def _mrr(gold_ranks: list[int]) -> float:
    return 1 / min(gold_ranks) if gold_ranks else 0.0


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("values must not be empty")
    return sorted(values)[int(fraction * (len(values) - 1))]


def paired_bootstrap_interval(
    deltas: list[float], *, iterations: int, seed: int
) -> tuple[float, float, float]:
    if not deltas:
        raise ValueError("at least one paired delta is required")
    if iterations < 1:
        raise ValueError("iterations must be positive")
    rng = random.Random(seed)
    sample_size = len(deltas)
    samples = [
        sum(deltas[rng.randrange(sample_size)] for _ in range(sample_size)) / sample_size
        for _ in range(iterations)
    ]
    return (
        round(sum(deltas) / sample_size, 4),
        round(percentile(samples, 0.025), 4),
        round(percentile(samples, 0.975), 4),
    )


def analyse_trace(payload: dict, *, iterations: int, seed: int) -> dict[str, object]:
    paired = [item for item in payload["traces"] if item["gold_evidence_chunk_ids"]]
    recall_deltas = [
        _recall_at_k(item["right_gold_ranks"], len(item["gold_evidence_chunk_ids"]))
        - _recall_at_k(item["left_gold_ranks"], len(item["gold_evidence_chunk_ids"]))
        for item in paired
    ]
    mrr_deltas = [
        _mrr(item["right_gold_ranks"]) - _mrr(item["left_gold_ranks"])
        for item in paired
    ]
    recall_changed = [
        item["question_id"]
        for item, delta in zip(paired, recall_deltas, strict=True)
        if delta != 0
    ]
    return {
        "left_run_id": payload["left_run_id"],
        "right_run_id": payload["right_run_id"],
        "paired_gold_questions": len(paired),
        "bootstrap_iterations": iterations,
        "seed": seed,
        "recall_at_5": paired_bootstrap_interval(
            recall_deltas, iterations=iterations, seed=seed
        ),
        "mrr": paired_bootstrap_interval(
            mrr_deltas, iterations=iterations, seed=seed + 1
        ),
        "recall_changed_question_ids": recall_changed,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=20_260_728)
    args = parser.parse_args()
    payload = json.loads(args.trace.read_text(encoding="utf-8"))
    print(json.dumps(analyse_trace(payload, iterations=args.iterations, seed=args.seed), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
