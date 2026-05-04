from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import re
from typing import Protocol


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


class EmbeddingProvider(Protocol):
    model_name: str
    dimensions: int

    def embed(self, text: str) -> list[float]:
        ...


@dataclass(frozen=True)
class LocalHashingEmbeddingProvider:
    """Deterministic local embedding provider for development and tests.

    This is not a semantic model. It is a small hashing-vector implementation
    that lets the retrieval pipeline be built and tested without API calls.
    """

    dimensions: int = 64
    model_name: str = "local-hashing-v1"

    def embed(self, text: str) -> list[float]:
        if self.dimensions <= 0:
            raise ValueError("dimensions must be positive")

        vector = [0.0] * self.dimensions
        tokens = TOKEN_RE.findall(text.lower())
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign

        return l2_normalise(vector)


def l2_normalise(vector: list[float]) -> list[float]:
    magnitude = math.sqrt(sum(value * value for value in vector))
    if magnitude == 0:
        return vector
    return [value / magnitude for value in vector]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("vectors must have the same dimensions")
    return sum(left_value * right_value for left_value, right_value in zip(left, right))
