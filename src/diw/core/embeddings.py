from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import math
import os
import re
from typing import Protocol


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


class EmbeddingProvider(Protocol):
    model_name: str
    dimensions: int

    def embed(self, text: str) -> list[float]:
        ...


def embed_document(provider: EmbeddingProvider, text: str) -> list[float]:
    """Embed corpus text, using a retrieval-document task when supported."""
    method = getattr(provider, "embed_document", None)
    if method is not None:
        return method(text)
    return provider.embed(text)


def embed_query(provider: EmbeddingProvider, text: str) -> list[float]:
    """Embed a search query, using a retrieval-query task when supported."""
    method = getattr(provider, "embed_query", None)
    if method is not None:
        return method(text)
    return provider.embed(text)


def embed_documents(provider: EmbeddingProvider, texts: list[str]) -> list[list[float]]:
    """Embed corpus texts in provider-supported batches while preserving order."""
    method = getattr(provider, "embed_documents", None)
    if method is not None:
        return method(texts)
    return [embed_document(provider, text) for text in texts]


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


class OpenAIEmbeddingProvider:
    """OpenAI text embedding provider for measured semantic retrieval runs.

    The local hashing provider remains the development/test default. This
    provider exists so retrieval quality can be measured with a real semantic
    embedding model on the frozen audit corpus. Embeddings are stored under
    the provider's model name, so they coexist with local hashing embeddings
    in the same database.
    """

    DEFAULT_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
    }

    def __init__(
        self,
        *,
        model_name: str = "text-embedding-3-small",
        dimensions: int | None = None,
        api_key: str | None = None,
    ) -> None:
        if dimensions is None:
            dimensions = self.DEFAULT_DIMENSIONS.get(model_name)
            if dimensions is None:
                raise ValueError(
                    f"dimensions must be set explicitly for embedding model: {model_name}"
                )
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self.model_name = model_name
        self.dimensions = dimensions
        self.api_key = os.getenv("OPENAI_API_KEY") if api_key is None else api_key
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAIEmbeddingProvider")
        self._client = None

    def embed(self, text: str) -> list[float]:
        if not text.strip():
            return [0.0] * self.dimensions
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self.api_key, timeout=45.0, max_retries=1)
        response = self._client.embeddings.create(
            model=self.model_name,
            input=text,
            dimensions=self.dimensions,
        )
        return [float(value) for value in response.data[0].embedding]

    def embed_documents(self, texts: list[str], *, batch_size: int = 64) -> list[list[float]]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not texts:
            return []
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self.api_key, timeout=45.0, max_retries=1)

        vectors: list[list[float] | None] = [None] * len(texts)
        nonempty = [(index, text) for index, text in enumerate(texts) if text.strip()]
        for index, text in enumerate(texts):
            if not text.strip():
                vectors[index] = [0.0] * self.dimensions
        for start in range(0, len(nonempty), batch_size):
            batch = nonempty[start : start + batch_size]
            response = self._client.embeddings.create(
                model=self.model_name,
                input=[text for _, text in batch],
                dimensions=self.dimensions,
            )
            ordered = sorted(response.data, key=lambda item: getattr(item, "index", 0))
            if len(ordered) != len(batch):
                raise ValueError("OpenAI returned an unexpected number of embeddings")
            for (original_index, _), item in zip(batch, ordered):
                vector = [float(value) for value in item.embedding]
                if len(vector) != self.dimensions:
                    raise ValueError(
                        f"OpenAI returned {len(vector)} dimensions; expected {self.dimensions}"
                    )
                vectors[original_index] = vector
        if any(vector is None for vector in vectors):
            raise ValueError("OpenAI embedding batch was incomplete")
        return [vector for vector in vectors if vector is not None]


class VertexAIEmbeddingProvider:
    """Vertex AI semantic embeddings through the Google Gen AI SDK.

    Application Default Credentials are resolved by the SDK. Query and document
    embeddings use distinct retrieval task types while sharing one stored model
    namespace.
    """

    def __init__(
        self,
        *,
        model_name: str = "gemini-embedding-001",
        dimensions: int = 768,
        project: str | None = None,
        location: str | None = None,
    ) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self.model_name = model_name
        self.dimensions = dimensions
        self.project = project or os.getenv("GOOGLE_CLOUD_PROJECT")
        self.location = location or os.getenv("GOOGLE_CLOUD_LOCATION", "global")
        if not self.project:
            raise ValueError("GOOGLE_CLOUD_PROJECT is required for VertexAIEmbeddingProvider")
        self._client = None

    def embed(self, text: str) -> list[float]:
        return self.embed_document(text)

    def embed_document(self, text: str) -> list[float]:
        return self._embed(text, task_type="RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text, task_type="RETRIEVAL_QUERY")

    def _embed(self, text: str, *, task_type: str) -> list[float]:
        if not text.strip():
            return [0.0] * self.dimensions
        if self._client is None:
            try:
                genai = importlib.import_module("google.genai")
            except ImportError as error:
                raise RuntimeError(
                    "Vertex AI support requires the 'cloud' extra: pip install -e '.[cloud]'"
                ) from error
            self._client = genai.Client(
                vertexai=True,
                project=self.project,
                location=self.location,
                http_options={"api_version": "v1"},
            )
        response = self._client.models.embed_content(
            model=self.model_name,
            contents=text,
            config={
                "task_type": task_type,
                "output_dimensionality": self.dimensions,
                "auto_truncate": False,
            },
        )
        embeddings = getattr(response, "embeddings", None) or []
        if not embeddings or getattr(embeddings[0], "values", None) is None:
            raise ValueError("Vertex AI returned no embedding vector")
        vector = [float(value) for value in embeddings[0].values]
        if len(vector) != self.dimensions:
            raise ValueError(
                f"Vertex AI returned {len(vector)} dimensions; expected {self.dimensions}"
            )
        return vector


def build_embedding_provider(
    provider: str = "local",
    *,
    dimensions: int | None = None,
    embedding_model: str | None = None,
) -> EmbeddingProvider:
    """Select an embedding provider by name.

    ``dimensions`` defaults to 64 for the local hashing provider and to the
    model's native size for known OpenAI embedding models.
    """

    if provider == "local":
        return LocalHashingEmbeddingProvider(dimensions=dimensions if dimensions is not None else 64)
    if provider == "openai":
        return OpenAIEmbeddingProvider(
            model_name=embedding_model or "text-embedding-3-small",
            dimensions=dimensions,
        )
    if provider == "vertex":
        return VertexAIEmbeddingProvider(
            model_name=embedding_model or "gemini-embedding-001",
            dimensions=dimensions if dimensions is not None else 768,
        )
    raise ValueError(f"unsupported embedding provider: {provider}")


def l2_normalise(vector: list[float]) -> list[float]:
    magnitude = math.sqrt(sum(value * value for value in vector))
    if magnitude == 0:
        return vector
    return [value / magnitude for value in vector]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("vectors must have the same dimensions")
    return sum(left_value * right_value for left_value, right_value in zip(left, right))
