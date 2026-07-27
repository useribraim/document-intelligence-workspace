from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from diw.core.chunking import chunk_markdown, chunks_as_records
from diw.core.normalisation import NormalisationReport, normalise_text_with_report


@dataclass(frozen=True)
class IngestedDocument:
    document_id: str
    version_id: str
    source_path: str
    source_name: str
    source_type: str
    content_hash: str
    ingested_at: str
    normalised_text: str
    normalisation_report: NormalisationReport
    chunks: list[dict[str, object]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "version_id": self.version_id,
            "source_path": self.source_path,
            "source_name": self.source_name,
            "source_type": self.source_type,
            "content_hash": self.content_hash,
            "ingested_at": self.ingested_at,
            "normalisation_report": asdict(self.normalisation_report),
            "chunk_count": len(self.chunks),
            "chunks": self.chunks,
        }

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), indent=2, ensure_ascii=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def infer_source_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".md", ".markdown"}:
        return "markdown"
    if suffix == ".txt":
        return "plain_text"
    return suffix.removeprefix(".") or "unknown"


def ingest_file(
    path: Path,
    *,
    target_chars: int = 1200,
    overlap_chars: int = 160,
) -> IngestedDocument:
    if not path.exists():
        raise FileNotFoundError(path)
    if not path.is_file():
        raise ValueError(f"Expected a file path, got: {path}")

    raw_text = path.read_text(encoding="utf-8")
    normalised = normalise_text_with_report(raw_text)
    content_hash = sha256_text(normalised.text)
    source_uri = path.resolve().as_uri()
    document_id = str(uuid5(NAMESPACE_URL, source_uri))
    version_id = str(uuid5(NAMESPACE_URL, f"{source_uri}:{content_hash}"))
    chunks = chunk_markdown(
        normalised.text,
        target_chars=target_chars,
        overlap_chars=overlap_chars,
    )

    return IngestedDocument(
        document_id=document_id,
        version_id=version_id,
        source_path=str(path),
        source_name=path.name,
        source_type=infer_source_type(path),
        content_hash=content_hash,
        ingested_at=datetime.now(UTC).isoformat(),
        normalised_text=normalised.text,
        normalisation_report=normalised.report,
        chunks=chunks_as_records(chunks),
    )
