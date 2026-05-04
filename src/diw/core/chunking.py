from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Iterable

from diw.core.normalisation import normalise_text


HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


@dataclass(frozen=True)
class SourceChunk:
    chunk_index: int
    text: str
    heading_path: tuple[str, ...]
    content_hash: str
    start_line: int
    end_line: int


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _heading_path_for_line(line: str, current: list[str | None]) -> tuple[str, ...]:
    match = HEADING_RE.match(line)
    if not match:
        return tuple(part for part in current if part)

    level = len(match.group(1))
    title = match.group(2).strip()
    current[level - 1] = title
    for idx in range(level, len(current)):
        current[idx] = None
    return tuple(part for part in current if part)


def chunk_markdown(
    text: str,
    *,
    target_chars: int = 1200,
    overlap_chars: int = 160,
) -> list[SourceChunk]:
    """Create heading-aware chunks suitable for retrieval and citation.

    The function is intentionally deterministic: the same source text produces
    stable chunk indexes and hashes, which matters for AI-run provenance.
    """
    if target_chars <= 200:
        raise ValueError("target_chars must be greater than 200")
    if overlap_chars < 0:
        raise ValueError("overlap_chars must be non-negative")
    if overlap_chars >= target_chars:
        raise ValueError("overlap_chars must be smaller than target_chars")

    cleaned = normalise_text(text)
    if not cleaned:
        return []

    lines = cleaned.split("\n")
    heading_stack: list[str | None] = [None] * 6
    chunks: list[SourceChunk] = []
    buffer: list[str] = []
    buffer_start_line = 1
    current_path: tuple[str, ...] = ()

    def flush(end_line: int) -> None:
        nonlocal buffer, buffer_start_line
        chunk_text = "\n".join(buffer).strip()
        if not chunk_text:
            buffer = []
            buffer_start_line = end_line + 1
            return

        chunks.append(
            SourceChunk(
                chunk_index=len(chunks),
                text=chunk_text,
                heading_path=current_path,
                content_hash=_content_hash(chunk_text),
                start_line=buffer_start_line,
                end_line=end_line,
            )
        )

        if overlap_chars and len(chunk_text) > overlap_chars:
            overlap = chunk_text[-overlap_chars:]
            buffer = [overlap]
            buffer_start_line = max(buffer_start_line, end_line)
        else:
            buffer = []
            buffer_start_line = end_line + 1

    for line_number, line in enumerate(lines, start=1):
        maybe_path = _heading_path_for_line(line, heading_stack)
        if HEADING_RE.match(line):
            if buffer:
                flush(line_number - 1)
            current_path = maybe_path
            buffer_start_line = line_number

        if not buffer:
            buffer_start_line = line_number
        buffer.append(line)

        if len("\n".join(buffer)) >= target_chars:
            flush(line_number)

    if buffer:
        flush(len(lines))

    return chunks


def chunks_as_records(chunks: Iterable[SourceChunk]) -> list[dict[str, object]]:
    return [
        {
            "chunk_index": chunk.chunk_index,
            "text": chunk.text,
            "heading_path": list(chunk.heading_path),
            "content_hash": chunk.content_hash,
            "start_line": chunk.start_line,
            "end_line": chunk.end_line,
        }
        for chunk in chunks
    ]
