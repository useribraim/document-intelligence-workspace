from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NormalisationReport:
    original_line_count: int
    normalised_line_count: int
    trailing_whitespace_lines: int
    collapsed_blank_lines: int


@dataclass(frozen=True)
class NormalisedText:
    text: str
    report: NormalisationReport


def normalise_text(text: str) -> str:
    """Return stable text for downstream chunking and hashing."""
    return normalise_text_with_report(text).text


def normalise_text_with_report(text: str) -> NormalisedText:
    """Normalise line endings and whitespace without changing document meaning.

    This function is intentionally conservative. It does not lowercase text,
    remove punctuation, rewrite spacing inside lines, or alter Markdown heading
    markers because those choices can destroy source meaning and provenance.
    """
    line_ending_normalised = text.replace("\r\n", "\n").replace("\r", "\n")
    raw_lines = line_ending_normalised.split("\n")

    trailing_whitespace_lines = sum(1 for line in raw_lines if line != line.rstrip())
    stripped_lines = [line.rstrip() for line in raw_lines]

    compacted: list[str] = []
    blank_seen = False
    collapsed_blank_lines = 0

    for line in stripped_lines:
        if line.strip():
            compacted.append(line)
            blank_seen = False
            continue

        if blank_seen:
            collapsed_blank_lines += 1
            continue

        compacted.append("")
        blank_seen = True

    normalised = "\n".join(compacted).strip()
    normalised_lines = normalised.split("\n") if normalised else []

    return NormalisedText(
        text=normalised,
        report=NormalisationReport(
            original_line_count=len(raw_lines),
            normalised_line_count=len(normalised_lines),
            trailing_whitespace_lines=trailing_whitespace_lines,
            collapsed_blank_lines=collapsed_blank_lines,
        ),
    )
