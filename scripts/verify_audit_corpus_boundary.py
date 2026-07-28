"""Fail if a tracked working-tree file reintroduces raw frozen audit-paper text.

This protects the current-tree boundary only.  It intentionally does not claim
to inspect or remediate historical Git objects; see the corpus-history incident
note for that required, separately authorised operation.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = Path("data/audit/corpus/text")
ALLOWED_TRACKED_PATHS = {CORPUS_DIR / "README.md"}


def tracked_corpus_paths() -> set[Path]:
    result = subprocess.run(
        ["git", "ls-files", "--", str(CORPUS_DIR)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return {Path(line) for line in result.stdout.splitlines() if line}


def main() -> int:
    unexpected = tracked_corpus_paths() - ALLOWED_TRACKED_PATHS
    if unexpected:
        rendered = ", ".join(str(path) for path in sorted(unexpected))
        print(f"raw audit corpus files must not be tracked: {rendered}", file=sys.stderr)
        return 1
    print("audit corpus boundary: current tracked tree contains no raw paper text")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
