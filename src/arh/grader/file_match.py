"""Outcome-based grading: inspect the final state of the trial work dir.

Path-agnostic by construction - only the end state matters, never the
trajectory the agent took to get there.
"""

from pathlib import Path

from arh.models import GraderSpec


def grade_file_match(work_dir: Path, spec: GraderSpec) -> tuple[bool, str]:
    for name in spec.expect_files:
        if not (work_dir / name).is_file():
            return False, f"expected file missing: {name}"
    for pattern in spec.forbid_files:
        hits = sorted(p.name for p in work_dir.rglob(pattern))
        if hits:
            return False, f"forbidden files present ({pattern}): {hits}"
    for name, substrings in spec.expect_content.items():
        f = work_dir / name
        if not f.is_file():
            return False, f"expected file missing: {name}"
        text = f.read_text(encoding="utf-8")
        for substring in substrings:
            if substring not in text:
                return False, f"content check failed: {name} does not contain {substring!r}"
    return True, "all checks passed"
