"""Gold-solution tests: for each task, directly produce a known-correct end
state (no LLM/MCP/Docker involved) and assert the real grader marks it a
success. This is the automated form of "hand-verified gold solution" - it
runs in CI and catches grader regressions.
"""

import shutil
from collections.abc import Callable
from pathlib import Path

import pytest

from arh.grader.file_match import grade_file_match
from arh.tasks import load_task

TASKS_DIR = Path(__file__).parent.parent / "tasks"


def _solve_fs_rename_001(work_dir: Path) -> None:
    (work_dir / "notes.txt").rename(work_dir / "notes.md")
    (work_dir / "report.txt").rename(work_dir / "report.md")


def _solve_fs_create_001(work_dir: Path) -> None:
    (work_dir / "index.md").write_text(
        "apple.txt\nbanana.txt\ncherry.txt\n", encoding="utf-8"
    )


GOLD_SOLVERS: dict[str, Callable[[Path], None]] = {
    "fs-rename-001": _solve_fs_rename_001,
    "fs-create-001": _solve_fs_create_001,
}


def _task_ids() -> list[str]:
    return sorted(p.stem for p in TASKS_DIR.glob("*.yaml"))


@pytest.mark.parametrize("task_id", _task_ids())
def test_gold_solution_passes_grader(task_id, tmp_path):
    task = load_task(TASKS_DIR / f"{task_id}.yaml")
    assert task_id in GOLD_SOLVERS, f"no gold solver registered for {task_id}"
    work_dir = tmp_path / "work"
    shutil.copytree(task.initial_state, work_dir)
    GOLD_SOLVERS[task_id](work_dir)
    ok, detail = grade_file_match(work_dir, task.grader)
    assert ok, detail
