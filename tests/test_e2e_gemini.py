"""Real end-to-end: Gemini 2.5 Flash driving the dockerized MCP filesystem server.

Run explicitly with:  uv run pytest -m e2e -v
Requires: Docker Desktop running, mcp/filesystem pulled, GEMINI_API_KEY set.
"""

import os
from pathlib import Path

import pytest

from arh.runner.trial import run_task_trials
from arh.store.jsonl import JsonlStore
from arh.tasks import load_task

pytestmark = pytest.mark.e2e

TASK_YAML = Path(__file__).parent.parent / "tasks" / "fs-rename-001.yaml"


@pytest.fixture(autouse=True)
def require_key():
    if not os.environ.get("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY not set")


async def test_single_real_trial(tmp_path):
    task = load_task(TASK_YAML)
    store = JsonlStore(tmp_path / "r.jsonl")
    results = await run_task_trials(
        task, "gemini/gemini-2.5-flash", 1, tmp_path / "runs", store
    )
    (r,) = results
    # The trial must complete and be graded - success either way is informative,
    # but an infra failure here means the harness itself is broken.
    assert r.failure_source in ("none", "agent")
    assert r.trajectory is not None
    assert len(r.trajectory.tool_calls) > 0  # the model actually used MCP tools
    assert r.trajectory.input_tokens > 0
