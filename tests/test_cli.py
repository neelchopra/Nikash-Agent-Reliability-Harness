from typer.testing import CliRunner

import arh.cli as cli_mod
from arh.cli import app
from arh.models import TrialResult
from arh.store.jsonl import JsonlStore

runner = CliRunner()


def test_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_report_command_renders_table(tmp_path):
    store = JsonlStore(tmp_path / "r.jsonl")
    for i in range(3):
        store.append(TrialResult(
            trial_id=f"t{i}", task_id="task-a", task_version=1, model="m",
            success=i < 2, failure_source="none" if i < 2 else "agent",
        ))
    result = runner.invoke(app, ["report", str(tmp_path / "r.jsonl")])
    assert result.exit_code == 0
    assert "task-a" in result.output
    assert "pass^1" in result.output


def test_run_command_invokes_runner(tmp_path, monkeypatch):
    called = {}

    async def fake_run_task_trials(task, model, n_trials, runs_root, store, on_result=None):
        called["task_id"] = task.id
        called["model"] = model
        called["n"] = n_trials
        return []

    monkeypatch.setattr(cli_mod, "run_task_trials", fake_run_task_trials)
    result = runner.invoke(app, [
        "run",
        "--task", "tasks/fs-rename-001.yaml",
        "--model", "fake/model",
        "--n", "2",
        "--out", str(tmp_path / "results"),
    ])
    assert result.exit_code == 0, result.output
    assert called == {"task_id": "fs-rename-001", "model": "fake/model", "n": 2}
