import arh.runner.trial as trial_mod
from arh.errors import InfraError
from arh.models import GraderSpec, Task, Trajectory
from arh.runner.trial import run_task_trials
from arh.store.jsonl import JsonlStore


def make_task(tmp_path, timeout_s: int = 120) -> Task:
    state = tmp_path / "state"
    state.mkdir(exist_ok=True)
    (state / "notes.txt").write_text("Buy milk", encoding="utf-8")
    return Task(
        id="t1", prompt="p", initial_state=state, timeout_s=timeout_s,
        grader=GraderSpec(kind="file_match", expect_files=["notes.md"]),
    )


class FakeMCPSession:
    def __init__(self, params):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        pass


def patch_agent(monkeypatch, behavior):
    """Replace the real agent loop; behavior() is invoked once per trial attempt
    and its exceptions propagate like real agent-loop exceptions."""

    async def fake_run_agent(model, task, mcp, temperature=0.7):
        return behavior()

    monkeypatch.setattr(trial_mod, "MCPSession", FakeMCPSession)
    monkeypatch.setattr(trial_mod, "run_agent", fake_run_agent)


async def test_successful_trial_graded_by_outcome(tmp_path, monkeypatch):
    task = make_task(tmp_path)
    current_sandbox = {}
    orig_sandbox_cls = trial_mod.TrialSandbox

    class SpySandbox(orig_sandbox_cls):
        def seed(self):
            super().seed()
            current_sandbox["dir"] = self.work_dir

    def behavior():
        # simulate the agent renaming the file inside the sandbox
        d = current_sandbox["dir"]
        (d / "notes.txt").rename(d / "notes.md")
        return Trajectory(task_id="t1", stop_reason="completed")

    monkeypatch.setattr(trial_mod, "TrialSandbox", SpySandbox)
    patch_agent(monkeypatch, behavior)
    store = JsonlStore(tmp_path / "r.jsonl")
    results = await run_task_trials(task, "m", 1, tmp_path / "runs", store)
    assert results[0].success
    assert results[0].failure_source == "none"
    assert store.load()[0].success


async def test_failed_trial_counts_as_agent_failure(tmp_path, monkeypatch):
    task = make_task(tmp_path)
    patch_agent(monkeypatch, lambda: Trajectory(task_id="t1", stop_reason="completed"))
    store = JsonlStore(tmp_path / "r.jsonl")
    results = await run_task_trials(task, "m", 1, tmp_path / "runs", store)
    assert not results[0].success
    assert results[0].failure_source == "agent"
    assert "notes.md" in results[0].grade_detail


async def test_infra_error_retried_then_succeeds(tmp_path, monkeypatch):
    task = make_task(tmp_path)
    attempts = {"n": 0}

    def behavior():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise InfraError("provider 429")
        return Trajectory(task_id="t1", stop_reason="completed")

    patch_agent(monkeypatch, behavior)
    monkeypatch.setattr(trial_mod, "_RETRY_WAIT_S", 0)  # no real sleeping in tests
    store = JsonlStore(tmp_path / "r.jsonl")
    results = await run_task_trials(task, "m", 1, tmp_path / "runs", store)
    assert attempts["n"] == 3
    assert len(results) == 1
    assert results[0].failure_source in ("agent", "none")  # graded, not infra


async def test_persistent_infra_error_excluded_not_agent_failure(tmp_path, monkeypatch):
    task = make_task(tmp_path)

    def behavior():
        raise InfraError("docker daemon unreachable")

    patch_agent(monkeypatch, behavior)
    monkeypatch.setattr(trial_mod, "_RETRY_WAIT_S", 0)
    store = JsonlStore(tmp_path / "r.jsonl")
    results = await run_task_trials(task, "m", 2, tmp_path / "runs", store)
    assert all(r.failure_source == "infra" for r in results)
    assert all(not r.success for r in results)


async def test_timeout_is_agent_failure(tmp_path, monkeypatch):
    import asyncio

    task = make_task(tmp_path, timeout_s=1)

    async def slow_run_agent(model, task, mcp, temperature=0.7):
        await asyncio.sleep(5)

    monkeypatch.setattr(trial_mod, "MCPSession", FakeMCPSession)
    monkeypatch.setattr(trial_mod, "run_agent", slow_run_agent)
    store = JsonlStore(tmp_path / "r.jsonl")
    results = await run_task_trials(task, "m", 1, tmp_path / "runs", store)
    assert not results[0].success
    assert results[0].failure_source == "agent"
    assert results[0].trajectory.stop_reason == "timeout"
