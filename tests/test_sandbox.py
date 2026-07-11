from pathlib import Path

from arh.models import GraderSpec, Task
from arh.runner.sandbox import TrialSandbox


def make_task(state_dir: Path) -> Task:
    return Task(
        id="t1",
        prompt="p",
        initial_state=state_dir,
        grader=GraderSpec(kind="file_match"),
    )


def test_seed_copies_initial_state_into_fresh_work_dir(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    (state / "a.txt").write_text("hello", encoding="utf-8")
    sb = TrialSandbox(make_task(state), runs_root=tmp_path / "runs")
    sb.seed()
    assert (sb.work_dir / "a.txt").read_text(encoding="utf-8") == "hello"
    # mutating the work dir must not touch the seed state
    (sb.work_dir / "a.txt").write_text("mutated", encoding="utf-8")
    assert (state / "a.txt").read_text(encoding="utf-8") == "hello"


def test_two_sandboxes_are_isolated(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    (state / "a.txt").write_text("hello", encoding="utf-8")
    task = make_task(state)
    sb1 = TrialSandbox(task, runs_root=tmp_path / "runs")
    sb2 = TrialSandbox(task, runs_root=tmp_path / "runs")
    assert sb1.trial_id != sb2.trial_id
    assert sb1.work_dir != sb2.work_dir


def test_cleanup_removes_work_dir(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    sb = TrialSandbox(make_task(state), runs_root=tmp_path / "runs")
    sb.seed()
    assert sb.work_dir.exists()
    sb.cleanup()
    assert not sb.work_dir.exists()


def test_server_params_build_docker_run_command(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    sb = TrialSandbox(make_task(state), runs_root=tmp_path / "runs")
    params = sb.server_params()
    assert params.command == "docker"
    assert params.args[:3] == ["run", "-i", "--rm"]
    assert "--network" in params.args and "none" in params.args
    assert f"{sb.work_dir}:/work" in params.args
    assert params.args[-2:] == ["mcp/filesystem", "/work"]
