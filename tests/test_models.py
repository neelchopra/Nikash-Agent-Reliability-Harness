from pathlib import Path

from arh.models import Trajectory, TrialResult
from arh.tasks import load_task

TASKS_DIR = Path(__file__).parent.parent / "tasks"


def test_trial_result_json_roundtrip():
    r = TrialResult(
        trial_id="abc123",
        task_id="fs-rename-001",
        task_version=1,
        model="gemini/gemini-2.5-flash",
        success=True,
        failure_source="none",
        trajectory=Trajectory(task_id="fs-rename-001", stop_reason="completed"),
    )
    line = r.model_dump_json()
    back = TrialResult.model_validate_json(line)
    assert back == r


def test_load_example_task_resolves_state_dir():
    task = load_task(TASKS_DIR / "fs-rename-001.yaml")
    assert task.id == "fs-rename-001"
    assert task.grader.kind == "file_match"
    assert task.initial_state.is_absolute()
    assert (task.initial_state / "notes.txt").is_file()
    assert task.n_trials == 10
    assert task.docker_image == "mcp/filesystem"
