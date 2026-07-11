from arh.models import TrialResult
from arh.store.jsonl import JsonlStore


def make_result(i: int, success: bool) -> TrialResult:
    return TrialResult(
        trial_id=f"t{i}", task_id="task-a", task_version=1,
        model="m", success=success, failure_source="none" if success else "agent",
    )


def test_append_and_load_roundtrip(tmp_path):
    store = JsonlStore(tmp_path / "sub" / "results.jsonl")
    store.append(make_result(1, True))
    store.append(make_result(2, False))
    rows = store.load()
    assert [r.trial_id for r in rows] == ["t1", "t2"]
    assert rows[0].success and not rows[1].success
