import pytest

from arh.models import TrialResult
from arh.report import summarize


def make(i, success, source=None):
    return TrialResult(
        trial_id=f"t{i}", task_id="task-a", task_version=1, model="m",
        success=success,
        failure_source=source or ("none" if success else "agent"),
    )


def test_summarize_excludes_infra_from_denominator():
    results = [make(i, True) for i in range(8)]          # 8 successes
    results += [make(8, False), make(9, False)]          # 2 agent failures
    results += [make(10, False, source="infra")]         # 1 infra row
    (s,) = summarize(results)
    assert s["n"] == 10          # infra row excluded
    assert s["c"] == 8
    assert s["infra"] == 1
    assert s["pass_hat"][1] == pytest.approx(0.8)
    # C(8,3)/C(10,3) = 56/120
    assert s["pass_hat"][3] == pytest.approx(56 / 120)
    assert s["pass_hat"][10] == 0.0
    low, high = s["wilson"]
    assert low == pytest.approx(0.49018, abs=1e-4)
    assert high == pytest.approx(0.94330, abs=1e-4)


def test_summarize_skips_k_larger_than_n():
    results = [make(i, True) for i in range(3)]
    (s,) = summarize(results)
    assert set(s["pass_hat"].keys()) == {1, 3}  # k=5,10 not computable at n=3


def test_summarize_groups_by_task_and_model():
    results = [make(1, True)]
    other = TrialResult(
        trial_id="x", task_id="task-b", task_version=1, model="m2",
        success=False, failure_source="agent",
    )
    summaries = summarize(results + [other])
    assert len(summaries) == 2
