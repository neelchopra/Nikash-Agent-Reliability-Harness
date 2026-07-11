"""Aggregate TrialResults into pass^k summaries and rich tables.

Infra-failed trials are excluded from n (the pass^k denominator) - they are
reported in their own column so the exclusion is always visible, never silent.
"""

from collections import defaultdict

from rich.table import Table

from arh.metrics.passk import pass_at_k, pass_hat_k, wilson_interval
from arh.models import TrialResult

K_VALUES = (1, 3, 5, 10)


def summarize(results: list[TrialResult]) -> list[dict]:
    groups: dict[tuple[str, str], list[TrialResult]] = defaultdict(list)
    for r in results:
        groups[(r.task_id, r.model)].append(r)

    summaries = []
    for (task_id, model), rows in sorted(groups.items()):
        counted = [r for r in rows if r.failure_source != "infra"]
        infra = len(rows) - len(counted)
        n = len(counted)
        c = sum(1 for r in counted if r.success)
        pass_hat = {k: pass_hat_k(n, c, k) for k in K_VALUES if k <= n}
        summaries.append(
            {
                "task_id": task_id,
                "model": model,
                "n": n,
                "c": c,
                "infra": infra,
                "pass_hat": pass_hat,
                "pass_at_1": pass_at_k(n, c, 1) if n >= 1 else 0.0,
                "wilson": wilson_interval(c, n),
            }
        )
    return summaries


def render_table(summaries: list[dict]) -> Table:
    table = Table(title="ARH pass^k report")
    table.add_column("task")
    table.add_column("model")
    table.add_column("n", justify="right")
    table.add_column("c", justify="right")
    table.add_column("infra", justify="right")
    for k in K_VALUES:
        table.add_column(f"pass^{k}", justify="right")
    table.add_column("Wilson 95% (p)", justify="right")
    for s in summaries:
        low, high = s["wilson"]
        table.add_row(
            s["task_id"],
            s["model"],
            str(s["n"]),
            str(s["c"]),
            str(s["infra"]),
            *[
                f"{s['pass_hat'][k]:.3f}" if k in s["pass_hat"] else "-"
                for k in K_VALUES
            ],
            f"[{low:.3f}, {high:.3f}]",
        )
    return table
