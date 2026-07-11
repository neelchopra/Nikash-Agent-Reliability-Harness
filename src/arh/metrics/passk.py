"""Unbiased pass^k / pass@k estimators and Wilson score intervals.

pass^k  = C(c,k)/C(n,k)      probability that k trials drawn without
                             replacement from n observed trials are ALL successes.
pass@k  = 1 - C(n-c,k)/C(n,k) probability that AT LEAST ONE of k drawn trials succeeds.
"""

import math


def _validate(n: int, c: int, k: int) -> None:
    if not 0 <= c <= n:
        raise ValueError(f"need 0 <= c <= n, got c={c}, n={n}")
    if not 1 <= k <= n:
        raise ValueError(f"need 1 <= k <= n, got k={k}, n={n}")


def pass_hat_k(n: int, c: int, k: int) -> float:
    """Probability all k of k drawn trials succeed (tau-bench pass^k, unbiased)."""
    _validate(n, c, k)
    if c < k:
        return 0.0
    return math.comb(c, k) / math.comb(n, k)


def pass_at_k(n: int, c: int, k: int) -> float:
    """Probability at least one of k drawn trials succeeds (HumanEval pass@k, unbiased)."""
    _validate(n, c, k)
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def wilson_interval(c: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion c/n."""
    if n == 0:
        return (0.0, 1.0)
    p = c / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))
