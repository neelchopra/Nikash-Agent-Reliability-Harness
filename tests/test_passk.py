import pytest

from arh.metrics.passk import pass_at_k, pass_hat_k, wilson_interval


def test_pass_hat_k_hand_computed():
    # C(5,3)/C(10,3) = 10/120
    assert pass_hat_k(n=10, c=5, k=3) == pytest.approx(10 / 120)
    # all successes -> 1.0
    assert pass_hat_k(n=10, c=10, k=10) == 1.0
    # c < k -> impossible to draw k successes
    assert pass_hat_k(n=10, c=2, k=3) == 0.0
    # zero successes
    assert pass_hat_k(n=10, c=0, k=1) == 0.0


def test_pass_at_k_hand_computed():
    # 1 - C(5,3)/C(10,3) = 1 - 10/120
    assert pass_at_k(n=10, c=5, k=3) == pytest.approx(1 - 10 / 120)
    # n-c < k -> at least one success guaranteed in any draw of k
    assert pass_at_k(n=10, c=8, k=3) == 1.0
    assert pass_at_k(n=10, c=0, k=3) == 0.0


def test_passk_rejects_k_greater_than_n():
    with pytest.raises(ValueError):
        pass_hat_k(n=5, c=5, k=6)
    with pytest.raises(ValueError):
        pass_at_k(n=5, c=5, k=6)


def test_wilson_interval_hand_computed():
    # c=8, n=10, z=1.96: p=0.8, denom=1.38416,
    # center=(0.8+0.19208)/1.38416=0.716738...
    # margin=1.96*sqrt(0.016+0.009604)/1.38416=1.96*0.16/1.38416=0.226563...
    low, high = wilson_interval(c=8, n=10)
    assert low == pytest.approx(0.49018, abs=1e-4)
    assert high == pytest.approx(0.94330, abs=1e-4)


def test_wilson_interval_edges():
    low, high = wilson_interval(c=0, n=10)
    assert low == 0.0
    assert 0.0 < high < 0.35
    low, high = wilson_interval(c=10, n=10)
    assert 0.65 < low < 1.0
    assert high == 1.0
    assert wilson_interval(c=0, n=0) == (0.0, 1.0)
