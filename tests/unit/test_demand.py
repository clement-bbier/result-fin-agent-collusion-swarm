import numpy as np
import pytest

from finswarm.config.schema import MarketConfig
from finswarm.market.demand import (
    DemandParams,
    benchmarks,
    best_response_price,
    delta,
    delta_price,
    monopoly_price,
    nash_price,
    profits,
    quantities,
    shares,
)

BETA = 100.0


def test_golden_benchmarks_calvano_baseline() -> None:
    b = benchmarks(MarketConfig())  # a=2, a0=0, mu=0.25, c=1, alpha=1, beta=100, n=2
    assert b.p_nash == pytest.approx(1.4729, abs=1e-3)
    assert b.p_mono == pytest.approx(1.9250, abs=1e-3)
    assert b.pi_nash == pytest.approx(0.2229 * BETA, rel=1e-3)
    assert b.pi_mono == pytest.approx(0.3375 * BETA, rel=1e-3)


def test_benchmarks_solve_their_first_order_conditions() -> None:
    m = MarketConfig()
    b = benchmarks(m)
    # Nash: no unilateral deviation improves profit
    for eps in (1e-4, -1e-4):
        dev = profits([b.p_nash + eps, b.p_nash], m)[0]
        assert dev <= b.pi_nash + 1e-9
    # Monopoly: no common deviation improves joint profit
    for eps in (1e-4, -1e-4):
        joint = profits([b.p_mono + eps] * 2, m).sum()
        assert joint <= 2 * b.pi_mono + 1e-9
    assert best_response_price([b.p_nash], m) == pytest.approx(b.p_nash, abs=1e-9)


@pytest.mark.parametrize("alpha", [0.5, 1.0, 3.2, 10.0])
@pytest.mark.parametrize("beta", [1.0, 100.0, 250.0])
def test_exact_alpha_beta_scaling(alpha: float, beta: float) -> None:
    base = benchmarks(MarketConfig())
    m = MarketConfig(alpha=alpha, demand={"beta": beta})
    b = benchmarks(m)
    assert b.p_nash == pytest.approx(alpha * base.p_nash, rel=1e-9)
    assert b.p_mono == pytest.approx(alpha * base.p_mono, rel=1e-9)
    assert b.pi_nash == pytest.approx(alpha * beta / BETA * base.pi_nash, rel=1e-9)
    assert b.pi_mono == pytest.approx(alpha * beta / BETA * base.pi_mono, rel=1e-9)
    p = np.array([1.6, 1.8])
    assert np.allclose(quantities(alpha * p, m), beta / BETA * quantities(p, MarketConfig()))


def test_delta_at_1_7_alpha() -> None:
    for alpha in (1.0, 2.5):
        m = MarketConfig(alpha=alpha)
        b = benchmarks(m)
        pi = profits([1.7 * alpha] * 2, m)[0]
        assert float(delta(pi, b)) == pytest.approx(0.70933, abs=1e-4)
    b = benchmarks(MarketConfig())
    assert float(delta(b.pi_nash, b)) == 0.0
    assert float(delta(b.pi_mono, b)) == pytest.approx(1.0)
    assert float(delta_price(b.p_mono, b)) == pytest.approx(1.0)


def test_shares_handle_absent_dealers_and_batches() -> None:
    m = MarketConfig()
    s = shares([1.5, np.inf], m)
    assert s[1] == 0.0
    assert s[0] == pytest.approx(shares([1.5], DemandParams())[0])
    batch = shares(np.array([[1.5, 1.7], [1.7, 1.5]]), m)
    assert np.allclose(batch[0], batch[1][::-1])
    assert profits([1.5, np.inf], m)[1] == 0.0


def test_more_firms_and_monopoly_bisection_is_stable() -> None:
    m = MarketConfig(n_dealers=4)
    b = benchmarks(m)
    assert b.p_nash < b.p_mono
    assert nash_price(m) == pytest.approx(b.p_nash)
    assert monopoly_price(DemandParams(a=6.0, mu=0.05), n=2) > 1.0
