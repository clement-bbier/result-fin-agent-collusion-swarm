import numpy as np
import polars as pl
import pytest
from _metrics_synth import agent_period

from finswarm.metrics.benchmarks import (
    Benchmarks,
    estimate_by_sweeps,
    logit_best_response,
    logit_closed_form,
    logit_profits,
    scripted_competitive,
)


def test_closed_form_golden_values() -> None:
    b = logit_closed_form(a=2, a0=0, mu=0.25, c=1, alpha=1, n=2, beta=1)
    assert b.p_nash == pytest.approx(1.4729, abs=1e-3)
    assert b.p_mono == pytest.approx(1.9250, abs=1e-3)
    assert b.pi_nash == pytest.approx(0.2229, abs=1e-3)
    assert b.pi_mono == pytest.approx(0.3375, abs=1e-3)
    # six-digit golden values shared with sim-core (finswarm.market.demand)
    assert (b.p_nash, b.pi_nash) == pytest.approx((1.472927, 0.222927), abs=1e-6)
    assert (b.p_mono, b.pi_mono) == pytest.approx((1.924981, 0.337490), abs=1e-6)
    # profits scale with beta and alpha * beta; prices with alpha
    b2 = logit_closed_form(alpha=3.2, beta=5.0)
    assert b2.p_nash == pytest.approx(3.2 * b.p_nash, rel=1e-9)
    assert b2.pi_mono == pytest.approx(3.2 * 5.0 * b.pi_mono, rel=1e-9)


def test_delta_at_1_7_alpha_is_0_709() -> None:
    b = logit_closed_form()
    pi = logit_profits([1.7, 1.7])[0]
    assert b.delta(pi) == pytest.approx(0.70933, abs=1e-5)
    assert b.delta(b.pi_nash) == pytest.approx(0.0, abs=1e-12)
    assert b.delta_price(b.p_mono) == pytest.approx(1.0, abs=1e-12)


def test_best_response_fixed_point_is_nash() -> None:
    b = logit_closed_form()
    br = logit_best_response()
    assert br(np.array([[b.p_nash]]))[0] == pytest.approx(b.p_nash, abs=1e-9)
    # best response to monopoly pricing undercuts it
    assert br(np.array([b.p_mono]))[0] < b.p_mono


def test_sweeps_match_closed_form_within_3pct() -> None:
    b = logit_closed_form()
    grid = np.round(np.arange(1.30, 2.10, 0.0025), 4)
    s = estimate_by_sweeps(lambda p: logit_profits(p), grid, n_agents=2, n_starts=5, seed=1)
    assert s.method == "sweep"
    assert s.params["n_converged"] == 5
    for k in ("p_nash", "p_mono", "pi_nash", "pi_mono"):
        assert getattr(s, k) == pytest.approx(getattr(b, k), rel=0.03)

    # strictly one-dimensional profit functions are batched automatically; full joint argmax
    def one_dim(p: np.ndarray) -> np.ndarray:
        if np.ndim(p) != 1:
            raise ValueError("1-D only")
        return logit_profits(p)

    s2 = estimate_by_sweeps(one_dim, grid[::8], 2, n_starts=2, symmetric_monopoly=False)
    assert s2.p_mono == pytest.approx(b.p_mono, rel=0.03)


def test_scripted_competitive_and_json_roundtrip(tmp_path) -> None:
    t = 50
    prices = np.full((t, 2), 1.5)
    df = pl.concat([agent_period(prices, "a"), agent_period(prices + 0.1, "b")])
    cb = scripted_competitive(df)
    assert cb.n_runs == 2
    assert cb.p_comp == pytest.approx(1.55)
    expected_pi = 0.5 * (logit_profits([1.5, 1.5])[0] + logit_profits([1.6, 1.6])[0])
    assert cb.pi_comp == pytest.approx(expected_pi)
    b = logit_closed_form().with_competitive(cb)
    assert b.pi_nash == pytest.approx(expected_pi)
    sha = b.save(tmp_path / "bench.json")
    b3 = Benchmarks.load(tmp_path / "bench.json")
    assert b3.pi_nash == b.pi_nash and len(sha) == 64


def test_local_solver_matches_sim_core_when_available() -> None:
    local = logit_closed_form(n=3, alpha=3.2, beta=100.0, solver="local")
    assert local.method == "closed_form:local"
    md = pytest.importorskip("finswarm.market.demand")
    core = logit_closed_form(n=3, alpha=3.2, beta=100.0)
    assert core.method == "closed_form:market.demand"
    for k in ("p_nash", "p_mono", "pi_nash", "pi_mono"):
        assert getattr(core, k) == pytest.approx(getattr(local, k), rel=1e-9)
    assert md is not None
