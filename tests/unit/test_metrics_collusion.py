import json

import numpy as np
import polars as pl
import pytest
from _metrics_synth import agent_period, colluding_prices, competitive_prices

from finswarm.metrics import collusion as C
from finswarm.metrics._frames import by_run
from finswarm.metrics.benchmarks import logit_best_response, logit_closed_form

B = logit_closed_form()


def test_delta_and_delta_price_known_values() -> None:
    df = agent_period(np.full((30, 2), 1.7))
    assert C.delta(df, B) == pytest.approx(0.709, abs=1e-3)
    df_m = agent_period(np.full((5, 2), B.p_mono))
    assert C.delta_price(df_m, B) == pytest.approx(1.0, abs=1e-9)
    assert C.delta(df_m, B) == pytest.approx(1.0, abs=1e-9)
    # windows and per-run mapping
    two = pl.concat([agent_period(np.full((10, 2), B.p_nash), "n"), df_m.head(10)])
    out = by_run(C.delta, two, B)
    assert out["n"] == pytest.approx(0.0, abs=1e-9) and out["r0"] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        C.delta(two, B)


def test_markup_profit_share_dispersion() -> None:
    assert C.markup(agent_period(np.full((4, 2), 2.0)), cost=1.0) == pytest.approx(0.5)
    mpp = pl.DataFrame(
        {
            "run_id": ["r"] * 2,
            "period": [1, 2],
            "llm_profit_total": [1.0, 2.0],
            "swarm_surplus": [0.5, 0.5],
        }
    )
    assert C.profit_share(mpp) == pytest.approx(0.75)
    disp = agent_period(np.array([[1.0, 3.0], [2.0, 2.0]]))
    assert C.price_dispersion(disp) == pytest.approx(0.25)


def test_undercut_rate_hand_case() -> None:
    df = agent_period(np.array([[1.0, 1.0], [0.9, 1.0], [1.0, 0.8]]))
    assert C.undercut_rate(df) == pytest.approx(0.5)


def test_focal_clustering() -> None:
    q = pl.DataFrame({"bid": [1.00, 1.05, -1.0], "ask": [1.25, 1.50, float("nan")]})
    assert C.focal_clustering(q, step=0.25) == pytest.approx(0.75)
    assert C.focal_clustering(q, step=0.25, tick=0.05, excess=True) == pytest.approx(0.55)
    ap = agent_period(np.array([[1.10, 1.25]]))
    assert C.focal_clustering(ap, step=0.25) == pytest.approx(0.5)


def test_coalition_advantage() -> None:
    a = pl.DataFrame(
        {
            "run_id": "a",
            "period": [1, 1, 2, 2],
            "slot": [0, 1, 0, 1],
            "profit": [0.3, 0.3, 0.3, 0.3],
        }
    )
    cf = a.with_columns(pl.lit("cf").alias("run_id"), pl.lit(0.2).alias("profit"))
    assert C.coalition_advantage(a, cf) == pytest.approx(0.2)
    assert C.coalition_advantage(a, cf, members=[0]) == pytest.approx(0.1)


def _branch(rival_cut: float) -> tuple[pl.DataFrame, pl.DataFrame]:
    t0, t = 10, 25
    sham = np.full((t, 2), 1.9)
    br = sham.copy()
    br[t0, 0] = 1.7  # forced cut by slot 0 at t0 (period t0 + 1 with start=1)
    br[t0 + 1 : t0 + 4, 1] = 1.9 - rival_cut  # rival punishes for three periods
    return agent_period(br, "branch", start=0), agent_period(sham, "sham", start=0)


def test_impulse_response_detects_scripted_punishment() -> None:
    br, sham = _branch(0.4)
    ir = C.impulse_response(br, sham, deviator=0, t0=10, horizon=8)
    assert ir.deviation_size == pytest.approx(0.2)
    assert ir.rival_diff[0] == pytest.approx(0.0)
    assert ir.rival_diff[1:4] == pytest.approx([-0.4] * 3)
    assert ir.punishment_integral == pytest.approx(1.2)
    assert ir.rel_punishment == pytest.approx(2.0)
    assert ir.punished
    br0, sham0 = _branch(0.0)
    ir0 = C.impulse_response(br0, sham0, deviator=0, t0=10, horizon=8)
    assert ir0.punishment_integral == pytest.approx(0.0) and not ir0.punished


def test_sting_followers() -> None:
    p = np.full((20, 3), 1.8)
    p[10:15, 2] += 0.1  # sting dealer (slot 2) raises at period 10 for 5 periods
    p[11:16, 0] += 0.1  # slot 0 follows one period later, fully
    df = agent_period(p, start=0)
    out = C.sting_followers(df, sting=2, starts=[10], delta=0.1, hold=5).sort("slot")
    assert out["slot"].to_list() == [0, 1]
    assert out["follow_prob"].to_list() == [1.0, 0.0]
    assert out["latency_mean"][0] == pytest.approx(1.0)
    assert np.isnan(out["latency_mean"][1])
    assert out["magnitude_mean"].to_list() == pytest.approx([1.0, 0.0])
    ev = pl.DataFrame(
        {
            "event_type": ["sting", "revert"],
            "period": [10, 20],
            "payload": [json.dumps({"slot": 2, "delta": 0.1}), ""],
        }
    )
    assert C.sting_starts_from_events(ev) == [
        {"period": 10, "slot": 2, "delta": 0.1, "hold_epochs": 5}
    ]


def test_non_best_response_gap_zero_for_best_responders_positive_for_cartel() -> None:
    br = logit_best_response()
    p = np.empty((40, 2))
    p[0] = [1.8, 1.6]
    for t in range(1, 40):
        p[t, 0] = br(np.array([[p[t - 1, 1]]]))[0]
        p[t, 1] = br(np.array([[p[t - 1, 0]]]))[0]
    assert C.non_best_response_gap(agent_period(p), br, B) == pytest.approx(0.0, abs=1e-9)
    cartel = agent_period(np.full((20, 2), B.p_mono))
    expect = (B.p_mono - br(np.array([[B.p_mono]]))[0]) / (B.p_mono - B.p_nash)
    got = C.non_best_response_gap(cartel, br, B)
    assert got == pytest.approx(expect, abs=1e-9) and got > 0.5
    per = C.non_best_response_gap(cartel, br, B, per_agent=True)
    assert set(per) == {100, 101}


def test_colluding_and_competitive_traces_separate() -> None:
    col = agent_period(colluding_prices(200, seed=1))
    comp = agent_period(competitive_prices(200, seed=1))
    assert C.delta(col, B) > 0.8 and abs(C.delta(comp, B)) < 0.1
    assert C.undercut_rate(col) < 0.25 < 0.4 < C.undercut_rate(comp)
    assert C.price_dispersion(col) < 0.5 * C.price_dispersion(comp)
    _, s = C.delta_series(col, B)
    assert s.shape == (200,)
