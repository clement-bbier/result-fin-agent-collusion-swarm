"""Primary estimand (incumbents' Delta_p), frozen-quote mechanical decomposition and D7."""

import math

import numpy as np
import polars as pl
import pytest
from _metrics_synth import market_frame as frame

from finswarm.metrics import collusion as C
from finswarm.metrics.benchmarks import LogitParams, MarketModel, d7_cartel, logit_closed_form

TRUNK = MarketModel(LogitParams(beta=100.0, tau=0.25), n_incumbents=4)
INC = [0, 1, 2, 3]


def sham_prices(t: int = 40, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.array([1.80, 1.85, 1.90, 1.95]) + rng.normal(0.0, 0.02, (t, 4))


def test_incumbents_delta_p_ignores_entrant_slot() -> None:
    tb = TRUNK.benchmarks()
    p = np.full((40, 4), 1.90)
    lever = MarketModel(TRUNK.params, n_incumbents=4, n_entrants=1)
    with_entrant = frame(p, lever, entrant=np.full(40, 1.10))  # entrant undercuts hard
    alone = frame(p, TRUNK)
    expect = (1.90 - tb.p_nash) / (tb.p_mono - tb.p_nash)
    got = C.delta_price_incumbents(with_entrant, tb, INC, 111, 150)
    assert got == pytest.approx(expect, abs=1e-12)
    assert got == pytest.approx(C.delta_price_incumbents(alone, tb, INC, 111, 150), abs=1e-12)
    # averaging every slot (the old shortcut) would let the entrant drag Delta_p down
    assert C.delta_price(with_entrant, tb, 111, 150, key="slot") < got - 0.1
    # a whitelisted incumbent that never quotes is an error, never silently dropped
    with pytest.raises(KeyError):
        C.delta_price_incumbents(alone.filter(pl.col("slot") != 3), tb, INC, 111, 150)
    with pytest.raises(ValueError):
        C.delta_price_incumbents(alone, tb, [], 111, 150)


def test_trunk_window_30_and_branch_window_40_periods() -> None:
    tb = TRUNK.benchmarks()
    p = np.vstack([np.full((30, 4), 1.70), np.full((10, 4), 1.50), np.full((40, 4), 1.90)])
    f = frame(p, TRUNK, start=71)  # periods 71..150
    trunk = C.delta_price_incumbents(f, tb, INC, *C.TRUNK_WINDOW)
    branch = C.delta_price_incumbents(f, tb, INC, *C.BRANCH_WINDOW)
    assert trunk == pytest.approx((1.70 - tb.p_nash) / (tb.p_mono - tb.p_nash))
    assert branch == pytest.approx((1.90 - tb.p_nash) / (tb.p_mono - tb.p_nash))


def test_volume_weighted_price_and_profit_estimand() -> None:
    tb = TRUNK.benchmarks()
    p = sham_prices()
    f = frame(p, TRUNK)
    q = TRUNK.quantities(p)
    vw = (p * q).sum(axis=1) / q.sum(axis=1)
    expect = np.mean((vw - tb.p_nash) / (tb.p_mono - tb.p_nash))
    assert C.delta_price_incumbents(f, tb, INC, weight="volume") == pytest.approx(expect)
    pi = TRUNK.profits(p).mean(axis=1)
    assert C.delta_profit_incumbents(f, tb, INC) == pytest.approx(np.mean(tb.delta(pi)))


def test_mechanical_decomposition_reviewer_example_l1() -> None:
    """Sham quotes frozen, L1 tau 0.25 -> 0.0625: nonzero mechanical Delta C, zero
    behavioural Delta C (the dealers did not change a single quote)."""
    tb = TRUNK.benchmarks()
    lever = MarketModel(LogitParams(beta=100.0, tau=0.0625), n_incumbents=4)
    lb = lever.benchmarks()
    p = sham_prices()
    sham_f, lever_f = frame(p, TRUNK), frame(p, lever)  # identical quotes, new routing

    def measures(f: pl.DataFrame, window_bench) -> dict[str, float]:
        return {
            "delta_p": C.delta_price_incumbents(f, tb, INC),
            "delta_p_window": C.delta_price_incumbents(f, window_bench, INC),
            "delta_p_vw": C.delta_price_incumbents(f, tb, INC, weight="volume"),
            "delta_pi": C.delta_profit_incumbents(f, tb, INC),
            "delta_pi_window": C.delta_profit_incumbents(f, window_bench, INC),
        }

    s, lv = measures(sham_f, tb), measures(lever_f, lb)
    total = {k: s[k] - lv[k] for k in s}
    dec = C.mechanical_delta_c(sham_f, lever, tb, members=INC, start=111, end=150)
    assert dec.n_periods == 40
    # the primary estimand (posted prices, trunk benchmarks) is mechanically inert ...
    assert dec.mechanical["delta_p"] == pytest.approx(0.0, abs=1e-12)
    # ... while the profit, window and volume-weighted variants move mechanically
    for k in ("delta_pi", "delta_pi_window", "delta_p_window", "delta_p_vw"):
        assert abs(dec.mechanical[k]) > 0.01, k
        assert dec.mechanical[k] == pytest.approx(total[k], abs=1e-9), k
    beh = dec.behavioural(total)
    assert set(beh) == set(C.MECHANICAL_MEASURES)
    assert all(abs(v) < 1e-9 for v in beh.values())
    # a genuine behavioural cut on top of the routing change shows up as behavioural
    cut = frame(p - 0.10, lever)
    total_cut = {"delta_p": s["delta_p"] - C.delta_price_incumbents(cut, tb, INC)}
    assert dec.behavioural(total_cut)["delta_p"] == pytest.approx(
        0.10 / (tb.p_mono - tb.p_nash), rel=1e-9
    )
    # matrix input and an explicit sham market give the same answer
    dec2 = C.mechanical_delta_c(p, lever.to_dict(), tb, sham_market_params=TRUNK)
    assert dec2.mechanical == pytest.approx(dec.mechanical)


def test_mechanical_decomposition_fringe_and_entrant() -> None:
    tb = TRUNK.benchmarks()
    p = sham_prices()
    l2 = MarketModel(TRUNK.params, n_incumbents=4, fringe_share=0.3)
    d2 = C.mechanical_delta_c(p, l2, tb)
    pi = TRUNK.profits(p).mean(axis=1)
    expect = np.mean(tb.delta(pi)) - np.mean(tb.delta(0.7 * pi))
    assert d2.mechanical["delta_pi"] == pytest.approx(expect, rel=1e-9) and expect > 0
    assert d2.mechanical["delta_p"] == pytest.approx(0.0, abs=1e-12)
    assert d2.mechanical["delta_p_vw"] == pytest.approx(0.0, abs=1e-12)  # flow diversion
    l6 = MarketModel(TRUNK.params, n_incumbents=4, n_entrants=1)
    with pytest.raises(ValueError):
        C.mechanical_delta_c(p, l6, tb)  # the entrant's quotes are needed
    d6 = C.mechanical_delta_c(p, l6, tb, entrant_quotes=np.full(40, 1.6))
    assert d6.mechanical["delta_pi"] > 0.05 and d6.mechanical["delta_p"] == pytest.approx(0.0)
    # the per-window benchmark of L6 counts the entrant (5 symmetric sellers)
    assert l6.benchmarks().p_nash == pytest.approx(
        logit_closed_form(n=5, mu=0.25, beta=100.0).p_nash
    )
    # L3b: the coarse tick re-rounds frozen quotes, a mechanical price change
    l3b = MarketModel(TRUNK.params, n_incumbents=4, tick=0.125)
    assert abs(C.mechanical_delta_c(p, l3b, tb).mechanical["delta_p"]) > 0


def test_d7_cartel_values() -> None:
    assert d7_cartel(LogitParams(), 2) == pytest.approx(0.5824, abs=1e-4)
    four = d7_cartel(TRUNK)
    assert four == pytest.approx(0.6213, abs=1e-4)
    from finswarm.config.schema import MarketConfig

    assert d7_cartel(MarketConfig(n_dealers=4)) == pytest.approx(four, abs=1e-9)


def test_d7_price_form_cartel_best_responders_and_permutation() -> None:
    tb = TRUNK.benchmarks()
    br = TRUNK.best_response()
    cartel = frame(np.full((40, 4), tb.p_mono), TRUNK)
    r = C.d7_price(cartel, br, tb, INC, d7_cartel=d7_cartel(TRUNK), n_perm=50)
    assert r.d7 == pytest.approx(d7_cartel(TRUNK), abs=1e-9)
    assert r.ratio == pytest.approx(1.0, abs=1e-9)
    # every gap positive: z is at its ceiling sqrt(#blocks) = sqrt(8) for 39 gaps
    assert r.z_perm > 2.5 and r.p_perm < 0.05
    # best responders to last period's rivals: D7 = 0 and the null has no spread
    p = np.empty((40, 4))
    p[0] = [1.8, 1.6, 1.7, 1.9]
    for t in range(1, 40):
        for i in range(4):
            p[t, i] = br(np.delete(p[t - 1], i)[None, :])[0]
    r0 = C.d7_price(frame(p, TRUNK), br, tb, INC, n_perm=50)
    assert r0.d7 == pytest.approx(0.0, abs=1e-9) and math.isnan(r0.z_perm)
    # an entrant is a rival for the best response but never an analysed member
    lever = MarketModel(TRUNK.params, n_incumbents=4, n_entrants=1)
    ent = frame(np.full((40, 4), tb.p_mono), lever, entrant=np.full(40, 1.5))
    r_ent = C.d7_price(ent, br, tb, INC, n_perm=0)
    expect = (tb.p_mono - br(np.array([[tb.p_mono] * 3 + [1.5]]))[0]) / (tb.p_mono - tb.p_nash)
    assert r_ent.d7 == pytest.approx(expect, abs=1e-9)
    # noisy best responders sit inside the sign-flip null; a markup over BR does not
    rng = np.random.default_rng(3)

    def play(markup: float) -> np.ndarray:
        q = np.full((80, 4), 1.6)
        for t in range(1, 80):
            for i in range(4):
                q[t, i] = br(np.delete(q[t - 1], i)[None, :])[0] + markup
            q[t] += rng.normal(0.0, 0.03, 4)
        return q

    noisy = C.d7_price(frame(play(0.0), TRUNK), br, tb, INC, n_perm=400, seed=1)
    above = C.d7_price(frame(play(0.08), TRUNK), br, tb, INC, n_perm=400, seed=1)
    assert noisy.n_perm == 400 and abs(noisy.z_perm) < 3.0
    assert above.z_perm > 3.0 and above.p_perm < 0.01 and above.d7 > noisy.d7
