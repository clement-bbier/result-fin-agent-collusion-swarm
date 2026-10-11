"""Levers L3 (live tick re-grid) and L2 (public limit-order fringe, mechanism B) on the dealer
market: rounding rules, determinism and checkpoint round trips across a re-grid, fringe
semantics (share 0 is mechanism A bit for bit, flow capture, expiry, logit routing LLN),
closed-form refusal, and numba / ``FINSWARM_NO_NUMBA=1`` parity."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from finswarm.config.schema import RunConfig
from finswarm.core.rng import RNGRegistry
from finswarm.core.types import Order, OrderType, Side
from finswarm.market.dealer import FRINGE_PATH, TICK_PATH, DealerMarket
from finswarm.market.fringe import FringeBook
from finswarm.market.grid import nearest_indices, regrid_indices, regrid_price
from finswarm.sim.integration import UnsupportedIntervention
from finswarm.swarm import kernels
from finswarm.swarm.state import SwarmState


def _cfg(
    n: int = 500,
    lam_total: float = 6.0,
    tpp: int = 10,
    tick: float = 0.01,
    share: float = 0.0,
    horizon: int = 50,
    pure: bool = False,
    mode: str = "agents",
    **mk: object,
) -> RunConfig:
    swarm: dict[str, object] = {
        "mode": mode,
        "n_agents": n,
        "arrivals": {"lam_mean": lam_total / n},
    }
    if pure:
        swarm.update(
            value_private_sd=0.0,
            signal_noise_sd=0.0,
            sentiment_weight=0.0,
            type_weights={"ZI_BUYER": 1.0},
        )
    market = {
        "n_dealers": 3,
        "price_grid": {"min": 1.0, "max": 4.0, "tick": tick},
        "fringe": {"share": share, "horizon_ticks": horizon},
        **mk,
    }
    return RunConfig(name="levers", ticks_per_period=tpp, swarm=swarm, market=market)


def _quote(m: DealerMarket, slot: int, price: float, oid: int, delay: int = 0) -> None:
    ticks = int(round((price - m.p_min) / m.tick_size))
    aid = m.dealer_agent_id(slot)
    o = Order(oid, aid, Side.SELL, OrderType.QUOTE, ticks, 1.0, -1, m.tick, m.tick + delay)
    ack = m.submit(o)
    assert ack.accepted, ack.reason


def _market(cfg: RunConfig, prices=(1.47, 1.62, 1.80), seed: int = 0) -> DealerMarket:
    m = DealerMarket(cfg, RNGRegistry(seed))
    for k, p in enumerate(prices):
        _quote(m, k, p, k + 1)
    m.advance(0)  # quotes due now are in force
    return m


def _same(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> None:
    assert a.keys() == b.keys()
    for key in a:
        assert np.array_equal(a[key], b[key], equal_nan=True), key


# ------------------------------------------------------------------------- L3: re-grid
def test_grid_rounding_rules() -> None:
    idx = np.array([47, 50, 52, -1, 300])
    assert regrid_indices(idx, 0.01, 0.05, 61, up=True).tolist() == [10, 10, 11, -1, 60]
    assert regrid_indices(idx, 0.01, 0.05, 61, up=False).tolist() == [9, 10, 10, -1, 60]
    assert regrid_indices(np.array([9, 10]), 0.05, 0.01, 301, up=True).tolist() == [45, 50]
    # an ask whose rounded level falls off the grid has no image
    assert regrid_indices(np.array([299]), 0.01, 0.125, 24, up=True).tolist() == [-1]
    assert regrid_price(1.47, 1.0, 0.01, 0.05, 61, up=True) == pytest.approx(1.50)
    assert regrid_price(1.47, 1.0, 0.01, 0.05, 61, up=False) == pytest.approx(1.45)
    assert math.isnan(regrid_price(float("nan"), 1.0, 0.01, 0.05, 61, up=True))
    assert nearest_indices(np.array([47, 52, -1]), 0.01, 0.05, 61).tolist() == [9, 10, -1]


@pytest.mark.parametrize("mode", ["agents", "closed_form"])
def test_regrid_requantises_quotes_up_and_reports(mode: str) -> None:
    m = _market(_cfg(mode=mode))
    _quote(m, 2, 1.93, 9, delay=5)  # pending (speed bump) on the old grid
    assert TICK_PATH in m.settable_params
    assert m.set_param(TICK_PATH, 0.05) == pytest.approx(0.01)
    assert m.tick_size == 0.05 and m.n_levels == 61 and m.p_min == 1.0
    asks = [m.p_min + k * m.tick_size for k in m.asks]
    assert asks == pytest.approx([1.50, 1.65, 1.80])  # never below the old ask
    rep = m.last_param_report
    assert rep is not None and rep["kind"] == "regrid" and rep["old_tick"] == 0.01
    assert [(d["old_price"], d["new_price"]) for d in rep["dealers"]] == pytest.approx(
        [(1.47, 1.50), (1.62, 1.65), (1.80, 1.80)]
    )
    assert rep["pending"] == [{"order_id": 9, "old_ticks": 93, "new_ticks": 19}]
    m.advance(10)
    assert m.best_ask() == pytest.approx(1.50)
    assert m.p_min + m.asks[2] * m.tick_size == pytest.approx(1.95)  # 1.93 -> 1.95 applied
    # finer grid: exact, and back to the original tick
    m.set_param(TICK_PATH, 0.01)
    assert [m.p_min + k * m.tick_size for k in m.asks] == pytest.approx([1.50, 1.65, 1.95])
    with pytest.raises(ValueError):
        m.set_param(TICK_PATH, 0.0)


def test_regrid_keeps_undrained_trades_on_their_grid() -> None:
    m = _market(_cfg())
    m.advance(30)
    before = m.state()["trades"]["price"].copy()
    m.set_param(TICK_PATH, 0.05)
    m.advance(30)
    tr = m.drain_trades_arrays()
    k = before.shape[0]
    assert k > 0 and np.allclose(tr["price"][:k], 1.0 + before * 0.01)
    assert np.allclose(np.round((tr["price"][k:] - 1.0) / 0.05) * 0.05 + 1.0, tr["price"][k:])
    assert np.array_equal(tr["trade_id"], np.arange(tr["trade_id"].shape[0]))


def _capture(m: DealerMarket) -> tuple[dict, dict, dict]:
    assert m.swarm is not None
    return m.state(), m.swarm.state(), m.rng.capture()


def _rebuild(cfg: RunConfig, seed: int, snap: tuple[dict, dict, dict]) -> DealerMarket:
    rng = RNGRegistry(seed)
    m = DealerMarket(cfg, rng, SwarmState.from_state(snap[1]))
    m.restore(snap[0])
    rng.restore(snap[2])
    return m


@pytest.mark.parametrize("share", [0.0, 0.3])
def test_regrid_determinism_and_checkpoint_round_trip(share: float) -> None:
    cfg = _cfg(share=share, horizon=7)
    a, b = _market(cfg), _market(cfg)
    a.advance(100)  # 10 periods on tick 0.01
    b.advance(40)
    b.advance(60)
    _same(a.drain_trades_arrays(), b.drain_trades_arrays())
    for m in (a, b):
        m.set_param(TICK_PATH, 0.05)
    snap = _capture(a)
    c = _rebuild(cfg, 0, snap)  # built from the ORIGINAL config (tick 0.01), state re-gridded
    assert c.tick_size == 0.05 and c.n_levels == 61
    for m in (a, b, c):
        m.advance(80)
    ta = a.drain_trades_arrays()
    assert ta["tick"].shape[0] > 0
    _same(ta, b.drain_trades_arrays())
    _same(ta, c.drain_trades_arrays())
    _same(a.drain_flow(), c.drain_flow())
    assert [p.to_dict() for p in a.drain_periods()] == [p.to_dict() for p in c.drain_periods()]
    if share > 0:
        assert (ta["seller_id"] == -2).any()
        assert a.fringe is not None and c.fringe is not None
        a.fringe.check()
        assert a.fringe.stats() == c.fringe.stats()


# ----------------------------------------------------------------------- L2: the fringe
def test_fringe_share_zero_is_mechanism_a_bit_for_bit() -> None:
    cfg = _cfg(n=400, lam_total=8.0, tpp=7)
    a = _market(cfg)
    b = _market(cfg)
    assert b.set_param(FRINGE_PATH, 0.0) == 0.0 and b.fringe is None
    c = _market(cfg)
    c.set_param(FRINGE_PATH, 0.3)
    c.set_param(FRINGE_PATH, 0.0)  # allocated, empty, off
    for m in (a, b, c):
        m.advance(350)
    ta = a.drain_trades_arrays()
    _same(ta, b.drain_trades_arrays())
    _same(ta, c.drain_trades_arrays())
    fa, fc = a.drain_flow(), c.drain_flow()
    _same(fa, b.drain_flow())
    dealer_rows = fc["dealer_id"] >= 0
    _same(fa, {k: v[dealer_rows] for k, v in fc.items()})
    assert not fc["buy_qty"][~dealer_rows].any() and not fc["sell_qty"][~dealer_rows].any()
    assert [p.to_dict() for p in a.drain_periods()] == [p.to_dict() for p in c.drain_periods()]


def test_fringe_captures_flow_dealer_profits_fall_and_conservation() -> None:
    off = _market(_cfg(n=1000, lam_total=20.0, tpp=100), prices=(1.80, 1.85, 1.90))
    on = _market(_cfg(n=1000, lam_total=20.0, tpp=100, share=0.3), prices=(1.80, 1.85, 1.90))
    for m in (off, on):
        m.advance(2000)
    p_off, p_on = off.drain_periods(), on.drain_periods()
    t_on = on.drain_trades_arrays()
    fr = t_on["seller_id"] == -2
    assert fr.sum() > 0.15 * fr.shape[0]
    assert sum(p.fringe_qty for p in p_on) == fr.sum()
    assert sum(p.fringe_revenue for p in p_on) == pytest.approx(t_on["price"][fr].sum())
    assert np.array_equal(t_on["seller_cost"][fr], t_on["price"][fr])
    assert np.all(t_on["price"][fr] >= on.marginal_cost - 1e-12)
    assert sum(p.profit.sum() for p in p_on) < 0.8 * sum(p.profit.sum() for p in p_off)
    sw = on.swarm
    assert sw is not None and on.fringe is not None
    assert on.dealer_inv.sum() + sw.inventory.sum() + on.fringe_inv == 0.0
    total = on.dealer_cash.sum() + sw.cash.sum() + on.fringe_cash
    assert total == pytest.approx(0.0, abs=1e-6)
    on.fringe.check()
    flow = on.drain_flow()
    rows = flow["dealer_id"] == -2
    assert rows.sum() == 2000 and flow["buy_qty"][rows].sum() == fr.sum()
    assert flow["sell_qty"][rows].sum() == on.fringe.stats()["posted"]
    snap = on.snapshot(500)
    levels = dict((round(p, 9), a) for p, _, a in snap)
    for price, lots in on.fringe_levels():
        assert levels[round(price, 9)] >= lots
    best = on.best_ask()
    assert best is not None and best <= 1.80 + 1e-12


def test_fringe_wheel_expires_after_horizon_and_fifo_take() -> None:
    fb = FringeBook.empty(horizon=4, n_levels=10)
    args = (fb.depth, fb.wheel, fb.meta, fb.st)
    # kernel order per tick: expire(t), then the posts of tick t
    assert kernels.fringe_expire(10, *args) == 0
    kernels.fringe_post(10, 3, *args)  # can be hit during ticks 10..13
    assert kernels.fringe_expire(11, *args) == 0
    kernels.fringe_post(11, 3, *args)  # ticks 11..14
    kernels.fringe_post(11, 5, *args)
    fb.check()
    assert fb.best == 3 and fb.resting == 3
    for t in (12, 13):
        assert kernels.fringe_expire(t, *args) == 0
    kernels.fringe_take(13, 3, fb.depth, fb.wheel, fb.meta, fb.st)  # consumes the tick-10 lot
    fb.check()
    assert kernels.fringe_expire(14, fb.depth, fb.wheel, fb.meta, fb.st) == 0  # already taken
    assert kernels.fringe_expire(15, fb.depth, fb.wheel, fb.meta, fb.st) == 2
    fb.check()
    assert fb.resting == 0 and fb.best == -1
    assert fb.stats()["filled"] == 1 and fb.stats()["expired"] == 2


def test_fringe_depth_expires_in_the_market_and_switches_off() -> None:
    m = _market(_cfg(share=0.3, horizon=3), prices=(1.20, 1.25, 1.30))
    m.advance(200)
    fb = m.fringe
    assert fb is not None
    fb.check()
    st = fb.stats()
    assert st["expired"] > 0 and st["posted"] == st["filled"] + st["expired"] + st["resting"]
    m.set_param(FRINGE_PATH, 0.0)
    m.advance(3)  # no new lots; everything resting expires within the horizon
    assert fb.resting == 0 and fb.best == -1 and m.fringe_levels() == []
    posted = fb.stats()["posted"]
    m.advance(50)
    assert fb.stats()["posted"] == posted
    assert not (m.drain_trades_arrays()["seller_id"] == -2)[-10:].any()


def test_lln_routing_kernel_over_dealers_fringe_and_outside() -> None:
    asks = np.array([50, 70, -1], dtype=np.int64)
    prices = 1.0 + asks * 0.01
    v, f_price, u0, tau = 2.0, 1.60, 0.0, 0.25
    util, w = np.empty(3), np.empty(3)
    n = 100_000
    counts = np.zeros(5)
    for ur in np.random.default_rng(1).random(n):
        d = kernels.route_choice_fringe(v, asks, prices, f_price, u0, tau, float(ur), util, w)
        counts[d if d >= 0 else 4] += 1
    e = np.exp(np.array([v - 1.5, v - 1.7, -np.inf, v - f_price, u0]) / tau)
    expected = e / e.sum()
    assert counts[2] == 0
    se = np.sqrt(expected * (1 - expected) / n)
    assert np.all(np.abs(counts / n - expected) < 5 * se + 1e-12), (counts / n, expected)
    # tau -> 0: exact price priority, the fringe included
    assert kernels.route_choice_fringe(2.0, asks, prices, 1.45, 0.0, 0.0, 0.7, util, w) == 3
    picks = {
        kernels.route_choice_fringe(2.0, asks, prices, 1.5, 0.0, 0.0, u, util, w)
        for u in (0.1, 0.9)
    }
    assert picks == {0, 3}


@pytest.mark.skipif(
    os.environ.get("FINSWARM_NO_NUMBA", "") not in ("", "0", "false"),
    reason="180k arrivals: numba only (the kernel itself is covered in both modes)",
)
def test_lln_market_shares_over_dealers_and_fringe_match_logit() -> None:
    cfg = _cfg(n=1000, lam_total=20.0, tpp=100, horizon=10_000, pure=True)
    m = _market(cfg, prices=(1.50, 1.70, 3.90))
    m.set_param(FRINGE_PATH, 0.3)
    m.set_param(FRINGE_PATH, 0.0)  # book allocated, no posting: fixed fringe ask below
    fb = m.fringe
    assert fb is not None
    # 1.60, deep enough never to deplete, in the wheel slot that expires last (tick H - 1)
    lv, lots, s = 60, 10**9, fb.horizon - 1
    fb.wheel[s, lv] = fb.depth[lv] = lots
    fb.meta[s] = (lots, lv, lv)
    fb.st[:] = (lv, lots, lots, 0, 0)
    m.advance(9000)
    periods = m.drain_periods()
    arrivals = sum(p.n_arrivals for p in periods)
    assert arrivals > 170_000
    realised = np.array(
        [*np.sum([p.qty for p in periods], axis=0), sum(p.fringe_qty for p in periods)]
    )
    u = np.array([2.0 - 1.50, 2.0 - 1.70, 2.0 - 3.90, 2.0 - 1.60]) / m.tau
    expected = np.exp(u) / (np.exp(u).sum() + np.exp(m.u0 / m.tau))
    share = realised / arrivals
    assert np.all(np.abs(share[[0, 1, 3]] / expected[[0, 1, 3]] - 1.0) < 0.02), (share, expected)


def test_fringe_refused_outside_agents_mode() -> None:
    m = _market(_cfg(mode="closed_form"))
    assert FRINGE_PATH not in m.settable_params
    with pytest.raises(UnsupportedIntervention, match="agents"):
        m.set_param(FRINGE_PATH, 0.3)
    assert m.set_param(FRINGE_PATH, 0.0) == 0.0  # switching it "off" is harmless
    with pytest.raises(UnsupportedIntervention, match="agents"):
        DealerMarket(_cfg(mode="closed_form", share=0.3), RNGRegistry(0))
    with pytest.raises(ValueError):
        _market(_cfg()).set_param(FRINGE_PATH, 1.5)


def test_fringe_checkpoint_round_trip_mid_run() -> None:
    cfg = _cfg(share=0.3, horizon=9, tpp=7)
    a = _market(cfg)
    a.advance(53)  # mid-period, with resting depth and an open period
    snap = _capture(a)
    b = _rebuild(cfg, 0, snap)
    for m in (a, b):
        m.set_param(FRINGE_PATH, 0.1)
        m.advance(61)
    _same(a.drain_trades_arrays(), b.drain_trades_arrays())
    _same(a.drain_flow(), b.drain_flow())
    assert a.fringe is not None and b.fringe is not None
    assert a.fringe.stats() == b.fringe.stats()


def test_numpy_fallback_matches_numba_with_fringe_and_regrid() -> None:
    here = str(Path(__file__).parent)
    code = (
        "import json,sys\n"
        f"sys.path.insert(0, {here!r})\n"
        "from test_levers_market import scenario\n"
        "print(json.dumps(scenario()))\n"
    )
    env = dict(os.environ, FINSWARM_NO_NUMBA="1")
    out = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
    )
    assert json.loads(out.stdout.strip().splitlines()[-1]) == scenario()


def test_fixed_quote_profits_crn_and_fringe_lowers_dealer_profit() -> None:
    from finswarm.market.sweep import fixed_quote_profits, profit_fn

    base = _cfg(n=500, lam_total=10.0, tpp=20, horizon=20)
    p = np.array([1.60, 1.70])
    a = fixed_quote_profits(base, p, periods=10)
    assert np.array_equal(a, fixed_quote_profits(base, p, periods=10))  # CRN, deterministic
    assert a.shape == (2,) and a[0] > a[1] > 0
    batch = profit_fn(base, periods=10)(np.array([[1.60, 1.70], [1.70, 1.60]]))
    assert np.array_equal(batch[0], a) and batch.shape == (2, 2)
    on = fixed_quote_profits(
        _cfg(n=500, lam_total=10.0, tpp=20, horizon=20, share=0.3), p, periods=10
    )
    assert np.all(on < a)


def test_bench_fringe_mode_runs() -> None:
    from finswarm.market.bench import run_bench

    dealer = run_bench(agents=500, ticks=2_000, mechanism="dealer")
    fringe = run_bench(agents=500, ticks=2_000, mechanism="fringe")
    assert dealer["fringe_trades"] == 0 and fringe["fringe_trades"] > 0
    assert fringe["fringe_share"] == 0.3 and fringe["ticks"] == dealer["ticks"] == 2_000


def scenario() -> dict[str, list]:
    """Fringe on, re-grid, fringe off: trades as lists (compared across JIT modes)."""
    m = _market(_cfg(n=200, lam_total=4.0, share=0.3, horizon=5))
    m.advance(40)
    m.set_param(TICK_PATH, 0.05)
    m.advance(40)
    m.set_param(FRINGE_PATH, 0.0)
    m.advance(20)
    t = m.drain_trades_arrays()
    keys = ("tick", "buyer_id", "seller_id", "price_ticks")
    return {k: t[k].tolist() for k in keys}
