import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from finswarm.config.schema import RunConfig
from finswarm.core.rng import RNGRegistry
from finswarm.core.types import Order, OrderType, Side
from finswarm.market.dealer import DealerMarket
from finswarm.market.demand import shares
from finswarm.swarm import kernels


def _cfg(
    n: int = 1000, lam_total: float = 20.0, tpp: int = 100, pure: bool = True, **mk: object
) -> RunConfig:
    swarm: dict[str, object] = {
        "mode": "agents",
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
    return RunConfig(name="agents", ticks_per_period=tpp, swarm=swarm, market=mk)


def _quote(m: DealerMarket, slot: int, price: float, oid: int) -> None:
    ticks = int(round((price - m.p_min) / m.tick_size))
    ack = m.submit(
        Order(
            oid, m.dealer_agent_id(slot), Side.SELL, OrderType.QUOTE, ticks, 1.0, -1, m.tick, m.tick
        )
    )
    assert ack.accepted, ack.reason


def _run(cfg: RunConfig, prices: list[float], ticks: int, seed: int = 0) -> DealerMarket:
    m = DealerMarket(cfg, RNGRegistry(seed))
    for k, p in enumerate(prices):
        _quote(m, k, p, k + 1)
    m.advance(ticks)
    return m


def test_lln_shares_match_closed_form_logit() -> None:
    cfg = _cfg()
    prices = [1.50, 1.70]
    m = _run(cfg, prices, 10_000)  # ~200k arrivals
    periods = m.drain_periods()
    qty = np.sum([p.qty for p in periods], axis=0)
    arrivals = sum(p.n_arrivals for p in periods)
    assert arrivals > 190_000
    realised = qty / arrivals
    expected = shares(prices, cfg.market)
    assert np.all(np.abs(realised / expected - 1.0) < 0.01), (realised, expected)
    assert m.dealer_agent_id(0) == 1000  # tagged ids follow the swarm


def test_tau_to_zero_is_price_priority() -> None:
    util, w = np.empty(3), np.empty(3)
    asks = np.array([50, 40, 45], dtype=np.int64)
    prices = 1.0 + asks * 0.01
    for tau in (0.0, 1e-9):
        for ur in np.linspace(0, 0.999, 7):
            assert kernels.route_choice(2.0, asks, prices, 0.0, tau, ur, util, w) == 1
        # nobody buys above the reservation value
        assert kernels.route_choice(1.3, asks, prices, 0.0, tau, 0.5, util, w) == -1
    tie = np.array([40, 40, -1], dtype=np.int64)
    picks = {
        kernels.route_choice(2.0, tie, 1.0 + tie * 0.01, 0.0, 0.0, u, util, w) for u in (0.1, 0.9)
    }
    assert picks == {0, 1}
    m = _run(_cfg(tpp=50, routing={"tau": 1e-9}), [1.55, 1.50], 200)
    q = np.sum([p.qty for p in m.drain_periods()], axis=0)
    assert q[0] == 0 and q[1] > 0


def test_chunking_invariance_bit_identical() -> None:
    cfg = _cfg(n=300, lam_total=8.0, tpp=7, pure=False)
    a = _run(cfg, [1.6, 1.7], 0)
    b = _run(cfg, [1.6, 1.7], 0)
    a.advance(500)
    for k in (1, 2, 3, 50, 1, 200, 243):
        b.advance(k)
    ta, tb = a.drain_trades_arrays(), b.drain_trades_arrays()
    for key in ta:
        assert np.array_equal(ta[key], tb[key]), key
    fa, fb = a.drain_flow(), b.drain_flow()
    for key in fa:
        assert np.array_equal(fa[key], fb[key]), key
    assert a.swarm is not None and b.swarm is not None
    for key, arr in a.swarm.state().items():
        assert np.array_equal(arr, b.swarm.state()[key]), key


def test_conservation_and_flow_tables() -> None:
    m = _run(_cfg(n=500, lam_total=5.0, tpp=10, pure=False), [1.6, 1.65], 300)
    sw = m.swarm
    assert sw is not None
    assert m.dealer_inv.sum() + sw.inventory.sum() == 0.0
    assert m.dealer_cash.sum() + sw.cash.sum() == pytest.approx(0.0, abs=1e-8)
    trades = m.drain_trades_arrays()
    flow = m.drain_flow()
    assert flow["buy_qty"].sum() == trades["size"].sum() == -m.dealer_inv.sum()
    assert set(np.unique(trades["seller_id"])) <= {500, 501}
    assert np.all(trades["buyer_id"] < 500) and np.all(np.isfinite(trades["buyer_value"]))
    assert len(flow["tick"]) == 2 * 300


def test_info_shock_raises_demand_with_attention_lag() -> None:
    cfg = _cfg(n=2000, lam_total=40.0, tpp=50)
    base = _run(cfg, [1.6, 1.6], 0, seed=3)
    shocked = _run(cfg, [1.6, 1.6], 0, seed=3)
    base.advance(100)
    shocked.advance(100)
    shocked.apply_info_shock(0.5, "attention")
    base.advance(500)
    shocked.advance(500)
    qb = [p.qty.sum() for p in base.drain_periods()]
    qs = [p.qty.sum() for p in shocked.drain_periods()]
    assert qb[:2] == qs[:2]  # identical before the shock (common random numbers)
    assert qs[-1] > qb[-1] * 1.02


def test_numpy_fallback_matches_numba(tmp_path: Path) -> None:
    here = str(Path(__file__).parent)
    code = (
        "import json,sys\n"
        f"sys.path.insert(0, {here!r})\n"
        "from test_dealer_agents import _cfg, _run\n"
        "m = _run(_cfg(n=200, lam_total=4.0, tpp=10, pure=False), [1.6, 1.7], 120)\n"
        "t = m.drain_trades_arrays()\n"
        "keys = ('tick', 'buyer_id', 'seller_id', 'price_ticks')\n"
        "print(json.dumps({k: t[k].tolist() for k in keys}))\n"
    )
    env = dict(os.environ, FINSWARM_NO_NUMBA="1")
    out = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
    )
    fallback = json.loads(out.stdout.strip().splitlines()[-1])
    m = _run(_cfg(n=200, lam_total=4.0, tpp=10, pure=False), [1.6, 1.7], 120)
    t = m.drain_trades_arrays()
    for k, v in fallback.items():
        assert t[k].tolist() == v, k
