"""Market throughput benchmark (``finswarm bench`` delegates here).

Same protocol as ``finswarm.sim.bench`` / ``scripts/bench_swarm.py``: two dealers with fixed
asks 1.50 and 1.70, ``agents`` swarm agents (20 arrivals per tick on average), 100 ticks per
period, trades/flow/periods drained every period as the runner does, JIT warm-up excluded.

``mechanism="dealer"`` is mechanism A; ``mechanism="fringe"`` is the same dealer market with
the public limit-order fringe on (``market.fringe.share = FRINGE_SHARE`` unless
``fringe_share`` is given, mechanism B).
"""

from __future__ import annotations

import time
from typing import Any

ARRIVALS_PER_TICK = 20.0
TICKS_PER_PERIOD = 100
FRINGE_SHARE = 0.3


def run_bench(
    agents: int = 10_000,
    ticks: int = 100_000,
    mechanism: str = "dealer",
    fringe_share: float | None = None,
) -> dict[str, Any]:
    from finswarm.config.schema import RunConfig
    from finswarm.core.rng import RNGRegistry
    from finswarm.core.types import Order, OrderType, Side
    from finswarm.market.registry import make_market

    if mechanism not in ("dealer", "fringe"):
        raise NotImplementedError(f"bench: mechanism {mechanism!r} is not implemented")
    share = float(fringe_share) if fringe_share is not None else 0.0
    if mechanism == "fringe" and fringe_share is None:
        share = FRINGE_SHARE
    tpp = TICKS_PER_PERIOD
    cfg = RunConfig.model_validate(
        {
            "name": "bench",
            "ticks_per_period": tpp,
            "market": {"mechanism": "dealer", "fringe": {"share": share}},
            "swarm": {
                "mode": "agents",
                "n_agents": int(agents),
                "arrivals": {"lam_mean": ARRIVALS_PER_TICK / max(1, int(agents))},
            },
        }
    )
    m: Any = make_market(cfg, RNGRegistry(0))
    for slot, price in enumerate((1.50, 1.70)):
        idx = int(round((price - m.p_min) / m.tick_size))
        aid = m.dealer_agent_id(slot)
        m.submit(Order(slot + 1, aid, Side.SELL, OrderType.QUOTE, idx, 1.0, -1, 0, 0))
    m.advance(tpp)  # JIT warm-up
    m.drain_trades_arrays(), m.drain_flow(), m.drain_periods()
    n_periods = max(1, int(ticks) // tpp)
    n_trades = n_fringe = 0
    t0 = time.perf_counter()
    for _ in range(n_periods):
        m.advance(tpp)
        tr = m.drain_trades_arrays()
        n_trades += int(tr["tick"].shape[0])
        n_fringe += int((tr["seller_id"] < 0).sum())
        m.drain_flow()
        m.drain_periods()
    dt = time.perf_counter() - t0
    done = n_periods * tpp
    out = {
        "impl": "market.bench",
        "mechanism": mechanism,
        "fringe_share": share,
        "agents": int(agents),
        "ticks": done,
        "seconds": dt,
        "ticks_per_s": done / dt if dt > 0 else float("inf"),
        "trades": n_trades,
        "fringe_trades": n_fringe,
    }
    print(
        f"[{mechanism} market, fringe share {share:g}, {agents} swarm agents, "
        f"{ARRIVALS_PER_TICK:g} arrivals/tick] {done} ticks in {dt:.3f}s = "
        f"{out['ticks_per_s']:.0f} ticks/s ({n_trades} trades, {n_fringe} at the fringe)"
    )
    return out
