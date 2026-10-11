"""``finswarm bench``: throughput of the real market with its background swarm.

Delegates to ``finswarm.market.bench.run_bench(agents=, ticks=, mechanism=)`` when the market
package provides one; otherwise runs the same protocol as ``scripts/bench_swarm.py``: two
dealers with fixed asks, ``agents`` swarm agents (20 arrivals per tick on average), 100 ticks
per period, trades/flow/periods drained every period as the runner does, JIT warm-up excluded.
"""

from __future__ import annotations

import time
from typing import Any

ARRIVALS_PER_TICK = 20.0
TICKS_PER_PERIOD = 100


def run_bench(
    agents: int = 10_000, ticks: int = 100_000, mechanism: str = "dealer"
) -> dict[str, Any]:
    import importlib

    try:
        market_bench = importlib.import_module("finswarm.market.bench").run_bench
    except ImportError:
        market_bench = None
    if market_bench is not None:
        return market_bench(agents=agents, ticks=ticks, mechanism=mechanism)
    from finswarm.config.schema import RunConfig
    from finswarm.core.rng import RNGRegistry
    from finswarm.core.types import Order, OrderType, Side
    from finswarm.market.registry import make_market

    tpp = TICKS_PER_PERIOD
    cfg = RunConfig.model_validate(
        {
            "name": "bench",
            "ticks_per_period": tpp,
            "market": {"mechanism": mechanism},
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
    n_trades = 0
    t0 = time.perf_counter()
    for _ in range(n_periods):
        m.advance(tpp)
        n_trades += int(m.drain_trades_arrays()["tick"].shape[0])
        m.drain_flow()
        m.drain_periods()
    dt = time.perf_counter() - t0
    done = n_periods * tpp
    out = {
        "impl": "market.registry",
        "mechanism": mechanism,
        "agents": int(agents),
        "ticks": done,
        "seconds": dt,
        "ticks_per_s": done / dt if dt > 0 else float("inf"),
        "trades": n_trades,
    }
    print(
        f"[{mechanism} market, {agents} swarm agents, {ARRIVALS_PER_TICK:g} arrivals/tick] "
        f"{done} ticks in {dt:.3f}s = {out['ticks_per_s']:.0f} ticks/s ({n_trades} trades)"
    )
    return out
