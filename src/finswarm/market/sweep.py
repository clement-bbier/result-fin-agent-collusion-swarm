"""Fixed-quote profits of the dealers: the market-side input of numerical benchmarks.

With the public fringe on (lever L2) the dealers face the residual demand left by the fringe,
and Calvano's closed-form Nash / joint-monopoly prices no longer apply. The competitive and
collusive benchmarks of such a market are the static Nash equilibrium and the symmetric
joint-profit maximum of the dealers' **one-shot pricing game in that market** (fringe share,
horizon, routing ``tau``, grid and swarm as in the window being evaluated), estimated by
``finswarm.metrics.benchmarks.estimate_by_sweeps(profit_fn(cfg), grid, n_agents)``.

:func:`fixed_quote_profits` evaluates one price vector: a fresh market with the run's seed
(common random numbers across price vectors), every dealer holding a fixed ask, a warm-up
long enough for the fringe depth to reach its stationary level, then the mean profit per
period of each dealer (``(price - c * alpha) * quantity``).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from finswarm.config.schema import RunConfig
from finswarm.core.rng import RNGRegistry
from finswarm.core.types import Order, OrderType, Side
from finswarm.market.dealer import DealerMarket


def fixed_quote_profits(
    cfg: RunConfig,
    prices: np.ndarray,
    *,
    periods: int = 50,
    warmup_periods: int | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Mean profit per period of each dealer when dealer ``i`` holds the ask ``prices[i]``.

    ``len(prices)`` sets the number of dealers; asks are snapped UP to the grid. The warm-up
    defaults to ``max(2, ceil(10 * fringe.horizon_ticks / ticks_per_period))`` periods.
    """
    p = np.asarray(prices, dtype=np.float64)
    n = int(p.shape[0])
    mcfg = cfg.model_copy(
        update={"market": cfg.market.model_copy(update={"n_dealers": n})}, deep=True
    )
    tpp = int(cfg.ticks_per_period)
    if warmup_periods is None:
        warmup_periods = max(2, -(-10 * int(cfg.market.fringe.horizon_ticks) // tpp))
    m = DealerMarket(mcfg, RNGRegistry(cfg.seed if seed is None else int(seed)))
    for i, price in enumerate(p.tolist()):
        k = int(np.ceil((price - m.p_min) / m.tick_size - 1e-9))
        k = min(max(k, 0), m.n_levels - 1)
        o = Order(i + 1, m.dealer_agent_id(i), Side.SELL, OrderType.QUOTE, k, 1.0, -1, 0, 0)
        ack = m.submit(o)
        if not ack.accepted:  # pragma: no cover - snapped onto the grid above
            raise ValueError(ack.reason)
    m.advance(warmup_periods * tpp)
    m.drain_trades_arrays(), m.drain_flow(), m.drain_periods()
    profit = np.zeros(n)
    for _ in range(int(periods)):
        m.advance(tpp)
        m.drain_trades_arrays(), m.drain_flow()
        for ps in m.drain_periods():
            profit += ps.profit
    return profit / max(1, int(periods))


def profit_fn(
    cfg: RunConfig,
    *,
    periods: int = 50,
    warmup_periods: int | None = None,
    seed: int | None = None,
) -> Callable[[np.ndarray], np.ndarray]:
    """``(m, n) -> (m, n)`` batch version of :func:`fixed_quote_profits` (CRN across rows),
    the ``profit_fn`` of ``finswarm.metrics.benchmarks.estimate_by_sweeps``."""

    def fn(batch: np.ndarray) -> np.ndarray:
        b = np.atleast_2d(np.asarray(batch, dtype=np.float64))
        return np.stack(
            [
                fixed_quote_profits(
                    cfg, row, periods=periods, warmup_periods=warmup_periods, seed=seed
                )
                for row in b
            ]
        )

    return fn
