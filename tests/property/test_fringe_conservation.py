"""Property: with live re-grids and fringe switches, cash and inventory stay conserved between
dealers, the public fringe and the swarm; the fringe book keeps its invariants; standing asks
are always on the current grid and never move down at a re-grid."""

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from finswarm.config.schema import RunConfig
from finswarm.core.rng import RNGRegistry
from finswarm.core.types import Order, OrderType, Side
from finswarm.market.dealer import FRINGE_PATH, TICK_PATH, DealerMarket

TICKS = (0.01, 0.02, 0.05, 0.1, 0.125)
STEP = st.tuples(
    st.sampled_from(["quote", "cancel", "regrid", "fringe"]),
    st.integers(0, 2),  # dealer slot
    st.floats(0.0, 0.999),  # ask as a fraction of the grid / share / tick choice
    st.integers(0, 3),  # speed bump
    st.integers(0, 25),  # ticks to advance
)


@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    steps=st.lists(STEP, min_size=1, max_size=20),
    seed=st.integers(0, 2**16),
    horizon=st.integers(1, 12),
    tpp=st.integers(1, 9),
)
def test_conservation_with_fringe_and_regrids(
    steps: list[tuple[str, int, float, int, int]], seed: int, horizon: int, tpp: int
) -> None:
    cfg = RunConfig(
        name="prop_fringe",
        seed=seed,
        ticks_per_period=tpp,
        market={"n_dealers": 3, "fringe": {"share": 0.3, "horizon_ticks": horizon}},
        swarm={
            "mode": "agents",
            "n_agents": 60,
            "arrivals": {"lam_mean": 0.1},
            "type_weights": {"ZI_BUYER": 0.6, "INFORMED": 0.2, "NOISE": 0.1, "MM": 0.1},
        },
    )
    m = DealerMarket(cfg, RNGRegistry(seed))
    paid, sold = np.zeros(3), np.zeros(3)
    f_paid = f_sold = 0.0
    for oid, (action, slot, x, bump, n) in enumerate(steps):
        if action in ("quote", "cancel"):
            otype = OrderType.QUOTE if action == "quote" else OrderType.CANCEL
            k = int(x * m.n_levels)
            o = Order(
                oid, m.dealer_agent_id(slot), Side.SELL, otype, k, 1.0, -1, m.tick, m.tick + bump
            )
            assert m.submit(o).accepted
        elif action == "regrid":
            before = np.where(m.asks >= 0, m.p_min + m.asks * m.tick_size, np.nan)
            m.set_param(TICK_PATH, TICKS[int(x * len(TICKS))])
            after = np.where(m.asks >= 0, m.p_min + m.asks * m.tick_size, np.nan)
            live = ~np.isnan(after)
            assert np.all(after[live] >= before[live] - 1e-9)
        else:
            m.set_param(FRINGE_PATH, round(x, 2) if x > 0.5 else 0.0)
        m.advance(n)
        tr = m.drain_trades_arrays()
        fr = tr["seller_id"] == -2
        slots = tr["seller_id"][~fr] - m.agent_id_offset
        np.add.at(paid, slots, tr["price"][~fr] * tr["size"][~fr])
        np.add.at(sold, slots, tr["size"][~fr])
        f_paid += float((tr["price"][fr] * tr["size"][fr]).sum())
        f_sold += float(tr["size"][fr].sum())
        assert np.all(tr["price"][fr] >= m.marginal_cost - 1e-9)
        assert np.all((m.asks >= -1) & (m.asks < m.n_levels))
        if m.fringe is not None:
            m.fringe.check()
            assert m.fringe.depth.shape[0] == m.n_levels
    sw = m.swarm
    assert sw is not None
    volume = float(np.abs(m.dealer_cash).sum() + abs(m.fringe_cash)) + 1.0
    assert abs(m.dealer_inv.sum() + sw.inventory.sum() + m.fringe_inv) <= 1e-9 * volume
    assert abs(m.dealer_cash.sum() + sw.cash.sum() + m.fringe_cash) <= 1e-9 * volume
    assert np.allclose(m.dealer_cash, paid, rtol=1e-9, atol=1e-9)
    assert np.allclose(-m.dealer_inv, sold, rtol=1e-9, atol=1e-9)
    assert abs(m.fringe_cash - f_paid) <= 1e-9 * volume and -m.fringe_inv == f_sold
