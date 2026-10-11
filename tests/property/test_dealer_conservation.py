"""Property: the dealer market conserves cash and inventory between dealers and the swarm."""

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from finswarm.config.schema import RunConfig
from finswarm.core.rng import RNGRegistry
from finswarm.core.types import Order, OrderType, Side
from finswarm.market.dealer import DealerMarket

STEP = st.tuples(
    st.integers(0, 2),  # dealer slot
    st.integers(0, 300),  # ask in ticks (grid has 301 levels)
    st.integers(0, 3),  # speed bump
    st.integers(0, 25),  # ticks to advance
    st.booleans(),  # cancel instead of quoting
)


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    steps=st.lists(STEP, min_size=1, max_size=20),
    seed=st.integers(0, 2**16),
    mode=st.sampled_from(["agents", "closed_form"]),
    tpp=st.integers(1, 9),
)
def test_cash_and_inventory_are_conserved(
    steps: list[tuple[int, int, int, int, bool]], seed: int, mode: str, tpp: int
) -> None:
    cfg = RunConfig(
        name="prop",
        seed=seed,
        ticks_per_period=tpp,
        market={"n_dealers": 3},
        swarm={
            "mode": mode,
            "n_agents": 60,
            "arrivals": {"lam_mean": 0.1},
            "type_weights": {
                "ZI_BUYER": 0.5,
                "INFORMED": 0.2,
                "NOISE": 0.1,
                "TREND": 0.1,
                "MM": 0.1,
            },
        },
    )
    m = DealerMarket(cfg, RNGRegistry(seed))
    paid = np.zeros(3)
    sold = np.zeros(3)
    for oid, (slot, ticks, bump, n, cancel) in enumerate(steps):
        otype = OrderType.CANCEL if cancel else OrderType.QUOTE
        ack = m.submit(
            Order(
                oid,
                m.dealer_agent_id(slot),
                Side.SELL,
                otype,
                ticks,
                1.0,
                -1,
                m.tick,
                m.tick + bump,
            )
        )
        assert ack.accepted, ack.reason
        m.advance(n)
        tr = m.drain_trades_arrays()
        slots = tr["seller_id"] - m.agent_id_offset
        np.add.at(paid, slots, tr["price"] * tr["size"])
        np.add.at(sold, slots, tr["size"])
    sw = m.swarm
    swarm_inv = sw.inventory.sum() if sw is not None else 0.0
    swarm_cash = sw.cash.sum() if sw is not None else 0.0
    volume = float(np.abs(m.dealer_cash).sum()) + 1.0
    assert abs(m.dealer_inv.sum() + swarm_inv + m.bg_inv) <= 1e-9 * volume
    assert abs(m.dealer_cash.sum() + swarm_cash + m.bg_cash) <= 1e-9 * volume
    assert np.allclose(m.dealer_cash, paid, rtol=1e-9, atol=1e-9)
    assert np.allclose(-m.dealer_inv, sold, rtol=1e-9, atol=1e-9)
    if sw is not None:
        assert np.all(sw.inventory >= 0) and np.all(sw.inventory <= sw.max_inv)
    periods = m.drain_periods()
    assert [p.period for p in periods] == list(range(m.tick // tpp))
    assert np.allclose(sum((p.qty for p in periods), np.zeros(3)), sold[:] - _open_period(m))


def _open_period(m: DealerMarket) -> np.ndarray:
    """Quantity sold in the current (not yet closed) period."""
    return m.state()["period_acc"]["qty"]
