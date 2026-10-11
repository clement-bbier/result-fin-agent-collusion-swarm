import numpy as np
import pytest

from finswarm.config.schema import RunConfig
from finswarm.core.rng import RNGRegistry
from finswarm.core.types import MechanismKind, Order, OrderType, Side
from finswarm.market.base import Market
from finswarm.market.dealer import DealerMarket
from finswarm.market.demand import profits, quantities
from finswarm.market.registry import REGISTRY, make_market


def _cfg(tpp: int = 1, **market: object) -> RunConfig:
    return RunConfig(name="t", ticks_per_period=tpp, market=market)


def _quote(m: DealerMarket, slot: int, price: float, oid: int, delay: int = 0) -> None:
    ticks = int(round((price - m.p_min) / m.tick_size))
    ack = m.submit(
        Order(
            oid,
            m.dealer_agent_id(slot),
            Side.SELL,
            OrderType.QUOTE,
            ticks,
            1.0,
            -1,
            m.tick,
            m.tick + delay,
        )
    )
    assert ack.accepted, ack.reason


def test_registry_and_protocol() -> None:
    m = make_market(_cfg(), RNGRegistry(0))
    assert isinstance(m, Market)
    assert isinstance(m, DealerMarket)
    assert REGISTRY[MechanismKind.DEALER] is DealerMarket
    with pytest.raises(NotImplementedError, match="lob"):
        make_market(_cfg(mechanism="lob"), RNGRegistry(0))
    with pytest.raises(NotImplementedError, match="two_sided"):
        make_market(_cfg(two_sided=True), RNGRegistry(0))


@pytest.mark.parametrize("tpp", [1, 5])
def test_closed_form_period_profit_equals_formula(tpp: int) -> None:
    cfg = _cfg(tpp)
    m = make_market(cfg, RNGRegistry(0))
    assert isinstance(m, DealerMarket)
    prices = [(1.47, 1.93), (1.70, 1.70), (1.55, 1.62)]
    for k, (p0, p1) in enumerate(prices):
        _quote(m, 0, p0, 2 * k)
        _quote(m, 1, p1, 2 * k + 1)
        m.advance(tpp)
    periods = m.drain_periods()
    assert [p.period for p in periods] == [0, 1, 2]
    total = np.zeros(2)
    for rec, pr in zip(periods, prices, strict=True):
        assert np.allclose(rec.profit, profits(list(pr), cfg.market), rtol=1e-12, atol=0)
        assert np.allclose(rec.qty, quantities(list(pr), cfg.market), rtol=1e-12, atol=0)
        assert np.allclose(rec.ask, pr)
        total += rec.revenue
    trades = m.drain_trades()
    assert len(trades) == 6
    assert all(t.buyer_id == -1 and t.tick % tpp == tpp - 1 for t in trades)
    assert np.allclose(m.dealer_cash, total)
    assert m.dealer_inv.sum() + m.bg_inv == pytest.approx(0.0, abs=1e-9)
    assert m.dealer_cash.sum() + m.bg_cash == pytest.approx(0.0, abs=1e-9)


def test_speed_bump_and_order_validation() -> None:
    m = make_market(_cfg(), RNGRegistry(0))
    assert isinstance(m, DealerMarket)
    _quote(m, 0, 1.50, 1)
    _quote(m, 1, 1.50, 2)
    _quote(m, 0, 1.80, 3, delay=3)  # applies at tick 3
    m.advance(5)
    asks = [p.ask[0] for p in m.drain_periods()]
    assert asks == pytest.approx([1.5, 1.5, 1.5, 1.8, 1.8])
    bad = [
        Order(9, 999, Side.SELL, OrderType.QUOTE, 10, 1.0, -1, 5, 5),  # not a dealer
        Order(10, 0, Side.BUY, OrderType.QUOTE, 10, 1.0, -1, 5, 5),  # bid on one-sided
        Order(11, 0, Side.SELL, OrderType.QUOTE, 10_000, 1.0, -1, 5, 5),  # off grid
        Order(12, 0, Side.SELL, OrderType.LIMIT, 10, 1.0, -1, 5, 5),  # wrong type
        Order(13, 0, Side.SELL, OrderType.QUOTE, 10, 0.0, -1, 5, 5),  # zero size
        Order(14, 0, Side.SELL, OrderType.QUOTE, 10, 1.0, -1, 5, 4),  # apply < submit
    ]
    assert not any(m.submit(o).accepted for o in bad)


def test_cancel_pending_and_live_quote_and_tif() -> None:
    m = make_market(_cfg(), RNGRegistry(0))
    assert isinstance(m, DealerMarket)
    _quote(m, 0, 1.50, 1)
    _quote(m, 0, 1.90, 2, delay=2)
    assert m.cancel(2)
    m.advance(1)
    assert m.best_ask() == pytest.approx(1.50)
    assert m.cancel(1)
    assert m.best_ask() is None and not m.cancel(1)
    m.submit(Order(3, 0, Side.SELL, OrderType.QUOTE, 60, 2.0, 2, m.tick, m.tick))
    m.advance(1)
    assert m.snapshot(5) == [(pytest.approx(1.60), 0.0, 2.0)]
    m.advance(1)
    assert m.best_ask() is None  # expired after tif_ticks = 2
    m.submit(Order(4, 1, Side.SELL, OrderType.QUOTE, 60, 1.0, -1, m.tick, m.tick))
    m.submit(Order(5, 1, Side.SELL, OrderType.CANCEL, -1, 0.0, -1, m.tick, m.tick + 2))
    m.advance(1)
    assert m.best_ask() == pytest.approx(1.60)
    m.advance(1)
    assert m.best_ask() is None


def test_info_shock_and_set_param_closed_form() -> None:
    cfg = _cfg()
    m = make_market(cfg, RNGRegistry(0))
    assert isinstance(m, DealerMarket)
    _quote(m, 0, 1.6, 1)
    _quote(m, 1, 1.6, 2)
    m.advance(1)
    m.apply_info_shock(0.5)
    m.advance(1)
    q0, q1 = (p.qty[0] for p in m.drain_periods())
    assert q1 > q0
    assert m.set_param("market.routing.tau", 0.1) is None and m.tau == 0.1
    with pytest.raises(KeyError):
        m.set_param("market.price_grid.min", 0.5)
