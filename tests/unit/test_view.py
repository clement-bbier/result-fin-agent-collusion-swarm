import math

import pytest

from finswarm.config.schema import ViewPolicy
from finswarm.core.rng import RNGRegistry
from finswarm.core.types import PeriodRecord, Quote, Trade
from finswarm.market.view import PublicView, RawView, view_hash

P_MIN, TICK = 1.0, 0.01


def _trade(i: int, tick: int, seller: int, buyer: int = 3) -> Trade:
    return Trade(i, tick, 47, 1.0, buyer, seller, 1, "dealer", buyer_value=2.1, seller_cost=1.0)


def _raw(tick: int = 6, period: int = 0) -> RawView:
    return RawView(
        tick=tick,
        period=period,
        viewer_id=100,
        rival_quotes=[
            (0, Quote(101, ask_ticks=50, ask_size=1.0)),
            (5, Quote(101, ask_ticks=47, ask_size=1.0)),
            (2, Quote(102, ask_ticks=60, ask_size=1.0)),
            (5, Quote(100, ask_ticks=40, ask_size=1.0)),  # the viewer's own quote
        ],
        trades=[_trade(1, 1, 101), _trade(2, 5, 102)],
        book=[
            (0, [(1.47, 0.0, 1.0)]),
            (5, [(1.40, 0.0, 1.0), (1.41, 0.0, 2.0), (1.42, 0.0, 3.0), (1.43, 0.0, 4.0)]),
        ],
        history=[PeriodRecord(k, 1.5, 40.0, 20.0, (1.6,), 1.55) for k in range(5)],
    )


def _view(seed: int = 0, **kw: object) -> PublicView:
    return PublicView(
        ViewPolicy(**kw), RNGRegistry(seed), p_min=P_MIN, tick_size=TICK, tagged_ids=[100, 101, 102]
    )


def test_identity_policy_and_private_fields() -> None:
    res = _view().apply(_raw())
    p = res.parts
    assert [(q.agent_id, q.ask_ticks) for q in p.rival_quotes] == [(101, 47), (102, 60)]
    assert len(p.trades) == 2 and all(math.isnan(t.buyer_value) for t in p.trades)
    assert len(p.book_levels) == 4 and len(p.history) == 5
    assert len(res.view_hash) == 64


def test_delay_aggregate_truncate_in_order() -> None:
    p = _view(delay_ticks=3).apply(_raw()).parts
    assert [(q.agent_id, q.ask_ticks) for q in p.rival_quotes] == [(101, 50), (102, 60)]
    assert [t.trade_id for t in p.trades] == [1]
    assert p.book_levels == [(1.47, 0.0, 1.0)]
    p = _view(aggregate_levels=2, depth_truncation=1).apply(_raw()).parts
    assert p.book_levels == [(pytest.approx(1.40), 0.0, 3.0)]
    p = _view(history_periods=2, show_trades=False, show_rival_quotes=False).apply(_raw()).parts
    assert [h.period for h in p.history] == [3, 4] and p.trades == [] and p.rival_quotes == []


def test_canonical_rounding() -> None:
    p = _view(price_round=0.05, size_round=1.0, canonicalise=True).apply(_raw()).parts
    assert [q.ask_ticks for q in p.rival_quotes] == [45, 60]
    assert all(t.price_ticks == 45 for t in p.trades)
    assert [lv[0] for lv in p.book_levels] == [1.4, 1.4, 1.4, 1.45]
    assert p.history[0].own_price == 1.5 and p.history[0].own_quantity == 40.0


def test_anonymisation_stable_and_reshuffled() -> None:
    v = _view(seed=3, anonymise_ids=True)
    a = v.apply(_raw(period=0)).parts
    b = v.apply(_raw(period=7)).parts
    ids_a = [q.agent_id for q in a.rival_quotes]
    assert ids_a == [q.agent_id for q in b.rival_quotes]  # stable within the run
    assert not {101, 102} & set(ids_a) and all(1000 <= i < 10_000 for i in ids_a)
    assert {t.buyer_id for t in a.trades} == {-1}  # swarm ids hidden
    same = _view(seed=3, anonymise_ids=True).apply(_raw(period=0)).parts
    assert ids_a == [q.agent_id for q in same.rival_quotes]
    r = _view(seed=3, anonymise_ids=True, reshuffle_every=2)
    seen = {
        tuple(q.agent_id for q in r.apply(_raw(period=k)).parts.rival_quotes)
        for k in range(0, 20, 2)
    }
    assert len(seen) > 1
    own = RawView(tick=6, period=0, viewer_id=100, trades=[_trade(9, 1, 100)])
    assert v.apply(own).parts.trades[0].seller_id == 100


def test_view_hash_is_content_addressed() -> None:
    h1 = _view().apply(_raw()).view_hash
    assert h1 == _view(seed=99).apply(_raw()).view_hash
    raw = _raw()
    raw.trades[0].size = 2.0
    assert _view().apply(raw).view_hash != h1
    assert _view(id="V1").apply(_raw()).view_hash != h1
    res = _view().apply(_raw())
    assert view_hash(res.parts, "V0") == res.view_hash
