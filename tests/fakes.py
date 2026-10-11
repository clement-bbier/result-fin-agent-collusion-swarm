"""Test doubles for the runner: a closed-form logit two-dealer market.

``FakeMarket`` implements the ``finswarm.market.base.Market`` protocol with the Calvano et al.
(2020) demand (Fish et al. 2024 alpha scaling): every tick a customer mass ``beta`` splits over
the dealers' standing asks with probabilities ``exp((a - p_i/alpha)/mu) / (sum_j ... +
exp(a0/mu))``. It is deliberately independent of ``finswarm.sim`` (no shared code) and has no
``set_tick`` hook, so a tick change goes through ``migrate_from``.
"""

from __future__ import annotations

import copy
import math
from typing import Any

from finswarm.config.schema import RunConfig
from finswarm.core.types import MechanismKind, Order, OrderAck, OrderType, Side, Trade


class FakeMarket:
    kind = MechanismKind.DEALER

    def __init__(self, cfg: RunConfig, rngs: Any = None) -> None:
        m = cfg.market
        self.p_min = float(m.price_grid.min)
        self.tick_size = float(m.price_grid.tick)
        self.alpha = float(m.alpha)
        self.a, self.a0, self.mu = m.demand.a, m.demand.a0, m.demand.mu
        self.beta = m.demand.beta / max(1, cfg.ticks_per_period)
        self.cost = m.cost * m.alpha
        self.tick = 0
        self.quotes: dict[int, tuple[int, int]] = {}  # agent -> (ask_ticks, order_id)
        self.pending: list[Order] = []
        self.trades: list[Trade] = []
        self.next_trade_id = 0
        self.submitted = 0

    def _price(self, idx: int) -> float:
        return self.p_min + idx * self.tick_size

    def demand(self, prices: list[float]) -> list[float]:
        w = [math.exp((self.a - p / self.alpha) / self.mu) for p in prices]
        denom = sum(w) + math.exp(self.a0 / self.mu)
        return [self.beta * x / denom for x in w]

    def submit(self, order: Order) -> OrderAck:
        self.submitted += 1
        if order.otype != OrderType.QUOTE or order.side != Side.SELL:
            return OrderAck(False, order.order_id, "fake market: sell quotes only")
        if order.apply_tick > self.tick:
            self.pending.append(order)
        else:
            self.quotes[order.agent_id] = (order.price_ticks, order.order_id)
        return OrderAck(True, order.order_id)

    def cancel(self, order_id: int) -> bool:
        for aid, (_, oid) in list(self.quotes.items()):
            if oid == order_id:
                del self.quotes[aid]
                return True
        return False

    def advance(self, n_ticks: int) -> None:
        for _ in range(n_ticks):
            for o in [o for o in self.pending if o.apply_tick <= self.tick]:
                self.quotes[o.agent_id] = (o.price_ticks, o.order_id)
            self.pending = [o for o in self.pending if o.apply_tick > self.tick]
            ids = sorted(self.quotes)
            qty = self.demand([self._price(self.quotes[i][0]) for i in ids])
            for aid, q in zip(ids, qty, strict=True):
                self.trades.append(
                    Trade(
                        trade_id=self.next_trade_id,
                        tick=self.tick,
                        price_ticks=self.quotes[aid][0],
                        size=q,
                        buyer_id=-1,
                        seller_id=aid,
                        aggressor=1,
                        mechanism="dealer",
                        seller_cost=self.cost,
                    )
                )
                self.next_trade_id += 1
            self.tick += 1

    def best_bid(self) -> float | None:
        return None

    def best_ask(self) -> float | None:
        return min((self._price(q[0]) for q in self.quotes.values()), default=None)

    def mid(self) -> float | None:
        return None

    def snapshot(self, depth: int) -> list[tuple[float, float, float]]:
        asks = sorted({q[0] for q in self.quotes.values()})[:depth]
        return [
            (self._price(k), 0.0, float(sum(1 for q in self.quotes.values() if q[0] == k)))
            for k in asks
        ]

    def drain_trades(self) -> list[Trade]:
        out, self.trades = self.trades, []
        return out

    def state(self) -> dict[str, Any]:
        return copy.deepcopy(
            {
                "tick": self.tick,
                "tick_size": self.tick_size,
                "quotes": self.quotes,
                "pending": self.pending,
                "trades": self.trades,
                "next_trade_id": self.next_trade_id,
            }
        )

    def restore(self, state: dict[str, Any]) -> None:
        st = copy.deepcopy(state)
        self.tick, self.tick_size = st["tick"], st["tick_size"]
        self.quotes, self.pending = st["quotes"], st["pending"]
        self.trades, self.next_trade_id = st["trades"], st["next_trade_id"]

    def migrate_from(self, other: Any) -> None:
        """Take ``other``'s state onto this market's grid (quotes snap to the new tick)."""
        st = other.state()
        old_tick, new_tick = st["tick_size"], self.tick_size
        self.restore(st)
        self.tick_size = new_tick
        self.quotes = {
            aid: (round((idx * old_tick) / new_tick), oid)
            for aid, (idx, oid) in self.quotes.items()
        }
        for o in self.pending:
            o.price_ticks = round((o.price_ticks * old_tick) / new_tick)


def fake_market_factory(cfg: RunConfig, rngs: Any = None) -> FakeMarket:
    return FakeMarket(cfg, rngs)


def base_config(**over: Any) -> RunConfig:
    """A small 2-dealer run on the fake market (scripted mock dealers, 30 periods)."""
    data: dict[str, Any] = {
        "name": "t",
        "periods": 30,
        "seed": 0,
        "market": {"price_grid": {"min": 1.0, "max": 4.0, "tick": 0.01}},
        "llm": {
            "mode": "mock",
            "lane": "mock",
            "slots": [
                {
                    "name": "A",
                    "model": "scripted",
                    "family": "mock",
                    "policy": "tit_for_tat(target=1.9, trigger=0.01, punish=3)",
                },
                {
                    "name": "B",
                    "model": "scripted",
                    "family": "mock",
                    "policy": "undercutter(eps=0.02, start=1.9, floor=1.4729)",
                },
            ],
        },
        "checkpoint": {"every_periods": 10, "keep_last": 2},
        "logging": {"snapshot_every": 5, "snapshot_levels": 4},
    }
    for k, v in over.items():
        data[k] = v
    return RunConfig.model_validate(data)
