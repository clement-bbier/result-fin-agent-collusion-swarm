"""Market protocol (frozen interface, 2026-10-09).

Every mechanism (dealer, fringe, lob, batch) implements this protocol and registers in
``finswarm.market.registry``. State must be fully serialisable for checkpoints.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from finswarm.core.types import MechanismKind, Order, OrderAck, Trade


@runtime_checkable
class Market(Protocol):
    kind: MechanismKind
    tick_size: float
    p_min: float
    tick: int  # current simulation tick

    def submit(self, order: Order) -> OrderAck: ...

    def cancel(self, order_id: int) -> bool: ...

    def advance(self, n_ticks: int) -> None:
        """Run background flow and matching for ``n_ticks`` ticks (pure given captured RNG)."""
        ...

    def best_bid(self) -> float | None: ...

    def best_ask(self) -> float | None: ...

    def mid(self) -> float | None: ...

    def snapshot(self, depth: int) -> list[tuple[float, float, float]]:
        """Top ``depth`` levels as (price, bid_depth, ask_depth)."""
        ...

    def drain_trades(self) -> list[Trade]:
        """Return and clear trades since the last drain (tagged-agent trades always included)."""
        ...

    def state(self) -> dict[str, Any]: ...

    def restore(self, state: dict[str, Any]) -> None: ...

    def migrate_from(self, other: Market) -> None:
        """Mid-run mechanism switch: build this market's state from ``other`` (pure)."""
        ...


def price_from_ticks(idx: int, p_min: float, tick_size: float) -> float:
    return p_min + idx * tick_size


def ticks_from_price(price: float, p_min: float, tick_size: float) -> int:
    return int(round((price - p_min) / tick_size))
