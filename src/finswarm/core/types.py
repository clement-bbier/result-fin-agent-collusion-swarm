"""Core value types shared by every module (frozen interface, 2026-10-09).

Prices inside markets are integer tick indices; floats appear only at the boundary
(``price = p_min + idx * tick``). Agent ids: swarm agents ``0..N-1``; tagged agents
(LLM and scripted) ``N..N+S-1`` with ``slot = agent_id - N``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from typing import Any


class Side(IntEnum):
    BUY = 1
    SELL = -1


class AgentKind(IntEnum):
    SWARM = 0
    LLM = 1
    SCRIPTED = 2


class OrderType(IntEnum):
    MARKET = 0
    LIMIT = 1
    QUOTE = 2
    CANCEL = 3
    REPLACE = 4


class MechanismKind(StrEnum):
    DEALER = "dealer"  # (A) posted quotes + logit routing of background flow
    FRINGE = "fringe"  # (B) dealers + public limit orders from a fringe of the swarm
    LOB = "lob"  # (C) continuous double auction, price-level book
    BATCH = "batch"  # (D) frequent batch auctions (uniform price) or pairwise midpoint


class SwarmType(IntEnum):
    ZI_BUYER = 0
    ZI_SELLER = 1
    INFORMED = 2
    NOISE = 3
    TREND = 4
    MM = 5


@dataclass(slots=True)
class Order:
    """An order from a tagged agent. Swarm orders never exist as Python objects."""

    order_id: int
    agent_id: int
    side: Side
    otype: OrderType
    price_ticks: int  # -1 for market orders
    size: float
    tif_ticks: int  # time in force in ticks; -1 = good till cancelled
    submit_tick: int
    apply_tick: int  # >= submit_tick (speed bump / modelled latency)


@dataclass(slots=True)
class OrderAck:
    accepted: bool
    order_id: int
    reason: str = ""


@dataclass(slots=True)
class Trade:
    trade_id: int
    tick: int
    price_ticks: int
    size: float
    buyer_id: int
    seller_id: int
    aggressor: int  # Side value of the aggressor, 0 for batch clearing
    mechanism: str  # MechanismKind value
    batch_id: int = -1
    buyer_value: float = float("nan")  # reservation value when known (swarm), for GS efficiency
    seller_cost: float = float("nan")


@dataclass(slots=True)
class Quote:
    """A dealer's two-sided (or one-sided) quote; -1 means no quote on that side."""

    agent_id: int
    bid_ticks: int = -1
    ask_ticks: int = -1
    bid_size: float = 0.0
    ask_size: float = 0.0


@dataclass(slots=True)
class PeriodRecord:
    """What a dealer learns about one past period (Fish et al. style history row)."""

    period: int
    own_price: float
    own_quantity: float
    own_profit: float
    rival_prices: tuple[float, ...]
    market_price: float  # mean traded price, NaN if none


@dataclass(slots=True)
class Observation:
    """What an agent sees at a decision epoch, AFTER the public-view policy was applied."""

    tick: int
    period: int
    agent_id: int
    slot: int
    role: str  # "dealer" | "seller" | "buyer" | "trader"
    inventory: float
    cash: float
    cost: float  # marginal cost (sellers/dealers) or value (buyers)
    history: list[PeriodRecord] = field(default_factory=list)
    rival_quotes: list[Quote] = field(default_factory=list)
    trades_since_epoch: list[Trade] = field(default_factory=list)
    book_levels: list[tuple[float, float, float]] = field(default_factory=list)  # (price, bid, ask)
    fundamental_hint: float | None = None
    notices: list[str] = field(default_factory=list)
    view_policy_id: str = "V0"
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Decision:
    """A parsed agent decision. ``action`` is one of quote|limit|market|cancel|hold."""

    action: str
    price: float | None = None
    size: float | None = None
    side: Side | None = None
    bid: float | None = None  # two-sided dealers
    ask: float | None = None
    plans: str = ""
    insights: str = ""
    message: str | None = None  # only when a text channel is enabled
    raw: str = ""
    parse_ok: bool = True
    repair_used: bool = False
    refusal: bool = False
