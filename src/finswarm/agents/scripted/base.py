"""Base class and helpers for scripted (non-LLM) agents.

Scripted agents implement the ``Agent`` protocol. ``decide`` is ``async`` for protocol
compatibility but synchronous inside: it calls ``act(obs)``. All randomness comes from a
private ``numpy`` generator seeded by the runner (deterministic per run seed and slot) whose
state is part of ``state()`` so checkpoints restore bit-exactly.

Prices are floats in market units (``price = p_min + idx * tick``); the runner snaps them to
the live grid. Dealers in the one-sided Calvano market post an ask (``Decision.ask``), buyers a
bid (``Decision.bid``).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from finswarm.agents.base import AgentMemory
from finswarm.core.types import AgentKind, Decision, Observation, PeriodRecord

NAN = float("nan")


def make_generator(seed: int | Sequence[int] | None) -> np.random.Generator:
    if seed is None:
        seed = 0
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))


def finite(xs: Sequence[float]) -> list[float]:
    return [float(x) for x in xs if x is not None and math.isfinite(float(x))]


def rival_prices(obs: Observation) -> list[float]:
    """Rivals' latest prices: last history row first, else the standing rival quotes."""
    if obs.history:
        xs = finite(obs.history[-1].rival_prices)
        if xs:
            return xs
    return standing_rival_prices(obs)


def standing_rival_prices(obs: Observation) -> list[float]:
    """Rivals' standing quotes at decision time (ask for sellers, bid for buyers)."""
    p_min = float(obs.extra.get("p_min", 0.0))
    tick = float(obs.extra.get("tick_size", 0.0))
    out: list[float] = []
    for q in obs.rival_quotes:
        idx = q.bid_ticks if obs.role == "buyer" else q.ask_ticks
        if idx is not None and idx >= 0 and tick > 0:
            out.append(p_min + idx * tick)
    return out


def own_last_price(obs: Observation, fallback: float = NAN) -> float:
    if obs.history and math.isfinite(obs.history[-1].own_price):
        return float(obs.history[-1].own_price)
    return fallback


class ScriptedAgent:
    """Common plumbing: memory, private RNG, (de)serialisation, quote helper."""

    policy: str = "scripted"
    kind: AgentKind = AgentKind.SCRIPTED

    def __init__(
        self,
        agent_id: int,
        slot: int,
        *,
        name: str | None = None,
        role: str = "dealer",
        memory_periods: int = 100,
        seed: int | Sequence[int] | None = None,
    ) -> None:
        self.agent_id = int(agent_id)
        self.slot = int(slot)
        self.name = name or f"{self.policy}#{slot}"
        self.role = role
        self.memory = AgentMemory(memory_periods=memory_periods)
        self._rng = make_generator(seed)
        self.last_price = NAN
        self.n_decisions = 0

    # -- protocol ------------------------------------------------------------------------
    async def decide(self, obs: Observation) -> Decision:
        d = self.act(obs)
        self.n_decisions += 1
        self.memory.last_decision = d
        p = d.ask if d.ask is not None else (d.bid if d.bid is not None else d.price)
        if p is not None and math.isfinite(p):
            self.last_price = float(p)
        return d

    def act(self, obs: Observation) -> Decision:  # pragma: no cover - abstract
        raise NotImplementedError

    def record_period(self, record: PeriodRecord) -> None:
        self.memory.history.append(record)
        self.on_period(record)

    def on_period(self, record: PeriodRecord) -> None:
        """Hook for learners (called after accounting of each period)."""

    def state(self) -> dict[str, Any]:
        return {
            "memory": self.memory.to_state(),
            "rng": self._rng.bit_generator.state,
            "last_price": self.last_price,
            "n_decisions": self.n_decisions,
            "extra": self._extra_state(),
        }

    def restore(self, state: dict[str, Any]) -> None:
        self.memory = AgentMemory.from_state(state["memory"])
        self._rng.bit_generator.state = state["rng"]
        self.last_price = float(state["last_price"])
        self.n_decisions = int(state.get("n_decisions", 0))
        self._restore_extra(state.get("extra", {}))

    def _extra_state(self) -> dict[str, Any]:
        return {}

    def _restore_extra(self, st: dict[str, Any]) -> None:
        return None

    # -- helpers -------------------------------------------------------------------------
    def quote(self, price: float, *, plans: str = "", insights: str = "") -> Decision:
        price = float(price)
        if self.role == "buyer":
            return Decision(action="quote", price=price, bid=price, plans=plans, insights=insights)
        return Decision(action="quote", price=price, ask=price, plans=plans, insights=insights)

    def hold(self, plans: str = "") -> Decision:
        return Decision(action="hold", plans=plans)

    @staticmethod
    def clamp(price: float, lo: float, hi: float | None = None) -> float:
        price = max(float(lo), float(price))
        if hi is not None and math.isfinite(hi):
            price = min(float(hi), price)
        return price

    @staticmethod
    def ceiling(obs: Observation) -> float | None:
        v = obs.extra.get("price_max")
        return float(v) if v is not None else None
