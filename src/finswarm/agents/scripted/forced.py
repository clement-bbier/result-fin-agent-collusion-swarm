"""Forced deviation: an impulse-response probe (Calvano et al. 2020, section 5.2 style).

``ForcedDeviationWrapper`` overrides one agent's price with ``base - delta`` for ``periods``
periods (``base`` = the agent's last own price when the deviation starts), forwards period
records to the wrapped agent so its memory shows what it actually did, and then reports
``done`` so the runner hands control back. The wrapped agent is not consulted while forced
(no LLM calls are spent on overridden decisions).
"""

from __future__ import annotations

import math
from typing import Any

from finswarm.agents.base import Agent, AgentMemory
from finswarm.agents.scripted.base import NAN, rival_prices
from finswarm.core.types import AgentKind, Decision, Observation, PeriodRecord


class ForcedDeviationWrapper:
    policy = "forced_deviation"
    skip_llm_log = True

    def __init__(self, inner: Agent, *, delta: float, periods: int = 1, event_id: int = -1) -> None:
        self.inner = inner
        self.delta = float(delta)
        self.periods = int(periods)
        self.event_id = int(event_id)
        self.left = int(periods)
        self.forced_price = NAN

    # protocol attributes delegate to the wrapped agent
    @property
    def agent_id(self) -> int:
        return self.inner.agent_id

    @property
    def slot(self) -> int:
        return self.inner.slot

    @property
    def kind(self) -> AgentKind:
        return self.inner.kind

    @property
    def name(self) -> str:
        return self.inner.name

    @property
    def role(self) -> str:
        return self.inner.role

    @property
    def memory(self) -> AgentMemory:
        return self.inner.memory

    @property
    def done(self) -> bool:
        return self.left <= 0

    def _base_price(self, obs: Observation) -> float:
        if obs.history and math.isfinite(obs.history[-1].own_price):
            return float(obs.history[-1].own_price)
        last = getattr(self.inner, "last_price", NAN)
        if isinstance(last, float) and math.isfinite(last):
            return last
        xs = rival_prices(obs)
        return sum(xs) / len(xs) if xs else obs.cost

    async def decide(self, obs: Observation) -> Decision:
        if not math.isfinite(self.forced_price):
            self.forced_price = max(obs.cost, self._base_price(obs) - self.delta)
        p = self.forced_price
        if self.role == "buyer":
            return Decision(action="quote", price=p, bid=p, plans="forced_deviation")
        return Decision(action="quote", price=p, ask=p, plans="forced_deviation")

    def record_period(self, record: PeriodRecord) -> None:
        self.inner.record_period(record)
        self.left -= 1

    def state(self) -> dict[str, Any]:
        return {"inner": self.inner.state(), "left": self.left, "forced_price": self.forced_price}

    def restore(self, state: dict[str, Any]) -> None:
        self.inner.restore(state["inner"])
        self.left = int(state["left"])
        self.forced_price = float(state["forced_price"])
