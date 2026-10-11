"""Agent protocol and memory (frozen interface, 2026-10-09)."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from finswarm.core.types import AgentKind, Decision, Observation, PeriodRecord


@dataclass
class AgentMemory:
    """Fish et al. style memory: overwritten scratchpads plus a bounded history window."""

    memory_periods: int = 100
    plans: str = ""
    insights: str = ""
    history: deque[PeriodRecord] = field(default_factory=lambda: deque(maxlen=100))
    notices: list[str] = field(default_factory=list)
    last_decision: Decision | None = None

    def __post_init__(self) -> None:
        if self.history.maxlen != self.memory_periods:
            self.history = deque(self.history, maxlen=self.memory_periods)

    def to_state(self) -> dict[str, Any]:
        return {
            "memory_periods": self.memory_periods,
            "plans": self.plans,
            "insights": self.insights,
            "history": [
                {
                    "period": r.period,
                    "own_price": r.own_price,
                    "own_quantity": r.own_quantity,
                    "own_profit": r.own_profit,
                    "rival_prices": list(r.rival_prices),
                    "market_price": r.market_price,
                }
                for r in self.history
            ],
            "notices": list(self.notices),
        }

    @classmethod
    def from_state(cls, st: dict[str, Any]) -> AgentMemory:
        mem = cls(
            memory_periods=int(st["memory_periods"]),
            plans=st["plans"],
            insights=st["insights"],
        )
        for r in st["history"]:
            mem.history.append(PeriodRecord(**{**r, "rival_prices": tuple(r["rival_prices"])}))
        mem.notices = list(st.get("notices", []))
        return mem


@runtime_checkable
class Agent(Protocol):
    agent_id: int
    slot: int
    kind: AgentKind
    name: str
    role: str
    memory: AgentMemory

    async def decide(self, obs: Observation) -> Decision: ...

    def record_period(self, record: PeriodRecord) -> None: ...

    def state(self) -> dict[str, Any]: ...

    def restore(self, state: dict[str, Any]) -> None: ...
