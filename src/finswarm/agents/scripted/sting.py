"""Sting dealer: an active probe operated by the regulator (H8).

Per probe: shadow the mean rival quote for ``shadow_epochs`` decisions, then hold a price
``delta`` above the last shadow price for ``hold_epochs`` decisions, then revert to shadowing.
After ``probes`` probes it keeps shadowing. Phase transitions are exposed through
``drain_events()`` and written to the ``events`` table (``event_type="sting_phase"``) so the
follow rate of the other dealers can be measured against exact probe times.
"""

from __future__ import annotations

from typing import Any

from finswarm.agents.scripted.base import NAN, ScriptedAgent, rival_prices, standing_rival_prices
from finswarm.core.types import Decision, Observation


class StingDealer(ScriptedAgent):
    policy = "sting_dealer"

    def __init__(
        self,
        agent_id: int,
        slot: int,
        *,
        delta: float,
        hold_epochs: int = 5,
        shadow_epochs: int = 5,
        probes: int = 1,
        event_id: int = -1,
        start: float | None = None,
        **kw: Any,
    ) -> None:
        super().__init__(agent_id, slot, **kw)
        self.delta = float(delta)
        self.hold_epochs = int(hold_epochs)
        self.shadow_epochs = int(shadow_epochs)
        self.probes = int(probes)
        self.event_id = int(event_id)
        self.start = start
        self.step = 0  # decisions taken so far
        self.phase = "init"
        self.probe = -1
        self.shadow_price = NAN
        self.raise_price = NAN
        self.phase_log: list[dict[str, Any]] = []
        self._pending: list[dict[str, Any]] = []

    def _phase_at(self, step: int) -> tuple[str, int]:
        cycle = self.shadow_epochs + self.hold_epochs
        probe, pos = divmod(step, cycle) if cycle > 0 else (self.probes, 0)
        if probe >= self.probes:
            return "done", self.probes - 1
        return ("shadow", probe) if pos < self.shadow_epochs else ("raise", probe)

    def _mean_rival(self, obs: Observation) -> float:
        xs = standing_rival_prices(obs) or rival_prices(obs)
        if xs:
            return sum(xs) / len(xs)
        if self.start is not None:
            return float(self.start)
        return self.last_price if self.last_price == self.last_price else obs.cost

    def act(self, obs: Observation) -> Decision:
        phase, probe = self._phase_at(self.step)
        if phase in ("shadow", "done"):
            price = self._mean_rival(obs)
            self.shadow_price = price
        else:
            if self.phase != "raise":
                base = (
                    self.shadow_price
                    if self.shadow_price == self.shadow_price
                    else self._mean_rival(obs)
                )
                self.raise_price = base + self.delta
            price = self.raise_price
        price = self.clamp(price, obs.cost, self.ceiling(obs))
        if phase != self.phase or probe != self.probe:
            rec = {
                "event_id": self.event_id,
                "slot": self.slot,
                "period": obs.period,
                "tick": obs.tick,
                "step": self.step,
                "phase": phase,
                "probe": probe,
                "price": price,
            }
            self.phase_log.append(rec)
            self._pending.append(rec)
        self.phase, self.probe = phase, probe
        self.step += 1
        return self.quote(price, plans=f"sting:{phase}:{probe}")

    def drain_events(self) -> list[dict[str, Any]]:
        out, self._pending = self._pending, []
        return out

    def _extra_state(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "phase": self.phase,
            "probe": self.probe,
            "shadow_price": self.shadow_price,
            "raise_price": self.raise_price,
            "phase_log": list(self.phase_log),
            "pending": list(self._pending),
        }

    def _restore_extra(self, st: dict[str, Any]) -> None:
        self.step = int(st["step"])
        self.phase = str(st["phase"])
        self.probe = int(st["probe"])
        self.shadow_price = float(st["shadow_price"])
        self.raise_price = float(st["raise_price"])
        self.phase_log = list(st["phase_log"])
        self._pending = list(st["pending"])
