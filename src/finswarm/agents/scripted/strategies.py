"""Classic repeated-game pricing strategies used as rivals, controls and sanity checks.

Defaults (filled by ``registry.make_scripted`` from the market benchmarks when available):
cooperative ``target`` = joint-monopoly price, punishment = one-shot Nash price.
"""

from __future__ import annotations

from typing import Any

from finswarm.agents.scripted.base import NAN, ScriptedAgent, own_last_price, rival_prices
from finswarm.core.types import Decision, Observation


class TitForTat(ScriptedAgent):
    """Price at ``target``; after a rival undercuts ``target - trigger``, mirror the lowest rival
    price (or ``punish_price``) and keep punishing ``punish`` more periods once rivals return."""

    policy = "tit_for_tat"

    def __init__(
        self,
        agent_id: int,
        slot: int,
        *,
        target: float,
        trigger: float = 0.05,
        punish: int = 3,
        punish_price: float | None = None,
        **kw: Any,
    ) -> None:
        super().__init__(agent_id, slot, **kw)
        self.target = float(target)
        self.trigger = float(trigger)
        self.punish = int(punish)
        self.punish_price = punish_price
        self._left = 0
        self._price_now = NAN

    def act(self, obs: Observation) -> Decision:
        rivals = rival_prices(obs)
        if rivals and min(rivals) < self.target - self.trigger:
            p = self.punish_price if self.punish_price is not None else min(rivals)
            self._price_now = self.clamp(p, obs.cost, self.ceiling(obs))
            self._left = self.punish
            return self.quote(self._price_now, plans="punish")
        if self._left > 0:
            self._left -= 1
            return self.quote(self._price_now, plans="punish")
        return self.quote(self.target, plans="cooperate")

    def _extra_state(self) -> dict[str, Any]:
        return {"left": self._left, "price_now": self._price_now}

    def _restore_extra(self, st: dict[str, Any]) -> None:
        self._left = int(st["left"])
        self._price_now = float(st["price_now"])


class GrimTrigger(ScriptedAgent):
    """Cooperate at ``target`` until any rival prices below ``target - trigger``; then price at
    ``punish_price`` forever."""

    policy = "grim_trigger"

    def __init__(
        self,
        agent_id: int,
        slot: int,
        *,
        target: float,
        punish_price: float,
        trigger: float = 0.05,
        **kw: Any,
    ) -> None:
        super().__init__(agent_id, slot, **kw)
        self.target = float(target)
        self.punish_price = float(punish_price)
        self.trigger = float(trigger)
        self.triggered = False

    def act(self, obs: Observation) -> Decision:
        rivals = rival_prices(obs)
        if not self.triggered and rivals and min(rivals) < self.target - self.trigger:
            self.triggered = True
        if self.triggered:
            return self.quote(self.punish_price, plans="punish")
        return self.quote(self.target, plans="cooperate")

    def _extra_state(self) -> dict[str, Any]:
        return {"triggered": self.triggered}

    def _restore_extra(self, st: dict[str, Any]) -> None:
        self.triggered = bool(st["triggered"])


class Undercutter(ScriptedAgent):
    """Price ``eps`` below the lowest rival price, never below ``floor`` (default: own cost)."""

    policy = "undercutter"

    def __init__(
        self,
        agent_id: int,
        slot: int,
        *,
        eps: float = 0.01,
        start: float | None = None,
        floor: float | None = None,
        **kw: Any,
    ) -> None:
        super().__init__(agent_id, slot, **kw)
        self.eps = float(eps)
        self.start = start
        self.floor = floor

    def act(self, obs: Observation) -> Decision:
        floor = self.floor if self.floor is not None else obs.cost
        rivals = rival_prices(obs)
        if not rivals:
            p = self.start if self.start is not None else own_last_price(obs, obs.cost * 1.5)
            return self.quote(self.clamp(p, floor, self.ceiling(obs)))
        return self.quote(self.clamp(min(rivals) - self.eps, floor, self.ceiling(obs)))


class StaticNash(ScriptedAgent):
    """Always quote the one-shot Nash price (competitive benchmark rival)."""

    policy = "static_nash"

    def __init__(self, agent_id: int, slot: int, *, price: float, **kw: Any) -> None:
        super().__init__(agent_id, slot, **kw)
        self.price = float(price)

    def act(self, obs: Observation) -> Decision:
        return self.quote(self.price)


class RandomPricer(ScriptedAgent):
    """Uniform random price in ``[lo, hi]`` every decision (noise control)."""

    policy = "random_pricer"

    def __init__(self, agent_id: int, slot: int, *, lo: float, hi: float, **kw: Any) -> None:
        super().__init__(agent_id, slot, **kw)
        if hi < lo:
            raise ValueError("random_pricer: hi < lo")
        self.lo, self.hi = float(lo), float(hi)

    def act(self, obs: Observation) -> Decision:
        return self.quote(float(self._rng.uniform(self.lo, self.hi)))


class ZIMarketMaker(ScriptedAgent):
    """Zero-intelligence constrained market maker (Gode & Sunder 1993 ZI-C spirit): a random
    markup over cost, never selling below cost. Two-sided when the market is two-sided."""

    policy = "zi_mm"

    def __init__(
        self,
        agent_id: int,
        slot: int,
        *,
        hi: float,
        spread: float = 0.0,
        two_sided: bool = False,
        size: float | None = None,
        **kw: Any,
    ) -> None:
        super().__init__(agent_id, slot, **kw)
        self.hi = float(hi)
        self.spread = float(spread)
        self.two_sided = bool(two_sided)
        self.size = size

    def act(self, obs: Observation) -> Decision:
        lo = obs.cost
        hi = max(self.hi, lo)
        p = float(self._rng.uniform(lo, hi))
        if self.two_sided:
            half = 0.5 * self.spread
            return Decision(action="quote", price=p, bid=p - half, ask=p + half, size=self.size)
        d = self.quote(p)
        d.size = self.size
        return d


def demand_profit_fn(
    *, alpha: float, a: float, a0: float, tau: float, beta: float, cost: float
) -> Any:
    """Counterfactual one-period profit for the closed-form logit market (used by Hedge)."""
    from finswarm.sim.online_metrics import logit_shares

    def profit(p: float, rivals: list[float]) -> float:
        s = logit_shares([p, *rivals], alpha=alpha, a=a, a0=a0, tau=tau)[0]
        return (p - cost) * beta * s

    return profit


__all__ = [
    "GrimTrigger",
    "RandomPricer",
    "StaticNash",
    "TitForTat",
    "Undercutter",
    "ZIMarketMaker",
    "demand_profit_fn",
]
