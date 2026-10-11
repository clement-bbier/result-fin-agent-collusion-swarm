"""Registered scripted controls SC1, SC2, SC4 (PREREG section 4) and the SC5 single dealer.

* ``best_responder`` (SC1, the competitive benchmark and the MQC base): at every decision,
  the static best response to the rivals' latest visible prices under the logit demand of the
  market as it is now (``finswarm.market.demand.best_response_price``). It sees the world
  like any agent: standing rival quotes as the public view shows them (anonymised, rounded or
  delayed), else the last period's rival prices; it decides whenever the clock asks it to.
  The live demand and routing come from ``obs.extra["demand_live"]`` (set by the runner, so a
  routing lever moves the best response); the build-time market config is the fallback. With
  no visible rival it quotes ``start`` (default: one-shot Nash price), except in a
  single-dealer market, where the best response to nobody is the stand-alone monopoly price
  (SC5: the market has no captive-flow segment, so captive flow is a single-dealer market).
* ``cost_plus`` (SC2, common shock): price = cost + s_t + markup + e_i, where s_t is a
  common AR(1) cost shock (one path per run seed, shared by every cost_plus dealer whatever
  its slot or decision time) and e_i idiosyncratic Gaussian noise from the agent's own stream.
  Default markup: one-shot Nash price minus cost. Prices never go below the true unit cost.
* ``leader_follower`` (SC4, price leadership): the leader moves its price by ``-step, 0,
  +step`` every ``hold`` periods inside ``[lo, hi]`` (default Nash and monopoly prices); a
  follower matches the leader's visible quote minus ``undercut`` (the leader is found by agent
  id, or as the highest visible rival price when ids are hidden).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import numpy as np

from finswarm.agents.scripted.base import (
    ScriptedAgent,
    finite,
    make_generator,
    standing_rival_prices,
)
from finswarm.core.types import Decision, Observation

COMMON_SHOCK_SALT = 0xC057


def latest_rival_prices(obs: Observation) -> list[float]:
    """Standing rival quotes as the view shows them, else the last period's rival prices."""
    standing = standing_rival_prices(obs)
    if standing:
        return standing
    if obs.history:
        return finite(obs.history[-1].rival_prices)
    return []


class BestResponder(ScriptedAgent):
    policy = "best_responder"

    def __init__(
        self,
        agent_id: int,
        slot: int,
        *,
        demand: dict[str, float | None],
        n_dealers: int = 2,
        start: float = 1.4729,
        **kw: Any,
    ) -> None:
        super().__init__(agent_id, slot, **kw)
        self.demand = dict(demand)
        self.n_dealers = int(n_dealers)
        self.start = float(start)

    def params(self, obs: Observation) -> Any:
        """Routed demand parameters of the market now (live values win over build-time)."""
        from finswarm.market.demand import DemandParams

        d = {**self.demand, **(obs.extra.get("demand_live") or {})}
        alpha = float(d["alpha"])
        tau = d.get("tau")
        mu = float(d["mu"]) if tau is None else float(tau) / alpha
        a0 = float(d["a0"]) + float(d.get("outside_utility") or 0.0) / alpha
        prm = DemandParams(a=float(d["a"]), a0=a0, mu=mu, beta=float(d["beta"]), alpha=alpha)
        return replace(prm, cost=float(obs.cost) / alpha)  # the dealer's live unit cost

    def act(self, obs: Observation) -> Decision:
        from finswarm.market.demand import best_response_price

        rivals = latest_rival_prices(obs)
        if not rivals and self.n_dealers > 1:
            return self.quote(self.clamp(self.start, obs.cost, self.ceiling(obs)), plans="start")
        p = best_response_price(rivals, self.params(obs))
        return self.quote(self.clamp(p, obs.cost, self.ceiling(obs)), plans="best response")


class CostPlus(ScriptedAgent):
    policy = "cost_plus"

    def __init__(
        self,
        agent_id: int,
        slot: int,
        *,
        markup: float,
        common_seed: int = 0,
        rho: float = 0.9,
        sigma: float = 0.05,
        idio_sd: float = 0.02,
        **kw: Any,
    ) -> None:
        super().__init__(agent_id, slot, **kw)
        if not 0.0 <= rho < 1.0:
            raise ValueError("cost_plus: rho must be in [0, 1)")
        self.markup = float(markup)
        self.common_seed = int(common_seed)
        self.rho, self.sigma, self.idio_sd = float(rho), float(sigma), float(idio_sd)
        self._common = make_generator([self.common_seed, COMMON_SHOCK_SALT])
        self._t = -1  # last period of the common path drawn
        self.shock = 0.0

    def common_shock(self, period: int) -> float:
        """s_period of the run's common AR(1) path (s_-1 = 0; one draw per period)."""
        if period < self._t:
            raise ValueError(f"cost_plus: period {period} is before {self._t} (no rewind)")
        while self._t < period:
            self.shock = self.rho * self.shock + self.sigma * float(self._common.standard_normal())
            self._t += 1
        return self.shock

    def act(self, obs: Observation) -> Decision:
        s = self.common_shock(obs.period)
        e = self.idio_sd * float(self._rng.standard_normal()) if self.idio_sd > 0 else 0.0
        p = obs.cost + s + self.markup + e
        return self.quote(self.clamp(p, obs.cost, self.ceiling(obs)), plans=f"shock {s:.4f}")

    def _extra_state(self) -> dict[str, Any]:
        return {"common": self._common.bit_generator.state, "t": self._t, "shock": self.shock}

    def _restore_extra(self, st: dict[str, Any]) -> None:
        self._common.bit_generator.state = st["common"]
        self._t = int(st["t"])
        self.shock = float(st["shock"])


class LeaderFollower(ScriptedAgent):
    policy = "leader_follower"

    def __init__(
        self,
        agent_id: int,
        slot: int,
        *,
        lo: float,
        hi: float,
        position: str = "follower",
        leader_slot: int = 0,
        start: float | None = None,
        step: float = 0.05,
        hold: int = 10,
        undercut: float = 0.0,
        **kw: Any,
    ) -> None:
        super().__init__(agent_id, slot, **kw)
        if position not in ("leader", "follower"):
            raise ValueError("leader_follower: position must be 'leader' or 'follower'")
        if hi < lo:
            raise ValueError("leader_follower: hi < lo")
        self.position = position
        self.leader_slot = int(leader_slot)
        self.lo, self.hi = float(lo), float(hi)
        self.start = float(start) if start is not None else self.hi
        self.step, self.hold, self.undercut = float(step), max(1, int(hold)), float(undercut)
        self.price = self.start
        self._moved = -1  # last period the leader moved

    def _leader_price(self, obs: Observation) -> float | None:
        leader_id = self.agent_id - self.slot + self.leader_slot  # runner: id = offset + slot
        p_min = float(obs.extra.get("p_min", 0.0))
        tick = float(obs.extra.get("tick_size", 0.0))
        for q in obs.rival_quotes:
            if q.agent_id == leader_id and q.ask_ticks is not None and q.ask_ticks >= 0:
                return p_min + q.ask_ticks * tick
        rivals = latest_rival_prices(obs)  # ids hidden by the view: the price leader is highest
        return max(rivals) if rivals else None

    def act(self, obs: Observation) -> Decision:
        if self.position == "leader":
            p = obs.period
            if p > 0 and p % self.hold == 0 and p != self._moved:
                move = float(self._rng.integers(-1, 2)) * self.step
                self.price = min(self.hi, max(self.lo, self.price + move))
                self._moved = p
            return self.quote(self.clamp(self.price, obs.cost, self.ceiling(obs)), plans="lead")
        lead = self._leader_price(obs)
        target = self.start if lead is None else lead - self.undercut
        return self.quote(self.clamp(target, obs.cost, self.ceiling(obs)), plans="follow")

    def _extra_state(self) -> dict[str, Any]:
        return {"price": self.price, "moved": self._moved}

    def _restore_extra(self, st: dict[str, Any]) -> None:
        self.price = float(st["price"])
        self._moved = int(st["moved"])


def run_seed_of(seed: int | Sequence[int] | None) -> int:
    """The run seed inside a runner slot seed ``[run_seed, salt, slot]`` (or a plain int)."""
    if seed is None:
        return 0
    if isinstance(seed, int | np.integer):
        return int(seed)
    seq = list(seed)
    return int(seq[0]) if seq else 0


def demand_dict(market: Any) -> dict[str, float | None]:
    """Build-time demand and routing of a ``MarketConfig`` (``BestResponder.demand``)."""
    d, r = market.demand, market.routing
    return {
        "a": float(d.a),
        "a0": float(d.a0),
        "mu": float(d.mu),
        "beta": float(d.beta),
        "alpha": float(market.alpha),
        "tau": None if r.tau is None else float(r.tau),
        "outside_utility": float(r.outside_utility),
    }


__all__ = [
    "BestResponder",
    "CostPlus",
    "LeaderFollower",
    "demand_dict",
    "latest_rival_prices",
    "run_seed_of",
]
