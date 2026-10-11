"""No-regret market maker over a fixed price grid.

``mode="exp3"`` (default): bandit feedback, EXP3 of Auer, Cesa-Bianchi, Freund & Schapire
(2002) with importance-weighted rewards; regret O(sqrt(T K log K)) against the best fixed
price in hindsight, whatever the rivals do. ``mode="hedge"``: full-information Hedge
(Freund & Schapire 1997) with counterfactual profits from the closed-form logit demand given
the rivals' observed prices. The point (in the spirit of Collina et al. 2025) is a seller that
provably cannot be steered into supra-competitive play by its rivals' strategies.

Rewards are profits normalised to [0, 1] by ``reward_scale`` and clipped.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np

from finswarm.agents.scripted.base import ScriptedAgent, finite
from finswarm.core.types import Decision, Observation, PeriodRecord


class NoRegretMM(ScriptedAgent):
    policy = "no_regret_mm"

    def __init__(
        self,
        agent_id: int,
        slot: int,
        *,
        lo: float,
        hi: float,
        n_prices: int = 21,
        mode: str = "exp3",
        horizon: int = 1000,
        gamma: float | None = None,
        eta: float | None = None,
        reward_scale: float = 100.0,
        profit_fn: Callable[[float, list[float]], float] | None = None,
        **kw: Any,
    ) -> None:
        super().__init__(agent_id, slot, **kw)
        if mode not in ("exp3", "hedge"):
            raise ValueError(f"no_regret_mm: unknown mode {mode!r}")
        if mode == "hedge" and profit_fn is None:
            raise ValueError("no_regret_mm: hedge mode needs a counterfactual profit_fn")
        self.grid = np.linspace(float(lo), float(hi), int(n_prices))
        self.k = int(n_prices)
        self.mode = mode
        self.reward_scale = float(reward_scale)
        self.profit_fn = profit_fn
        t = max(1, int(horizon))
        if gamma is None:
            gamma = min(1.0, math.sqrt(self.k * math.log(self.k) / ((math.e - 1.0) * t)))
        if eta is None:
            eta = math.sqrt(8.0 * math.log(self.k) / t)
        self.gamma = float(gamma)
        self.eta = float(eta)
        self.log_w = np.zeros(self.k)
        self.arm = -1
        self.probs = np.full(self.k, 1.0 / self.k)
        self.cum_reward = 0.0

    def distribution(self) -> np.ndarray:
        w = np.exp(self.log_w - self.log_w.max())
        p = w / w.sum()
        if self.mode == "exp3":
            p = (1.0 - self.gamma) * p + self.gamma / self.k
        return p

    def act(self, obs: Observation) -> Decision:
        self.probs = self.distribution()
        self.arm = int(self._rng.choice(self.k, p=self.probs))
        return self.quote(float(self.grid[self.arm]), plans=f"arm={self.arm}")

    def _reward(self, profit: float) -> float:
        if not math.isfinite(profit):
            return 0.0
        return min(1.0, max(0.0, profit / self.reward_scale))

    def on_period(self, record: PeriodRecord) -> None:
        if self.arm < 0:
            return
        r = self._reward(record.own_profit)
        self.cum_reward += r
        if self.mode == "exp3":
            xhat = r / self.probs[self.arm]
            self.log_w[self.arm] += self.gamma * xhat / self.k
        else:
            rivals = finite(record.rival_prices)
            assert self.profit_fn is not None
            rewards = np.array([self._reward(self.profit_fn(float(p), rivals)) for p in self.grid])
            self.log_w += self.eta * rewards
        self.log_w -= self.log_w.max()

    def _extra_state(self) -> dict[str, Any]:
        return {
            "log_w": self.log_w.tolist(),
            "arm": self.arm,
            "probs": self.probs.tolist(),
            "cum_reward": self.cum_reward,
        }

    def _restore_extra(self, st: dict[str, Any]) -> None:
        self.log_w = np.array(st["log_w"], dtype=float)
        self.arm = int(st["arm"])
        self.probs = np.array(st["probs"], dtype=float)
        self.cum_reward = float(st["cum_reward"])
