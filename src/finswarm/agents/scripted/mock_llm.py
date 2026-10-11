"""Stand-in for an LLM dealer while ``finswarm.agents.llm`` is not available.

It has ``kind = AgentKind.LLM`` and exposes ``last_call`` / ``last_prompt`` exactly like the
runner expects from a real LLM agent, so the ``llm_decisions`` table, the prompts sidecar and
the manifest LLM totals are exercised end to end at zero cost. Policy: a noisy move towards
the highest rival price, clipped to ``[cost, hi]``; ``fail_rate`` injects parse failures
(the runner then keeps the previous quote and flags the decision).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from finswarm.agents.scripted.base import ScriptedAgent, own_last_price, rival_prices
from finswarm.core.types import AgentKind, Decision, Observation


class MockLLMAgent(ScriptedAgent):
    policy = "mock_llm"
    kind = AgentKind.LLM

    def __init__(
        self,
        agent_id: int,
        slot: int,
        *,
        lo: float,
        hi: float,
        sigma: float = 0.05,
        fail_rate: float = 0.0,
        model: str = "mock-1",
        lane: str = "mock",
        prompt_version: str = "mock_v1",
        **kw: Any,
    ) -> None:
        super().__init__(agent_id, slot, **kw)
        self.lo, self.hi = float(lo), float(hi)
        self.sigma = float(sigma)
        self.fail_rate = float(fail_rate)
        self.model = model
        self.lane = lane
        self.prompt_version = prompt_version
        self.last_call: dict[str, Any] | None = None
        self.last_prompt: dict[str, Any] | None = None

    def act(self, obs: Observation) -> Decision:
        prompt = f"{self.prompt_version}|slot={self.slot}|period={obs.period}|cost={obs.cost:.6f}"
        rivals = rival_prices(obs)
        own = own_last_price(obs, float(self._rng.uniform(self.lo, self.hi)))
        target = max(rivals) if rivals else own
        price = 0.5 * own + 0.5 * target + float(self._rng.normal(0.0, self.sigma))
        price = self.clamp(price, max(self.lo, obs.cost), self.hi)
        fail = self.fail_rate > 0 and float(self._rng.random()) < self.fail_rate
        self.last_prompt = {"system": "mock dealer", "user": prompt}
        self.last_call = {
            "provider": "mock",
            "model": self.model,
            "lane": self.lane,
            "prompt_version": self.prompt_version,
            "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest()[:16],
            "cache_hit": False,
            "latency_ms": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cached_tokens": 0,
            "cost_usd": 0.0,
            "request_id": None,
            "error": None,
        }
        if fail:
            return Decision(action="hold", raw="{not json", parse_ok=False, plans="", insights="")
        plans = f"move toward {target:.4f}"
        insights = f"rivals={','.join(f'{x:.4f}' for x in rivals)}"
        raw = json.dumps({"price": round(price, 6), "plans": plans, "insights": insights})
        d = self.quote(price, plans=plans, insights=insights)
        d.raw = raw
        return d

    def _extra_state(self) -> dict[str, Any]:
        return {}
