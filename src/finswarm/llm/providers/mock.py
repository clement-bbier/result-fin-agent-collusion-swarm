"""Zero-cost mock provider with scripted pricing policies, and a ``Flaky`` wrapper.

``MockLLM`` reads the compact observation that :class:`finswarm.agents.llm.LLMAgent`
puts in ``req.tags["obs_json"]`` and answers in the output format the template asks for
(``obs["output"]``): Fish text sections or JSON. Answers are deterministic functions of
the observation and ``req.seed``, so mock runs are bit-reproducible.

Benchmarks default to Calvano et al. (2020) baseline values scaled by the marginal cost
(``p_N = 1.4729 c``, ``p_M = 1.9250 c`` with ``c = alpha``) unless the observation
carries ``nash_price`` / ``monopoly_price``.
"""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any

from finswarm.llm.client import Capabilities, LLMRequest, LLMResponse
from finswarm.llm.errors import RateLimited, RetryableError
from finswarm.llm.timing import Clock, SystemClock

POLICIES: tuple[str, ...] = (
    "tit_for_tat",
    "grim_trigger",
    "undercutter",
    "random",
    "static_nash",
)

NASH_OVER_COST = 1.4729
MONOPOLY_OVER_COST = 1.9250

BAD_JSON_TEXT = 'I think the best price is somewhere around... {"price": '


def _rival_prices(row: dict[str, Any]) -> list[float]:
    return [float(x) for x in row.get("rival_prices", []) if x == x]  # drop NaN


def _round_to_tick(p: float, tick: float) -> float:
    if tick <= 0:
        return round(p, 4)
    return round(round(p / tick) * tick, 6)


class MockLLM:
    """Scripted pricing policy behind the :class:`LLMClient` protocol."""

    name = "mock"
    capabilities = Capabilities(
        supports_seed=True,
        supports_temperature=True,
        supports_json_schema=True,
        supports_cache_control=False,
        free=True,
    )

    def __init__(
        self,
        policy: str = "static_nash",
        *,
        latency_ms: float = 0.0,
        clock: Clock | None = None,
    ) -> None:
        if policy not in POLICIES:
            raise ValueError(f"unknown mock policy {policy!r}; choose from {POLICIES}")
        self.policy = policy
        self.latency_ms = latency_ms
        self.clock: Clock = clock or SystemClock()
        self.n_calls = 0

    # ----------------------------------------------------------------- policy
    def choose_price(
        self, obs: dict[str, Any], seed: int | None, policy: str | None = None
    ) -> float:
        pol = policy or self.policy
        cost = float(obs.get("cost", 1.0))
        nash = float(obs.get("nash_price") or NASH_OVER_COST * cost)
        mono = float(obs.get("monopoly_price") or MONOPOLY_OVER_COST * cost)
        tick = float(obs.get("tick_size") or 0.01 * max(cost, 1e-9))
        hist: list[dict[str, Any]] = list(obs.get("history", []))  # oldest first
        last_rivals = _rival_prices(hist[-1]) if hist else []

        if pol == "static_nash":
            p = nash
        elif pol == "tit_for_tat":
            p = mono if not last_rivals else sum(last_rivals) / len(last_rivals)
            p = min(max(p, nash), mono)
        elif pol == "grim_trigger":
            threshold = mono - 2.0 * tick
            defected = any((min(r) < threshold) for r in (_rival_prices(row) for row in hist) if r)
            p = nash if defected else mono
        elif pol == "undercutter":
            p = mono if not last_rivals else max(nash, min(last_rivals) - tick)
        elif pol == "random":
            digest = hashlib.sha256(
                f"{seed}|{obs.get('period')}|{obs.get('slot')}".encode()
            ).digest()
            rng = random.Random(int.from_bytes(digest[:8], "big"))
            p = rng.uniform(nash, mono)
        else:  # pragma: no cover - guarded in __init__
            raise ValueError(pol)
        return _round_to_tick(p, tick)

    def render(self, obs: dict[str, Any], price: float, policy: str) -> str:
        fmt = obs.get("output", "price_json")
        note = f"mock policy {policy}"
        plans = f"Keep following {policy}."
        insights = f"Last price {price:g}."
        if fmt == "fish_text":
            return (
                f"My observations and thoughts:\n{note}\n\n"
                f"New content for PLANS.txt:\n{plans}\n\n"
                f"New content for INSIGHTS.txt:\n{insights}\n\n"
                f"My chosen price:\n{price:g}"
            )
        if fmt == "order_json":
            cost = float(obs.get("cost", 1.0))
            half = max(price - cost, 0.0)
            bid = round(cost - half, 6) if obs.get("two_sided") else None
            payload: dict[str, Any] = {
                "observations": note,
                "plans": plans,
                "insights": insights,
                "bid": bid,
                "ask": price,
                "bid_size": None,
                "ask_size": None,
            }
        elif fmt == "cda_json":
            payload = {
                "observations": note,
                "plans": plans,
                "insights": insights,
                "price": price,
                "message": None,
            }
        elif fmt == "judge_json":
            payload = {"score": 1, "rationale": note, "evidence": ""}
        else:
            payload = {"observations": note, "plans": plans, "insights": insights, "price": price}
        return json.dumps(payload)

    # --------------------------------------------------------------- protocol
    async def complete(self, req: LLMRequest) -> LLMResponse:
        self.n_calls += 1
        if self.latency_ms > 0:
            await self.clock.sleep(self.latency_ms / 1000.0)
        raw = req.tags.get("obs_json")
        if raw is None:
            text = json.dumps({"ok": True})
        else:
            obs = json.loads(raw)
            policy = req.tags.get("mock_policy", self.policy)
            price = self.choose_price(obs, req.seed, policy)
            text = self.render(obs, price, policy)
        prompt_chars = len(req.system) + sum(len(str(m.get("content", ""))) for m in req.messages)
        return LLMResponse(
            text=text,
            input_tokens=prompt_chars // 4,
            output_tokens=len(text) // 4,
            latency_ms=self.latency_ms,
            cost_usd=0.0,
            request_id=f"mock-{self.n_calls}",
        )


class Flaky:
    """Wrap a client with seeded failures: 429s, 5xx, latency and malformed JSON.

    ``calls`` records the clock time of every attempt that reached the provider, which the
    scheduler tests use to check that configured rates are never exceeded.
    """

    def __init__(
        self,
        inner: Any,
        p_bad_json: float = 0.0,
        p_429: float = 0.0,
        latency_ms: float = 0.0,
        *,
        p_5xx: float = 0.0,
        retry_after: float | None = 1.0,
        seed: int = 0,
        clock: Clock | None = None,
    ) -> None:
        self.inner = inner
        self.name = getattr(inner, "name", "mock")
        self.capabilities = inner.capabilities
        self.p_bad_json = p_bad_json
        self.p_429 = p_429
        self.p_5xx = p_5xx
        self.latency_ms = latency_ms
        self.retry_after = retry_after
        self.clock: Clock = clock or SystemClock()
        self._rng = random.Random(seed)
        self.calls: list[float] = []
        self.n_429 = 0
        self.n_5xx = 0
        self.n_bad = 0

    async def complete(self, req: LLMRequest) -> LLMResponse:
        self.calls.append(self.clock.now())
        u_fail, u_json = self._rng.random(), self._rng.random()
        if u_fail < self.p_429:
            self.n_429 += 1
            raise RateLimited("flaky 429", retry_after=self.retry_after)
        if u_fail < self.p_429 + self.p_5xx:
            self.n_5xx += 1
            raise RetryableError("flaky 503", kind="server", status=503)
        if self.latency_ms > 0:
            await self.clock.sleep(self.latency_ms / 1000.0)
        resp: LLMResponse = await self.inner.complete(req)
        if u_json < self.p_bad_json:
            self.n_bad += 1
            resp = resp.model_copy(update={"text": BAD_JSON_TEXT})
        return resp.model_copy(update={"latency_ms": resp.latency_ms + self.latency_ms})
