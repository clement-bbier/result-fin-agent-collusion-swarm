"""Offline fakes shared by the ``test_llm_*`` modules (never touch the network)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from finswarm.config.schema import LaneConfig
from finswarm.llm.client import Capabilities, LLMRequest, LLMResponse
from finswarm.llm.timing import Clock

Script = LLMResponse | BaseException | str | Callable[[LLMRequest], LLMResponse]


class ScriptedClient:
    """LLMClient that plays a script (responses, texts or exceptions), then ``default``."""

    name = "fake"
    capabilities = Capabilities(free=True)

    def __init__(
        self,
        script: list[Script] | None = None,
        *,
        default: Script = '{"ok": true}',
        clock: Clock | None = None,
        cost: float = 0.0,
    ) -> None:
        self.script = list(script or [])
        self.default = default
        self.clock = clock
        self.cost = cost
        self.calls: list[tuple[float, LLMRequest]] = []

    async def complete(self, req: LLMRequest) -> LLMResponse:
        self.calls.append((self.clock.now() if self.clock else 0.0, req))
        item = self.script.pop(0) if self.script else self.default
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, LLMResponse):
            return item
        if isinstance(item, str):
            return LLMResponse(text=item, input_tokens=10, output_tokens=5, cost_usd=self.cost)
        return item(req)

    @property
    def times(self) -> list[float]:
        return [t for t, _ in self.calls]


def mock_lane(**kw: Any) -> LaneConfig:
    return LaneConfig(provider="mock", models=["mock-v1"], **kw)


def req(i: int = 0, **kw: Any) -> LLMRequest:
    base: dict[str, Any] = {
        "provider": "mock",
        "model": "mock-v1",
        "system": "sys",
        "messages": [{"role": "user", "content": f"hello {i}"}],
        "seed": i,
    }
    base.update(kw)
    return LLMRequest(**base)


def max_in_window(times: list[float], window: float, eps: float = 1e-6) -> int:
    """Largest number of events inside any half-open window of length ``window``."""
    ts = sorted(times)
    best = 0
    for i, t in enumerate(ts):
        n = sum(1 for u in ts[: i + 1] if u > t - window + eps)
        best = max(best, n)
    return best
