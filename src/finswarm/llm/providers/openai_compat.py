"""OpenAI-compatible ``/chat/completions`` adapter over ``httpx`` (Gemini, OpenRouter).

Never used for Claude. One instance per provider: ``base_url`` already contains the
version prefix (``.../openai/v1``), the adapter appends ``/chat/completions``. The last
``x-ratelimit-*`` headers seen are kept in ``last_limits`` for the lane health check.
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from finswarm.llm.client import Capabilities, LLMRequest, LLMResponse
from finswarm.llm.errors import FatalLLMError, RateLimited, RetryableError, parse_retry_after
from finswarm.llm.keys import redact


class OpenAICompatClient:
    def __init__(
        self,
        name: str,
        base_url: str,
        *,
        api_key: str | None,
        json_mode: bool = True,
        capabilities: Capabilities | None = None,
        timeout_s: float = 120.0,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.name = name
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self.json_mode = json_mode
        self.capabilities = capabilities or Capabilities(supports_seed=True, free=True)
        self.timeout_s = timeout_s
        self._http = http_client
        self.last_limits: dict[str, str] = {}

    @property
    def url(self) -> str:
        return f"{self.base_url}/chat/completions"

    @staticmethod
    def _cost(model: str, in_tokens: int, out_tokens: int) -> float:
        """USD of one call at the catalogue price (0.0 when the model is unpriced or free)."""
        from finswarm.llm.pricing import estimate_call_cost

        try:
            return float(estimate_call_cost(model, in_tokens, out_tokens))
        except Exception:  # unpriced model: never fail a completed call on accounting
            return 0.0

    def build_body(self, req: LLMRequest) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": req.model,
            "messages": [{"role": "system", "content": req.system}, *req.messages],
            "max_tokens": int(req.max_tokens),
        }
        if req.temperature is not None and self.capabilities.supports_temperature:
            body["temperature"] = float(req.temperature)
        if req.seed is not None and self.capabilities.supports_seed:
            body["seed"] = int(req.seed)
        if req.schema_ is not None and self.json_mode:
            body["response_format"] = {"type": "json_object"}
        return body

    def _client(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=self.timeout_s)
        return self._http

    async def aclose(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    async def complete(self, req: LLMRequest) -> LLMResponse:
        if not self._api_key:
            raise FatalLLMError(f"{self.name}: missing API key", status=401)
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        t0 = time.perf_counter()
        try:
            r = await self._client().post(self.url, json=self.build_body(req), headers=headers)
        except httpx.TimeoutException as e:
            raise RetryableError(f"{self.name}: timeout", kind="timeout") from e
        except httpx.TransportError as e:
            raise RetryableError(f"{self.name}: {type(e).__name__}", kind="connection") from e
        latency_ms = (time.perf_counter() - t0) * 1000.0
        self.last_limits = {
            k.lower(): v for k, v in r.headers.items() if k.lower().startswith("x-ratelimit")
        }
        retry_after = parse_retry_after(r.headers.get("retry-after"))
        detail = redact(r.text[:400])
        if r.status_code == 429:
            raise RateLimited(f"{self.name}: 429 {detail}", retry_after=retry_after)
        if r.status_code >= 500 or r.status_code in (408, 409):
            raise RetryableError(
                f"{self.name}: {r.status_code} {detail}",
                kind="server",
                status=r.status_code,
                retry_after=retry_after,
            )
        if r.status_code >= 400:
            raise FatalLLMError(f"{self.name}: {r.status_code} {detail}", status=r.status_code)
        data = r.json()
        choices = data.get("choices") or [{}]
        choice = choices[0]
        text = (choice.get("message") or {}).get("content") or ""
        usage = data.get("usage") or {}
        cached = int(((usage.get("prompt_tokens_details") or {}).get("cached_tokens")) or 0)
        in_tok = int(usage.get("prompt_tokens") or 0)
        out_tok = int(usage.get("completion_tokens") or 0)
        return LLMResponse(
            text=str(text),
            input_tokens=in_tok,
            output_tokens=out_tok,
            cached_input_tokens=cached,
            latency_ms=latency_ms,
            # catalogue price of the model (0 for the :free models, paid for openrouter_llama);
            # the cost was hard-wired to 0.0 until 2026-10-10, so paid OpenRouter calls were
            # invisible to the lane budget guard and the ledger
            cost_usd=self._cost(req.model, in_tok, out_tok),
            request_id=data.get("id") or r.headers.get("x-request-id"),
            stop_reason=str(choice.get("finish_reason") or "stop"),
        )
