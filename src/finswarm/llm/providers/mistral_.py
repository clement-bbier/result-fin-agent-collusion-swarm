"""Mistral adapter: official ``mistralai`` SDK, ``chat.complete_async``.

JSON is requested with ``response_format={"type": "json_object"}`` when the request has a
schema (the template itself describes the JSON fields). ``random_seed`` carries the
request seed. The SDK's retries stay off; the lane scheduler owns retries.
"""

from __future__ import annotations

import time
from typing import Any

from finswarm.llm.client import Capabilities, LLMRequest, LLMResponse
from finswarm.llm.errors import FatalLLMError, RateLimited, RetryableError, parse_retry_after
from finswarm.llm.keys import redact


def _content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for chunk in content:
        text = getattr(chunk, "text", None)
        if text is None and isinstance(chunk, dict):
            text = chunk.get("text")
        if text:
            parts.append(str(text))
    return "".join(parts)


def map_mistral_error(exc: BaseException) -> BaseException:
    """Translate SDK / transport exceptions into finswarm LLM errors."""
    from mistralai.client.errors import MistralError, NoResponseError

    if isinstance(exc, MistralError):
        status = int(exc.status_code)
        retry_after = parse_retry_after(exc.headers.get("retry-after") if exc.headers else None)
        if status == 429:
            return RateLimited(redact(exc.message), retry_after=retry_after)
        if status >= 500 or status in (408, 409):
            return RetryableError(
                redact(exc.message), kind="server", status=status, retry_after=retry_after
            )
        return FatalLLMError(redact(exc.message), status=status)
    name = type(exc).__name__
    if "Timeout" in name:
        return RetryableError(redact(str(exc)), kind="timeout")
    if isinstance(exc, NoResponseError) or name in (
        "ConnectError",
        "ReadError",
        "WriteError",
        "RemoteProtocolError",
        "NetworkError",
        "TransportError",
    ):
        return RetryableError(redact(str(exc)), kind="connection")
    return exc


class MistralClient:
    name = "mistral"
    capabilities = Capabilities(
        supports_seed=True,
        supports_temperature=True,
        supports_json_schema=False,
        supports_cache_control=False,
        free=True,
    )

    def __init__(
        self,
        *,
        api_key: str | None = None,
        client: Any | None = None,
        json_mode: bool = True,
        timeout_s: float = 120.0,
    ) -> None:
        if client is None:
            from mistralai.client import Mistral

            client = Mistral(api_key=api_key, timeout_ms=int(timeout_s * 1000))
        self._client = client
        self.json_mode = json_mode

    def build_params(self, req: LLMRequest) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": req.model,
            "messages": [{"role": "system", "content": req.system}, *req.messages],
            "max_tokens": int(req.max_tokens),
        }
        if req.temperature is not None:
            params["temperature"] = float(req.temperature)
        if req.seed is not None:
            params["random_seed"] = int(req.seed)
        if req.schema_ is not None and self.json_mode:
            params["response_format"] = {"type": "json_object"}
        return params

    async def complete(self, req: LLMRequest) -> LLMResponse:
        t0 = time.perf_counter()
        try:
            resp = await self._client.chat.complete_async(**self.build_params(req))
        except Exception as e:  # mapped below; unknown errors are re-raised
            mapped = map_mistral_error(e)
            if mapped is e:
                raise
            raise mapped from e
        latency_ms = (time.perf_counter() - t0) * 1000.0
        choice = resp.choices[0] if resp.choices else None
        text = _content_text(getattr(getattr(choice, "message", None), "content", None))
        usage = getattr(resp, "usage", None)
        return LLMResponse(
            text=text,
            input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            latency_ms=latency_ms,
            cost_usd=0.0,
            request_id=getattr(resp, "id", None),
            stop_reason=str(getattr(choice, "finish_reason", None) or "stop"),
        )
