"""Anthropic adapter: official ``anthropic`` SDK (``AsyncAnthropic``), nothing else.

Request shape for Haiku / Sonnet / Opus 5.5 (see ``build_params``):

* ``system`` is a single text block with ``cache_control: {"type": "ephemeral"}`` (stable
  prefix first); the volatile market state lives in the final user message;
* when the first user message is a list of text blocks (revision-3 prompts: MARKET_DATA
  header, one block per round, then the volatile block), the second-to-last block (the end of
  the append-only history) gets a second ``cache_control`` breakpoint, so the next round's
  request reads this one's entry (one new block per round, inside the 20-block lookback).
  String content gets no message breakpoint. Empty text blocks are dropped (the API rejects
  them);
* ``output_config = {"effort": "low", "format": {"type": "json_schema", "schema": ...}}``;
  if the API rejects the schema the adapter falls back, once per model, to
  prompt-instructed JSON (the caller's parser repairs if needed);
* ``max_tokens >= 2048`` (thinking is on by default on these models);
* never ``temperature`` / ``top_p`` / ``top_k``, never ``thinking.budget_tokens``, never an
  assistant prefill, never a forced ``tool_choice``.

The SDK's own retries are disabled (``max_retries=0``): the lane scheduler owns retries
and honours ``retry-after``.
"""

from __future__ import annotations

import json
import time
from typing import Any

from finswarm.llm.client import Capabilities, LLMRequest, LLMResponse
from finswarm.llm.errors import FatalLLMError, RateLimited, RetryableError, parse_retry_after
from finswarm.llm.keys import redact
from finswarm.llm.pricing import Catalog, Price, cost_usd, load_catalog

MIN_MAX_TOKENS = 2048
DEFAULT_EFFORT = "low"
EFFORTS = ("low", "medium", "high", "xhigh", "max")
JSON_INSTRUCTION = (
    "\n\nRespond with only a JSON object (no prose, no code fence) that validates against "
    "this JSON schema:\n"
)


class SchemaRejected(FatalLLMError):
    """The API rejected ``output_config.format`` (internal; triggers the JSON fallback)."""


def _retry_after(exc: Any) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    return parse_retry_after(headers.get("retry-after"))


def _is_schema_error(message: str) -> bool:
    m = message.lower()
    return any(s in m for s in ("output_config", "json_schema", "schema", "output format"))


def mark_history_breakpoint(messages: list[dict[str, Any]]) -> None:
    """Put ``cache_control`` on the last history block of the first user message (in place).

    The first user message, when it is a list of at least two text blocks, ends with the
    volatile block; the block before it closes the append-only history.
    """
    first = messages[0] if messages else None
    if first is None or first["role"] != "user" or isinstance(first["content"], str):
        return
    blocks = first["content"]
    if len(blocks) >= 2 and blocks[-2].get("type") == "text":
        blocks[-2] = {**blocks[-2], "cache_control": {"type": "ephemeral"}}


class AnthropicClient:
    name = "anthropic"
    capabilities = Capabilities(
        supports_seed=False,
        supports_temperature=False,
        supports_json_schema=True,
        supports_cache_control=True,
        free=False,
    )

    def __init__(
        self,
        *,
        api_key: str | None = None,
        workspace_id: str | None = None,
        client: Any | None = None,
        catalog: Catalog | None = None,
        prices: dict[str, Price] | None = None,
        timeout_s: float = 120.0,
        default_effort: str = DEFAULT_EFFORT,
        capture_limits: bool = False,
    ) -> None:
        if client is None:
            import anthropic

            # Keys that are not scoped to a workspace must send ``anthropic-workspace-id``
            # (API 400 otherwise); workspace-scoped keys ignore the header.
            headers = {"anthropic-workspace-id": workspace_id} if workspace_id else None
            client = anthropic.AsyncAnthropic(
                api_key=api_key, max_retries=0, timeout=timeout_s, default_headers=headers
            )
        self._client = client
        self._catalog = catalog
        self._prices = prices
        self.default_effort = default_effort
        self.schema_rejected: set[str] = set()
        # capture_limits: read ``anthropic-ratelimit-*`` headers via ``with_raw_response``
        # (used by the lane health check)
        self.capture_limits = capture_limits
        self.last_limits: dict[str, str] = {}

    # ---------------------------------------------------------------- request
    def _price(self, model: str) -> Price:
        if self._prices is not None and model in self._prices:
            return self._prices[model]
        catalog = self._catalog or load_catalog()
        price = catalog.price(model)
        if price is None:
            raise FatalLLMError(f"no price for {model!r} in configs/models.yaml (budget guard)")
        return price

    def build_params(self, req: LLMRequest, *, use_schema: bool = True) -> dict[str, Any]:
        if req.temperature is not None:
            raise FatalLLMError("temperature must be None on the anthropic lane", status=400)
        effort = req.effort or self.default_effort
        if effort not in EFFORTS:
            raise FatalLLMError(f"unknown effort {effort!r}", status=400)
        messages: list[dict[str, Any]] = []
        for m in req.messages:
            role = m.get("role")
            if role not in ("user", "assistant"):
                raise FatalLLMError(f"unsupported message role {role!r}", status=400)
            content = m.get("content", "")
            if not isinstance(content, str):
                content = [
                    dict(b) for b in content if not (b.get("type") == "text" and not b.get("text"))
                ]
            messages.append({"role": role, "content": content})
        if not messages or messages[-1]["role"] != "user":
            raise FatalLLMError("the last message must be a user turn (no prefill)", status=400)
        mark_history_breakpoint(messages)

        output_config: dict[str, Any] = {"effort": effort}
        if req.schema_ is not None and use_schema:
            output_config["format"] = {"type": "json_schema", "schema": req.schema_}
        elif req.schema_ is not None:
            instruction = JSON_INSTRUCTION + json.dumps(req.schema_, sort_keys=True)
            last = dict(messages[-1])
            content = last["content"]
            if isinstance(content, str):
                last["content"] = content + instruction
            else:
                last["content"] = [*content, {"type": "text", "text": instruction.strip()}]
            messages[-1] = last

        return {
            "model": req.model,
            "max_tokens": max(int(req.max_tokens), MIN_MAX_TOKENS),
            "system": [
                {"type": "text", "text": req.system, "cache_control": {"type": "ephemeral"}}
            ],
            "messages": messages,
            "output_config": output_config,
        }

    # --------------------------------------------------------------- transport
    async def _create(self, params: dict[str, Any]) -> Any:
        import anthropic

        try:
            if not self.capture_limits:
                return await self._client.messages.create(**params)
            raw = await self._client.messages.with_raw_response.create(**params)
            self.last_limits = {
                k.lower(): v
                for k, v in raw.headers.items()
                if k.lower().startswith("anthropic-ratelimit")
            }
            return await raw.parse()
        except anthropic.RateLimitError as e:
            raise RateLimited(redact(str(e)), retry_after=_retry_after(e)) from e
        except anthropic.APITimeoutError as e:
            raise RetryableError(redact(str(e)), kind="timeout") from e
        except anthropic.APIConnectionError as e:
            raise RetryableError(redact(str(e)), kind="connection") from e
        except anthropic.BadRequestError as e:
            msg = redact(str(e))
            if "output_config" in params and "format" in params["output_config"]:
                if _is_schema_error(msg):
                    raise SchemaRejected(msg, status=400) from e
            raise FatalLLMError(msg, status=400) from e
        except anthropic.APIStatusError as e:
            status = int(getattr(e, "status_code", 0) or 0)
            if status >= 500 or status in (408, 409):
                raise RetryableError(
                    redact(str(e)), kind="server", status=status, retry_after=_retry_after(e)
                ) from e
            raise FatalLLMError(redact(str(e)), status=status) from e

    async def complete(self, req: LLMRequest) -> LLMResponse:
        price = self._price(req.model)  # fail before spending if the model is not priced
        use_schema = req.schema_ is not None and req.model not in self.schema_rejected
        t0 = time.perf_counter()
        try:
            msg = await self._create(self.build_params(req, use_schema=use_schema))
        except SchemaRejected:
            self.schema_rejected.add(req.model)
            msg = await self._create(self.build_params(req, use_schema=False))
        latency_ms = (time.perf_counter() - t0) * 1000.0

        text = "".join(
            getattr(b, "text", "") for b in msg.content if getattr(b, "type", None) == "text"
        )
        usage = msg.usage
        uncached = int(getattr(usage, "input_tokens", 0) or 0)
        cache_read = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
        cache_write = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
        output = int(getattr(usage, "output_tokens", 0) or 0)
        stop_reason = str(getattr(msg, "stop_reason", None) or "end_turn")
        refusal = stop_reason == "refusal"
        error = None
        if refusal:
            details = getattr(msg, "stop_details", None)
            error = f"refusal:{getattr(details, 'category', None)}"
        return LLMResponse(
            text=text,
            input_tokens=uncached + cache_read + cache_write,
            output_tokens=output,
            cached_input_tokens=cache_read,
            latency_ms=latency_ms,
            cost_usd=cost_usd(
                price,
                uncached_input=uncached,
                output=output,
                cache_read=cache_read,
                cache_write=cache_write,
            ),
            request_id=getattr(msg, "_request_id", None) or getattr(msg, "id", None),
            stop_reason=stop_reason,
            refusal=refusal,
            error=error,
        )
