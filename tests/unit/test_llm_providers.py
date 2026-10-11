"""Provider adapters, offline: request shapes and error mapping with fake SDK clients."""

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import anthropic
import httpx
import httpx2
import pytest
from llm_fakes import req

from finswarm.agents.llm.schemas import PriceDecision, strict_json_schema
from finswarm.config.schema import LaneConfig
from finswarm.llm.client import LLMClient
from finswarm.llm.errors import FatalLLMError, RateLimited, RetryableError
from finswarm.llm.providers import make_client
from finswarm.llm.providers.anthropic_ import AnthropicClient
from finswarm.llm.providers.mistral_ import MistralClient, map_mistral_error
from finswarm.llm.providers.mock import Flaky, MockLLM
from finswarm.llm.providers.openai_compat import OpenAICompatClient
from finswarm.llm.scheduler import LaneScheduler
from finswarm.llm.timing import VirtualClock

SCHEMA = strict_json_schema(PriceDecision)
ANSWER = '{"observations": "o", "plans": "p", "insights": "i", "price": 1.8}'


def _message(text: str = ANSWER, stop_reason: str = "end_turn", category: str | None = None):
    return SimpleNamespace(
        content=[
            SimpleNamespace(type="thinking", thinking=""),
            SimpleNamespace(type="text", text=text),
        ],
        usage=SimpleNamespace(
            input_tokens=100,
            output_tokens=50,
            cache_read_input_tokens=600,
            cache_creation_input_tokens=0,
        ),
        stop_reason=stop_reason,
        stop_details=SimpleNamespace(category=category) if category else None,
        _request_id="req_123",
        id="msg_1",
    )


class _FakeMessages:
    def __init__(self, script: list[Any]) -> None:
        self.script = list(script)
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class _FakeAnthropic:
    def __init__(self, *script: Any) -> None:
        self.messages = _FakeMessages(list(script))


def _haiku_req(**kw: Any):
    base = {
        "provider": "anthropic",
        "model": "claude-haiku-5-5",
        "system": "STABLE SYSTEM PROMPT",
        "messages": [{"role": "user", "content": "volatile market state"}],
        "schema": SCHEMA,
        "max_tokens": 512,
        "effort": None,
    }
    base.update(kw)
    return req(**base)


def _status_error(cls: type, status: int, msg: str, headers: dict[str, str] | None = None):
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx2.Response(status, headers=headers or {}, request=request)
    return cls(msg, response=response, body=None)


def test_anthropic_request_shape() -> None:
    fake = _FakeAnthropic(_message())
    client = AnthropicClient(client=fake)
    assert isinstance(client, LLMClient)
    resp = asyncio.run(client.complete(_haiku_req()))
    (kw,) = fake.messages.calls
    for forbidden in ("temperature", "top_p", "top_k", "thinking", "tool_choice", "tools"):
        assert forbidden not in kw
    assert kw["model"] == "claude-haiku-5-5"
    assert kw["max_tokens"] == 2048  # raised to the floor for thinking models
    assert kw["output_config"]["effort"] == "low"
    assert kw["output_config"]["format"] == {"type": "json_schema", "schema": SCHEMA}
    assert kw["system"] == [
        {"type": "text", "text": "STABLE SYSTEM PROMPT", "cache_control": {"type": "ephemeral"}}
    ]
    assert kw["messages"] == [{"role": "user", "content": "volatile market state"}]
    assert resp.text == ANSWER  # thinking blocks are skipped
    assert resp.input_tokens == 700 and resp.cached_input_tokens == 600
    assert resp.output_tokens == 50 and resp.request_id == "req_123"
    assert resp.cost_usd == pytest.approx((100 * 0.10 + 600 * 0.01 + 50 * 0.50) / 1e6)
    assert not resp.refusal


def test_anthropic_never_sends_temperature_or_prefill() -> None:
    fake = _FakeAnthropic(_message())
    client = AnthropicClient(client=fake)
    with pytest.raises(FatalLLMError):
        asyncio.run(client.complete(_haiku_req(temperature=1.0)))
    prefill = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "{"}]
    with pytest.raises(FatalLLMError):
        asyncio.run(client.complete(_haiku_req(messages=prefill)))
    with pytest.raises(FatalLLMError):
        asyncio.run(client.complete(_haiku_req(model="claude-unknown-model")))
    assert fake.messages.calls == []
    kw = client.build_params(_haiku_req(effort="medium", max_tokens=4096))
    assert kw["output_config"]["effort"] == "medium" and kw["max_tokens"] == 4096


def test_anthropic_capture_limits_uses_raw_response() -> None:
    class _Raw:
        headers = {"anthropic-ratelimit-requests-limit": "50", "request-id": "req_1"}

        async def parse(self) -> Any:
            return _message()

    class _RawCreate:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        async def create(self, **kw: Any) -> Any:
            self.calls.append(kw)
            return _Raw()

    fake = _FakeAnthropic()
    fake.messages.with_raw_response = _RawCreate()  # type: ignore[attr-defined]
    client = AnthropicClient(client=fake, capture_limits=True)
    resp = asyncio.run(client.complete(_haiku_req()))
    assert resp.text == ANSWER and fake.messages.calls == []
    assert client.last_limits == {"anthropic-ratelimit-requests-limit": "50"}
    assert "temperature" not in fake.messages.with_raw_response.calls[0]


def test_health_check_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    from finswarm.llm import keys
    from finswarm.llm.lanes import health_check

    monkeypatch.setattr(keys, "_LOADED", True)  # never read a real .env in tests
    for name in ("ANTHROPIC_API_KEY", "MISTRAL_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    lines: list[str] = []
    res = health_check(out=lines.append)
    assert res["mock"]["status"] == "ok"
    assert res["anthropic_haiku"] == {
        "status": "unconfigured",
        "detail": "ANTHROPIC_API_KEY not set",
    }
    assert all(res[n]["status"] == "unconfigured" for n in res if n != "mock")
    assert any(line.startswith("unconfigured lanes (6)") for line in lines)


def test_anthropic_refusal_is_flagged_and_not_retried() -> None:
    fake = _FakeAnthropic(_message(text="", stop_reason="refusal", category="cyber"))
    clock = VirtualClock()
    lane = LaneConfig(provider="anthropic", models=["claude-haiku-5-5"])
    s = LaneScheduler({"h": lane}, {"h": AnthropicClient(client=fake)}, clock=clock)

    async def go():
        try:
            return await s.submit(_haiku_req(), "h")
        finally:
            await s.aclose()

    resp = asyncio.run(go())
    assert resp.refusal and resp.stop_reason == "refusal" and resp.error == "refusal:cyber"
    assert len(fake.messages.calls) == 1
    row = s.ledger.aggregate("lane")[0]
    assert row["non_ok"] == 1 and row["attempts"] == 1


def test_anthropic_error_mapping() -> None:
    rl = _status_error(anthropic.RateLimitError, 429, "slow down", {"retry-after": "3"})
    client = AnthropicClient(client=_FakeAnthropic(rl))
    with pytest.raises(RateLimited) as ei:
        asyncio.run(client.complete(_haiku_req()))
    assert ei.value.retry_after == 3.0

    overloaded = _status_error(anthropic.APIStatusError, 529, "overloaded")
    client = AnthropicClient(client=_FakeAnthropic(overloaded))
    with pytest.raises(RetryableError) as ei2:
        asyncio.run(client.complete(_haiku_req()))
    assert ei2.value.status == 529 and not isinstance(ei2.value, RateLimited)

    timeout = anthropic.APITimeoutError(request=httpx2.Request("POST", "https://x.invalid"))
    client = AnthropicClient(client=_FakeAnthropic(timeout))
    with pytest.raises(RetryableError) as ei3:
        asyncio.run(client.complete(_haiku_req()))
    assert ei3.value.kind == "timeout"

    auth = _status_error(anthropic.AuthenticationError, 401, "bad key")
    client = AnthropicClient(client=_FakeAnthropic(auth))
    with pytest.raises(FatalLLMError):
        asyncio.run(client.complete(_haiku_req()))


def test_anthropic_schema_rejection_falls_back_to_prompted_json() -> None:
    bad = _status_error(
        anthropic.BadRequestError, 400, "output_config.format.schema: unsupported keyword"
    )
    fake = _FakeAnthropic(bad, _message(), _message())
    client = AnthropicClient(client=fake)
    resp = asyncio.run(client.complete(_haiku_req()))
    assert resp.text == ANSWER
    first, second = fake.messages.calls
    assert "format" in first["output_config"]
    assert second["output_config"] == {"effort": "low"}
    assert "JSON schema" in second["messages"][-1]["content"]
    assert second["system"] == first["system"]  # cached prefix untouched
    assert "claude-haiku-5-5" in client.schema_rejected
    asyncio.run(client.complete(_haiku_req()))  # later calls skip the schema directly
    assert "format" not in fake.messages.calls[2]["output_config"]


def test_mistral_request_shape_and_errors() -> None:
    captured: list[dict[str, Any]] = []

    class _Chat:
        async def complete_async(self, **kw: Any) -> Any:
            captured.append(kw)
            return SimpleNamespace(
                id="m1",
                choices=[
                    SimpleNamespace(message=SimpleNamespace(content=ANSWER), finish_reason="stop")
                ],
                usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7),
            )

    client = MistralClient(client=SimpleNamespace(chat=_Chat()))
    r = req(5, provider="mistral", model="ministral-8b-latest", schema=SCHEMA, temperature=None)
    resp = asyncio.run(client.complete(r))
    (kw,) = captured
    assert kw["messages"][0] == {"role": "system", "content": "sys"}
    assert kw["random_seed"] == 5 and "temperature" not in kw
    assert kw["response_format"] == {"type": "json_object"}
    assert resp.text == ANSWER and resp.input_tokens == 11 and resp.cost_usd == 0.0
    kw2 = client.build_params(req(1, temperature=1.0))
    assert kw2["temperature"] == 1.0 and "response_format" not in kw2

    from mistralai.client.errors import SDKError

    raw = httpx2.Response(
        429,
        headers={"retry-after": "2"},
        request=httpx2.Request("POST", "https://api.mistral.ai/v1/chat/completions"),
    )
    mapped = map_mistral_error(SDKError("API error occurred", raw))
    assert isinstance(mapped, RateLimited) and mapped.retry_after == 2.0
    raw500 = httpx2.Response(503, request=raw.request)
    assert isinstance(map_mistral_error(SDKError("down", raw500)), RetryableError)
    raw400 = httpx2.Response(400, request=raw.request)
    assert isinstance(map_mistral_error(SDKError("bad", raw400)), FatalLLMError)


def test_openai_compat_request_shape_errors_and_limits() -> None:
    seen: list[httpx.Request] = []
    script = [
        httpx.Response(429, headers={"retry-after": "4"}, json={"error": "rate"}),
        httpx.Response(
            200,
            headers={"x-ratelimit-limit-requests": "1000", "x-ratelimit-remaining-requests": "9"},
            json={
                "id": "g1",
                "choices": [{"message": {"content": ANSWER}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 9,
                    "prompt_tokens_details": {"cached_tokens": 4},
                },
            },
        ),
        httpx.Response(401, json={"error": "bad key sk-test-123456789"}),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return script.pop(0)

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = OpenAICompatClient(
        "openrouter", "https://openrouter.ai/api/v1/", api_key="sk-test", http_client=http
    )
    r = req(
        3,
        provider="openrouter",
        model="meta-llama/llama-3.3-70b-instruct",
        schema=SCHEMA,
        temperature=0.7,
    )

    async def go():
        with pytest.raises(RateLimited) as ei:
            await client.complete(r)
        assert ei.value.retry_after == 4.0
        resp = await client.complete(r)
        assert client.last_limits["x-ratelimit-limit-requests"] == "1000"
        with pytest.raises(FatalLLMError):
            await client.complete(r)
        await client.aclose()
        return resp

    resp = asyncio.run(go())
    assert str(seen[0].url) == "https://openrouter.ai/api/v1/chat/completions"
    assert seen[0].headers["authorization"] == "Bearer sk-test"
    body = json.loads(seen[0].content)
    assert body["messages"][0] == {"role": "system", "content": "sys"}
    assert body["seed"] == 3 and body["temperature"] == 0.7
    assert body["response_format"] == {"type": "json_object"}
    assert resp.text == ANSWER and resp.cached_input_tokens == 4 and resp.request_id == "g1"

    keyless = OpenAICompatClient("gemini", "https://x.invalid/v1", api_key=None)
    with pytest.raises(FatalLLMError):
        asyncio.run(keyless.complete(r))


def test_mock_policies_and_flaky() -> None:
    hist = [
        {
            "period": 0,
            "own_price": 1.9,
            "own_quantity": 30,
            "own_profit": 20,
            "rival_prices": [1.92],
        },
        {
            "period": 1,
            "own_price": 1.9,
            "own_quantity": 30,
            "own_profit": 20,
            "rival_prices": [1.6],
        },
    ]
    obs = {"period": 2, "slot": 0, "cost": 1.0, "history": hist, "tick_size": 0.01}
    expected = {
        "static_nash": 1.47,
        "tit_for_tat": 1.6,
        "grim_trigger": 1.47,
        "undercutter": 1.59,
    }
    for policy, price in expected.items():
        assert MockLLM(policy).choose_price(obs, seed=0) == pytest.approx(price)
    first = {**obs, "history": []}
    assert MockLLM("tit_for_tat").choose_price(first, 0) == pytest.approx(1.92)
    assert MockLLM("grim_trigger").choose_price({**obs, "history": hist[:1]}, 0) == pytest.approx(
        1.92
    )
    rnd = MockLLM("random")
    assert rnd.choose_price(obs, 1) == rnd.choose_price(obs, 1)
    assert 1.47 <= rnd.choose_price(obs, 2) <= 1.93
    with pytest.raises(ValueError):
        MockLLM("nope")

    fish = asyncio.run(
        MockLLM("static_nash").complete(
            req(1, tags={"obs_json": json.dumps({**obs, "output": "fish_text"})})
        )
    )
    assert fish.text.strip().endswith("My chosen price:\n1.47") and fish.cost_usd == 0.0

    clock = VirtualClock()
    flaky = Flaky(MockLLM(), p_bad_json=1.0, latency_ms=250, clock=clock)
    out = asyncio.run(
        flaky.complete(req(1, tags={"obs_json": json.dumps({**obs, "output": "price_json"})}))
    )
    assert out.text.startswith("I think") and clock.now() == pytest.approx(0.25)
    always_429 = Flaky(MockLLM(), p_429=1.0, retry_after=2.0, clock=clock)
    with pytest.raises(RateLimited):
        asyncio.run(always_429.complete(req(1)))


def test_make_client_builds_each_kind_without_network(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "gm-test")
    assert isinstance(make_client("mock"), MockLLM)
    gemini = make_client("gemini")
    assert isinstance(gemini, OpenAICompatClient)
    assert gemini.url.endswith("/v1beta/openai/chat/completions")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    assert isinstance(make_client("anthropic"), AnthropicClient)
    with pytest.raises(KeyError):
        make_client("nope")


def test_anthropic_cache_control_on_system_and_history_blocks() -> None:
    from finswarm.agents.llm.llm_agent import LLMAgent
    from finswarm.core.types import Observation, PeriodRecord
    from finswarm.llm.providers.anthropic_ import AnthropicClient

    agent = LLMAgent(
        agent_id=1, slot=0, name="Dealer A", model="claude-haiku-5-5", provider="anthropic",
        lane="anthropic_haiku", scheduler=None, prompt="dealer_v1",  # type: ignore[arg-type]
    )  # fmt: skip
    hist = [PeriodRecord(t, 1.8, 10.0, 8.0, (1.75, 1.7, 1.9), float("nan")) for t in range(9)]
    obs = Observation(
        tick=9, period=9, agent_id=1, slot=0, role="dealer", inventory=0.0, cash=0.0,
        cost=1.0, history=hist, extra={"max_wtp": 2.5, "tick_size": 0.05, "p_min": 1.0},
    )  # fmt: skip
    req = agent.build_request(obs)
    kw = AnthropicClient(client=object()).build_params(req)
    assert kw["system"][-1]["cache_control"] == {"type": "ephemeral"}
    blocks = kw["messages"][0]["content"]
    marked = [i for i, b in enumerate(blocks) if "cache_control" in b]
    assert marked == [len(blocks) - 2]  # the last history block, never the volatile one
    assert blocks[-2]["text"].startswith("Round 9:\n") and blocks[-2]["cache_control"] == {
        "type": "ephemeral"
    }
    assert blocks[0]["text"].startswith("First I will show you the market data")
    assert "My current state:" in blocks[-1]["text"] and "Round" not in blocks[-1]["text"]
    assert "cache_control" not in req.messages[0]["content"][-2]  # the request is not mutated
    # repair turns and the prompted-JSON fallback keep the breakpoint on the history block
    rep = req.model_copy(update={"messages": [*req.messages, {"role": "user", "content": "fix"}]})
    kw2 = AnthropicClient(client=object()).build_params(rep, use_schema=False)
    assert "cache_control" in kw2["messages"][0]["content"][-2]
    # string content (other templates, sliding window) gets no message breakpoint
    plain = req.model_copy(update={"messages": [{"role": "user", "content": "x"}]})
    kw3 = AnthropicClient(client=object()).build_params(plain)
    assert kw3["messages"][0]["content"] == "x"
