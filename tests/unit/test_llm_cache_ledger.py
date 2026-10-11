import math
from pathlib import Path

import pytest
from llm_fakes import req

from finswarm.llm.cache import ResponseCache, cache_key
from finswarm.llm.client import LLMResponse
from finswarm.llm.errors import ReplayMiss
from finswarm.llm.ledger import Ledger
from finswarm.llm.pricing import Price, cost_usd, estimate_cost, load_catalog


def test_cache_key_is_stable_and_ignores_tags() -> None:
    a = req(1, tags={"run_id": "parent", "obs_json": "{}"})
    b = req(1, tags={"run_id": "branch", "agent": "x"})
    assert cache_key(a) == cache_key(b)
    # same content built in a different key order -> same key
    c = req(1, messages=[{"content": "hello 1", "role": "user"}])
    assert cache_key(a) == cache_key(c)
    assert len(cache_key(a)) == 64


@pytest.mark.parametrize(
    "change",
    [
        {"seed": 2},
        {"model": "other"},
        {"prompt_version": "fish_v1@abc"},
        {"system": "sys2"},
        {"max_tokens": 4096},
        {"temperature": 1.0},
        {"effort": "low"},
        {"schema": {"type": "object"}},
        {"messages": [{"role": "user", "content": "other"}]},
    ],
)
def test_cache_key_changes_with_each_field(change: dict) -> None:
    assert cache_key(req(1)) != cache_key(req(1, **change))


def test_live_then_replay(tmp_path: Path) -> None:
    path = tmp_path / "cache.sqlite"
    live = ResponseCache(path, mode="live")
    r = req(3)
    assert live.get(r) is None
    live.put(r, LLMResponse(text="hi", input_tokens=5, cost_usd=0.1, request_id="x"))
    live.put(req(4), LLMResponse(text="", error="boom"))  # errors are never cached
    assert len(live) == 1
    live.close()

    replay = ResponseCache(path, mode="replay")
    hit = replay.get(r)
    assert hit is not None and hit.cache_hit and hit.text == "hi" and hit.cost_usd == 0.0
    with pytest.raises(ReplayMiss):
        replay.get(req(4))
    replay.put(req(5), LLMResponse(text="new"))  # replay never writes
    assert len(replay) == 1

    mock = ResponseCache(path, mode="mock")
    assert mock.get(r) is None


def test_refusals_are_cached() -> None:
    c = ResponseCache()
    c.put(req(1), LLMResponse(text="", refusal=True, error="refusal:cyber"))
    hit = c.get(req(1))
    assert hit is not None and hit.refusal


def test_ledger_aggregates_and_budget() -> None:
    led = Ledger(lane_budgets={"anthropic_haiku": 1.0}, global_cap_usd=2.0)
    t0 = 1_760_000_000.0
    h = {"lane": "anthropic_haiku", "provider": "anthropic", "model": "claude-haiku-5-5"}
    led.record(**h, run_id="r1", cost_usd=0.25, attempts=2, ts=t0)
    led.record(**h, run_id="r2", cost_usd=0.5, ts=t0 + 86_400)
    led.record(**h, run_id="r2", status="cache_hit", attempts=0, ts=t0 + 86_400)
    led.record(lane="mistral_free", provider="mistral", model="m", run_id="r1", ts=t0)
    assert led.spend(lane="anthropic_haiku") == pytest.approx(0.75)
    assert led.spend(run_id="r2") == pytest.approx(0.5)
    assert led.remaining_budget("anthropic_haiku") == pytest.approx(0.25)
    assert led.remaining_budget() == pytest.approx(1.25)
    assert math.isinf(led.remaining_budget("mistral_free"))
    by_day = led.aggregate("day")
    assert [row["calls"] for row in by_day] == [2, 2]
    by_lane = {row["lane"]: row for row in led.aggregate("lane")}
    assert by_lane["anthropic_haiku"]["cache_hits"] == 1
    assert by_lane["anthropic_haiku"]["attempts"] == 3
    day0 = by_day[0]["day"]
    assert led.requests_on_day("anthropic_haiku", day0) == 2
    with pytest.raises(ValueError):
        led.aggregate("ts; DROP TABLE calls")  # type: ignore[arg-type]


def test_ledger_mirrors_provider_calls_to_the_jsonl_cost_ledger(tmp_path: Path) -> None:
    import json

    jsonl = tmp_path / "results" / "cost_ledger.jsonl"
    led = Ledger(tmp_path / "cache" / "ledger.sqlite", jsonl_path=jsonl)
    h = {"lane": "anthropic_haiku", "provider": "anthropic", "model": "claude-haiku-5-5"}
    t0 = 1_760_000_000.0
    led.record(**h, run_id="r1", input_tokens=1500, output_tokens=300, cost_usd=0.25, ts=t0)
    led.record(**h, run_id="r1", status="cache_hit", attempts=0, ts=t0 + 1)  # not a call
    led.record(**h, run_id="r1", status="error", error="boom", cost_usd=0.0, ts=t0 + 2)
    lines = [json.loads(ln) for ln in jsonl.read_text(encoding="utf-8").splitlines()]
    assert [ln["status"] for ln in lines] == ["ok", "error"]
    first = lines[0]
    # the keys scripts/sprint_status.py and scripts/cost_report.py read
    assert first["usd"] == first["cost_usd"] == 0.25
    assert (first["model"], first["lane"], first["run_id"]) == (
        "claude-haiku-5-5",
        "anthropic_haiku",
        "r1",
    )
    assert (first["input_tokens"], first["output_tokens"]) == (1500, 300)
    assert first["ts"].startswith("2025-10-09T") and first["ts"].endswith("Z")  # ISO UTC
    assert sum(ln["usd"] for ln in lines) == pytest.approx(led.spend())
    assert Ledger().jsonl_path is None  # in-memory ledgers (mock, tests) never mirror


def test_cache_dir_and_cost_ledger_follow_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from finswarm.llm.pricing import cache_dir, cost_ledger_path, repo_root
    from finswarm.llm.scheduler import LaneScheduler

    monkeypatch.delenv("FINSWARM_CACHE_DIR", raising=False)
    monkeypatch.delenv("FINSWARM_COST_LEDGER", raising=False)
    assert cache_dir() == repo_root() / "cache"
    assert cost_ledger_path() == repo_root() / "results" / "cost_ledger.jsonl"
    shared = tmp_path / "root" / "cache"
    monkeypatch.setenv("FINSWARM_CACHE_DIR", str(shared))
    monkeypatch.setenv("FINSWARM_COST_LEDGER", str(tmp_path / "root" / "results" / "cl.jsonl"))
    assert cache_dir() == shared
    # every worktree's scheduler then shares one cache, one SQLite ledger and one JSONL mirror
    import asyncio

    s = LaneScheduler.from_catalog(lanes=["mock"], mode="live")
    try:
        assert Path(s.cache.path) == shared / "llm_cache.sqlite"
        assert Path(s.ledger.path) == shared / "llm_ledger.sqlite"
        assert s.ledger.jsonl_path == tmp_path / "root" / "results" / "cl.jsonl"
    finally:
        asyncio.run(s.aclose())


def test_pricing_from_catalog() -> None:
    cat = load_catalog()
    haiku = cat.price("claude-haiku-5-5")
    assert haiku is not None and haiku.verified
    assert (haiku.input, haiku.output, haiku.cache_read, haiku.cache_write) == (
        0.10,
        0.50,
        0.01,
        0.125,
    )
    assert cat.price("claude-sonnet-5-5").input == 2.00
    assert cat.price("claude-opus-5-5").output == 20.00
    assert cat.price("ministral-8b-latest").free
    assert not cat.price("meta-llama/llama-3.3-70b-instruct").free
    assert set(cat.lanes) >= {
        "anthropic_haiku",
        "anthropic_sonnet",
        "mistral_free",
        "openrouter_llama",
        "gemini_free",
        "openrouter_free",
        "mock",
    }
    c = cost_usd(haiku, uncached_input=1000, output=500, cache_read=10_000, cache_write=2000)
    assert c == pytest.approx((1000 * 0.10 + 500 * 0.50 + 10_000 * 0.01 + 2000 * 0.125) / 1e6)
    # long-context rate card above 100k prompt tokens
    long = cost_usd(haiku, uncached_input=150_000, output=1000)
    assert long == pytest.approx((150_000 * 0.50 + 1000 * 2.50) / 1e6)
    assert estimate_cost(Price(), req(1)) == 0.0
    assert estimate_cost(haiku, req(1, max_tokens=2048)) >= 2048 * 0.50 / 1e6
