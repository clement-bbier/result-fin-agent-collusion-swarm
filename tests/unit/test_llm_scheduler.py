"""LaneScheduler under a virtual clock: rates, retries, AIMD, budgets, quotas, cache."""

import asyncio
import calendar
import json

import pytest
from llm_fakes import ScriptedClient, max_in_window, mock_lane, req

from finswarm.config.schema import LaneConfig
from finswarm.llm.cache import ResponseCache
from finswarm.llm.client import LLMResponse
from finswarm.llm.errors import LaneExhausted, RateLimited, ReplayMiss, RetryableError
from finswarm.llm.ledger import Ledger
from finswarm.llm.providers.mock import Flaky, MockLLM
from finswarm.llm.scheduler import (
    BudgetPolicy,
    LaneScheduler,
    LaneState,
    SlidingWindow,
    load_budget_policy,
)
from finswarm.llm.timing import VirtualClock


def _sched(lane: LaneConfig, client, clock: VirtualClock, **kw) -> LaneScheduler:
    return LaneScheduler({"L": lane}, {"L": client}, clock=clock, **kw)


async def _gather(s: LaneScheduler, n: int, **kw) -> list[LLMResponse]:
    try:
        return await asyncio.gather(*(s.submit(req(i, **kw), "L") for i in range(n)))
    finally:
        await s.aclose()


def test_rates_never_exceeded_with_flaky_provider() -> None:
    clock = VirtualClock()
    flaky = Flaky(MockLLM(), p_429=0.25, p_5xx=0.05, latency_ms=150, seed=7, clock=clock)
    s = _sched(mock_lane(rps=2, rpm=40, max_in_flight=4), flaky, clock, seed=1)
    out = asyncio.run(_gather(s, 60))
    assert len(out) == 60 and all(r.error is None for r in out)
    assert flaky.n_429 > 0 and flaky.n_5xx > 0
    assert max_in_window(flaky.calls, 1.0) <= 2
    assert max_in_window(flaky.calls, 60.0) <= 40
    st = s.lane_stats("L")
    assert st.rate_limited == flaky.n_429
    assert st.dispatched == len(flaky.calls) == 60 + st.retries


def test_sliding_windows_rates_tokens_and_scale() -> None:
    w = SlidingWindow.per_second(2.5)  # 2 requests per 0.8 s == 2.5 rps exactly
    assert (w.limit, w.window) == (2, 0.8)
    assert SlidingWindow.per_minute(30).window == 60.0
    tpm = SlidingWindow(3000, 60.0, kind="tpm")
    tpm.add(0.0, 1018)
    second = tpm.add(0.0, 1018)
    assert tpm.wait_time(0.0, 1018) == pytest.approx(60.0)
    assert tpm.wait_time(30.0, 1018) == pytest.approx(30.0)
    assert tpm.wait_time(30.0, 1018, scale=0.5) == pytest.approx(90.0)  # window stretched
    assert tpm.wait_time(30.0, 900) == 0.0
    second[1] = 15.0  # corrected with the actual token count after the response
    assert tpm.wait_time(1.0, 1018) == 0.0
    assert tpm.wait_time(1.0, 5000) == pytest.approx(59.0)  # oversized: wait for empty window
    assert tpm.wait_time(61.0, 5000) == 0.0


def test_tpm_lane_delays_dispatch() -> None:
    clock = VirtualClock()
    heavy = LLMResponse(text="ok", input_tokens=2000, output_tokens=900)
    client = ScriptedClient(default=heavy, clock=clock)
    s = _sched(mock_lane(tpm=3000, max_in_flight=2), client, clock)
    # each call really uses 2900 tokens: one call per minute fits in tpm=3000
    asyncio.run(_gather(s, 3, max_tokens=2900))
    assert client.times == pytest.approx([0.0, 60.0, 120.0], abs=1e-3)

    light = ScriptedClient(clock=clock)  # 15 tokens actually used per call
    s2 = _sched(mock_lane(tpm=3000, max_in_flight=1), light, clock)
    t0 = clock.now()
    asyncio.run(_gather(s2, 3, max_tokens=2900))
    # the estimate (prompt + max_tokens) is corrected with the actual usage after each call
    assert light.times == pytest.approx([t0, t0, t0])


def test_retry_after_is_honoured_and_lane_cools() -> None:
    clock = VirtualClock()
    client = ScriptedClient(
        [RateLimited(retry_after=7.0), RateLimited(retry_after=7.0), "done"], clock=clock
    )
    s = _sched(mock_lane(max_in_flight=1), client, clock, seed=3)

    async def go() -> LLMResponse:
        try:
            return await s.submit(req(1), "L")
        finally:
            await s.aclose()

    resp = asyncio.run(go())
    assert resp.text == "done"
    t = client.times
    assert len(t) == 3
    assert t[1] - t[0] >= 7.0 - 1e-9 and t[2] - t[1] >= 7.0 - 1e-9
    assert s.lane_stats("L").retries == 2
    rows = s.ledger.aggregate("lane")
    assert rows[0]["attempts"] == 3
    # two 429s: scale 0.7 * 0.7 (plus a little additive recovery during the waits)
    assert 0.49 <= s.lane_scale("L") < 0.6


def test_aimd_decrease_and_additive_recovery() -> None:
    clock = VirtualClock()
    client = ScriptedClient([RateLimited(retry_after=1.0), "ok"], clock=clock)
    s = _sched(mock_lane(rps=10, max_in_flight=1), client, clock, backoff_base_s=0.001)

    async def go() -> None:
        try:
            await s.submit(req(1), "L")
        finally:
            await s.aclose()

    asyncio.run(go())
    scale = s.lane_scale("L")
    assert scale == pytest.approx(0.7, abs=0.002)
    clock.advance(60.0)  # +5 % of the configured rate per minute
    assert s.lane_scale("L") == pytest.approx(0.75, abs=0.002)
    clock.advance(600.0)
    assert s.lane_scale("L") == 1.0
    assert s.lane_state("L") == LaneState.OPEN


def test_cooling_state_after_429() -> None:
    clock = VirtualClock()
    s = _sched(mock_lane(max_in_flight=1), ScriptedClient(), clock)
    delay = s._on_retryable(s._lane("L"), RateLimited(retry_after=30.0), attempt=1)
    assert delay >= 30.0
    assert s.lane_state("L") == LaneState.COOLING
    clock.advance(delay + 0.1)
    assert s.lane_state("L") == LaneState.OPEN
    # 5xx without retry-after: backoff only, no cooling and no AIMD decrease
    delay = s._on_retryable(s._lane("L"), RetryableError("503", status=503), attempt=3)
    assert 0.0 <= delay <= 4.0
    assert s.lane_state("L") == LaneState.OPEN


def test_retries_exhausted_raises_lane_exhausted() -> None:
    clock = VirtualClock()
    client = ScriptedClient(default=RetryableError("503", kind="server", status=503), clock=clock)
    s = _sched(mock_lane(max_in_flight=1), client, clock, seed=0)

    async def go() -> None:
        try:
            await s.submit(req(1), "L")
        finally:
            await s.aclose()

    with pytest.raises(LaneExhausted) as ei:
        asyncio.run(go())
    assert ei.value.reason == "retries_exhausted"
    assert len(client.calls) == 6
    gaps = [b - a for a, b in zip(client.times, client.times[1:], strict=False)]
    assert all(0.0 <= g <= 60.0 + 1e-9 for g in gaps)


def test_long_retry_after_exhausts_lane_with_resume_time() -> None:
    clock = VirtualClock()
    client = ScriptedClient([RateLimited(retry_after=3600.0)], clock=clock)
    s = _sched(mock_lane(max_in_flight=1), client, clock)

    async def go() -> None:
        try:
            await s.submit(req(1), "L")
        finally:
            await s.aclose()

    with pytest.raises(LaneExhausted) as ei:
        asyncio.run(go())
    assert ei.value.reason == "quota" and ei.value.resume_at == pytest.approx(clock.wall() + 3600.0)
    assert s.lane_state("L") == LaneState.EXHAUSTED


def test_lane_budget_cap_raises() -> None:
    clock = VirtualClock()
    client = ScriptedClient(clock=clock, cost=0.0004)
    s = _sched(mock_lane(max_in_flight=2, budget_usd=0.001), client, clock)

    async def go() -> int:
        ok = 0
        try:
            for i in range(10):
                await s.submit(req(i), "L")
                ok += 1
        finally:
            await s.aclose()
        return ok

    with pytest.raises(LaneExhausted) as ei:
        asyncio.run(go())
    assert ei.value.reason == "budget"
    assert len(client.calls) == 3  # 0.0008 <= 0.001 allowed the third call
    assert s.lane_state("L") == LaneState.EXHAUSTED
    assert s.ledger.spend(lane="L") == pytest.approx(0.0012)


def test_global_paid_cap_with_reservations_and_warnings() -> None:
    clock = VirtualClock()
    client = ScriptedClient(clock=clock, cost=0.0006)
    lane = LaneConfig(provider="anthropic", models=["claude-haiku-5-5"], max_in_flight=4)
    alerts: list[str] = []
    budget = BudgetPolicy(anthropic_total_usd=200.0, warn_usd=(0.001,), stop_usd=0.004)
    s = _sched(lane, client, clock, budget=budget, on_alert=alerts.append)
    haiku_req = {"provider": "anthropic", "model": "claude-haiku-5-5", "max_tokens": 2048}

    async def go() -> None:
        try:
            for i in range(50):
                await s.submit(req(i, **haiku_req), "L")
        finally:
            await s.aclose()

    with pytest.raises(LaneExhausted) as ei:
        asyncio.run(go())
    assert ei.value.reason == "global_cap"
    # the guard reserves max_tokens (2048) of output (~$0.00103): spend + reservation of
    # the 6th call would exceed $0.004, so exactly 5 calls went out
    assert len(client.calls) == 5
    assert s.global_spend() == pytest.approx(0.003)
    assert s.lane_state("L") == LaneState.EXHAUSTED
    assert len(alerts) == 1 and "crossed" in alerts[0]


def test_rpd_quota_and_utc_rollover() -> None:
    midnight = calendar.timegm((2026, 10, 10, 0, 0, 0))
    clock = VirtualClock(wall_start=midnight - 120.0)
    client = ScriptedClient(clock=clock)
    ledger = Ledger()
    s = _sched(mock_lane(rpd=3, max_in_flight=1), client, clock, ledger=ledger)

    async def submit(i: int) -> None:
        await s.submit(req(i), "L")

    async def day1() -> None:
        for i in range(3):
            await submit(i)
        with pytest.raises(LaneExhausted) as ei:
            await submit(3)
        assert ei.value.reason == "rpd" and ei.value.resume_at == midnight
        # a second scheduler sharing the ledger sees today's count
        s2 = _sched(mock_lane(rpd=3), ScriptedClient(clock=clock), clock, ledger=ledger)
        with pytest.raises(LaneExhausted):
            await s2.submit(req(9), "L")
        await s2.aclose()

    asyncio.run(day1())
    assert s.lane_state("L") == LaneState.EXHAUSTED
    clock.advance(200.0)  # past 00:00 UTC

    async def day2() -> None:
        try:
            await submit(4)
        finally:
            await s.aclose()

    asyncio.run(day2())
    assert len(client.calls) == 4
    assert s.lane_state("L") == LaneState.OPEN


def test_cache_hits_skip_buckets_and_cost() -> None:
    clock = VirtualClock()
    client = ScriptedClient(clock=clock, cost=0.01)
    s = _sched(mock_lane(rps=1, rpd=2, max_in_flight=1), client, clock)

    async def go() -> list[LLMResponse]:
        try:
            return [await s.submit(req(1), "L") for _ in range(5)]
        finally:
            await s.aclose()

    out = asyncio.run(go())
    assert len(client.calls) == 1
    assert [r.cache_hit for r in out] == [False, True, True, True, True]
    assert clock.now() == 0.0  # no rate-limit wait happened
    st = s.lane_stats("L")
    assert st.cache_hits == 4 and st.dispatched == 1
    assert s.ledger.spend(lane="L") == pytest.approx(0.01)
    assert s.ledger.aggregate("lane")[0]["cache_hits"] == 4


def test_replay_mode_hits_and_misses() -> None:
    clock = VirtualClock()
    cache = ResponseCache(mode="live")
    live = _sched(mock_lane(), ScriptedClient(["first"], clock=clock), clock, cache=cache)

    async def run_live() -> None:
        await live.submit(req(1), "L")
        await live.aclose()

    asyncio.run(run_live())
    cache.mode = "replay"
    replay_client = ScriptedClient(clock=clock)
    replay = _sched(mock_lane(), replay_client, clock, cache=cache)

    async def run_replay() -> LLMResponse:
        try:
            r = await replay.submit(req(1), "L")
            with pytest.raises(ReplayMiss):
                await replay.submit(req(2), "L")
            return r
        finally:
            await replay.aclose()

    r = asyncio.run(run_replay())
    assert r.text == "first" and r.cache_hit
    assert replay_client.calls == []


def test_fifo_within_run_fair_across_runs_and_priority() -> None:
    clock = VirtualClock()
    client = ScriptedClient(clock=clock)
    s = _sched(mock_lane(max_in_flight=1), client, clock)
    plan = [("A", 0)] * 4 + [("B", 0)] * 2 + [("C", -1)]

    async def go() -> None:
        try:
            await asyncio.gather(
                *(
                    s.submit(req(i, tags={"run_id": run, "priority": str(p)}), "L")
                    for i, (run, p) in enumerate(plan)
                )
            )
        finally:
            await s.aclose()

    asyncio.run(go())
    order = [(r.tags["run_id"], r.seed) for _, r in client.calls]
    assert order == [("C", 6), ("A", 0), ("B", 4), ("A", 1), ("B", 5), ("A", 2), ("A", 3)]


def test_mock_mode_serves_mock_without_ledger() -> None:
    clock = VirtualClock()
    lane = LaneConfig(provider="anthropic", models=["claude-haiku-5-5"])
    s = _sched(lane, ScriptedClient(clock=clock), clock, cache=ResponseCache(mode="mock"))
    obs = {"period": 0, "slot": 0, "cost": 1.0, "history": [], "output": "price_json"}
    r = req(1, provider="anthropic", model="claude-haiku-5-5", tags={"obs_json": json.dumps(obs)})
    resp = asyncio.run(s.submit(r, "L"))
    assert json.loads(resp.text)["price"] == pytest.approx(1.47)
    assert len(s.ledger) == 0


def test_request_must_match_lane() -> None:
    s = _sched(mock_lane(), ScriptedClient(), VirtualClock())
    with pytest.raises(ValueError):
        asyncio.run(s.submit(req(1, provider="anthropic"), "L"))
    with pytest.raises(ValueError):
        asyncio.run(s.submit(req(1, model="other"), "L"))
    with pytest.raises(KeyError):
        asyncio.run(s.submit(req(1), "nope"))


def test_from_catalog_and_burst_of_429s_is_one_event(tmp_path) -> None:
    s = LaneScheduler.from_catalog(
        lanes=["mock", "anthropic_haiku"],
        cache_path=tmp_path / "c.sqlite",
        ledger_path=tmp_path / "l.sqlite",
        budget_path=tmp_path / "none.yaml",
        clock=VirtualClock(),
    )
    assert s.cache.mode == "live" and s.ledger.lane_budgets["anthropic_haiku"] == 60.0
    assert s.budget.hard_stop == 195.0 and s.max_attempts == 6 and s.backoff_cap_s == 60.0
    lane = s._lane("anthropic_haiku")
    for _ in range(5):  # five concurrent workers hit 429 at the same instant
        s._on_retryable(lane, RateLimited(retry_after=0.5), attempt=1)
    assert s.lane_scale("anthropic_haiku") == pytest.approx(0.7)
    assert s.lane_stats("anthropic_haiku").rate_limited == 5


def test_budget_policy_defaults_and_file(tmp_path) -> None:
    d = load_budget_policy(tmp_path / "missing.yaml")
    assert (d.anthropic_total_usd, d.warn_usd, d.stop_usd) == (200.0, (100.0, 150.0), 195.0)
    p = tmp_path / "budget.yaml"
    p.write_text("anthropic_total_usd: 50\nwarn_usd: [10, 20]\nstop_usd: 45\nother: 1\n")
    b = load_budget_policy(p)
    assert b.hard_stop == 45.0 and b.warn_usd == (10.0, 20.0)
