import asyncio
import math

import numpy as np
import pytest

from finswarm.agents.base import Agent
from finswarm.agents.scripted import (
    ForcedDeviationWrapper,
    GrimTrigger,
    NoRegretMM,
    RandomPricer,
    ReplayRival,
    StingDealer,
    TitForTat,
    Undercutter,
    make_scripted,
)
from finswarm.agents.scripted.registry import SCRIPTED_POLICIES, parse_policy
from finswarm.agents.scripted.strategies import demand_profit_fn
from finswarm.config.schema import MarketConfig
from finswarm.core.types import AgentKind, Observation, PeriodRecord, Quote
from finswarm.sim.online_metrics import Benchmarks, closed_form_benchmarks


def obs(period=0, rivals=(), own=float("nan"), cost=1.0, standing=None):
    hist = []
    if rivals:
        hist = [PeriodRecord(period - 1, own, 0.0, 0.0, tuple(rivals), float("nan"))]
    quotes = []
    if standing is not None:
        quotes = [
            Quote(agent_id=9 + i, ask_ticks=round((p - 1.0) / 0.01)) for i, p in enumerate(standing)
        ]
    return Observation(
        tick=period,
        period=period,
        agent_id=0,
        slot=0,
        role="dealer",
        inventory=0.0,
        cash=0.0,
        cost=cost,
        history=hist,
        rival_quotes=quotes,
        extra={"p_min": 1.0, "tick_size": 0.01, "price_max": 4.0},
    )


def decide(agent, o):
    return asyncio.run(agent.decide(o))


def test_closed_form_benchmarks_match_calvano() -> None:
    b = closed_form_benchmarks(MarketConfig())
    assert b.p_nash == pytest.approx(1.4729, abs=1e-4)
    assert b.p_mono == pytest.approx(1.9250, abs=1e-4)
    assert b.pi_nash == pytest.approx(22.29, abs=1e-2)  # 0.2229 * alpha * beta
    assert b.pi_mono == pytest.approx(33.75, abs=1e-2)  # 0.3375 * alpha * beta
    b3 = closed_form_benchmarks(MarketConfig(alpha=3.2))
    assert b3.p_nash == pytest.approx(1.4729 * 3.2, abs=1e-3)


def test_every_policy_implements_the_agent_protocol() -> None:
    required = {
        "replay_rival": {"schedule": {0: [(0, float("nan"), 1.5)]}},
        "sting_dealer": {"delta": 0.1},
    }
    for name in SCRIPTED_POLICIES:
        a = make_scripted(name, required.get(name, {}), slot=1, agent_id=7, seed=[1, 2])
        assert isinstance(a, Agent)
        assert a.slot == 1 and a.agent_id == 7
        d = decide(a, obs(1, rivals=(1.8,)))
        assert d.action in ("quote", "hold")


def test_tit_for_tat_punishes_then_forgives() -> None:
    a = TitForTat(0, 0, target=1.9, trigger=0.05, punish=2)
    assert decide(a, obs(1, rivals=(1.9,))).ask == 1.9
    assert decide(a, obs(2, rivals=(1.88,))).ask == 1.9  # within tolerance
    assert decide(a, obs(3, rivals=(1.6,))).ask == pytest.approx(1.6)  # mirrors the undercut
    assert decide(a, obs(4, rivals=(1.9,))).ask == pytest.approx(1.6)  # punish 1/2
    assert decide(a, obs(5, rivals=(1.9,))).ask == pytest.approx(1.6)  # punish 2/2
    assert decide(a, obs(6, rivals=(1.9,))).ask == 1.9


def test_grim_trigger_never_forgives() -> None:
    a = GrimTrigger(0, 0, target=1.9, punish_price=1.47)
    assert decide(a, obs(1, rivals=(1.9,))).ask == 1.9
    assert decide(a, obs(2, rivals=(1.5,))).ask == 1.47
    assert all(decide(a, obs(t, rivals=(1.9,))).ask == 1.47 for t in range(3, 10))


def test_undercutter_floors_at_cost_or_floor() -> None:
    a = Undercutter(0, 0, eps=0.05, start=1.9)
    assert decide(a, obs(0)).ask == 1.9
    assert decide(a, obs(1, rivals=(1.7, 1.8))).ask == pytest.approx(1.65)
    assert decide(a, obs(2, rivals=(1.02,))).ask == 1.0  # cost floor
    b = Undercutter(0, 0, eps=0.05, floor=1.47)
    assert decide(b, obs(2, rivals=(1.3,))).ask == 1.47


def test_random_pricer_bounds_and_determinism() -> None:
    a, b = (
        RandomPricer(0, 0, lo=1.5, hi=1.6, seed=[3]),
        RandomPricer(0, 0, lo=1.5, hi=1.6, seed=[3]),
    )
    xs = [decide(a, obs(t)).ask for t in range(50)]
    assert all(1.5 <= x <= 1.6 for x in xs)
    assert xs == [decide(b, obs(t)).ask for t in range(50)]


def _profit(rival: float):
    return demand_profit_fn(alpha=1.0, a=2.0, a0=0.0, tau=0.25, beta=100.0, cost=1.0)


@pytest.mark.parametrize("mode", ["exp3", "hedge"])
def test_no_regret_average_regret_decreases_on_fixed_opponent(mode: str) -> None:
    rival = 1.7
    profit = _profit(rival)
    horizon = 3000
    a = NoRegretMM(
        0,
        0,
        lo=1.0,
        hi=2.2,
        n_prices=13,
        mode=mode,
        horizon=horizon,
        reward_scale=60.0,
        profit_fn=profit,
        seed=[11],
    )
    best = max(profit(float(p), [rival]) for p in a.grid)
    realised = 0.0
    avg_regret = {}
    for t in range(1, horizon + 1):
        d = a.act(obs(t, rivals=(rival,)))
        pi = profit(d.ask, [rival])
        realised += pi
        a.record_period(PeriodRecord(t, d.ask, 0.0, pi, (rival,), float("nan")))
        if t in (100, 500, horizon):
            avg_regret[t] = (t * best - realised) / t
    assert avg_regret[horizon] < avg_regret[500] < avg_regret[100]
    assert avg_regret[horizon] < 0.25 * best


def test_no_regret_state_roundtrip_reproduces_choices() -> None:
    a = NoRegretMM(0, 0, lo=1.0, hi=2.0, n_prices=11, seed=[5])
    for t in range(20):
        d = a.act(obs(t, rivals=(1.7,)))
        a.record_period(PeriodRecord(t, d.ask, 1.0, 20.0 + t % 3, (1.7,), float("nan")))
    st = a.state()
    xs = [a.act(obs(100 + t, rivals=(1.7,))).ask for t in range(10)]
    b = NoRegretMM(0, 0, lo=1.0, hi=2.0, n_prices=11, seed=[999])
    b.restore(st)
    assert xs == [b.act(obs(100 + t, rivals=(1.7,))).ask for t in range(10)]


def test_sting_dealer_phases_and_log() -> None:
    a = StingDealer(0, 0, delta=0.2, shadow_epochs=2, hold_epochs=3, probes=2, event_id=4)
    prices = []
    for t in range(12):
        prices.append(decide(a, obs(t, standing=(1.70, 1.80))).ask)
    shadow = pytest.approx(1.75)
    assert prices[0:2] == [shadow, shadow]
    assert prices[2:5] == [pytest.approx(1.95)] * 3
    assert prices[5:7] == [shadow, shadow]
    assert prices[7:10] == [pytest.approx(1.95)] * 3
    assert prices[10:] == [shadow, shadow]
    phases = [(e["phase"], e["probe"], e["step"]) for e in a.drain_events()]
    assert phases == [
        ("shadow", 0, 0),
        ("raise", 0, 2),
        ("shadow", 1, 5),
        ("raise", 1, 7),
        ("done", 1, 10),
    ]
    assert a.drain_events() == []
    assert all(e["event_id"] == 4 for e in a.phase_log)


def test_replay_rival_replays_open_loop() -> None:
    nan = float("nan")
    sched = {0: [(0, nan, 1.8)], 1: [(1, nan, 1.7)], 3: [(3, nan, nan)], 4: [(5, nan, 1.6)]}
    a = ReplayRival(0, 1, schedule=sched)
    assert decide(a, obs(0, rivals=(1.0,))).ask == 1.8  # ignores the market
    assert decide(a, obs(1)).ask == 1.7
    assert decide(a, obs(2)).action == "hold"  # no row logged for the period: keep the quote
    # a logged decision with NaN on both sides is a withdrawn quote: replayed as a cancel, so
    # the ghost's rival does not keep a quote the source run no longer had
    d3 = decide(a, obs(3))
    assert d3.action == "cancel" and d3.price is None and d3.ask is None
    # with the global epoch in the observation, rows are matched by epoch
    o4 = obs(4)
    o4.extra["epoch"] = 4
    assert decide(a, o4).action == "hold"  # the source agent did not decide at epoch 4
    o4.extra["epoch"] = 5
    assert decide(a, o4).ask == 1.6


def test_forced_deviation_wrapper_cuts_then_hands_back() -> None:
    inner = TitForTat(0, 0, target=1.9, trigger=0.05, punish=1)
    decide(inner, obs(0, rivals=(1.9,)))
    w = ForcedDeviationWrapper(inner, delta=0.3, periods=2, event_id=1)
    assert w.kind == AgentKind.SCRIPTED and w.slot == 0 and w.memory is inner.memory
    n = inner.n_decisions
    for t in (1, 2):
        d = decide(w, obs(t, rivals=(1.9,), own=1.9))
        assert d.ask == pytest.approx(1.6)
        w.record_period(PeriodRecord(t, d.ask, 1.0, 1.0, (1.9,), float("nan")))
    assert w.done and inner.n_decisions == n  # inner never consulted while forced
    assert len(inner.memory.history) == 2
    st = w.state()
    w2 = ForcedDeviationWrapper(TitForTat(0, 0, target=1.9), delta=0.3, periods=2)
    w2.restore(st)
    assert w2.left == 0 and math.isclose(w2.forced_price, 1.6)


def test_registry_inline_params_defaults_and_strictness() -> None:
    assert parse_policy("undercutter(eps=0.02, start=1.9)") == (
        "undercutter",
        {"eps": 0.02, "start": 1.9},
    )
    assert parse_policy("static_nash") == ("static_nash", {})
    b = Benchmarks(p_nash=2.0, pi_nash=1.0, p_mono=3.0, pi_mono=2.0)
    tft = make_scripted("tft", None, slot=0, benchmarks=b)
    assert isinstance(tft, TitForTat) and tft.target == 3.0
    tft2 = make_scripted("tit_for_tat(target=2.5)", None, slot=0, benchmarks=b)
    assert tft2.target == 2.5
    assert make_scripted("static_nash", {}, slot=0, benchmarks=b).price == 2.0
    with pytest.raises(KeyError):
        make_scripted("teleporter", {}, slot=0)
    with pytest.raises(TypeError):
        make_scripted("undercutter", {"epsilon": 0.1}, slot=0)
    hedge = make_scripted("no_regret_mm", {"mode": "hedge"}, slot=0, market=MarketConfig())
    assert hedge.mode == "hedge" and hedge.profit_fn is not None


def test_mock_llm_standin_emits_call_metadata() -> None:
    a = make_scripted("mock_llm(fail_rate=0.0)", {"model": "m"}, slot=0, seed=[1])
    assert a.kind == AgentKind.LLM
    d = decide(a, obs(1, rivals=(1.8,), own=1.7))
    assert d.parse_ok and d.raw.startswith("{")
    assert a.last_call["provider"] == "mock" and a.last_call["cost_usd"] == 0.0
    flaky = make_scripted("mock_llm(fail_rate=1.0)", {}, slot=0, seed=[1])
    assert not decide(flaky, obs(1)).parse_ok


def test_scripted_agent_state_roundtrip_with_memory() -> None:
    a = make_scripted("random_pricer", {"lo": 1.0, "hi": 2.0}, slot=0, seed=[4])
    for t in range(5):
        decide(a, obs(t))
        a.record_period(PeriodRecord(t, 1.5, 2.0, 3.0, (1.6,), 1.55))
    st = a.state()
    b = make_scripted("random_pricer", {"lo": 1.0, "hi": 2.0}, slot=0, seed=[77])
    b.restore(st)
    assert len(b.memory.history) == 5
    assert np.isclose(decide(a, obs(9)).ask, decide(b, obs(9)).ask)
