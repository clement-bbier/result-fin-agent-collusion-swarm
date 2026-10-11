"""LLMAgent end to end with the mock lane: decisions, repair, fallback, memory, replace."""

import asyncio
import json

import pytest
from llm_fakes import ScriptedClient, mock_lane

from finswarm.agents.base import Agent
from finswarm.agents.llm.llm_agent import LLMAgent
from finswarm.config.schema import LLMSlotConfig, LLMTierConfig
from finswarm.core.types import Observation, PeriodRecord, Side
from finswarm.llm.cache import ResponseCache
from finswarm.llm.client import LLMResponse
from finswarm.llm.providers.mock import MockLLM
from finswarm.llm.scheduler import LaneScheduler
from finswarm.llm.timing import VirtualClock
from finswarm.store.schemas import LLM_DECISIONS

GOOD = '{"observations": "o", "plans": "plan A", "insights": "ins A", "price": 1.81}'


def _obs(period: int, history: list[PeriodRecord] | None = None, **kw) -> Observation:
    base = {
        "tick": period,
        "period": period,
        "agent_id": 2,
        "slot": 0,
        "role": "dealer",
        "inventory": 0.0,
        "cash": 0.0,
        "cost": 1.0,
        "history": history or [],
        "extra": {"max_wtp": 4.51, "tick_size": 0.01},
    }
    base.update(kw)
    return Observation(**base)


def _sched(client) -> LaneScheduler:
    return LaneScheduler(
        {"mock": mock_lane(max_in_flight=4)}, {"mock": client}, clock=VirtualClock()
    )


def _agent(sched, prompt: str = "fish_v1_json", slot: int = 0, **kw) -> LLMAgent:
    return LLMAgent(
        agent_id=2 + slot,
        slot=slot,
        name=f"Firm {slot}",
        model="mock-v1",
        provider="mock",
        lane="mock",
        scheduler=sched,
        prompt=prompt,
        run_id="run-1",
        base_seed=42,
        **kw,
    )


def _run(coro):
    return asyncio.run(coro)


def test_agent_protocol_and_mock_decision() -> None:
    sched = _sched(MockLLM("tit_for_tat"))
    agent = _agent(sched, prompt="fish_v1")
    assert isinstance(agent, Agent)
    d = _run(agent.decide(_obs(0)))
    assert d.action == "quote" and d.price == pytest.approx(1.92) and d.ask == d.price
    assert d.parse_ok and not d.repair_used and not d.refusal
    assert agent.memory.plans == "Keep following tit_for_tat."
    assert agent.memory.last_decision is d
    (row,) = agent.drain_logs()
    assert set(row) == set(LLM_DECISIONS)
    assert row["obs_tick"] == 0 and row["prompt_version"] == agent.template.version
    assert row["parse_ok"] and row["action_type"] == "quote" and row["cost_usd"] == 0.0
    assert agent.drain_logs() == []


def test_symmetric_agents_get_distinct_seeds_and_cache_keys() -> None:
    client = ScriptedClient(default=GOOD)
    sched = _sched(client)
    a, b = _agent(sched, slot=0), _agent(sched, slot=1)

    async def go():
        return await asyncio.gather(a.decide(_obs(0)), b.decide(_obs(0)))

    _run(go())
    assert len(client.calls) == 2  # identical prompts, no cross-agent cache hit
    seeds = {r.seed for _, r in client.calls}
    assert len(seeds) == 2
    assert all(json.loads(r.tags["obs_json"])["output"] == "price_json" for _, r in client.calls)


def test_repair_path_uses_second_call() -> None:
    client = ScriptedClient(["Sure! I would go with a high price.", GOOD])
    agent = _agent(_sched(client))
    d = _run(agent.decide(_obs(3)))
    assert d.parse_ok and d.repair_used and d.price == pytest.approx(1.81)
    assert len(client.calls) == 2
    repair_req = client.calls[1][1]
    assert repair_req.prompt_version.endswith("+repair") and repair_req.tags["repair"] == "1"
    assert len(repair_req.messages) == 2 and repair_req.messages[-1]["role"] == "user"
    assert "Sure! I would go" in repair_req.messages[-1]["content"]
    assert repair_req.schema_ == client.calls[0][1].schema_
    assert agent.memory.plans == "plan A"


def test_fallback_holds_previous_quote_and_keeps_memory() -> None:
    client = ScriptedClient([GOOD], default="garbage")
    agent = _agent(_sched(client), max_repairs=1)
    first = _run(agent.decide(_obs(0)))
    second = _run(agent.decide(_obs(1)))
    assert first.parse_ok and not second.parse_ok and second.repair_used
    assert second.action == "quote" and second.price == pytest.approx(1.81)
    assert agent.memory.plans == "plan A"  # a failed answer never overwrites the files
    assert len(client.calls) == 3
    logs = agent.drain_logs()
    assert [r["parse_ok"] for r in logs] == [True, False] and logs[1]["error"]


def test_fallback_without_history_uses_nash_default() -> None:
    agent = _agent(_sched(ScriptedClient(default="garbage")), max_repairs=0)
    d = _run(agent.decide(_obs(0)))
    assert not d.parse_ok and d.price == pytest.approx(1.4729)
    d2 = _run(
        _agent(_sched(ScriptedClient(default="x")), max_repairs=0).decide(
            _obs(0, extra={"max_wtp": 4.51, "nash_price": 1.5})
        )
    )
    assert d2.price == 1.5


def test_refusal_is_flagged_without_repair() -> None:
    refusal = LLMResponse(text="", refusal=True, stop_reason="refusal", error="refusal:None")
    client = ScriptedClient([refusal])
    agent = _agent(_sched(client))
    d = _run(agent.decide(_obs(0)))
    assert d.refusal and not d.parse_ok and len(client.calls) == 1
    assert agent.drain_logs()[0]["refusal"]


def test_memory_truncation_and_state_roundtrip() -> None:
    agent = _agent(_sched(MockLLM("static_nash")), memory_periods=5)
    for t in range(8):
        agent.record_period(PeriodRecord(t, 1.5, 10.0, 5.0, (1.5,), float("nan")))
    assert [r.period for r in agent.memory.history] == [3, 4, 5, 6, 7]
    _run(agent.decide(_obs(8)))
    st = json.loads(json.dumps(agent.state()))  # JSON-serialisable checkpoint
    clone = _agent(_sched(MockLLM("static_nash")), memory_periods=5)
    clone.restore(st)
    assert clone.memory.plans == agent.memory.plans
    assert [r.period for r in clone.memory.history] == [3, 4, 5, 6, 7]
    assert clone.memory.last_decision.price == agent.memory.last_decision.price
    assert clone.render(_obs(9)) == agent.render(_obs(9))


def test_replace_agent_keep_memory_semantics() -> None:
    agent = _agent(_sched(ScriptedClient(default=GOOD)))
    agent.record_period(PeriodRecord(0, 1.8, 10.0, 8.0, (1.8,), float("nan")))
    agent.memory.notices.append("Regulator watching.")
    _run(agent.decide(_obs(1)))
    seed_before = agent._seed(_obs(5), 0)

    agent.replace(model="mock-v1", lane="mock", provider="mock", keep_memory=True)
    assert agent.memory.plans == "plan A" and len(agent.memory.history) == 1
    assert agent.generation == 1 and agent._seed(_obs(5), 0) != seed_before

    agent.replace(
        model="claude-haiku-5-5", lane="anthropic_haiku", keep_memory=False, keep_name=False
    )
    assert agent.provider == "anthropic"  # inferred from configs/models.yaml
    assert agent.memory.plans == "" and agent.memory.insights == ""
    assert len(agent.memory.history) == 0 and agent.memory.last_decision is None
    assert agent.memory.notices == ["Regulator watching."]  # environment, not memory
    assert agent.memory.memory_periods == 100
    assert agent.name == "Firm 0~claude-haiku-5-5" and agent.generation == 2


def test_from_slot_and_cda_decisions() -> None:
    slot_cfg = LLMSlotConfig(
        name="S1", role="seller", model="mock-v1", family="mock", prompt="agrawal_v1"
    )
    tier = LLMTierConfig(lane="mock", comm_channel="sellers", oversight_notice=True)
    client = ScriptedClient(
        [
            '{"plans": "p", "insights": "i", "price": 95, "message": "stay at 95"}',
            '{"plans": "p", "insights": "i", "price": null, "message": null}',
        ]
    )
    agent = LLMAgent.from_slot(slot_cfg, agent_id=5, slot=0, scheduler=_sched(client), tier=tier)
    assert agent.comm_enabled and agent.oversight_notice and agent.lane == "mock"
    obs = _obs(0, role="seller", cost=80.0, extra={"n_rounds": 30})
    d = _run(agent.decide(obs))
    assert d.action == "limit" and d.side == Side.SELL and d.price == 95 and d.size == 1.0
    assert d.message == "stay at 95"
    d2 = _run(agent.decide(_obs(1, role="seller", cost=80.0, extra={"n_rounds": 30})))
    assert d2.action == "hold" and d2.message is None


def test_mock_mode_runs_are_deterministic() -> None:
    def run_once() -> list[float]:
        sched = LaneScheduler(
            {"mock": mock_lane()}, clock=VirtualClock(), cache=ResponseCache(mode="mock")
        )
        agent = _agent(sched, mock_policy="random")
        prices = []
        for t in range(5):
            d = _run(agent.decide(_obs(t)))
            prices.append(d.price)
            agent.record_period(PeriodRecord(t, d.price, 10.0, 5.0, (1.7,), float("nan")))
        return prices

    a, b = run_once(), run_once()
    assert a == b and len(set(a)) > 1
