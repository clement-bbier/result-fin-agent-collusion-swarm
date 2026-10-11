"""The runner-facing LLM factory and the real LLMAgent + MockLLM path inside a run."""

import asyncio
import json

import polars as pl
import pytest

from finswarm.agents.llm import RunnerLLMAgent, make_llm_agent
from finswarm.agents.llm.factory import POOL, resolve_lane
from finswarm.config.schema import RunConfig
from finswarm.core.rng import RNGRegistry
from finswarm.llm.pricing import load_catalog
from finswarm.sim.integration import slot_is_scripted
from finswarm.sim.run import Runner, run_config
from finswarm.store.reader import load_run


def llm_config(periods: int = 5) -> RunConfig:
    return RunConfig.model_validate(
        {
            "name": "llm_mock",
            "periods": periods,
            "seed": 7,
            "llm": {
                "mode": "mock",
                "lane": "mock",
                "slots": [
                    {"name": "A", "model": "mock-v1", "family": "mock", "policy": "tit_for_tat"},
                    {"name": "B", "model": "mock-v1", "family": "mock", "policy": "undercutter"},
                ],
            },
            "checkpoint": {"every_periods": 2},
        }
    )


def test_slot_rule_and_lane_resolution() -> None:
    cfg = llm_config()
    assert not any(slot_is_scripted(s) for s in cfg.llm.slots)  # mock-v1 -> real LLMAgent
    scripted = cfg.llm.slots[0].model_copy(update={"model": "scripted"})
    assert slot_is_scripted(scripted)
    cat = load_catalog()
    assert resolve_lane(cat, "claude-haiku-5-5", "haiku", "live") == "anthropic_haiku"
    assert resolve_lane(cat, "claude-haiku-5-5", None, "mock") == "mock"
    with pytest.raises(ValueError):
        resolve_lane(cat, "no-such-model", None, "live")


def test_factory_builds_runner_agent_with_resolved_model_and_shared_scheduler() -> None:
    cfg = llm_config()
    sc = cfg.llm.slots[0].model_copy(update={"model": "haiku", "family": "anthropic"})

    async def go() -> None:
        kw = dict(run_cfg=cfg, run_id="r", rngs=RNGRegistry(0), provider="mock", lane=None)
        a = make_llm_agent(sc, slot=0, agent_id=0, mode="mock", **kw)
        b = make_llm_agent(cfg.llm.slots[1], slot=1, agent_id=1, mode="mock", **kw)
        assert isinstance(a, RunnerLLMAgent) and a.model == "claude-haiku-5-5"
        assert a.lane == "mock" and a.base_seed == cfg.seed and a.scheduler is b.scheduler
        assert a.last_call is None and a.last_prompt is None
        await a.aclose()
        assert "mock" in POOL._entries  # b still holds it
        await b.aclose()
        assert "mock" not in POOL._entries

    asyncio.run(go())


def test_mock_llm_dealers_run_through_the_real_llm_agent(tmp_path) -> None:
    seen = []

    def spy(obs, policy, rng, **kw):
        from finswarm.market.view import apply_view

        out = apply_view(obs, policy, rng, **kw)
        seen.append(dict(out.extra))
        return out

    runner = Runner(llm_config(), out=tmp_path)
    runner.view_fn, runner.view_impl = spy, "market.view"  # observe what agents receive
    res = asyncio.run(runner.run())
    assert res.status == "ok" and res.manifest["impl"]["llm_agents"] == "agents.llm"
    data = load_run(res.run_dir)
    llm = data.frame("llm_decisions")
    assert llm.height == 10 and llm["parse_ok"].all()
    assert set(llm["prompt_version"].to_list()) - {None} and llm["prompt_version"].null_count() == 0
    assert llm["prompt_hash"].null_count() == 0 and (llm["model"] == "mock-v1").all()
    assert (llm["apply_tick"] == llm["obs_tick"]).all()  # no speed bump
    assert llm["decision_id"].n_unique() == 10 and (llm["cost_usd"] == 0.0).all()
    first = llm.filter(pl.col("period") == 0).sort("slot")["price"].to_list()
    assert first == pytest.approx([1.92, 1.92])  # both open at the monopoly price
    prompts = (res.run_dir / "prompts.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(prompts) == 10 and "user" in json.loads(prompts[0])
    ex = seen[0]
    assert ex["max_wtp"] == pytest.approx(4.51) and ex["n_rounds"] == 5
    assert ex["nash_price"] == pytest.approx(1.4729, abs=1e-4)
    assert ex["monopoly_price"] == pytest.approx(1.9250, abs=1e-4)
    assert ex["default_price"] == ex["nash_price"] and len(ex["view_hash"]) == 64
    assert not any(k.startswith("_raw") for k in ex)
    assert res.manifest["llm"]["calls"] == 10 and res.manifest["cost_usd"] == 0.0


def test_max_wtp_tag_override_and_resume_identity(tmp_path) -> None:
    data = llm_config(periods=6).model_dump(mode="json")
    data["tags"] = {"max_wtp": "3.5"}
    cfg = RunConfig.model_validate(data)
    ref = run_config(cfg, out=tmp_path / "ref")
    crashed = run_config(cfg, out=tmp_path / "k", halt_after_period=3)
    assert crashed.status == "failed"
    resumed = run_config(cfg, out=tmp_path / "k")
    assert resumed.status == "ok"
    a = load_run(ref.run_dir).frame("quotes").drop("run_id")
    b = load_run(resumed.run_dir).frame("quotes").drop("run_id")
    assert a.equals(b)
    prompts = (ref.run_dir / "prompts.jsonl").read_text(encoding="utf-8")
    assert "3.5" in prompts
