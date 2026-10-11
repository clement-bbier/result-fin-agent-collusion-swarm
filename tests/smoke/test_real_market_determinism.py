"""Determinism contract on the real dealer market, closed-form AND agents mode.

Agents mode: 2,000 swarm agents, 10 ticks per period, 20 periods; ``lam_mean = beta / (N *
ticks_per_period)`` gives Calvano's ``beta`` arrivals per period. Dealers: a scripted
tit-for-tat and a real ``LLMAgent`` answered by ``MockLLM("undercutter")`` (mock mode).
(a) same config + seed -> identical ``trades``/``quotes`` bytes; (b) kill at period 12 + resume
-> identical; (c) ``branch()`` at period 10 with ``set_param market.routing.tau`` -> identical
before, different after.
"""

import hashlib
from pathlib import Path

import polars as pl
import pytest

from finswarm.config.schema import RunConfig
from finswarm.perturb.branch import branch
from finswarm.sim.run import run_config
from finswarm.store.reader import load_run

pytestmark = pytest.mark.smoke
N_AGENTS, TPP, PERIODS, BETA = 2_000, 10, 20, 100.0


def config(mode: str) -> RunConfig:
    data = {
        "name": f"det_{mode}",
        "periods": PERIODS,
        "ticks_per_period": TPP,
        "seed": 3,
        "market": {"demand": {"beta": BETA}},
        "swarm": {"mode": mode},
        "llm": {
            "mode": "mock",
            "lane": "mock",
            "slots": [
                {
                    "name": "A",
                    "model": "scripted",
                    "family": "mock",
                    "policy": "tit_for_tat(target=1.9, trigger=0.01, punish=3)",
                },
                {"name": "B", "model": "mock-v1", "family": "mock", "policy": "undercutter"},
            ],
        },
        "clock": {"mode": "sync", "epoch_ticks": TPP},
        "checkpoint": {"every_periods": 5, "keep_last": 2},
        "logging": {"snapshot_every": 5, "snapshot_levels": 4},
    }
    if mode == "agents":
        data["swarm"] = {
            "mode": "agents",
            "n_agents": N_AGENTS,
            "arrivals": {"lam_mean": BETA / (N_AGENTS * TPP)},
        }
    return RunConfig.model_validate(data)


def part_bytes(run_dir: Path, tables=("trades", "quotes")) -> dict[str, str]:
    return {
        f"{t}/{p.name}": hashlib.sha256(p.read_bytes()).hexdigest()
        for t in tables
        for p in sorted((run_dir / t).glob("part-*.parquet"))
    }


@pytest.mark.parametrize("mode", ["closed_form", "agents"])
def test_same_seed_kill_resume_and_branch(tmp_path, mode) -> None:
    cfg = config(mode)
    a = run_config(cfg, out=tmp_path / "a")
    b = run_config(cfg, out=tmp_path / "b")
    assert a.status == b.status == "ok" and a.periods_completed == PERIODS
    ha = part_bytes(a.run_dir)
    assert ha and ha == part_bytes(b.run_dir)  # (a)
    trades = load_run(a.run_dir).frame("trades")
    if mode == "agents":
        assert trades["buyer_id"].min() >= 0 and trades["buyer_id"].max() < N_AGENTS
        assert set(trades["seller_id"].unique().to_list()) <= {N_AGENTS, N_AGENTS + 1}
        assert (a.run_dir / "flow_per_tick").exists()
        flow = load_run(a.run_dir).frame("flow_per_tick")
        assert flow.height == PERIODS * TPP * 2  # one row per tick per dealer in use
    else:
        assert set(trades["buyer_id"].unique().to_list()) == {-1}
    llm = load_run(a.run_dir).frame("llm_decisions")
    assert llm.height == PERIODS and llm["parse_ok"].all()

    crashed = run_config(cfg, out=tmp_path / "k", halt_after_period=12)  # (b)
    assert crashed.status == "failed" and crashed.periods_completed == 13
    resumed = run_config(cfg, out=tmp_path / "k")
    assert resumed.status == "ok" and resumed.manifest["resumes"][0]["from_period"] == 10
    assert part_bytes(resumed.run_dir) == ha

    ev = {"type": "set_param", "path": "market.routing.tau", "value": 0.5}  # (c)
    child = load_run(branch(a.run_dir, 10, [ev], "tau"))
    par = load_run(a.run_dir)
    assert child.manifest["status"] == "ok" and child.manifest["trunk_id"] == a.run_id
    for t in ("trades", "quotes", "agent_period"):
        before_p = par.frame(t).filter(pl.col("period") < 10).drop("run_id")
        before_c = child.frame(t).filter(pl.col("period") < 10).drop("run_id")
        assert before_p.equals(before_c), t
    after_p = par.frame("trades").filter(pl.col("period") >= 10).drop("run_id")
    after_c = child.frame("trades").filter(pl.col("period") >= 10).drop("run_id")
    assert not after_p.equals(after_c)
