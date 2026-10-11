"""The runner on ``finswarm.core.clock`` (levers L5 staggered updates, L7 speed bump).

Real closed-form dealer market, scripted tit-for-tat in slot 0 and a mock-LLM stand-in in slot 1
(``llm_decisions`` rows carry ``obs_tick``/``apply_tick``). Sync-clock outputs are pinned
bit-for-bit to the pre-refactor runner by ``test_smoke_mock`` / ``test_real_market_determinism``
(same assertions, unchanged) and by the regression run recorded in the commit message.
"""

import hashlib
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from fakes import base_config

from finswarm.config.schema import RunConfig
from finswarm.perturb.branch import branch
from finswarm.sim.ghost import run_ghosts
from finswarm.sim.run import run_config
from finswarm.store.reader import load_run
from finswarm.store.schemas import TABLES

pytestmark = pytest.mark.smoke


def cfg(periods: int = 40, tpp: int = 1, swarm: str = "closed_form", **clock) -> RunConfig:
    data = base_config(periods=periods).model_dump(mode="json")
    data["ticks_per_period"] = tpp
    data["llm"]["slots"][1]["policy"] = "mock_llm(fail_rate=0.1)"
    data["llm"]["slots"][1]["model"] = "mock-1"
    data["clock"] = {"mode": "sync", "epoch_ticks": 1, **clock}
    if swarm == "agents":
        data["swarm"] = {"mode": "agents", "n_agents": 1000, "arrivals": {"lam_mean": 0.1 / tpp}}
        data["market"]["demand"] = {"kind": "swarm"}
    return RunConfig.model_validate(data)


def part_bytes(run_dir: Path, tables=tuple(TABLES)) -> dict[str, str]:
    return {
        f"{t}/{p.name}": hashlib.sha256(p.read_bytes()).hexdigest()
        for t in tables
        for p in sorted((run_dir / t).glob("part-*.parquet"))
    }


def decisions(run_dir: Path) -> pl.DataFrame:
    return load_run(run_dir).frame("quotes").filter(pl.col("decision_id") >= 0)


@pytest.mark.parametrize("swarm", ["closed_form", "agents"])
def test_poisson_clock_is_deterministic_staggered_and_resumable(tmp_path, swarm) -> None:
    c = cfg(periods=30, tpp=10, swarm=swarm, mode="poisson", epoch_ticks=4)
    a = run_config(c, out=tmp_path / "a")
    b = run_config(c, out=tmp_path / "b")
    assert a.status == b.status == "ok"
    ha = part_bytes(a.run_dir)
    assert ha == part_bytes(b.run_dir)
    q = decisions(a.run_dir)
    by_tick = q.group_by("tick").agg(pl.col("slot").n_unique().alias("n"))
    assert (by_tick["n"] == 1).sum() > 0  # dealers revise at staggered ticks
    sync = run_config(cfg(periods=30, tpp=10, swarm=swarm, epoch_ticks=4), out=tmp_path / "s")
    assert part_bytes(sync.run_dir, ("quotes",)) != part_bytes(a.run_dir, ("quotes",))
    crashed = run_config(c, out=tmp_path / "k", halt_after_period=17)
    assert crashed.status == "failed"
    resumed = run_config(c, out=tmp_path / "k")
    assert resumed.status == "ok" and resumed.manifest["resumes"][0]["from_period"] == 10
    assert part_bytes(resumed.run_dir) == ha


def test_poisson_mean_revision_interval_is_epoch_ticks(tmp_path) -> None:
    r = run_config(cfg(periods=150, tpp=10, mode="poisson", epoch_ticks=5), out=tmp_path)
    q = decisions(r.run_dir).sort(["slot", "tick"])
    for slot in (0, 1):
        ticks = q.filter(pl.col("slot") == slot)["tick"].to_numpy()
        gaps = np.diff(ticks)
        assert gaps.min() >= 1 and ticks[0] == 0
        assert gaps.mean() == pytest.approx(5.0, rel=0.12)  # ~300 revisions per slot
    llm = load_run(r.run_dir).frame("llm_decisions")
    assert (llm["apply_tick"] == llm["obs_tick"]).all()  # no speed bump


def test_set_param_clock_mode_via_branch(tmp_path) -> None:
    parent = run_config(cfg(periods=30, tpp=10, epoch_ticks=4), out=tmp_path)
    ev = {"type": "set_param", "path": "clock.mode", "value": "poisson"}
    child_dir = branch(parent.run_dir, 10, [ev], "L5")
    child, par = load_run(child_dir), load_run(parent.run_dir)
    assert child.manifest["status"] == "ok"
    for t in TABLES:
        before_p = par.frame(t).filter(pl.col("period") < 10).drop("run_id")
        before_c = child.frame(t).filter(pl.col("period") < 10).drop("run_id")
        assert before_p.equals(before_c), t
    qp = par.frame("quotes").filter(pl.col("period") >= 10).drop("run_id")
    qc = child.frame("quotes").filter(pl.col("period") >= 10).drop("run_id")
    assert not qp.equals(qc)
    sync_ticks = set(qp.filter(pl.col("decision_id") >= 0)["tick"].to_list())
    assert sync_ticks <= {10 * p + k for p in range(10, 30) for k in (0, 4, 8)}
    assert not set(qc.filter(pl.col("decision_id") >= 0)["tick"].to_list()) <= sync_ticks
    ev_rows = child.frame("events").filter(pl.col("source") == "branch")
    assert ev_rows.select(["period", "event_type", "tick"]).rows() == [(10, "set_param", 100)]
    assert '"clock.mode"' in ev_rows["payload"][0] and '"poisson"' in ev_rows["payload"][0]
    # resuming the child from its own checkpoints after the switch is exact
    again = branch(parent.run_dir, 10, [ev], "L5b", halt_after_period=24)
    assert load_run(again).manifest["status"] == "failed"
    again = branch(parent.run_dir, 10, [ev], "L5b")
    tables = ("trades", "quotes", "llm_decisions", "agent_period")
    a = {t: load_run(again).frame(t).drop("run_id") for t in tables}
    assert all(a[t].equals(child.frame(t).drop("run_id")) for t in tables)


@pytest.mark.parametrize("mode", ["sync", "poisson"])
def test_speed_bump_is_fifo_constant_delay_and_resumable(tmp_path, mode) -> None:
    d = 7
    c = cfg(periods=30, tpp=10, mode=mode, epoch_ticks=3, speed_bump_max=d)
    r = run_config(c, out=tmp_path / "a")
    data = load_run(r.run_dir)
    llm = data.frame("llm_decisions")
    assert ((llm["apply_tick"] - llm["obs_tick"]) == d).all()
    q = decisions(r.run_dir)
    lq = q.filter(pl.col("slot") == 1).sort("decision_id")
    assert lq["tick"].to_list() == llm.sort("decision_id")["apply_tick"].to_list()
    for slot in (0, 1):
        ticks = q.filter(pl.col("slot") == slot).sort("decision_id")["tick"].to_numpy()
        assert (np.diff(ticks) >= 0).all()  # FIFO
    # rivals and period prices see a quote only once in force
    orders = data.frame("orders")
    assert (orders["tick"] >= 0).all()
    ap = data.frame("agent_period").filter(pl.col("slot") == 0).sort("period")
    sent = q.filter(pl.col("slot") == 0).sort("tick")
    for p, price in zip(ap["period"].to_list(), ap["price"].to_list(), strict=True):
        last_tick = 10 * p + 9
        live = sent.filter(pl.col("tick") <= last_tick)["ask"].to_list()
        if live:
            assert price == pytest.approx(live[-1])
        else:
            assert np.isnan(price)
    crashed = run_config(c, out=tmp_path / "k", halt_after_period=16)
    assert crashed.status == "failed"
    resumed = run_config(c, out=tmp_path / "k")
    assert part_bytes(resumed.run_dir) == part_bytes(r.run_dir)


def test_speed_bump_lever_via_branch_and_jitter_tag(tmp_path) -> None:
    parent = run_config(cfg(periods=30, tpp=10, epoch_ticks=5), out=tmp_path)
    evs = [
        {"type": "set_param", "path": "clock.speed_bump_max", "value": 6},
        {"type": "set_param", "path": "tags.speed_bump_jitter", "value": "3"},
    ]
    child = load_run(branch(parent.run_dir, 10, evs, "L7"))
    assert child.manifest["status"] == "ok"
    llm = child.frame("llm_decisions")
    before = llm.filter(pl.col("period") < 10)
    after = llm.filter(pl.col("period") >= 10)
    assert (before["apply_tick"] == before["obs_tick"]).all()
    delay = (after["apply_tick"] - after["obs_tick"]).unique().to_list()
    assert len(delay) == 1 and 6 <= delay[0] <= 9  # one constant per-slot latency
    ticks = after.sort("decision_id")["apply_tick"].to_numpy()
    assert (np.diff(ticks) > 0).all()


def test_staleness_window_shares_observations(tmp_path) -> None:
    c = cfg(periods=40, tpp=10, mode="poisson", epoch_ticks=3, staleness_window=4)
    r = run_config(c, out=tmp_path)
    llm = load_run(r.run_dir).frame("llm_decisions")
    lag = llm["apply_tick"] - llm["obs_tick"]  # = decision_tick - obs_tick (no speed bump)
    assert lag.min() == 0 and lag.max() <= 3 and (lag > 0).any()


def test_ghosts_replay_rivals_under_a_staggered_clock(tmp_path) -> None:
    data = cfg(periods=40, tpp=1, mode="poisson", epoch_ticks=2).model_dump(mode="json")
    data["llm"]["slots"][0]["policy"] = "mock_llm"
    r = run_config(RunConfig.model_validate(data), out=tmp_path)
    ghosts = run_ghosts(r.run_dir)
    assert [g.status for g in ghosts] == ["ok", "ok"]
    parent_q = load_run(r.run_dir).frame("quotes")
    periods = parent_q["period"].unique().len()
    assert periods < 40  # some periods have no revision at all
    for k, rival in ((0, 1), (1, 0)):
        g = load_run(r.run_dir / "ghost" / str(k)).frame("quotes")
        cols = ["period", "epoch", "tick", "ask", "decision_id"]
        p_rows = parent_q.filter(pl.col("slot") == rival).select(cols)
        g_rows = g.filter(pl.col("slot") == rival).select(cols)
        assert p_rows.equals(g_rows)  # the rival replays its logged quotes, same schedule


def test_invalid_clock_events_fail_before_period_zero(tmp_path) -> None:
    bad = {"type": "set_param", "path": "clock.epoch_ticks", "value": 0, "when": {"at_period": 5}}
    data = cfg(periods=10).model_dump(mode="json")
    data["events"] = [bad]
    r = run_config(RunConfig.model_validate(data), out=tmp_path)
    assert r.status == "failed" and r.periods_completed == 0
    assert "UnsupportedIntervention" in r.error and "clock.epoch_ticks" in r.error
