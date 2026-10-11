"""End-to-end runner tests on the real closed-form dealer market (``market.registry``) with
scripted and mock-LLM agents. ``FakeMarket`` (tests/fakes.py) is kept only where a hook-less
test double is the point (re-grid through ``migrate_from``, demand cross-check)."""

import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from fakes import FakeMarket, base_config, fake_market_factory

from finswarm.config.schema import RunConfig
from finswarm.core.rng import RNGRegistry
from finswarm.core.types import Order, OrderType, Side
from finswarm.market.registry import make_market
from finswarm.perturb.branch import branch
from finswarm.sim.checkpoints import list_checkpoints
from finswarm.sim.ghost import run_ghosts
from finswarm.sim.run import run_config, run_from_path
from finswarm.store.reader import load_run
from finswarm.store.schemas import TABLES

pytestmark = pytest.mark.smoke
ROOT = Path(__file__).resolve().parents[2]

MANIFEST_KEYS = {
    "run_id", "name", "seed", "config_hash", "resolved_config", "git", "seeds", "slots",
    "packages", "environment", "started_at", "finished_at", "wall_time_s", "llm", "cost_usd",
    "parent_run_id", "branch_period", "branch_events_hash", "status", "periods_completed",
    "checkpoints", "impl", "alpha", "arm", "trunk_id", "condition", "label",
}  # fmt: skip


def mixed_config(periods: int = 30, **over) -> RunConfig:
    """Slot 0 scripted tit-for-tat, slot 1 mock-LLM stand-in (exercises llm_decisions)."""
    data = base_config(periods=periods, **over).model_dump(mode="json")
    data["llm"]["slots"][1]["policy"] = "mock_llm(fail_rate=0.1)"
    data["llm"]["slots"][1]["model"] = "mock-1"
    return RunConfig.model_validate(data)


def run(cfg, out, **kw):
    return run_config(cfg, out=out, **kw)  # real market (finswarm.market.registry)


def part_bytes(run_dir: Path, tables=("trades", "quotes")) -> dict[str, str]:
    out = {}
    for t in tables:
        for p in sorted((run_dir / t).glob("part-*.parquet")):
            out[f"{t}/{p.name}"] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def test_end_to_end_tables_manifest_checkpoints(tmp_path) -> None:
    r = run(mixed_config(), tmp_path)
    assert r.status == "ok" and r.periods_completed == 30
    assert r.manifest["impl"]["market"] == "market.registry"
    data = load_run(r.run_dir)  # validates dtypes of every part
    for t, lf in data.tables.items():
        assert dict(lf.collect_schema()) == TABLES[t]
    assert data.frame("trades").height == 60 and data.frame("quotes").height == 60
    assert set(data.frame("trades")["buyer_id"].to_list()) == {-1}  # closed-form buyer
    flow = data.frame("flow_per_tick")
    assert flow.height == 60 and set(flow["dealer_id"].to_list()) == {0, 1}
    llm = data.frame("llm_decisions")
    assert llm.height == 30 and set(llm["slot"].to_list()) == {1}
    assert llm["cost_usd"].sum() == 0.0 and llm["view_hash"].null_count() == 0
    assert (~llm["parse_ok"]).sum() == data.frame("metrics_per_period")["parse_fail_count"].sum()
    q = data.frame("quotes")
    assert (
        q.filter(pl.col("slot") == 1)["is_llm"].all()
        and not q.filter(pl.col("slot") == 0)["is_llm"].any()
    )
    m = r.manifest
    assert MANIFEST_KEYS <= set(m)
    assert m["status"] == "ok" and m["cost_usd"] == 0.0 and m["llm"]["calls"] == 30
    assert (m["alpha"], m["arm"], m["trunk_id"], m["condition"], m["label"]) == (
        1.0,
        "t",
        r.run_id,
        "t",
        "",
    )
    assert m["llm"]["parse_failures"] == int((~llm["parse_ok"]).sum())
    assert [s["kind"] for s in m["slots"]] == ["SCRIPTED", "LLM"]
    assert json.loads((r.run_dir / "manifest.json").read_text())["run_id"] == r.run_id
    names = {n for c in list_checkpoints(r.run_dir) for n in c.names}
    assert {"initial", "final"} <= names
    assert [c.period for c in list_checkpoints(r.run_dir)] == [0, 10, 20, 30]
    assert (r.run_dir / "prompts.jsonl").exists()
    # agent_period quantities/profits come from the market's own period summaries
    ap = data.frame("agent_period").filter(pl.col("period") == 0).sort("slot")
    tr = data.frame("trades").filter(pl.col("period") == 0).sort("seller_id")
    assert ap["quantity"].to_list() == pytest.approx(tr["size"].to_list())
    # a finished run is skipped, not re-run
    again = run(mixed_config(), tmp_path)
    assert again.skipped and again.run_id == r.run_id


def test_same_config_and_seed_give_identical_parquet_bytes(tmp_path) -> None:
    a = run(mixed_config(), tmp_path / "a")
    b = run(mixed_config(), tmp_path / "b")
    assert a.run_id == b.run_id
    ha, hb = part_bytes(a.run_dir), part_bytes(b.run_dir)
    assert ha and ha == hb
    c = run(mixed_config(seed=1), tmp_path / "c")
    ta = load_run(a.run_dir).frame("quotes").drop("run_id")
    tc = load_run(c.run_dir).frame("quotes").drop("run_id")
    assert not ta.equals(tc)  # the mock LLM draws depend on the seed


def test_kill_then_resume_gives_identical_output(tmp_path) -> None:
    ref = run(mixed_config(), tmp_path / "ref")
    crashed = run(mixed_config(), tmp_path / "k", halt_after_period=16)
    assert crashed.status == "failed" and crashed.periods_completed == 17
    resumed = run(mixed_config(), tmp_path / "k")
    assert resumed.status == "ok" and resumed.manifest["resumes"][0]["from_period"] == 10
    tables = tuple(TABLES)
    assert part_bytes(ref.run_dir, tables) == part_bytes(resumed.run_dir, tables)
    assert ref.manifest["llm"] == resumed.manifest["llm"]
    assert (ref.run_dir / "prompts.jsonl").read_bytes() == (
        resumed.run_dir / "prompts.jsonl"
    ).read_bytes()


def test_branch_identical_before_and_different_after(tmp_path) -> None:
    parent = run(mixed_config(periods=40), tmp_path)
    ev = {"type": "set_param", "path": "market.routing.tau", "value": 0.5}
    child_dir = branch(parent.run_dir, 25, [ev], "soft")
    child = load_run(child_dir)
    par = load_run(parent.run_dir)
    m = child.manifest
    assert m["status"] == "ok" and m["parent_run_id"] == parent.run_id and m["branch_period"] == 25
    assert m["trunk_id"] == parent.run_id
    assert m["branch_events_hash"] and m["seeds"]["interventions_respawn"][0] == 0
    for t in TABLES:
        a = par.frame(t).filter(pl.col("period") < 25).drop("run_id")
        b = child.frame(t).filter(pl.col("period") < 25).drop("run_id")
        assert a.equals(b), t
    assert set(child.frame("trades")["run_id"].to_list()) == {m["run_id"]}
    after_p = par.frame("trades").filter(pl.col("period") >= 25).drop("run_id")
    after_c = child.frame("trades").filter(pl.col("period") >= 25).drop("run_id")
    assert after_p.height == after_c.height and not after_p.equals(after_c)
    bev = child.frame("events").filter(pl.col("source") == "branch")
    assert bev.select(["period", "event_type"]).rows() == [(25, "set_param")]
    # branching again is idempotent (child already ok)
    assert branch(parent.run_dir, 25, [ev], "soft") == child_dir


def test_unsupported_interventions_fail_the_run_with_a_reason(tmp_path) -> None:
    grid = {"type": "set_param", "path": "market.price_grid.min", "value": 0.5}
    r = run(mixed_config(periods=5, events=[{**grid, "when": {"at_period": 3}}]), tmp_path / "a")
    assert r.status == "failed" and r.periods_completed == 0  # refused before period 0
    assert "UnsupportedIntervention" in r.error and "price_grid.min" in r.error
    fringe = {"type": "set_param", "path": "market.fringe.share", "value": 0.3}
    r = run(mixed_config(periods=5, events=[{**fringe, "when": {"at_period": 3}}]), tmp_path / "f")
    assert r.status == "failed" and r.periods_completed == 0  # closed form: no fringe
    assert "UnsupportedIntervention" in r.error and "agents" in r.error
    assert json.loads((r.run_dir / "manifest.json").read_text())["status"] == "failed"
    liq = {"type": "remove_liquidity", "share": 0.5, "when": {"at_period": 2}}
    r = run(mixed_config(periods=5, events=[liq]), tmp_path / "b")  # closed form: no swarm
    assert r.status == "failed" and r.periods_completed == 2 and "remove_liquidity" in r.error
    two = mixed_config(periods=5).model_dump(mode="json")
    two["market"]["two_sided"] = True
    r = run(RunConfig.model_validate(two), tmp_path / "c")
    assert r.status == "failed" and "two-sided" in r.error


def test_branch_extends_horizon_and_resumes_from_its_own_checkpoints(tmp_path) -> None:
    parent = run(mixed_config(periods=20), tmp_path)
    evs = [
        {"type": "set_param", "path": "periods", "value": 35},
        {"type": "forced_deviation", "slot": 0, "delta": 0.2},
    ]
    d1 = branch(parent.run_dir, 20, evs, "probe", halt_after_period=27)
    assert json.loads((d1 / "manifest.json").read_text())["status"] == "failed"
    d2 = branch(parent.run_dir, 20, evs, "probe")
    assert d1 == d2
    c = load_run(d2)
    assert c.manifest["status"] == "ok" and c.manifest["periods_completed"] == 35
    assert c.frame("agent_period")["period"].max() == 34


def test_fake_market_regrid_goes_through_migrate_from(tmp_path) -> None:
    """Hook-less markets (no ``settable_params``/``set_param``) are rebuilt and migrated."""
    parent = run(mixed_config(periods=30), tmp_path, market_factory=fake_market_factory)
    ev = {"type": "set_param", "path": "market.price_grid.tick", "value": 0.05}
    child = load_run(branch(parent.run_dir, 20, [ev], "coarse", market_factory=fake_market_factory))
    assert child.manifest["status"] == "ok" and child.manifest["impl"]["market"] == "injected"
    asks = child.frame("quotes").filter(pl.col("period") >= 21)["ask"].to_list()
    assert all(abs(a / 0.05 - round(a / 0.05)) < 1e-9 for a in asks)


@pytest.mark.parametrize("clock", ["sync", "poisson"])
def test_ghosts_replay_rivals_open_loop(tmp_path, clock) -> None:
    data = mixed_config(periods=20).model_dump(mode="json")
    data["llm"]["slots"][0]["policy"] = "mock_llm"
    data["clock"]["mode"] = clock
    cfg = RunConfig.model_validate(data)
    r = run(cfg, tmp_path)
    ghosts = run_ghosts(r.run_dir)
    assert [g.status for g in ghosts] == ["ok", "ok"]
    assert (r.run_dir / "ghost" / "0" / "manifest.json").exists()
    parent_q = load_run(r.run_dir).frame("quotes")
    g0 = load_run(r.run_dir / "ghost" / "0")
    assert g0.manifest["ghost_of"] == r.run_id
    rival_parent = parent_q.filter(pl.col("slot") == 1).select(["period", "ask"])
    rival_ghost = g0.frame("quotes").filter(pl.col("slot") == 1).select(["period", "ask"])
    assert rival_parent.equals(rival_ghost)  # slot 1 replays its logged quotes exactly
    assert g0.frame("llm_decisions")["slot"].unique().to_list() == [0]


def test_real_closed_form_market_matches_fake_market_demand() -> None:
    cfg = base_config()
    real, fake = make_market(cfg, RNGRegistry(0)), FakeMarket(cfg)
    for m in (real, fake):
        m.submit(Order(0, 0, Side.SELL, OrderType.QUOTE, 47, 1.0, -1, 0, 0))
        m.submit(Order(1, 1, Side.SELL, OrderType.QUOTE, 92, 1.0, -1, 0, 0))
        m.advance(1)
    qr = np.sort(real.drain_trades_arrays()["size"])
    qf = np.sort([t.size for t in fake.drain_trades()])
    assert qr == pytest.approx(qf, rel=1e-12)


def test_cli_smoke_config_runs_on_the_real_market(tmp_path) -> None:
    results = run_from_path(ROOT / "configs" / "smoke.yaml", provider="mock", out=tmp_path / "runs")
    assert len(results) == 3
    for r in results:
        assert r.status == "ok" and r.manifest["cost_usd"] == 0.0
        assert r.manifest["impl"]["market"] == "market.registry"
        assert (tmp_path / "manifests" / f"{r.run_id}.json").exists()
        ev = load_run(r.run_dir).frame("events")
        assert ev.select(["period", "event_type"]).rows() == [(25, "set_param")]
