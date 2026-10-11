"""Pre-registered levers L2 (public fringe 0 -> 0.3) and L3 (tick 0.01 -> 0.05) as mid-run
``set_param`` events from a checkpoint (``finswarm.perturb.branch.branch``) on a 2,000-agent
swarm run: identical before the branch period, the lever in force after it, a kill + resume
of the child reproduces the uninterrupted child byte for byte, and ``branch_levers.yaml``
only uses paths the market can set live."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import polars as pl
import pytest

from finswarm.config.load import load_campaign_config
from finswarm.config.schema import RunConfig
from finswarm.core.rng import RNGRegistry
from finswarm.core.types import AgentKind
from finswarm.market.registry import make_market
from finswarm.perturb.branch import branch
from finswarm.perturb.events import SetParam
from finswarm.sim.integration import market_config
from finswarm.sim.run import run_config
from finswarm.store.reader import load_run

pytestmark = pytest.mark.smoke
ROOT = Path(__file__).resolve().parents[2]
N_AGENTS, TPP, PERIODS, BETA, BP = 2_000, 10, 20, 100.0, 10


def config() -> RunConfig:
    return RunConfig.model_validate(
        {
            "name": "levers",
            "periods": PERIODS,
            "ticks_per_period": TPP,
            "seed": 5,
            "market": {
                "demand": {"beta": BETA},
                "price_grid": {"min": 1.0, "max": 4.0, "tick": 0.01},
                "fringe": {"horizon_ticks": TPP},
            },
            "swarm": {
                "mode": "agents",
                "n_agents": N_AGENTS,
                "arrivals": {"lam_mean": BETA / (N_AGENTS * TPP)},
            },
            "llm": {
                "mode": "mock",
                "lane": "mock",
                "slots": [
                    {
                        "name": "A",
                        "model": "scripted",
                        "family": "mock",
                        "policy": "tit_for_tat(target=1.87, trigger=0.01, punish=3)",
                    },
                    {"name": "B", "model": "mock-v1", "family": "mock", "policy": "undercutter"},
                ],
            },
            "clock": {"mode": "sync", "epoch_ticks": TPP},
            "checkpoint": {"every_periods": 5, "keep_last": 3},
            "logging": {"snapshot_every": 5, "snapshot_levels": 4},
        }
    )


def part_bytes(run_dir: Path, tables=("trades", "quotes", "flow_per_tick")) -> dict[str, str]:
    return {
        f"{t}/{p.name}": hashlib.sha256(p.read_bytes()).hexdigest()
        for t in tables
        for p in sorted((run_dir / t).glob("part-*.parquet"))
    }


@pytest.fixture(scope="module")
def parent(tmp_path_factory):
    r = run_config(config(), out=tmp_path_factory.mktemp("parent"))
    assert r.status == "ok" and r.periods_completed == PERIODS
    return r


def _before_identical(parent_dir: Path, child) -> None:
    par = load_run(parent_dir)
    for t in ("trades", "quotes", "agent_period", "flow_per_tick"):
        a = par.frame(t).filter(pl.col("period") < BP).drop("run_id")
        b = child.frame(t).filter(pl.col("period") < BP).drop("run_id")
        assert a.height > 0 and a.equals(b), t


def _on_grid(x: pl.Series, tick: float) -> bool:
    k = ((x - 1.0) / tick).round(0)
    return bool(((k * tick + 1.0 - x).abs() < 1e-9).all())


def test_branch_l3_tick_regrid_and_resume(parent, tmp_path) -> None:
    ev = {"type": "set_param", "path": "market.price_grid.tick", "value": 0.05}
    child_dir = branch(parent.run_dir, BP, [ev], "L3_tick", out=tmp_path / "u")
    child = load_run(child_dir)
    assert child.manifest["status"] == "ok", child.manifest["error"]
    _before_identical(parent.run_dir, child)
    trades = child.frame("trades")
    before = trades.filter(pl.col("period") < BP)["price"]
    after = trades.filter(pl.col("period") >= BP)["price"]
    assert after.len() > 0 and _on_grid(after, 0.05) and not _on_grid(before, 0.05)
    asks = child.frame("quotes").filter(pl.col("period") >= BP)["ask"].drop_nulls().drop_nans()
    assert asks.len() > 0 and _on_grid(asks, 0.05)
    ev_rows = child.frame("events").filter(pl.col("period") == BP)
    regrid = ev_rows.filter(pl.col("event_type") == "regrid")
    assert regrid.height == 1 and regrid["source"][0] == "derived"
    rep = json.loads(regrid["payload"][0])
    assert rep["old_tick"] == 0.01 and rep["new_tick"] == 0.05 and rep["n_levels"] == [301, 61]
    for d in rep["dealers"]:
        assert d["new_price"] >= d["old_price"] - 1e-12 and _on_grid(
            pl.Series([d["new_price"]]), 0.05
        )
    assert ev_rows.filter(pl.col("event_type") == "set_param").height == 1
    # kill after the event, resume from the child's own checkpoint: same bytes
    crashed = branch(parent.run_dir, BP, [ev], "L3_tick", out=tmp_path / "k", halt_after_period=12)
    assert json.loads((crashed / "manifest.json").read_text())["status"] == "failed"
    resumed = branch(parent.run_dir, BP, [ev], "L3_tick", out=tmp_path / "k")
    m = json.loads((resumed / "manifest.json").read_text())
    assert m["status"] == "ok" and m["resumes"][0]["from_period"] == BP
    assert part_bytes(resumed) == part_bytes(child_dir)


def test_branch_l2_fringe_on_then_off(parent, tmp_path) -> None:
    evs = [
        {"type": "set_param", "path": "market.fringe.share", "value": 0.3},
        {
            "type": "set_param",
            "path": "market.fringe.share",
            "value": 0.0,
            "when": {"at_period": 15},
        },
    ]
    child = load_run(branch(parent.run_dir, BP, evs, "L2_fringe", out=tmp_path))
    assert child.manifest["status"] == "ok", child.manifest["error"]
    _before_identical(parent.run_dir, child)
    trades = child.frame("trades")
    fringe = trades.filter(pl.col("seller_id") == -2)
    periods = set(fringe["period"].unique().to_list())
    assert periods and min(periods) >= BP and max(periods) <= 15  # off at 15, horizon 1 period
    assert fringe.height > 0.1 * trades.filter(pl.col("period").is_between(BP, 14)).height
    assert set(fringe["seller_kind"].unique().to_list()) == {int(AgentKind.SWARM)}
    assert (fringe["seller_cost"] == fringe["price"]).all()
    flow = child.frame("flow_per_tick")
    frows = flow.filter(pl.col("dealer_id") == -2)
    assert frows["period"].min() == BP and frows.height == (PERIODS - BP) * TPP
    assert frows["buy_qty"].sum() == fringe.height
    assert frows.filter(pl.col("period") >= 16)["sell_qty"].sum() == 0
    derived = child.frame("events").filter(pl.col("event_type") == "fringe_share")
    assert derived["period"].to_list() == [BP, 15]
    # dealers sell less while the fringe is on (same swarm draws as the parent)
    par = load_run(parent.run_dir).frame("trades")
    window = (pl.col("period") >= BP) & (pl.col("period") < 15)
    dealer_child = trades.filter(window & (pl.col("seller_id") >= 0)).height
    assert dealer_child < par.filter(window).height


def test_branch_levers_yaml_l2_l3_paths_are_settable_live() -> None:
    cc = load_campaign_config(ROOT / "configs" / "examples" / "branch_levers.yaml")
    base = cc.base.model_copy(deep=True)
    m = make_market(market_config(base), RNGRegistry(0))
    seen = set()
    for spec in cc.branches:
        for ev in spec.events:
            if isinstance(ev, SetParam) and ev.path.startswith("market."):
                assert ev.path in m.settable_params, (spec.name, ev.path)
                seen.add(ev.path)
    assert {"market.fringe.share", "market.price_grid.tick"} <= seen
