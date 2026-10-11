"""Per-branch ghosts (``BranchSpec.ghost``): projection, execution and the GH-H/GH-S files."""

import json
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from finswarm.config.load import load_campaign_config, load_yaml
from finswarm.config.load import run_id as make_run_id
from finswarm.config.schema import CampaignConfig
from finswarm.perturb.branch import child_config
from finswarm.sim.campaign import plan_runs, project, run_campaign
from finswarm.store.reader import load_run

ROOT = Path(__file__).resolve().parents[2]
SPRINT = ROOT / "configs" / "sprint"


def shrink(path: Path, *, periods: int = 6, at: int = 4, horizon: int = 8) -> CampaignConfig:
    """The campaign file at test size: one seed, a 200-agent swarm, 10 ticks per period."""
    data: dict[str, Any] = load_yaml(path)
    data["seeds"] = [0]
    base = data["base"]
    base.update(periods=periods, ticks_per_period=10)
    base["swarm"].update(n_agents=200, arrivals={"lam_mean": 0.01, "lam_sd_log": 0.5})
    base["clock"]["epoch_ticks"] = 10
    base["checkpoint"]["every_periods"] = 2
    for b in data["branches"]:
        b["at"] = f"period:{at}"
        for e in b["events"]:
            if e.get("path") == "periods":
                e["value"] = horizon
    return CampaignConfig.model_validate(data)


def test_projection_counts_one_focal_slot_per_ghost() -> None:
    cc = load_campaign_config(SPRINT / "tier0" / "GH-H.yaml")
    p = project(cc)
    kinds = [r["kind"] for r in p.rows]
    assert (kinds.count("run"), kinds.count("branch"), kinds.count("ghost")) == (12, 12, 48)
    assert p.ghosts == 48 and "48 ghosts" in p.table()
    ghost_calls = sum(r["calls"] for r in p.rows if r["kind"] == "ghost")
    assert ghost_calls == 12 * 4 * 1 * 50  # the registered 2,400
    no_ghost = project(load_campaign_config(SPRINT / "tier0" / "SH-H.yaml"))
    assert p.calls - no_ghost.calls == ghost_calls and no_ghost.ghosts == 0


@pytest.mark.parametrize(
    ("ghost", "sham"), [("tier0/GH-H", "tier0/SH-H"), ("tier2/GH-S", "tier1/SH-S")]
)
def test_ghost_files_reuse_the_sham_children(ghost: str, sham: str) -> None:
    """GH-* branch = SH-* sham branch + ghost: the sham children have the same run ids, so a
    campaign run after SH-* skips them and only runs the ghosts."""
    g = load_campaign_config(SPRINT / f"{ghost}.yaml")
    s = load_campaign_config(SPRINT / f"{sham}.yaml")
    (gb,), (sb,) = g.branches, s.branches
    assert gb.ghost == "replay_parent" and sb.ghost is None
    assert gb.model_dump(exclude={"ghost"}) == sb.model_dump(exclude={"ghost"})

    def children(cc: CampaignConfig) -> list[str]:
        (spec,) = cc.branches
        period = int(spec.at.split(":")[1])
        return [
            make_run_id(child_config(r.cfg, r.run_id, period, spec.events, spec.name))
            for r in plan_runs(cc)
        ]

    assert children(g) == children(s)


def test_gh_h_in_mock_mode_produces_ghost_sub_runs(tmp_path) -> None:
    cc = shrink(SPRINT / "tier0" / "GH-H.yaml")
    res = run_campaign(cc, provider="mock", out=tmp_path, dashboard=False)
    assert [r.status for r in res] == ["ok"] * 6  # trunk, sham, 4 ghosts
    sham = next(r for r in res if r.manifest.get("branch_period") == 4)
    ghosts = [r for r in res if r.manifest.get("ghost_of")]
    assert sorted(g.manifest["ghost_slot"] for g in ghosts) == [0, 1, 2, 3]
    for g in ghosts:
        k = g.manifest["ghost_slot"]
        assert g.run_dir == sham.run_dir / "ghost" / str(k)
        assert g.manifest["ghost_of"] == sham.run_id and g.periods_completed == 8
        tags = g.manifest["resolved_config"]["tags"]
        assert tags["replicate_id"] == f"sham/ghost:{k}" and tags["ghost_rival_policy"] == (
            "replay_parent"
        )
        llm = load_run(g.run_dir).frame("llm_decisions").filter(pl.col("period") >= 4)
        assert llm["slot"].unique().to_list() == [k]  # rivals replay the sham's quotes
        quotes = load_run(g.run_dir).frame("quotes").filter(pl.col("period") >= 4)
        sham_q = load_run(sham.run_dir).frame("quotes").filter(pl.col("period") >= 4)
        for j in {0, 1, 2, 3} - {k}:
            a = quotes.filter(pl.col("slot") == j).select(["period", "ask"])
            b = sham_q.filter(pl.col("slot") == j).select(["period", "ask"])
            assert a.equals(b)
    summary = json.loads((tmp_path / "TR-H" / "campaign.json").read_text(encoding="utf-8"))
    assert summary["status"] == "ok" and len(summary["runs"]) == 6

    again = run_campaign(cc, provider="mock", out=tmp_path, dashboard=False)
    assert len(again) == 6 and all(r.skipped for r in again)  # resume skips ghosts too
