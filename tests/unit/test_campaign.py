import json
from pathlib import Path

import pytest
from fakes import base_config, fake_market_factory

from finswarm.config.load import load_campaign_config, load_yaml
from finswarm.config.schema import CampaignConfig, RunConfig
from finswarm.sim.campaign import (
    TokenProfile,
    check_prereg,
    expand_cells,
    match_parents,
    plan_runs,
    project,
    run_campaign,
)

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = sorted((ROOT / "configs" / "examples").glob("*.yaml"))


def campaign(**over) -> CampaignConfig:
    data = {"name": "camp", "base": base_config().model_dump(mode="json"), "seeds": [0, 1]}
    data.update(over)
    return CampaignConfig.model_validate(data)


def test_example_configs_validate_and_project() -> None:
    assert {p.name for p in EXAMPLES} == {
        "a1_fish.yaml",
        "a2_swarm.yaml",
        "c_agrawal.yaml",
        "branch_levers.yaml",
    }
    for path in [*EXAMPLES, ROOT / "configs" / "smoke.yaml"]:
        cc = load_campaign_config(path)
        runs = plan_runs(cc)
        assert runs and all(isinstance(r.cfg, RunConfig) for r in runs)
        assert len({r.run_id for r in runs}) == len(runs)
        proj = project(cc)
        assert proj.runs == len(runs)
        for spec in cc.branches:
            assert match_parents(spec, runs, cc.base.name), spec.name


def test_a1_fish_zipped_grid_scales_price_grid_with_alpha() -> None:
    cc = load_campaign_config(ROOT / "configs" / "examples" / "a1_fish.yaml")
    cells = expand_cells(cc)
    assert len(cells) == 3
    runs = plan_runs(cc)
    assert len(runs) == 3 * 8
    by_alpha = {r.cfg.market.alpha: r.cfg.market.price_grid for r in runs}
    assert by_alpha[3.2].max == pytest.approx(12.8) and by_alpha[10.0].tick == pytest.approx(0.1)
    assert all(r.n_llm == 2 and r.calls == 600 for r in runs)


def test_dry_run_projection_counts_calls_and_usd() -> None:
    smoke = load_campaign_config(ROOT / "configs" / "smoke.yaml")
    p = project(smoke)
    assert (p.runs, p.calls, p.usd) == (3, 0, 0.0)  # scripted dealers make no LLM calls
    cc = campaign(
        base={
            **base_config().model_dump(mode="json"),
            "periods": 40,
            "llm": {
                "mode": "live",
                "lane": "haiku",
                "slots": [{"name": n, "model": "haiku", "family": "anthropic"} for n in "ABCD"],
            },
        },
        seeds=[0, 1, 2],
        lanes={"haiku": {"provider": "anthropic", "models": ["haiku"], "budget_usd": 0.01}},
        branches=[
            {
                "name": "lever",
                "parent": "t",
                "at": "period:30",
                "events": [{"type": "set_param", "path": "periods", "value": 50}],
            }
        ],
    )
    prof = TokenProfile(input_tokens=1000, output_tokens=200)
    p = project(cc, profile=prof)
    per_call = (1000 * 0.10 + 200 * 0.50) / 1e6  # claude-haiku-5-5 in configs/models.yaml
    assert p.usd == pytest.approx(p.calls * per_call)
    assert "models.yaml" in p.pricing
    assert p.runs == 3 and p.branches == 3
    assert p.calls == 3 * 4 * 40 + 3 * 4 * (50 - 30)
    assert p.by_lane == {"haiku": pytest.approx(p.usd)} and p.over_budget == ["haiku"]
    assert project(cc, provider="mock", profile=prof).usd == 0.0
    assert "TOTAL" in p.table() and "OVER BUDGET" in p.table()


def test_default_profile_prices_haiku_at_the_catalogue_rate() -> None:
    from finswarm.llm.pricing import estimate_call_cost

    assert TokenProfile() == TokenProfile(input_tokens=1500, output_tokens=300)
    assert estimate_call_cost("haiku", 1500, 300) == pytest.approx(0.0003)
    assert estimate_call_cost("mock-v1", 1500, 300) == 0.0
    assert estimate_call_cost("ministral-8b", 1500, 300) == 0.0
    with pytest.raises(KeyError):
        estimate_call_cost("no-such-model", 1, 1)
    a1 = project(load_campaign_config(ROOT / "configs" / "examples" / "a1_fish.yaml"))
    assert a1.calls == 24 * 600 and a1.usd == pytest.approx(a1.calls * 0.0003)


def test_prereg_guard_blocks_registered_campaigns_without_frozen() -> None:
    if (ROOT / "docs" / "03_preregistration" / "FROZEN").exists():
        pytest.skip("pre-registration already frozen in this checkout")
    with pytest.raises(RuntimeError):
        check_prereg(campaign(prereg_id="PREREG-v1"))
    check_prereg(campaign())  # unregistered campaigns are not guarded


def test_campaign_runs_branches_and_resumes(tmp_path) -> None:
    cc = campaign(
        branches=[
            {
                "name": "coarse",
                "parent": "t",
                "at": "period:20",
                "events": [{"type": "set_param", "path": "market.price_grid.tick", "value": 0.05}],
            },
            {"name": "never", "parent": "t", "at": "checkpoint:not_there", "events": []},
        ]
    )
    res = run_campaign(cc, out=tmp_path, market_factory=fake_market_factory, dashboard=False)
    assert [r.status for r in res] == ["ok"] * 4
    summary = json.loads((tmp_path / "camp" / "campaign.json").read_text())
    assert summary["status"] == "ok" and len(summary["skipped_branches"]) == 2
    children = [r for r in res if r.manifest.get("parent_run_id")]
    assert len(children) == 2 and all(c.manifest["branch_period"] == 20 for c in children)
    again = run_campaign(cc, out=tmp_path, market_factory=fake_market_factory, dashboard=False)
    assert all(r.skipped for r in again)


def test_nested_branches_hang_off_an_earlier_child(tmp_path) -> None:
    """A branch may name the children of an earlier branch (run name or branch name) as its
    parents: the trend-rule extension (T0 -> 120) and the levers shifted onto it."""
    cc = campaign(
        branches=[
            {
                "name": "ext",
                "parent": "t",
                "at": "period:20",
                "events": [{"type": "set_param", "path": "periods", "value": 26}],
            },
            {
                "name": "coarse",
                "parent": "t--ext",
                "at": "period:26",
                "events": [
                    {"type": "set_param", "path": "periods", "value": 30},
                    {"type": "set_param", "path": "market.price_grid.tick", "value": 0.05},
                ],
            },
            {"name": "late", "parent": "ext", "at": "period:26", "events": []},
        ]
    )
    proj = project(cc)
    names = sorted(r["name"] for r in proj.rows if r["kind"] == "branch")
    assert names == ["t--ext"] * 2 + ["t--ext--coarse"] * 2 + ["t--ext--late"] * 2
    by_name = {r["name"]: r for r in proj.rows if r["kind"] == "branch"}
    assert by_name["t--ext"]["calls"] == 0  # scripted slots: no LLM calls, but the rows exist
    res = run_campaign(cc, out=tmp_path, market_factory=fake_market_factory, dashboard=False)
    assert [r.status for r in res] == ["ok"] * 8
    ext = [r for r in res if r.manifest.get("name") == "t--ext"]
    levers = [r for r in res if r.manifest.get("name") == "t--ext--coarse"]
    late = [r for r in res if r.manifest.get("name") == "t--ext--late"]
    assert len(ext) == len(levers) == len(late) == 2
    assert {r.manifest["parent_run_id"] for r in levers} == {r.run_id for r in ext}
    assert all(r.manifest["branch_period"] == 26 and r.periods_completed == 30 for r in levers)
    assert all(r.manifest["branch_period"] == 26 and r.periods_completed == 26 for r in late)
    summary = json.loads((tmp_path / "camp" / "campaign.json").read_text())
    assert summary["status"] == "ok" and summary["skipped_branches"] == []
    again = run_campaign(cc, out=tmp_path, market_factory=fake_market_factory, dashboard=False)
    assert all(r.skipped for r in again)


def _reduced(path: Path, out: Path, seeds: list[int], n_agents: int) -> CampaignConfig:
    """A registered config with fewer trunks and a small swarm (mock-mode test copy)."""
    import yaml

    data = load_yaml(path)
    data["seeds"] = seeds
    data["base"]["swarm"]["n_agents"] = n_agents
    data["base"]["swarm"]["arrivals"]["lam_mean"] = 1.0 / n_agents
    data.pop("prereg_id", None)  # the FROZEN guard never applies to a test copy
    out.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return load_campaign_config(out)


def test_tr_h_ext_then_a_shifted_lever_in_mock_mode(tmp_path) -> None:
    """The registered trend-rule path: TR-H-ext extends two mock trunks to 120, then the
    shifted L1 branches from the extension children at 120 and runs to 170."""
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(
        "_resolve_registered", ROOT / "scripts" / "resolve_registered.py"
    )
    rr = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = rr
    spec.loader.exec_module(rr)
    tier0 = ROOT / "configs" / "sprint" / "tier0"
    assert rr.main([str(tier0 / "L1-H.yaml"), "--shift", "20", "--out-dir", str(tmp_path)]) == 0
    ext = _reduced(tier0 / "TR-H-ext.yaml", tmp_path / "ext.yaml", [0, 1], 400)
    lever = _reduced(tmp_path / "L1-H.shifted.yaml", tmp_path / "lever.yaml", [0, 1], 400)
    assert ext.base == lever.base and ext.branches[0] == lever.branches[0]
    out = tmp_path / "out"
    res = run_campaign(ext, provider="mock", out=out, dashboard=False)
    assert [r.status for r in res] == ["ok"] * 4
    ext_children = [r for r in res if r.manifest.get("branch_name") == "ext"]
    assert len(ext_children) == 2
    for c in ext_children:
        assert c.periods_completed == 120 and c.manifest["branch_period"] == 100
        assert (c.manifest["arm"], c.manifest["label"]) == ("trunk_ext", "TR-H-ext")
        names = [n for ck in c.manifest["checkpoints"] for n in ck["names"]]
        assert "collusion_established_120" in names
    res2 = run_campaign(lever, provider="mock", out=out, dashboard=False)
    assert [r.status for r in res2] == ["ok"] * 6
    assert sum(r.skipped for r in res2) == 4  # the trunks and the ext children are reused
    levers = [r for r in res2 if r.manifest.get("branch_name") == "L1"]
    assert len(levers) == 2
    for lv in levers:
        assert lv.manifest["parent_run_id"] in {c.run_id for c in ext_children}
        assert lv.manifest["branch_period"] == 120 and lv.periods_completed == 170
        assert (lv.manifest["arm"], lv.manifest["label"]) == ("L1", "L1-H")
        assert lv.manifest["name"] == "TR-H--ext--L1"


def test_from_scratch_ext_campaign_in_mock_mode(tmp_path) -> None:
    """AC-H-ext (1 LLM dealer + 3 best responders): every run gets its ext child to 120."""
    tier0 = ROOT / "configs" / "sprint" / "tier0"
    ext = _reduced(tier0 / "AC-H-ext.yaml", tmp_path / "ac_ext.yaml", [0, 1], 400)
    res = run_campaign(ext, provider="mock", out=tmp_path / "out", dashboard=False)
    assert [r.status for r in res] == ["ok"] * 4
    kids = [r for r in res if r.manifest.get("branch_name") == "ext"]
    assert len(kids) == 2
    for c in kids:
        assert c.periods_completed == 120 and c.manifest["branch_period"] == 100
        assert (c.manifest["arm"], c.manifest["label"]) == ("AC_ext", "AC-H-ext")
        assert c.manifest["name"] == "AC-H--ext"


def test_smoke_yaml_is_a_campaign_with_three_seeds() -> None:
    data = load_yaml(ROOT / "configs" / "smoke.yaml")
    cc = CampaignConfig.model_validate(data)
    assert cc.seeds == [0, 1, 2] and cc.base.periods == 50
    assert cc.base.checkpoint.every_periods == 10
    assert [s.policy.split("(")[0] for s in cc.base.llm.slots] == ["tit_for_tat", "undercutter"]
    (ev,) = cc.base.events
    assert ev.type == "set_param" and ev.when.at_period == 25 and ev.path == "market.routing.tau"
