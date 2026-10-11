"""Invariants of the pre-registered campaign files (configs/sprint, configs/controls)."""

from pathlib import Path

import pytest

from finswarm.config.load import load_campaign_config
from finswarm.config.schema import CampaignConfig
from finswarm.sim.campaign import plan_runs, project, run_campaign
from finswarm.store.reader import load_run

ROOT = Path(__file__).resolve().parents[2]
SPRINT = ROOT / "configs" / "sprint"
CONTROLS = ROOT / "configs" / "controls"
ALL = sorted([*SPRINT.rglob("*.yaml"), *CONTROLS.glob("*.yaml")])

#: registered routing levels (alpha = 1): lower tau = more price-sensitive routing
LEVELS = {"s1": 0.25, "s2": 0.125, "s3": 0.0625, "s4": 0.02}


def load(rel: str) -> CampaignConfig:
    return load_campaign_config(ROOT / "configs" / rel)


def tau_events(cc: CampaignConfig) -> dict[str, float]:
    return {
        b.name: float(e.value)
        for b in cc.branches
        for e in b.events
        if e.type == "set_param" and e.path == "market.routing.tau"
    }


@pytest.mark.parametrize("path", ALL, ids=lambda p: p.stem)
def test_every_file_validates_and_projects(path: Path) -> None:
    cc = load_campaign_config(path)
    # pilots run before the freeze, so they are unregistered on purpose
    assert cc.prereg_id == (None if path.stem.startswith("PILOT-") else "PREREG-v1")
    p = project(cc)
    assert p.runs == len(plan_runs(cc)) > 0


def test_routing_levels_and_l1_direction() -> None:
    for rel in ("sprint/tier0/TR-H.yaml", "sprint/tier1/TR-S.yaml"):
        cc = load(rel)
        assert cc.base.market.routing.tau is None  # s1 = alpha * mu = 0.25
        assert cc.base.market.alpha * cc.base.market.demand.mu == LEVELS["s1"]
    for rel, name in (
        ("sprint/tier0/L1-H.yaml", "L1"),
        ("controls/MQC-L1.yaml", "L1"),
        ("sprint/tier2/EXT-H.yaml", "L1"),
        ("sprint/tier2/HYST-H.yaml", "hyst_L1"),
        ("sprint/tier1/LT-S.yaml", "LT_2"),
    ):
        assert tau_events(load(rel))[name] == LEVELS["s3"]  # s1 -> s3: more price-sensitive
    for rel, levels in (
        ("sprint/tier1/GR-H.yaml", "s1 s2 s3 s4"),  # s1 CRN-paired with s2-s4 (second review)
        ("sprint/tier2/GR+-H.yaml", "s1 s2 s3 s4"),
    ):
        got = {(r.cfg.tags["level"], r.cfg.market.routing.tau) for r in plan_runs(load(rel))}
        want = {(lv, None if lv == "s1" else LEVELS[lv]) for lv in levels.split()}
        assert got == want


def test_no_file_uses_a_legacy_routing_value() -> None:
    for path in ALL:
        text = path.read_text(encoding="utf-8")
        assert "0.0833" not in text and "0.03125" not in text, path.name


def test_gate_1b_dealer_template_control() -> None:
    cc = load("sprint/tier0/PC-H-dealer.yaml")
    fish = load("sprint/tier0/PC-H.yaml")
    assert cc.seeds == list(range(8)) and not cc.grid and cc.base.periods == 300
    m = cc.base.market
    assert (m.n_dealers, m.alpha, m.demand.kind, m.two_sided) == (
        2,
        1.0,
        "closed_form_logit",
        False,
    )
    assert m == fish.base.market and cc.base.swarm.mode == "closed_form"
    assert {(s.prompt, s.prefix, s.model) for s in cc.base.llm.slots} == {
        ("dealer_v1", "P1", "claude-haiku-5-5")
    }
    assert project(cc).calls == 8 * 2 * 300


@pytest.mark.parametrize(
    ("name", "model"), [("PILOT-H", "claude-haiku-5-5"), ("PILOT-S", "claude-sonnet-5-5")]
)
def test_pilots_are_twenty_unregistered_calls(name: str, model: str) -> None:
    cc = load(f"sprint/tier0/{name}.yaml")
    trunk = load("sprint/tier0/TR-H.yaml")
    assert cc.prereg_id is None and project(cc).calls == 20
    assert {s.model for s in cc.base.llm.slots} == {model} and cc.base.periods == 5
    assert cc.base.market == trunk.base.market and not set(cc.seeds) & set(trunk.seeds)


def test_pilot_runs_in_mock_mode(tmp_path) -> None:
    (res,) = run_campaign(
        load("sprint/tier0/PILOT-H.yaml"), provider="mock", out=tmp_path, dashboard=False
    )
    assert res.status == "ok"
    llm = load_run(res.run_dir).frame("llm_decisions")
    assert llm.height == 20 and llm["parse_ok"].all()
    assert llm["prompt_version"].str.starts_with("dealer_v1@").all()


def test_pilot_m_is_an_unregistered_14b_instrument_check_outside_the_hashed_set() -> None:
    """A-005: configs/pilots/ is outside the freeze globs (configs/sprint, configs/controls)."""
    path = ROOT / "configs" / "pilots" / "PILOT-M.yaml"
    cc = load_campaign_config(path)
    trunk = load("sprint/tier1/TR-M.yaml")
    assert cc.prereg_id is None and project(cc).calls == 20 and project(cc).usd == 0.0
    assert cc.seeds == [9002] and not set(cc.seeds) & set(trunk.seeds)
    assert cc.base.periods == 5 and cc.base.market == trunk.base.market
    assert (cc.base.swarm, cc.base.view, cc.base.clock) == (
        trunk.base.swarm,
        trunk.base.view,
        trunk.base.clock,
    )
    assert {s.model for s in cc.base.llm.slots} == {"ministral-14b-latest"}
    assert {s.prompt for s in cc.base.llm.slots} == {"dealer_v1"}
    p = cc.base.llm.params
    assert (p.effort, p.max_tokens, p.temperature) == (None, 2048, 1.0)
    assert cc.base.llm.memory_step == 25
    lane = cc.lanes["mistral_free"]
    assert (lane.models, lane.rpm, lane.max_in_flight, lane.free) == (
        ["ministral-14b-latest"],
        30,
        2,
        True,
    )
    assert path not in ALL  # not hashed: the sprint/controls globs do not cover configs/pilots


def test_pilot_l_is_an_unregistered_llama_instrument_check_outside_the_hashed_set() -> None:
    """A-005: the second family moves to Llama-3.3-70B (lane L, paid) after both ministral pilots
    failed; the pilot mirrors PILOT-M on the paid lane with a 0.5 USD cap."""
    path = ROOT / "configs" / "pilots" / "PILOT-L.yaml"
    cc = load_campaign_config(path)
    pilot_m = load_campaign_config(ROOT / "configs" / "pilots" / "PILOT-M.yaml")
    trunk = load("sprint/tier1/TR-M.yaml")
    assert cc.prereg_id is None and project(cc).calls == 20
    assert 0.0 < project(cc).usd < 0.5  # paid lane, well inside the pilot's budget line
    assert cc.seeds == [9003] and not set(cc.seeds) & (set(trunk.seeds) | set(pilot_m.seeds))
    assert cc.base.periods == 5 and cc.base.market == trunk.base.market
    assert (cc.base.swarm, cc.base.view, cc.base.clock) == (
        trunk.base.swarm,
        trunk.base.view,
        trunk.base.clock,
    )
    assert {(s.model, s.family, s.prompt) for s in cc.base.llm.slots} == {
        ("meta-llama/llama-3.3-70b-instruct", "llama", "dealer_v1")
    }
    assert cc.base.llm.lane == "openrouter_llama" and cc.base.llm.memory_step == 25
    p = cc.base.llm.params
    assert (p.effort, p.max_tokens, p.temperature) == (None, 2048, 1.0)
    lane = cc.lanes["openrouter_llama"]
    assert (lane.provider, lane.models, lane.max_in_flight, lane.budget_usd, lane.free) == (
        "openrouter",
        ["meta-llama/llama-3.3-70b-instruct"],
        4,
        0.5,
        False,
    )
    assert path not in ALL


#: A-005 (2026-10-10): the second family is Llama-3.3-70B on the paid lane L; the Mistral free
#: lane is dropped (both ministral pilots failed as dealer_v1 instruments)
LLAMA_SET = ("TR-M", "SH-M", "LT-M")
LLAMA_BUDGETS = {"TR-M": 4.0, "SH-M": 2.0, "LT-M": 2.5}  # lane budget lines, 10 USD lane in all
NOT_RUN_M = ("CA-M", "GH-M", "S2-M", "TR-M-ext")  # moved to tier2, kept for the record


def _branch_calls(cc: CampaignConfig) -> int:
    """Calls of the branch children only (the shared trunks are counted by TR-M)."""
    trunk_calls = sum(
        len([s for s in r.cfg.llm.slots if s.model != "scripted"]) * r.cfg.periods
        for r in plan_runs(cc)
        if r.cfg.tags.get("arm") == "trunk"
    )
    return project(cc).calls - trunk_calls


def test_second_family_set_is_the_h_design_on_lane_l() -> None:
    """campaign_table.md second family after A-005: Llama-3.3-70B via OpenRouter, 6 trunks."""
    trunk_h = load("sprint/tier0/TR-H.yaml")
    for name in LLAMA_SET:
        cc = load(f"sprint/tier1/{name}.yaml")
        assert cc.tier == 1 and cc.prereg_id == "PREREG-v1", name
        assert cc.seeds == list(range(6)), name
        (lane_name, lane) = next(iter(cc.lanes.items()))
        assert lane_name == "openrouter_llama" and cc.base.llm.lane == "openrouter_llama", name
        assert (lane.provider, lane.models, lane.max_in_flight, lane.free) == (
            "openrouter",
            ["meta-llama/llama-3.3-70b-instruct"],
            4,
            False,
        ), name
        assert lane.budget_usd == LLAMA_BUDGETS[name], name
        llm = cc.base.llm
        p = llm.params  # no thinking effort, temperature 1.0, max_tokens 2048 (A-004)
        assert (p.effort, p.max_tokens, p.temperature) == (None, 2048, 1.0), name
        assert llm.memory_step == 25, name
        assert {(s.model, s.family, s.prompt) for s in llm.slots} == {
            ("meta-llama/llama-3.3-70b-instruct", "llama", "dealer_v1")
        }, name
        # same market, swarm, view and clock as the Haiku trunk (second family, not a new design)
        assert cc.base.market == trunk_h.base.market, name
        assert cc.base.swarm == trunk_h.base.swarm, name
        assert (cc.base.view, cc.base.clock) == (trunk_h.base.view, trunk_h.base.clock), name
        assert project(cc).usd > 0.0, name  # paid lane
    for name in NOT_RUN_M:
        path = ROOT / "configs" / "sprint" / "tier2" / f"{name}.yaml"
        cc = load_campaign_config(path)
        assert cc.tier == 2, name
        assert path.read_text(encoding="utf-8").startswith("# NOT RUN (A-005"), name
    pc = ROOT / "configs" / "sprint" / "tier0" / "PC-M.yaml"
    assert pc.read_text(encoding="utf-8").startswith("# DROPPED (A-004 / A-005")
    assert {s.model for s in load("sprint/tier0/PC-M.yaml").base.llm.slots} == {
        "ministral-8b-latest"
    }  # content unchanged below the header


def test_second_family_call_counts_and_pairing() -> None:
    trunk = load("sprint/tier1/TR-M.yaml")
    assert trunk.seeds == list(range(6)) and not trunk.branches
    assert project(trunk).calls == 6 * 4 * 100
    for name, calls in (("SH-M", 1200), ("LT-M", 1200)):
        cc = load(f"sprint/tier1/{name}.yaml")
        # branch files share the trunk campaign (same name, base and seeds -> same trunk run ids)
        assert cc.name == "TR-M" and cc.base == trunk.base and cc.seeds == trunk.seeds, name
        assert {b.parent for b in cc.branches} == {"TR-M"}, name
        assert {b.at for b in cc.branches} == {"period:100"}, name
        assert _branch_calls(cc) == calls, name
    sh_m, sh_h = load("sprint/tier1/SH-M.yaml"), load("sprint/tier0/SH-H.yaml")
    assert sh_m.branches[0].events[:2] == sh_h.branches[0].events[:2]  # horizon + arm
    lt = load("sprint/tier1/LT-M.yaml")
    assert [b.name for b in lt.branches] == ["LT_1"]  # A-005: L2 only, no rule lever
    (l2,) = lt.branches
    ev = {e.path: e.value for e in l2.events if e.type == "set_param"}
    assert ev["tags.arm"] == "L2" and ev["market.fringe.share"] == 0.3
    ext = load_campaign_config(ROOT / "configs" / "sprint" / "tier2" / "TR-M-ext.yaml")
    assert ext.base == trunk.base  # regenerated from the Llama TR-M, kept for the record
