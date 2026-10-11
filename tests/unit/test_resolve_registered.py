"""Rule-driven resolution of the registered templates (D-26): sting seeds from the qualified
trunks, LT lever branches from the Tier 0 lever files; resolved files validate and carry
provenance."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

from finswarm.config.load import load_campaign_config
from finswarm.sim.campaign import project

ROOT = Path(__file__).resolve().parents[2]
TIER1 = ROOT / "configs" / "sprint" / "tier1"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        f"_script_{name}", ROOT / "scripts" / f"{name}.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


rr = _load("resolve_registered")


def _events(cc, branch: str) -> dict[str, object]:
    (b,) = [b for b in cc.branches if b.name == branch]
    return {e.path: e.value for e in b.events if e.type == "set_param"}


def test_sting_seeds_are_the_first_six_qualified_in_seed_order(tmp_path: Path, capsys) -> None:
    rc = rr.main(
        [
            str(TIER1 / "S1-H.yaml"),
            "--qualified",
            "11",
            "3",
            "0",
            "8",
            "5",
            "2",
            "7",
            "9",
            "--out-dir",
            str(tmp_path),
        ]
    )
    assert rc == 0
    out = tmp_path / "S1-H.resolved.yaml"
    cc = load_campaign_config(out)
    assert cc.seeds == [0, 2, 3, 5, 7, 8]
    template = load_campaign_config(TIER1 / "S1-H.yaml")
    assert cc.base == template.base and cc.branches == template.branches  # only the seeds change
    assert project(cc).calls == project(template).calls  # 6 trunks + 6 S1 branches
    head = out.read_text(encoding="utf-8").splitlines()[:4]
    assert all(line.startswith("# ") for line in head)
    assert "sha256" in head[1] and "--qualified 11 3 0 8 5 2 7 9" in head[1]
    line = capsys.readouterr().out.strip()
    assert line.startswith("RESOLVED S1-H.yaml -> S1-H.resolved.yaml: seeds [0, 2, 3, 5, 7, 8]")
    assert "config_hash" in line


def test_fewer_than_six_qualified_keeps_them_all(tmp_path: Path) -> None:
    assert (
        rr.main([str(TIER1 / "S2-H.yaml"), "--qualified", "4", "1", "--out-dir", str(tmp_path)])
        == 0
    )
    assert load_campaign_config(tmp_path / "S2-H.resolved.yaml").seeds == [1, 4]


def test_competitive_side_takes_the_non_qualified_trunks(tmp_path: Path, capsys) -> None:
    qualified = [str(s) for s in (0, 1, 2, 3, 5, 6, 8, 11)]
    rc = rr.main(
        [str(TIER1 / "S1c-LLM-TR-H.yaml"), "--qualified", *qualified, "--out-dir", str(tmp_path)]
    )
    assert rc == 0
    cc = load_campaign_config(tmp_path / "S1c-LLM-TR-H.resolved.yaml")
    assert cc.seeds == [4, 7, 9]  # at most 3, in seed order, from TR-H's 12 trunks
    # every trunk qualified: nothing to run, no file
    rc = rr.main(
        [
            str(TIER1 / "S1c-LLM-TR-H.yaml"),
            "--qualified",
            *map(str, range(12)),
            "--out-dir",
            str(tmp_path / "none"),
        ]
    )
    assert rc == 2 and not (tmp_path / "none").exists()
    assert "NOTHING-TO-RUN" in capsys.readouterr().out


def test_lt_m_is_l2_only_since_a_005_and_no_longer_a_template(tmp_path: Path, capsys) -> None:
    """A-005: the second family (Llama, 10 USD lane budget) keeps L2 as its only lever."""
    assert "LT-M" not in rr.RULES
    assert rr.main([str(TIER1 / "LT-M.yaml"), "--lever", "L3", "--out-dir", str(tmp_path)]) == 1
    assert "not a registered template" in capsys.readouterr().err
    cc = load_campaign_config(TIER1 / "LT-M.yaml")
    assert [b.name for b in cc.branches] == ["LT_1"]
    ev = _events(cc, "LT_1")
    assert ev["tags.arm"] == "L2" and ev["market.fringe.share"] == 0.3
    assert project(cc).calls == 2400 + 1200


def test_lt_s_takes_two_distinct_eligible_levers(tmp_path: Path) -> None:
    assert (
        rr.main(
            [str(TIER1 / "LT-S.yaml"), "--lever", "L6", "--lever", "L1", "--out-dir", str(tmp_path)]
        )
        == 0
    )
    cc = load_campaign_config(tmp_path / "LT-S.resolved.yaml")
    assert _events(cc, "LT_1")["tags.arm"] == "L2"
    assert _events(cc, "LT_2")["tags.arm"] == "L6" and _events(cc, "LT_3")["tags.arm"] == "L1"
    assert _events(cc, "LT_3")["tags.level"] == "s3"  # L1 carries its registered level tag
    assert _events(cc, "LT_3")["market.routing.tau"] == 0.0625
    raw = yaml.safe_load((tmp_path / "LT-S.resolved.yaml").read_text(encoding="utf-8"))
    (lt2,) = [b for b in raw["branches"] if b["name"] == "LT_2"]
    assert lt2["events"][0] == {"type": "set_param", "path": "periods", "value": 150}


@pytest.mark.parametrize(
    "argv",
    [
        ["LT-M.yaml", "--lever", "L2"],  # L2 is the fixed first lever
        ["LT-M.yaml", "--lever", "L5"],  # Tier 1 lever: not eligible
        ["LT-S.yaml", "--lever", "L1"],  # LT-S needs two
        ["LT-S.yaml", "--lever", "L1", "--lever", "L1"],  # distinct
        ["LT-M.yaml", "--qualified", "0", "1"],  # wrong rule kind
        ["S1-H.yaml", "--lever", "L1"],
        ["S1-H.yaml"],
        ["TR-H.yaml", "--qualified", "0"],  # not a template
    ],
)
def test_misuse_is_refused_and_writes_nothing(tmp_path: Path, argv: list[str], capsys) -> None:
    path = (
        TIER1 / argv[0]
        if argv[0] != "TR-H.yaml"
        else ROOT / "configs" / "sprint" / "tier0" / "TR-H.yaml"
    )
    assert rr.main([str(path), *argv[1:], "--out-dir", str(tmp_path)]) == 1
    assert not list(tmp_path.glob("*.yaml"))
    assert "ERROR" in capsys.readouterr().err


def _branch(cc, name: str):
    (b,) = [b for b in cc.branches if b.name == name]
    return b


def test_shift_hangs_every_branch_off_the_ext_child(tmp_path: Path, capsys) -> None:
    """Trend rule: T0 100 -> 120, windows +20, parent = the TR-H-ext child."""
    l1 = ROOT / "configs" / "sprint" / "tier0" / "L1-H.yaml"
    hyst = ROOT / "configs" / "sprint" / "tier2" / "HYST-H.yaml"
    gh = ROOT / "configs" / "sprint" / "tier0" / "GH-H.yaml"
    rc = rr.main([str(l1), str(hyst), str(gh), "--shift", "20", "--out-dir", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out.splitlines()
    assert len(out) == 3 and all(ln.startswith("SHIFTED ") for ln in out)
    ext_cc = load_campaign_config(ROOT / "configs" / "sprint" / "tier0" / "TR-H-ext.yaml")
    cc = load_campaign_config(tmp_path / "L1-H.shifted.yaml")
    template = load_campaign_config(l1)
    assert cc.name == "TR-H" and cc.base == template.base and cc.seeds == template.seeds
    assert [b.name for b in cc.branches] == ["ext", "L1"]
    assert _branch(cc, "ext") == ext_cc.branches[0]  # the extension branch verbatim
    lever = _branch(cc, "L1")
    assert (lever.parent, lever.at) == ("TR-H--ext", "period:120")
    ev = _events(cc, "L1")
    assert (
        ev["periods"] == 170 and ev["tags.shift"] == "20" and ev["tags.t0"] == "120"
    )  # tags are strings
    assert ev["market.routing.tau"] == 0.0625 and ev["tags.arm"] == "L1"  # the lever itself
    assert ev["tags.label"] == "L1-H"  # labels do not move: the analysis keys on tags.shift
    # hysteresis: horizon 160 -> 180, the reverts fire at 150 instead of 130
    h = load_campaign_config(tmp_path / "HYST-H.shifted.yaml")
    assert _events(h, "hyst_L1")["periods"] == 180
    reverts = [e for e in _branch(h, "hyst_L1").events if e.type == "revert"]
    assert [e.when.at_period for e in reverts] == [150, 150]
    assert [e.event_id for e in reverts] == [2, 3]  # indices unchanged: the tag events come last
    # ghosts keep their rival policy and hang off the ext child too
    g = load_campaign_config(tmp_path / "GH-H.shifted.yaml")
    assert _branch(g, "sham").ghost == "replay_parent" and _branch(g, "sham").parent == "TR-H--ext"
    # projection: trunks + ext children (20 periods) + shifted levers (50 periods after 120)
    assert project(cc).calls == 12 * 4 * 100 + 12 * 4 * 20 + 12 * 4 * 50
    assert project(ext_cc).calls == 12 * 4 * 100 + 12 * 4 * 20
    head = (tmp_path / "L1-H.shifted.yaml").read_text(encoding="utf-8").splitlines()[:3]
    assert "SHIFTED from L1-H.yaml" in head[0] and "--shift 20" in head[1]


@pytest.mark.parametrize(
    "argv",
    [
        ["sprint/tier0/L1-H.yaml", "--shift", "0"],
        ["sprint/tier0/L1-H.yaml", "--shift", "20", "--lever", "L3"],
        ["sprint/tier0/L1-H.yaml", "--shift", "20", "--make-ext", "20"],
        ["sprint/tier0/PC-H.yaml", "--shift", "20"],  # 300 periods: not a 100-period campaign
        ["controls/SC-1.yaml", "--shift", "20"],  # scripted only: the rule is for LLM campaigns
        ["controls/SC-1.yaml", "--make-ext", "20"],
        [
            "sprint/tier1/L5-H.yaml",
            "--shift",
            "20",
            "--ext",
            "sprint/tier1/TR-S-ext.yaml",
        ],  # other trunk
    ],
)
def test_shift_misuse_is_refused(tmp_path: Path, argv: list[str], capsys) -> None:
    paths = [str(ROOT / "configs" / a) if a.endswith(".yaml") else a for a in argv]
    assert rr.main([*paths, "--out-dir", str(tmp_path)]) == 1
    assert not list(tmp_path.glob("*.yaml"))
    assert "ERROR" in capsys.readouterr().err


#: campaign -> (its directory, its ext file's directory); TR-M-ext sits in tier2 since A-005 (not run)
EXT_CAMPAIGNS = {
    "TR-H": ("sprint/tier0", "sprint/tier0"),
    "TR-S": ("sprint/tier1", "sprint/tier1"),
    "TR-M": ("sprint/tier1", "sprint/tier2"),
    "AC-H": ("sprint/tier0", "sprint/tier0"),
    "GR-H": ("sprint/tier1", "sprint/tier1"),
    "CA-H": ("sprint/tier1", "sprint/tier1"),
    "PREV-H": ("sprint/tier2", "sprint/tier2"),
}


def test_generic_ext_branch_matches_the_hand_written_tr_h_ext() -> None:
    data = yaml.safe_load((ROOT / "configs" / "sprint" / "tier0" / "TR-H.yaml").read_text())
    ext = yaml.safe_load((ROOT / "configs" / "sprint" / "tier0" / "TR-H-ext.yaml").read_text())
    assert rr.make_ext_branch(data, 20) == ext["branches"][0]
    assert rr.ext_campaign(data, 20)["base"] == ext["base"]


def test_every_100_period_llm_campaign_has_its_ext_file() -> None:
    """`<campaign>-ext.yaml` = the campaign (grid kept, branches dropped) + the generic ext."""
    for name, (rel, ext_rel) in EXT_CAMPAIGNS.items():
        src = ROOT / "configs" / rel / f"{name}.yaml"
        ext = ROOT / "configs" / ext_rel / f"{name}-ext.yaml"
        assert ext.exists(), ext
        cc, ec = load_campaign_config(src), load_campaign_config(ext)
        assert ec.name == cc.name and ec.base == cc.base and ec.seeds == cc.seeds
        assert ec.grid == cc.grid and ec.prereg_id == "PREREG-v1"
        (b,) = ec.branches
        assert b.name == "ext" and b.parent == cc.base.name and b.at == "period:100"
        ev = {e.path: e.value for e in b.events if e.type == "set_param"}
        assert ev["periods"] == 120 and ev["tags.arm"] == f"{cc.base.tags['arm']}_ext"
        assert ev["tags.label"] == f"{cc.base.tags['label']}-ext"
        assert b.events[-1].note == "collusion_established_120"
        runs = len(list(project(cc).rows))
        assert project(ec).calls == project(cc).calls + (project(cc).calls // 100) * 20, name
        assert runs >= 1


def test_shift_of_a_from_scratch_campaign_is_its_ext(tmp_path: Path) -> None:
    ac = ROOT / "configs" / "sprint" / "tier0" / "AC-H.yaml"
    assert rr.main([str(ac), "--shift", "20", "--out-dir", str(tmp_path)]) == 0
    cc = load_campaign_config(tmp_path / "AC-H.shifted.yaml")
    ext = load_campaign_config(ROOT / "configs" / "sprint" / "tier0" / "AC-H-ext.yaml")
    assert cc.branches == ext.branches and cc.base == ext.base  # AC-H-ext.yaml found and used
    gr = ROOT / "configs" / "sprint" / "tier1" / "GR-H.yaml"
    assert rr.main([str(gr), "--shift", "20", "--out-dir", str(tmp_path)]) == 0
    g = load_campaign_config(tmp_path / "GR-H.shifted.yaml")
    assert g.grid == load_campaign_config(gr).grid and g.branches[0].parent == "GR-H"
    assert project(g).calls == 16 * 4 * 100 + 16 * 4 * 20  # every grid cell gets its ext child


def test_shift_of_a_sonnet_branch_uses_tr_s_ext(tmp_path: Path) -> None:
    sh = ROOT / "configs" / "sprint" / "tier1" / "SH-S.yaml"
    assert rr.main([str(sh), "--shift", "20", "--out-dir", str(tmp_path)]) == 0
    cc = load_campaign_config(tmp_path / "SH-S.shifted.yaml")
    assert [b.name for b in cc.branches] == ["ext", "sham"]
    assert (
        cc.branches[0]
        == load_campaign_config(ROOT / "configs" / "sprint" / "tier1" / "TR-S-ext.yaml").branches[0]
    )
    assert cc.branches[1].parent == "TR-S--ext" and cc.branches[1].at == "period:120"
    assert _events(cc, "sham")["periods"] == 170


def test_every_registered_template_exists_and_validates() -> None:
    for stem in rr.RULES:
        cc = load_campaign_config(TIER1 / f"{stem}.yaml")
        assert cc.prereg_id == "PREREG-v1"
        assert "TEMPLATE RULE" in (TIER1 / f"{stem}.yaml").read_text(encoding="utf-8"), stem
    for lever in rr.ELIGIBLE_LEVERS:
        assert rr.lever_events(lever)  # every eligible lever has a registered Tier 0 file
