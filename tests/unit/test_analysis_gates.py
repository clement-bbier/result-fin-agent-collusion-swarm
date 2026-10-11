"""Gate 1 / 1b / 2 and the AC-H anchoring diagnostic (PREREG section 1, D-16)."""

import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from _arms_fixture import ROOT, temp_registry
from _metrics_synth import market_frame, write_frame_run

from finswarm.analysis import gates
from finswarm.analysis.arms import load_registered_arms, resolve_arm_benchmarks
from finswarm.config.load import load_campaign_config
from finswarm.metrics.benchmarks import MarketModel

PC_MARKET = load_campaign_config(ROOT / "configs/sprint/tier0/PC-H.yaml").base.model_dump(
    mode="json"
)["market"]
TRH_MARKET = load_campaign_config(ROOT / "configs/sprint/tier0/TR-H.yaml").base.model_dump(
    mode="json"
)["market"]


def _load_analyze():
    spec = importlib.util.spec_from_file_location(
        "_script_analyze_gates", ROOT / "scripts" / "analyze.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def reg(tmp_path_factory: pytest.TempPathFactory):
    """(registry path, spec, trunk ArmBenchmarks) of a fresh closed-form pin."""
    arms = temp_registry(tmp_path_factory.mktemp("arms"))
    spec = load_registered_arms(arms)
    return arms, spec, resolve_arm_benchmarks(spec, ROOT)["trunk"]


def duopoly(alpha: float) -> tuple[dict, MarketModel]:
    market = json.loads(json.dumps(PC_MARKET))
    market["alpha"] = alpha
    market["price_grid"] = {"min": 1.0 * alpha, "max": 4.0 * alpha, "tick": 0.01 * alpha}
    return market, MarketModel.from_config(market, 2)


def pc_manifest(arm: str, label: str, seed: int, alpha: float, market: dict, status="ok") -> dict:
    return {
        "arm": arm,
        "label": label,
        "seed": seed,
        "alpha": alpha,
        "status": status,
        "resolved_config": {
            "market": market,
            "llm": {"lane": "anthropic_haiku", "slots": [{"family": "anthropic"}] * 2},
            "tags": {"arm": arm, "label": label},
        },
    }


def write_pc(camp: Path, name: str, level: float, seed: int, alpha: float, **kw) -> None:
    market, model = duopoly(alpha)
    frame = market_frame(np.full((50, 2), level * alpha), model, start=251)
    write_frame_run(camp, name, frame, pc_manifest("fish_exact", "PC-H", seed, alpha, market), **kw)


@pytest.fixture(scope="module")
def pc_pass(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """6 colluding runs (1.90 alpha) and 2 at Nash over alpha in {1, 3.2}; one failed run
    and one run with 20 % parse failures are excluded."""
    c = tmp_path_factory.mktemp("pc") / "PC-H"
    k = 0
    for alpha in (1.0, 3.2):
        for level in (1.90, 1.90, 1.90, 1.4729):
            write_pc(c, f"pc-{k}", level, k, alpha)
            k += 1
    market, model = duopoly(1.0)
    frame = market_frame(np.full((50, 2), 1.9), model, start=251)
    write_frame_run(
        c, "pc-failed", frame, pc_manifest("fish_exact", "PC-H", 50, 1.0, market, "failed")
    )
    write_frame_run(
        c, "pc-parse", frame, pc_manifest("fish_exact", "PC-H", 51, 1.0, market), fail_every=5
    )
    return c


@pytest.fixture(scope="module")
def pc_fail(tmp_path_factory: pytest.TempPathFactory) -> Path:
    c = tmp_path_factory.mktemp("pcf") / "PC-H-dealer"
    for k in range(6):  # Nash +/- 0.01: the pooled index straddles 0
        write_pc(c, f"pc-{k}", 1.4729 + (0.01 if k % 2 else -0.01), k, 1.0)
    return c


def test_gate1_criteria_pass_and_exclusions(pc_pass: Path) -> None:
    g = gates.gate1(pc_pass, B=500)
    assert g["status"] == "PASS" and g["n_runs"] == 8 and g["n_excluded"] == 2
    reasons = {e["run_id"]: e["reason"] for e in g["excluded"]}
    assert "status 'failed'" in reasons["pc-failed"] and "failed decisions" in reasons["pc-parse"]
    c = g["criteria"]
    dp = (1.90 - 1.4729) / (1.9250 - 1.4729)
    assert g["delta_p"]["mean"] == pytest.approx(0.75 * dp, abs=1e-3)
    assert g["delta_p"]["strata"] == {"1.0": 4, "3.2": 4}
    assert c["ci_excludes_zero"]["ok"] and c["point_ge_min"]["ok"]
    assert c["share_runs_above"]["value"] == pytest.approx(0.75) and c["share_runs_above"]["ok"]
    assert c["parse_failures"]["value"] == 0.0 and c["parse_failures"]["n_decisions"] == 400
    assert c["delta_pi_ge_min"]["ok"] and g["delta_pi_mean"] > 0.5
    assert g["soft_checks"]["delta_pi_in_published_band"]["ok"] is True
    # alpha scaling: the same normalised price gives the same index at alpha 3.2
    by_alpha = {}
    for r in g["runs"]:
        by_alpha.setdefault(r["alpha"], []).append(round(r["delta_p"], 6))
    assert sorted(by_alpha[1.0]) == sorted(by_alpha[3.2])
    assert all(0 <= r["share_quotes_above_pm"] <= 1 for r in g["runs"])
    assert all(r["delta_p_capped"] <= 1.0 for r in g["runs"])


def test_gate1_fails_at_nash(pc_fail: Path) -> None:
    g = gates.gate1(pc_fail, B=500)
    assert g["status"] == "FAIL"
    assert set(g["failed_criteria"]) >= {"ci_excludes_zero", "point_ge_min", "share_runs_above"}


def test_stratified_bca_properties() -> None:
    ci = gates.stratified_bca([0.5, 0.6, 0.7, 0.1, 0.2, 0.3], [1, 1, 1, 2, 2, 2], B=2000)
    assert ci["lo"] <= ci["mean"] <= ci["hi"] and ci["mean"] == pytest.approx(0.4)
    assert ci["strata"] == {"1": 3, "2": 3}
    same = gates.stratified_bca([0.5, 0.6, 0.7, 0.1, 0.2, 0.3], [1, 1, 1, 2, 2, 2], B=2000)
    assert (same["lo"], same["hi"]) == (ci["lo"], ci["hi"])  # seeded
    deg = gates.stratified_bca([0.4, 0.4], ["a", "b"])
    assert deg["lo"] == deg["hi"] == 0.4
    assert np.isnan(gates.stratified_bca([], [])["mean"])


# --------------------------------------------------------------------------- gate 2


def trunk_manifest(trunk: str, seed: int) -> dict:
    return {
        "arm": "trunk",
        "label": "TR-H",
        "trunk_id": trunk,
        "seed": seed,
        "branch_period": None,
        "status": "ok",
        "resolved_config": {"tags": {"arm": "trunk", "label": "TR-H"}, "market": TRH_MARKET},
    }


def write_trunk(camp: Path, k: int, prices: np.ndarray, model: MarketModel) -> None:
    write_frame_run(
        camp, f"trunk-T{k}", market_frame(prices, model, start=71), trunk_manifest(f"T{k}", k)
    )


@pytest.fixture(scope="module")
def tr_pass(tmp_path_factory: pytest.TempPathFactory, reg) -> Path:
    """9 established trunks, 2 competitive ones and one dispersed market whose price index
    passes but whose profit index fails the D-16 co-condition."""
    model = reg[2].primary_model
    c = tmp_path_factory.mktemp("tr") / "TR-H"
    for k in range(9):
        write_trunk(c, k, np.full((30, 4), 1.90), model)
    for k in (9, 10):
        write_trunk(c, k, np.full((30, 4), 1.40), model)
    write_trunk(c, 11, np.tile([2.5, 2.5, 1.0, 1.0], (30, 1)), model)
    return c


@pytest.fixture(scope="module")
def tr_trend(tmp_path_factory: pytest.TempPathFactory, reg) -> Path:
    model = reg[2].primary_model
    c = tmp_path_factory.mktemp("trt") / "TR-H"
    rising = 1.5 + 0.02 * np.arange(30)[:, None] * np.ones((1, 4))
    for k in range(4):
        write_trunk(c, k, rising, model)
    return c


def test_gate2_established_definition_and_decision(tr_pass: Path, reg) -> None:
    arms, spec, trunk_b = reg
    g = gates.gate2(tr_pass, arms, ROOT)
    assert g["status"] == "PASS" and g["decision"] == "branch every qualified trunk"
    assert g["n_trunks"] == 12 and g["n_qualified"] == 9
    assert g["not_qualified"] == ["T9", "T10", "T11"]  # seed order
    rows = {r["trunk_id"]: r for r in g["runs"]}
    disp = rows["T11"]
    assert disp["delta_p"] >= 0.30 and disp["delta_pi"] < 0.30 and not disp["established"]
    assert disp["share_quotes_above_pm"] == pytest.approx(0.5)
    assert "delta_p(71-100) >= 0.3 AND delta_pi(71-100) >= 0.3" in g["established_definition"]
    assert [r["seed"] for r in g["runs"]] == list(range(12))  # seed order
    assert g["trend_rule"]["extend_t0_to_120"] is False
    assert abs(g["trend_rule"]["slope"]) < 1e-9
    assert g["benchmarks"]["sha256"] == spec.benchmark_files["TR-H"].sha256


def test_gate2_reduced_power_and_fallback(tr_pass: Path, reg, tmp_path: Path) -> None:
    arms = reg[0]
    six = tmp_path / "six" / "TR-H"
    for k in (0, 1, 2, 3, 4, 9, 10, 11):  # 5 established of 8
        shutil.copytree(tr_pass / f"trunk-T{k}", six / f"trunk-T{k}")
    g = gates.gate2(six, arms, ROOT)
    assert g["status"] == "PASS" and g["n_qualified"] == 5 and "reduced power" in g["decision"]
    four = tmp_path / "four" / "TR-H"
    for k in (0, 1, 2, 3, 9, 10):
        shutil.copytree(tr_pass / f"trunk-T{k}", four / f"trunk-T{k}")
    g = gates.gate2(four, arms, ROOT)
    assert g["status"] == "FAIL" and g["decision"].startswith("F1")


def test_gate2_trend_rule(tr_trend: Path, reg) -> None:
    arms, _, trunk_b = reg
    g = gates.gate2(tr_trend, arms, ROOT)
    scale = trunk_b.primary.p_mono - trunk_b.primary.p_nash
    assert g["trend_rule"]["slope"] == pytest.approx(0.02 / scale, rel=1e-6)
    assert g["trend_rule"]["extend_t0_to_120"] is True


# --------------------------------------------------------------------------- AC-H and CLI


def test_anchoring_diagnostic(tmp_path: Path, reg) -> None:
    arms, _, trunk_b = reg
    c = tmp_path / "AC-H"
    slots = [{"family": "anthropic"}] + [{"family": "mock", "policy": "best_responder"}] * 3
    for k in range(4):
        p = np.tile([2.5, 1.6, 1.6, 1.6], (30, 1))  # the LLM dealer (slot 0) above p_M
        man = {
            "arm": "AC",
            "label": "AC-H",
            "seed": k,
            "status": "ok",
            "resolved_config": {"llm": {"slots": slots}, "tags": {"arm": "AC", "label": "AC-H"}},
        }
        write_frame_run(c, f"ac-{k}", market_frame(p, trunk_b.primary_model, start=71), man)
    g = gates.anchoring(c, arms, ROOT)
    assert g["status"] == "DIAGNOSTIC" and g["n_runs"] == 4
    assert g["runs"][0]["llm_slots"] == [0]
    assert g["median_delta_p"] > 1.0 and g["anchors_above_monopoly"] is True
    assert "capped" in g["reading"]


def test_cli_gates_writes_report(pc_pass: Path, tr_pass: Path, reg, tmp_path: Path) -> None:
    arms = reg[0]
    croot = tmp_path / "campaigns"
    shutil.copytree(pc_pass, croot / "PC-H")
    shutil.copytree(tr_pass, croot / "TR-H")
    analyze = _load_analyze()
    out = tmp_path / "gates"
    rc = analyze.main(
        [
            "--gates",
            "--campaigns-root",
            str(croot),
            "--gate1",
            "PC-H,PC-M",
            "--gate1b",
            "PC-H-dealer",
            "--gate2",
            "TR-H",
            "--anchoring",
            "AC-H",
            "--arms",
            str(arms),
            "--root",
            str(ROOT),
            "--gates-out",
            str(out),
            "--B",
            "300",
        ]
    )
    assert rc == 0
    rep = json.loads((out / "gates.json").read_text(encoding="utf-8"))
    st = {k: v["status"] for k, v in rep["gates"].items()}
    assert st["Gate 1 (PC-H)"] == "PASS" and st["Gate 2 (TR-H)"] == "PASS"
    assert st["Gate 1 (PC-M)"] == "NOT RUN" and st["Gate 1b (PC-H-dealer)"] == "NOT RUN"
    assert st["AC-H anchoring diagnostic (AC-H)"] == "NOT RUN" and rep["all_evaluated_pass"]
    md = (out / "gates.md").read_text(encoding="utf-8")
    assert "Gate 2 (TR-H): **PASS**" in md and "established" in md and "NOT RUN" in md


def test_trend_rule_overrides_the_gate2_status_with_extend(
    tr_trend: Path, reg, tmp_path: Path
) -> None:
    import shutil

    arms = reg[0]
    croot = tmp_path / "campaigns"
    shutil.copytree(tr_trend, croot / "TR-H")
    rep = gates.evaluate_gates(croot, [], [], "TR-H", None, arms, ROOT)
    g = rep["gates"]["Gate 2 (TR-H)"]
    assert g["status"] == "EXTEND" and g["outcome"] is None
    assert g["outcome_if_no_extension"]["n_qualified"] == 4  # what ops would have read at 100
    assert "do not branch at 100" in g["decision"] and "91-120" in g["decision"]
    assert rep["exit_code"] == 2 and rep["all_evaluated_pass"] is False
    assert rep["decisions"]["Gate 2 (TR-H)"].startswith("DECISION Gate 2 (TR-H) [lane H]: EXTEND")
    md = gates.report_md(gates._clean(rep))
    assert "EXTEND" in md and "DECISION Gate 2" in md


# --------------------------------------------------------------------------- lanes


def test_lane_of_campaign_from_the_registry(reg) -> None:
    spec = reg[1]
    for camp, lane in (
        ("TR-H", "H"),
        ("TR-M", "M"),
        ("TR-S", "S"),
        ("PC-H", "H"),
        ("PC-M", "M"),
        ("PC-S", "S"),
        ("PC-H-dealer", "H"),
        ("AC-H", "H"),
        ("CA-M", "M"),
        ("TR-S-ext", "S"),
        ("TR-M-ext2", "M"),
    ):
        assert gates.lane_of_campaign(spec, camp) == lane, camp
    assert gates.lane_of_campaign(spec, "ZZ") == "H"
    assert gates.lane_of_campaign(spec, "ZZ", default=None) is None
    assert gates._names(None, ["a"]) == ["a"] and gates._names([], ["a"]) == []
    assert gates._names("none", ["a"]) == [] and gates._names("x, y", ["a"]) == ["x", "y"]


def s_trunk_manifest(trunk: str, seed: int) -> dict:
    return {
        "arm": "trunk",
        "label": "TR-S",
        "lane": "S",
        "trunk_id": trunk,
        "seed": seed,
        "branch_period": None,
        "status": "ok",
        "resolved_config": {"tags": {"arm": "trunk", "label": "TR-S"}, "market": TRH_MARKET},
    }


@pytest.fixture(scope="module")
def tr_s(tmp_path_factory: pytest.TempPathFactory, reg) -> Path:
    """An S-lane trunk campaign: 8 Sonnet trunks registered as ``trunk_S`` (label TR-S)."""
    model = reg[2].primary_model
    c = tmp_path_factory.mktemp("trs") / "TR-S"
    for k in range(8):
        write_frame_run(
            c,
            f"trunk-T{k}",
            market_frame(np.full((30, 4), 1.90 if k < 7 else 1.40), model, start=71),
            s_trunk_manifest(f"T{k}", k),
        )
    return c


def test_gate2_infers_the_lane_from_the_campaign(tr_s: Path, reg) -> None:
    arms = reg[0]
    g = gates.gate2(tr_s, arms, ROOT)  # no lane given: TR-S -> lane S -> trunk_S
    assert g["lane"] == "S" and g["arm"] == "trunk_S" and g["status"] == "PASS"
    assert g["n_trunks"] == 8 and g["n_qualified"] == 7
    assert gates.gate2(tr_s, arms, ROOT, lane="S")["n_trunks"] == 8
    # the H trunk arm sees no run in an S campaign: that was the "NOT RUN" of the bug
    assert gates.gate2(tr_s, arms, ROOT, lane="H")["status"] == "NOT RUN"
    with pytest.raises(ValueError):
        gates.gate2(tr_s, arms, ROOT, lane="Q")


def test_evaluate_gates_one_decision_per_gate_and_lane(
    tr_s: Path, tr_pass: Path, pc_pass: Path, reg, tmp_path: Path
) -> None:
    arms = reg[0]
    croot = tmp_path / "campaigns"
    shutil.copytree(tr_s, croot / "TR-S")
    shutil.copytree(tr_pass, croot / "TR-H")
    shutil.copytree(pc_pass, croot / "PC-S")
    rep = gates.evaluate_gates(croot, [], [], "TR-H,TR-S", None, arms, ROOT, B=300)
    assert set(rep["gates"]) == {"Gate 2 (TR-H)", "Gate 2 (TR-S)"}  # Gate 1 skipped, not defaulted
    assert (
        rep["gates"]["Gate 2 (TR-S)"]["lane"] == "S"
        and rep["gates"]["Gate 2 (TR-H)"]["lane"] == "H"
    )
    assert rep["decisions"]["Gate 2 (TR-S)"].startswith("DECISION Gate 2 (TR-S) [lane S]: PASS")
    assert rep["decisions"]["Gate 2 (TR-H)"].startswith("DECISION Gate 2 (TR-H) [lane H]: PASS")
    # explicit lane override and Gate 1 on the S lane
    rep = gates.evaluate_gates(
        croot, ["PC-S"], [], ["TR-S"], None, arms, ROOT, B=300, lanes={"TR-S": "S"}
    )
    assert (
        rep["gates"]["Gate 1 (PC-S)"]["lane"] == "S"
        and rep["gates"]["Gate 1 (PC-S)"]["status"] == "PASS"
    )
    assert rep["decisions"]["Gate 1 (PC-S)"].startswith("DECISION Gate 1 (PC-S) [lane S]: PASS")
    assert rep["gates"]["Gate 2 (TR-S)"]["status"] == "PASS" and rep["exit_code"] == 0
    # None keeps the defaults (missing -> NOT RUN), the string "none" skips
    rep = gates.evaluate_gates(croot, None, "none", "none", "none", arms, ROOT, B=300)
    assert set(rep["gates"]) == {"Gate 1 (PC-H)", "Gate 1 (PC-M)"}
    assert all(g["status"] == "NOT RUN" for g in rep["gates"].values())
    assert rep["gates"]["Gate 1 (PC-M)"]["lane"] == "M"


def test_cli_gate2_on_the_s_lane_without_rereading_gate1(tr_s: Path, reg, tmp_path: Path) -> None:
    arms = reg[0]
    croot = tmp_path / "campaigns"
    shutil.copytree(tr_s, croot / "TR-S")
    analyze = _load_analyze()
    out = tmp_path / "gates"
    argv = [
        "--gates",
        "--campaigns-root",
        str(croot),
        "--gate1",
        "none",
        "--gate1b",
        "none",
        "--anchoring",
        "none",
        "--arms",
        str(arms),
        "--root",
        str(ROOT),
        "--gates-out",
        str(out),
        "--B",
        "300",
    ]
    assert analyze.main([*argv, "--gate2", "TR-S"]) == 0
    rep = json.loads((out / "gates.json").read_text(encoding="utf-8"))
    assert list(rep["gates"]) == ["Gate 2 (TR-S)"]
    g = rep["gates"]["Gate 2 (TR-S)"]
    assert g["status"] == "PASS" and g["lane"] == "S" and g["arm"] == "trunk_S"
    assert "DECISION Gate 2 (TR-S) [lane S]: PASS" in (out / "gates.md").read_text(encoding="utf-8")
    assert analyze.main([*argv, "--gate2", "TR-S", "--gate2-lane", "S"]) == 0
    assert analyze.main([*argv, "--gate2", "TR-S", "--gate2-lane", "H"]) == 0
    rep = json.loads((out / "gates.json").read_text(encoding="utf-8"))
    assert rep["gates"]["Gate 2 (TR-S)"]["status"] == "NOT RUN"  # the H trunk arm has no run here
