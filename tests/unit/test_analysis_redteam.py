"""Red-team fixes before the stage-2 freeze (lead items B1, B2, m9, m11, m12 and the A-003 L2
rule): Holm-based family labels, the SAP s10 run filter, the L2 ask-fall rule, the MQC and
own-scoring guards, split-sample eligibility, window shift and override warnings."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from _arms_fixture import ROOT, temp_registry
from _metrics_synth import market_frame, write_frame_run

from finswarm.analysis import campaign as C
from finswarm.analysis import gates as G
from finswarm.analysis import stats as S
from finswarm.analysis.arms import load_registered_arms, resolve_arm_benchmarks
from finswarm.config.load import load_campaign_config

TRH_MARKET = load_campaign_config(ROOT / "configs/sprint/tier0/TR-H.yaml").base.model_dump(
    mode="json"
)["market"]
CA_LABEL = "price reduction consistent with competitive adjustment"
EXTEND_REPORT = {"gates": {"Gate 2 (TR-H)": {"status": "EXTEND", "trend_rule": {"shift": 20}}}}


def mqc_doc(sc1_l2: float | None) -> dict:
    l2 = {"status": "ok", "mean": 3.0, "bca_hi": 4.0}
    if sc1_l2 is not None:
        l2["sc1_price_response"] = sc1_l2
    return {
        "levers": {
            "L1": {"status": "ok", "mean": 1.0, "bca_hi": 1.5, "sc1_price_response": -0.01},
            "L2": l2,
        }
    }


def manifest(arm, label, trunk, seed, branch=True, status="ok", market=None) -> dict:
    m = {
        "arm": arm,
        "label": label,
        "trunk_id": trunk,
        "seed": seed,
        "status": status,
        "branch_period": 100 if branch else None,
        "resolved_config": {"tags": {"arm": arm, "label": label}},
    }
    if market is not None:
        m["resolved_config"]["market"] = market
    return m


@pytest.fixture(scope="module")
def reg(tmp_path_factory: pytest.TempPathFactory):
    arms = temp_registry(tmp_path_factory.mktemp("arms"))
    spec = load_registered_arms(arms)
    return arms, spec, resolve_arm_benchmarks(spec, ROOT)


def build(root: Path, reg, shift: int = 0) -> Path:
    """8 established trunks with sham, sham2 and L2 (cut 0.25 of p_M - p_N); plus a trunk
    with manifest status 'failed' (T8), a trunk with 25 of 30 periods (T9) and a sham with 35
    of 40 periods (T3). Windows start at 71 / 111 (+ ``shift``)."""
    arms, spec, ab = reg
    tm, l2m = ab["sham"].primary_model, ab["L2"].window_model
    pn, pm = ab["sham"].primary.p_nash, ab["sham"].primary.p_mono
    rng = np.random.default_rng(20261010)
    c = root / "TR-H"
    t0, b0 = 71 + shift, 111 + shift
    for k in range(10):
        t = f"T{k}"
        n_tr = 25 if k == 9 else 30
        tr = np.full((n_tr, 4), pn + 0.85 * (pm - pn)) + rng.normal(0, 0.01, (n_tr, 4))
        write_frame_run(
            c,
            f"trunk-{t}",
            market_frame(tr, tm, start=t0),
            manifest("trunk", "TR-H", t, k, branch=False, status="failed" if k == 8 else "ok"),
        )
        if k == 9:
            continue
        n_sh = 35 if k == 3 else 40
        sham = np.full((n_sh, 4), pn + 0.80 * (pm - pn)) + 0.02 * rng.normal(0, 1, (n_sh, 4))
        write_frame_run(
            c,
            f"sham-{t}",
            market_frame(sham, tm, start=b0),
            manifest("sham", "SH-H", t, k, market=TRH_MARKET),
        )
        sham40 = np.full((40, 4), pn + 0.80 * (pm - pn)) + 0.02 * rng.normal(0, 1, (40, 4))
        write_frame_run(
            c, f"sham2-{t}", market_frame(sham40, tm, start=b0), manifest("sham2", "SH2-H", t, k)
        )
        l2 = sham40 - 0.25 * (pm - pn) + 0.02 * rng.normal(0, 1, (40, 4))
        write_frame_run(c, f"L2-{t}", market_frame(l2, l2m, start=b0), manifest("L2", "L2-H", t, k))
    return c


@pytest.fixture(scope="module")
def camp(tmp_path_factory: pytest.TempPathFactory, reg) -> Path:
    return build(tmp_path_factory.mktemp("rt"), reg)


@pytest.fixture(scope="module")
def analysed(camp: Path, reg, tmp_path_factory: pytest.TempPathFactory) -> dict:
    out = tmp_path_factory.mktemp("an")
    res = C.analyze_campaign(camp, reg[0], ROOT, out, mqc=mqc_doc(-0.05), B=300, n_perm=0)
    res["_exclusions"] = json.loads((out / "TR-H" / "exclusions.json").read_text("utf-8"))
    return res


def test_b2_run_filter_status_and_complete_window(analysed: dict) -> None:
    ex = {e["run_id"]: e["reason"] for e in analysed["_exclusions"]}
    assert ex["trunk-T8"].startswith("manifest status 'failed'")
    assert ex["trunk-T9"].startswith("incomplete (SAP s10): 25 of 30")
    assert ex["sham-T3"].startswith("incomplete (SAP s10): 35 of 40")
    q = analysed["inference"]["qualification"]["by_lane"]["H"]
    assert q["n_qualified"] == 8 and "T8" not in q["qualified"] and "T9" not in q["qualified"]
    l2 = analysed["contrasts"]["L2"]
    assert l2["n_pairs"] == 7 and "T3" not in l2["trunks"]  # the dropped sham drops its pair
    assert analysed["contrasts"]["sham2"]["n_pairs"] == 7


def test_b1_family_labels_from_holm_and_bases(analysed: dict) -> None:
    lv = analysed["h1r"]["levers"]
    assert lv["L2"]["label_basis"].startswith("Holm")
    assert lv["L2"]["reject"] and lv["L2"]["own_scoring"]["reject"]
    assert lv["L1"]["placeholder"] and lv["L1"]["dual_benchmark_label"].startswith("not run")
    assert lv["L1"]["supported"] is False
    assert analysed["contrasts"]["L2"]["dual_benchmark"]["basis"].startswith("Holm")
    assert analysed["contrasts"]["sham2"]["dual_benchmark"]["basis"] == "raw p (single contrast)"
    assert S.label_from_decisions(True, True) == "collusion reduction"
    assert S.label_from_decisions(True, False) == CA_LABEL
    assert "unavailable" in S.label_from_decisions(True, None)
    assert S.label_from_decisions(False, True) == "reduction under the own scoring only"
    assert S.label_from_decisions(False, False) == "no significant reduction"


def test_a003_l2_rule_passes_with_a_small_sc1_fall(analysed: dict) -> None:
    h1 = analysed["h1"]
    rule = h1["a003_l2_rule"]
    assert rule["applies"] and rule["status"] == "ok" and rule["sc1_ask_fall"] == 0.05
    assert rule["llm_ask_fall"] == pytest.approx(0.25 * 0.7229, abs=0.03)  # 0.25 (p_M - p_N)
    assert rule["passes"] and rule["threshold"] == pytest.approx(0.10)
    assert h1["h1_supported"] and h1["dual_benchmark_label"] == "collusion reduction"
    assert h1["checks"]["a003_l2_passes"] and "a003" in h1["rule"]
    lv = analysed["h1r"]["levers"]["L2"]
    assert lv["a003_l2_rule"]["passes"] and lv["supported"]
    assert analysed["h1r"]["n_supported"] == 1
    c = analysed["contrasts"]["L2"]
    assert c["own_vs_trunk_max_abs_diff"] is not None and c["ask_fall"]["mean"] > 0.1
    assert "own_scoring_informative" in c and lv["own_vs_trunk_max_abs_diff"] is not None


def test_a003_l2_rule_fails_with_a_large_sc1_fall(camp: Path, reg, tmp_path) -> None:
    res = C.analyze_campaign(
        camp, reg[0], ROOT, tmp_path / "big", mqc=mqc_doc(-0.15), B=200, n_perm=0
    )
    h1 = res["h1"]
    assert h1["checks"]["p_lt_alpha"] and h1["a003_l2_rule"]["passes"] is False
    assert not h1["h1_supported"] and h1["dual_benchmark_label"] == CA_LABEL
    assert res["h1r"]["levers"]["L2"]["dual_benchmark_label"] == CA_LABEL
    assert res["h1r"]["levers"]["L2"]["supported"] is False and res["h1r"]["n_supported"] == 0
    assert res["contrasts"]["L2"]["dual_benchmark"]["label"] == CA_LABEL


def test_a003_l2_rule_not_evaluable_without_sc1(camp: Path, reg, tmp_path) -> None:
    res = C.analyze_campaign(
        camp, reg[0], ROOT, tmp_path / "none", mqc=mqc_doc(None), B=200, n_perm=0
    )
    h1 = res["h1"]
    assert h1["a003_l2_rule"]["passes"] is None and not h1["h1_supported"]
    assert h1["dual_benchmark_label"].startswith("not evaluable (A-003")


def test_m9_missing_primary_mqc_raises(camp: Path, reg, tmp_path) -> None:
    with pytest.raises(ValueError, match="MQC of the primary lever"):
        C.analyze_campaign(camp, reg[0], ROOT, tmp_path / "o", mqc={"L1": 1.0}, B=200, n_perm=0)


def test_m11_eligibility_ties_and_qualified_count() -> None:
    elig = C._mqc_eligible(
        {
            "levers": {
                "L1": {"status": "ok", "mean": 1.0, "bca_hi": 2.0},
                "L2": {"status": "ok", "mean": 1.0},  # no upper bound: ineligible
                "L3": {"status": "ok", "mean": 1.0, "bca_hi": 9.0},  # upper bound too high
            }
        },
        5.0,
        7.0,
    )
    assert elig == {"L1": 1.0, "L2": None, "L3": None}
    assert C._mqc_eligible({"L1": 1.0}, 5.0, 7.0) == {"L1": None}  # a bare point: no bound
    fam = ["L1", "L2", "L3"]
    eff = {"L1": 0.30, "L2": 0.30, "L3": 0.10}
    assert S.select_best(eff, fam, {"L1": 2.0, "L2": 1.0, "L3": 0.5}, 5.0) == "L2"  # lower MQC
    assert S.select_best(eff, fam, {"L1": 1.0, "L2": 2.0, "L3": 0.5}, 5.0) == "L1"
    assert S.select_best(eff, fam) == "L1"  # no MQC: ties keep the family order
    trunks = [f"T{k}" for k in range(8)]
    seeds = {t: k for k, t in enumerate(trunks)}
    diffs = {lev: {t: 0.3 for t in trunks} for lev in fam}
    off = S.split_sample_best(diffs, seeds, fam, B=100, min_total=8, n_qualified=6)
    assert off["status"].startswith("off: 6 qualified trunks")
    assert S.split_sample_best(diffs, seeds, fam, B=100, min_total=8)["status"] == "ok"


def test_m12_shift_windows_guard_and_arithmetic(reg, tmp_path) -> None:
    arms, spec, _ = reg
    gates = tmp_path / "gates.json"
    gates.write_text(json.dumps(EXTEND_REPORT), encoding="utf-8")
    with pytest.raises(ValueError, match="EXTEND"):
        C.shift_windows(spec, gates, 0)
    shifted, info = C.shift_windows(spec, gates, 20)
    assert shifted.window_of("trunk") == (91, 120) and shifted.window_of("sham") == (131, 170)
    assert shifted.windows["sens_late"] == (141, 170)
    assert shifted.windows["sens_full"] == (121, 170)
    assert shifted.windows["fish"] == tuple(spec.windows["fish"]) and info["required_shift"] == 20
    plain, info0 = C.shift_windows(spec, None, 0)
    assert plain.windows["sens_late"] == (121, 150) and info0["gate2_status"] is None


def test_m12_shifted_campaign_sensitivity_windows_and_warnings(reg, tmp_path) -> None:
    arms, spec, _ = reg
    gates = tmp_path / "gates.json"
    gates.write_text(json.dumps(EXTEND_REPORT), encoding="utf-8")
    camp = build(tmp_path / "ext", reg, shift=20)
    with pytest.raises(ValueError, match="EXTEND"):
        C.analyze_campaign(camp, arms, ROOT, tmp_path / "an2", mqc=mqc_doc(-0.05), gates_json=gates)
    res = C.analyze_campaign(
        camp,
        arms,
        ROOT,
        tmp_path / "an",
        mqc=mqc_doc(-0.05),
        B=200,
        n_perm=0,
        gates_json=gates,
        window_shift=20,
        seed=5,
    )
    assert res["windows"]["trunk"] == [91, 120] and res["windows"]["branch"] == [131, 170]
    assert res["window_shift"]["shift"] == 20 and res["window_shift"]["gate2_status"] == "EXTEND"
    sham = next(r for r in res["runs"] if r["run_id"] == "sham-T0")
    assert set(sham["delta_p_sens"]) == {"141-170", "121-170"}
    assert all(v is not None and v <= 1.0 for v in sham["delta_p_sens"].values())
    l2 = res["contrasts"]["L2"]
    assert l2["delta_p_sens_sens_late"]["window"] == "141-170"
    assert l2["delta_p_sens_sens_late"]["n"] == 7
    assert l2["delta_p_sens_sens_full"]["mean"] == pytest.approx(l2["delta_p"]["mean"], abs=0.05)
    assert res["inference"]["qualification"]["by_lane"]["H"]["n_qualified"] == 8
    assert any("non-registered arms file" in w for w in res["warnings"])
    assert any("seed override 5" in w for w in res["warnings"])


def test_b2_gate2_applies_the_same_filter(camp: Path, reg) -> None:
    g = G.gate2(camp, reg[0], ROOT)
    ex = {e["run_id"]: e["reason"] for e in g["excluded"]}
    assert ex["trunk-T8"].startswith("manifest status 'failed'")
    assert ex["trunk-T9"].startswith("incomplete (SAP s10)")
    assert g["n_trunks"] == 8 and g["n_qualified"] == 8 and g["status"] == "PASS"


def test_cli_window_shift_flag(camp: Path, reg, tmp_path) -> None:
    spec = importlib.util.spec_from_file_location("_an_rt", ROOT / "scripts" / "analyze.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mqc = tmp_path / "mqc.json"
    mqc.write_text(json.dumps(mqc_doc(-0.05)), encoding="utf-8")
    gates = tmp_path / "gates.json"
    gates.write_text(json.dumps(EXTEND_REPORT), encoding="utf-8")
    argv = [
        str(camp),
        "--arms",
        str(reg[0]),
        "--root",
        str(ROOT),
        "--out-root",
        str(tmp_path / "a"),
        "--mqc",
        str(mqc),
        "--B",
        "200",
        "--n-perm",
        "0",
        "--gates-json",
        str(gates),
    ]
    assert mod.main(argv) != 0  # EXTEND without --window-shift 20 is refused


def test_a022_reporting_fields_raw_tost_signs_and_sigma_label(analysed: dict) -> None:
    c = analysed["contrasts"]["L2"]
    raw = c["delta_p_raw"]
    assert {"mean", "bca_lo", "bca_hi", "n", "sd", "tost", "signs", "diffs"} <= set(raw)
    assert raw["tost"]["margin"] == 0.20 and "p" in raw["tost"]
    assert sum(raw["signs"].values()) == raw["n"] == c["n_pairs"]
    assert c["delta_p"]["signs"]["positive"] == c["n_pairs"]  # every trunk shows the cut
    assert C._sign_counts([0.0, -0.1, 0.2, 0.3]) == {"zero": 1, "negative": 1, "positive": 2}
    sig = analysed["inference"]["sigma_d"]
    assert "NOT a noise scale" in sig["label"] and "sham_vs_sham_sd_raw" in sig
