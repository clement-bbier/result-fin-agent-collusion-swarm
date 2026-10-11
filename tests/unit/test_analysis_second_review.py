"""Second red-team review in the analysis layer: capped index (D-16), dual-benchmark rule
(D-17), practical significance (D-19), split-sample guard (D-22)."""

import json
from pathlib import Path

import numpy as np
import pytest
from _arms_fixture import ROOT, temp_registry
from _metrics_synth import market_frame, write_frame_run

from finswarm.analysis import stats as S
from finswarm.analysis.arms import load_registered_arms, resolve_arm_benchmarks
from finswarm.analysis.campaign import analyze_campaign
from finswarm.config.load import load_campaign_config
from finswarm.metrics.collusion import capped, score_quotes

TRH_MARKET = load_campaign_config(ROOT / "configs/sprint/tier0/TR-H.yaml").base.model_dump(
    mode="json"
)["market"]


def _manifest(arm: str, label: str, trunk: str, seed: int, branch: bool = True) -> dict:
    m = {
        "arm": arm,
        "label": label,
        "trunk_id": trunk,
        "seed": seed,
        "branch_period": 100 if branch else None,
        "resolved_config": {"tags": {"arm": arm, "label": label}},
    }
    if arm == "sham":
        m["resolved_config"]["market"] = TRH_MARKET
    return m


@pytest.fixture(scope="module")
def reg(tmp_path_factory: pytest.TempPathFactory):
    arms = temp_registry(tmp_path_factory.mktemp("arms"))
    spec = load_registered_arms(arms)
    return arms, spec, resolve_arm_benchmarks(spec, ROOT)


@pytest.fixture(scope="module")
def anchored(tmp_path_factory: pytest.TempPathFactory, reg) -> Path:
    """Six qualified trunks (1.95) whose SHAM prices sit ABOVE monopoly (raw index > 1); L1 brings them down
    to just above p_M (raw still > 1), L2 brings them below p_M (a real reduction)."""
    _, _, ab = reg
    tm = ab["sham"].primary_model
    pm = ab["sham"].primary.p_mono
    c = tmp_path_factory.mktemp("camp") / "TR-H"
    rng = np.random.default_rng(1)
    for k in range(6):
        t = f"T{k}"
        write_frame_run(
            c,
            f"trunk-{t}",
            market_frame(np.full((30, 4), 1.95), tm, start=71),
            _manifest("trunk", "TR-H", t, k, branch=False),
        )
        sham = 2.60 + 0.02 * k + rng.normal(0, 0.01, (40, 4))  # ~2.65 > p_M 2.054
        write_frame_run(c, f"sham-{t}", market_frame(sham, tm), _manifest("sham", "SH-H", t, k))
        l1 = np.full((40, 4), pm + 0.10 + 0.01 * k)  # still above p_M: capped index stays 1
        write_frame_run(
            c, f"L1-{t}", market_frame(l1, ab["L1"].window_model), _manifest("L1", "L1-H", t, k)
        )
        l2 = sham - 0.90  # ~1.75: below p_M, a genuine fall of the capped index
        write_frame_run(
            c, f"L2-{t}", market_frame(l2, ab["L2"].window_model), _manifest("L2", "L2-H", t, k)
        )
    return c


MQC2 = {
    "levers": {
        "L1": {"status": "ok", "mean": 1.0, "bca_hi": 1.5, "sc1_price_response": -0.02},
        "L2": {"status": "ok", "mean": 2.0, "bca_hi": 3.0, "sc1_price_response": -0.05},
    }
}


def test_capped_index_and_raw_reported(anchored: Path, reg, tmp_path: Path) -> None:
    arms, spec, ab = reg
    res = analyze_campaign(anchored, arms, ROOT, tmp_path / "out", mqc=MQC2, B=300, n_perm=0)
    shams = [r for r in res["runs"] if r["registered_arm"] == "sham"]
    assert all(r["delta_p_raw"] > 1.0 and r["delta_p"] == 1.0 for r in shams)
    assert all(r["share_quotes_above_pm"] == 1.0 for r in shams)
    assert all(np.isfinite(r["trade_weighted_price"]) for r in shams)
    l1 = res["contrasts"]["L1"]
    # L1 only lowers prices from far above p_M to just above it: no capped reduction, a raw one
    assert l1["delta_p"]["mean"] == pytest.approx(0.0, abs=1e-12)
    assert l1["delta_p_raw"]["mean"] > 0.5
    assert l1["dual_benchmark"]["label"] == "no significant reduction"
    l2 = res["contrasts"]["L2"]
    assert l2["delta_p"]["mean"] > 0.3 and l2["delta_p"]["t"]["p"] < 0.01
    assert l2["dual_benchmark"]["label"] == "collusion reduction"
    assert l2["benchmark_shift"]["trunk"]["p_mono"] == pytest.approx(ab["sham"].primary.p_mono)
    assert "own" in l2["benchmark_shift"] and "d_p_nash" in l2["benchmark_shift"]
    h1 = res["h1"]
    assert (
        h1["lever"] == "L2"
        and h1["h1_supported"]
        and h1["dual_benchmark_label"] == "collusion reduction"
    )
    assert h1["own_scoring"]["available"] and h1["own_scoring"]["p_one_sided"] < 0.05
    assert h1["practical_significance"]["verdict"] == "practically significant"
    assert h1["rule"]["index"].startswith("capped")
    levers = res["h1r"]["levers"]
    assert levers["L1"]["dual_benchmark_label"] == "no significant reduction"
    assert levers["L2"]["own_scoring"]["reject"] is True and levers["L3"]["placeholder"]
    assert levers["L3"]["dual_benchmark_label"] == "not run (enters Holm at p = 1)"
    assert levers["L2"]["label_basis"].startswith("Holm") and levers["L2"]["supported"]
    assert levers["L2"]["a003_l2_rule"]["passes"] and res["h1r"]["n_supported"] == 1
    assert l2["dual_benchmark"]["basis"].startswith("Holm")
    # 6 qualified trunks: below the Gate 2 table's 8, the split estimate is off (D-25)
    best = res["h1r"]["best_lever_split_sample"]
    assert best["status"].startswith("off") and "fewer than 8" in best["status"]
    assert res["h1"]["gate2"]["reduced_power"] and res["h1"]["status"].startswith(
        "tested with reduced power"
    )
    assert res["inference"]["qualification"]["by_lane"]["H"]["n_qualified"] == 6
    md = (tmp_path / "out" / "TR-H" / "summary.md").read_text(encoding="utf-8")
    assert "dual-benchmark (D-17)" in md and "Practical significance (D-19)" in md


def test_score_quotes_caps_the_price_indices(reg) -> None:
    _, _, ab = reg
    model, tb = ab["sham"].primary_model, ab["sham"].primary
    high = np.full((5, 4), 3.0)  # far above monopoly
    s = score_quotes(high, model, tb, tb)
    assert s["delta_p"] == 1.0 and s["delta_p_window"] == 1.0 and s["delta_p_vw"] == 1.0
    assert capped(1.7) == 1.0 and capped(0.4) == 0.4 and np.isnan(capped(float("nan")))


def test_dual_benchmark_labels_and_practical_significance() -> None:
    assert S.dual_benchmark_label(0.01, 0.02) == "collusion reduction"
    assert (
        S.dual_benchmark_label(0.01, 0.02, mqc_ok=False)
        == "collusion reduction (MQC above the cap)"
    )
    assert (
        S.dual_benchmark_label(0.01, 0.30)
        == "price reduction consistent with competitive adjustment"
    )
    assert S.dual_benchmark_label(0.30, 0.01) == "reduction under the own scoring only"
    assert S.dual_benchmark_label(0.30, 0.40) == "no significant reduction"
    assert "unavailable" in S.dual_benchmark_label(0.01, None)
    assert S.dual_benchmark_label(float("nan"), None) == "no significant reduction"
    big = [0.40, 0.42, 0.38, 0.41, 0.39, 0.40, 0.43, 0.37]  # tight, far above 0.10
    assert S.practical_significance(big)["verdict"] == "practically significant"
    wide = [0.15, 0.30, 0.05, 0.25, 0.10, 0.20, 0.00, 0.35]  # 90 % interval straddles 0.10-0.20
    ps = S.practical_significance(wide)
    assert ps["verdict"] == "inconclusive"
    lo, hi = ps["t_interval_90"]
    assert lo <= 0.10 and hi >= 0.20 and ps["bca_90"][0] <= ps["bca_90"][1]
    small = [0.02, 0.03, 0.01, 0.02, 0.03, 0.02, 0.01, 0.02]
    assert S.practical_significance(small)["verdict"] == "below practical relevance"
    assert S.practical_significance([0.3])["verdict"] == "not applicable"


def test_primary_h1_test_dual_scoring_paths() -> None:
    d = [0.30, 0.35, 0.28, 0.32, 0.31, 0.29, 0.33, 0.30]
    flat = [0.00, 0.01, -0.01, 0.02, 0.00, -0.02, 0.01, 0.00]
    both = S.primary_h1_test(d, 2.0, B=300, d_own=d)
    assert both["h1_supported"] and both["dual_benchmark_label"] == "collusion reduction"
    assert both["h1_practically_significant"] and both["tost"] is None
    trunk_only = S.primary_h1_test(d, 2.0, B=300, d_own=flat)
    assert not trunk_only["h1_supported"]
    assert (
        trunk_only["dual_benchmark_label"]
        == "price reduction consistent with competitive adjustment"
    )
    assert trunk_only["tost"] is None  # one scoring is significant: no TOST
    neither = S.primary_h1_test(flat, 2.0, B=300, d_own=flat)
    assert not neither["h1_supported"] and neither["tost"] is not None
    assert neither["tost"]["equivalent"] is True
    mqc_bad = S.primary_h1_test(d, 9.0, B=300, d_own=d)
    assert not mqc_bad["h1_supported"] and "MQC above the cap" in mqc_bad["dual_benchmark_label"]
    legacy = S.primary_h1_test(d, 2.0, B=300)  # own scoring unavailable: never supported (m9)
    assert not legacy["h1_supported"] and "unavailable" in legacy["dual_benchmark_label"]
    assert legacy["own_scoring_available"] is False and legacy["practical_verdict"]
    assert legacy["own_scoring"] == {"available": False}


def test_split_sample_guard_and_eligibility() -> None:
    fam = ["L1", "L2", "L3", "L4a", "L5", "L6", "L7", "L8"]
    diffs = {"L1": {f"T{k}": 0.2 for k in range(6)}, "L2": {f"T{k}": 0.4 for k in range(6)}}
    seeds = {f"T{k}": k for k in range(6)}
    ok = S.split_sample_best(diffs, seeds, fam, B=200)
    assert ok["status"] == "ok" and ok["best_lever"] == "L2" and ok["estimate"]["n"] == 3
    # eligibility: L2 above the MQC cap -> L1 wins
    elig = S.split_sample_best(
        diffs, seeds, fam, B=200, mqc={"L1": 1.0, "L2": None}, mqc_max_pp=5.0
    )
    assert elig["best_lever"] == "L1"
    few = S.split_sample_best(diffs, {f"T{k}": k for k in range(4)}, fam, B=200)
    assert few["status"].startswith("off") and few["best_lever"] is None and few["estimate"] == {}


def test_mqc_sc1_price_response_in_document(tmp_path: Path, reg) -> None:
    from finswarm.analysis.mqc import analyze_mqc

    arms, spec, ab = reg
    tm = ab["sham"].primary_model
    c = tmp_path / "SC-1"
    for k in range(3):
        sham = np.full((40, 4), 1.40)
        lever = np.full((40, 4), 1.30)
        for arm, p in (("sham", sham), ("L2", lever)):
            d = write_frame_run(
                c, f"{arm}-S{k}", market_frame(p, tm), _manifest(arm, "MQC-L2", f"S{k}", k)
            )
            # a trades table: one unit per period at the posted price with a known buyer value
            import polars as pl

            from finswarm.store.schemas import TRADES

            per = np.arange(111, 151, dtype=np.int32)
            tr = {col: [None] * len(per) for col in TRADES}
            tr.update(
                run_id=[f"{arm}-S{k}"] * len(per),
                trade_id=np.arange(len(per)),
                tick=per.astype(np.int64) * 100,
                period=per - 1,  # store periods are 0-based (A-008)
                price=np.full(len(per), float(p[0, 0])),
                size=np.ones(len(per)),
                buyer_id=np.zeros(len(per), dtype=np.int32),
                seller_id=np.full(len(per), 100, dtype=np.int32),
                buyer_value=np.full(len(per), 2.0),
                seller_cost=np.ones(len(per)),
            )
            (d / "trades").mkdir(exist_ok=True)
            pl.DataFrame(tr, schema=TRADES).write_parquet(d / "trades" / "part-0000.parquet")
    doc = analyze_mqc(c, spec, B=200)
    l2 = doc["levers"]["L2"]
    assert l2["status"] == "ok" and l2["n"] == 3
    assert l2["sc1_price_response"] == pytest.approx(-0.10)
    assert l2["price_reference_mean"] == pytest.approx(1.40)
    json.dumps(doc)  # serialisable


def test_fallback_withdraws_every_lever_label(anchored: Path, reg, tmp_path: Path) -> None:
    import shutil

    arms, _, _ = reg
    four = tmp_path / "four" / "TR-H"
    for k in range(6):  # only 4 trunks qualify once T4 and T5 lose their trunk runs
        for a in ("trunk", "sham", "L1", "L2"):
            if a == "trunk" and k >= 4:
                continue
            shutil.copytree(anchored / f"{a}-T{k}", four / f"{a}-T{k}")
    res = analyze_campaign(four, arms, ROOT, tmp_path / "out", mqc=MQC2, B=200, n_perm=0)
    q = res["inference"]["qualification"]["by_lane"]["H"]
    assert q["n_qualified"] == 4 and q["outcome"]["h1_tested"] is False
    assert res["h1"]["status"].startswith("not tested") and res["h1"]["h1_supported"] is False
    assert res["h1"]["dual_benchmark_label"] == "not tested (Gate 2 fallback)"
    levers = res["h1r"]["levers"]
    assert all(
        v["dual_benchmark_label"].startswith("not tested (fallback") for v in levers.values()
    )
    assert not any(v["reject"] for v in levers.values())
    assert res["contrasts"]["L2"]["dual_benchmark"]["label"].startswith("not tested (fallback")
    assert res["h1r"]["status"].startswith("not tested")
    md = (tmp_path / "out" / "TR-H" / "summary.md").read_text(encoding="utf-8")
    assert "collusion reduction" not in md.replace("'collusion reduction'", "")
