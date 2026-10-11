"""Family S (D-20): threshold tests against the MEI, two one-sided tests, TOST at the MEI,
within-seed-group Spearman permutation, Welch s1 vs s4, cluster proportions, Holm m = 8."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from finswarm.analysis import family_s as FS

ROOT = Path(__file__).resolve().parents[2]


def _load_analyze():
    spec = importlib.util.spec_from_file_location(
        "_script_analyze_fs", ROOT / "scripts" / "analyze.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


BIG = [0.40, 0.42, 0.38, 0.41, 0.39, 0.40, 0.43, 0.37]
FLAT = [0.00, 0.01, -0.01, 0.02, 0.00, -0.02, 0.01, 0.00]
MID = [0.12, 0.10, 0.11, 0.09, 0.10, 0.12, 0.08, 0.11]


def test_threshold_tests_against_the_mei() -> None:
    assert FS.t_vs_mei(BIG, 0.15)["p"] < 1e-6
    assert FS.t_vs_mei(MID, 0.15)["p"] > 0.5  # mean 0.10 < 0.15: no evidence of >= MEI
    assert np.isnan(FS.t_vs_mei([0.3], 0.15)["p"])
    assert FS.two_one_sided_abs(BIG, 0.15)["p"] < 1e-6
    assert FS.two_one_sided_abs([-x for x in BIG], 0.15)["p"] < 1e-6  # either direction counts
    assert FS.two_one_sided_abs(FLAT, 0.15)["p"] > 0.9
    tost = FS.tost_mei(FLAT, 0.15)
    assert tost["equivalent"] is True and tost["mei"] == 0.15
    assert FS.tost_mei(BIG, 0.15)["equivalent"] is False


def test_spearman_permutation_within_seed_groups_and_welch() -> None:
    levels, values, groups = [], [], []
    for seed in (3001, 3002, 3003, 3004):
        for lev, v in zip((1, 2, 3, 4), (0.8, 0.6, 0.4, 0.1), strict=True):
            levels.append(lev)
            values.append(v + 0.01 * (seed % 7))
            groups.append(seed)
    sp = FS.spearman_permutation(levels, values, groups, n_perm=2000)
    assert sp["rho"] < -0.95 and sp["p_perm"] < 0.01  # tied level ranks cap |rho| below 1
    assert sp["groups"] == ["3001", "3002", "3003", "3004"]
    up = FS.spearman_permutation(levels, [-v for v in values], groups, n_perm=500)
    assert up["rho"] > 0.95 and up["p_perm"] > 0.99
    we = FS.welch_one_sided([0.80, 0.82, 0.78, 0.81], [0.10, 0.12, 0.08, 0.11], 0.30)
    assert we["diff"] == pytest.approx(0.70) and we["p"] < 1e-6
    weak = FS.welch_one_sided([0.40, 0.42, 0.38, 0.41], [0.10, 0.12, 0.08, 0.11], 0.30)
    assert weak["p"] >= 0.5 - 1e-9  # a difference of exactly 0.30 cannot beat H0 mu <= 0.30
    assert np.isnan(FS.welch_one_sided([0.5], [0.1, 0.2], 0.3)["p"])


def test_proportion_difference_and_sign_test() -> None:
    coll = [(8, 10), (9, 10), (7, 10), (8, 10), (9, 10), (8, 10)]
    comp = [(1, 10), (0, 10), (2, 10), (1, 10), (1, 10), (0, 10), (1, 10), (2, 10)]
    pd = FS.proportion_difference(coll, comp, 0.30)
    assert pd["diff"] == pytest.approx(49 / 60 - 8 / 80) and pd["p"] < 1e-3  # t, K-1 df (A-010)
    same = FS.proportion_difference(coll, coll, 0.30)
    assert same["diff"] == 0.0 and same["p"] > 0.99
    assert FS.sign_test(7, 8)["p"] < 0.05 and FS.sign_test(4, 8)["p"] > 0.3
    assert np.isnan(FS.sign_test(0, 0)["p"])


def _contrasts(**diffs):
    out = {}
    for arm, d in diffs.items():
        out[arm] = {
            "delta_p": {"diffs": {f"T{i}": v for i, v in enumerate(d)}, "mean": float(np.mean(d))}
        }
    return out


def test_family_s_holm_statuses_and_co_conditions() -> None:
    contrasts = _contrasts(L3=BIG, L5=FLAT, L4a=BIG, R1=FLAT, R3=MID, L9=FLAT)
    gr = [
        {"arm": f"GR-s{k}", "seed": s, "delta_p": v + 0.01 * (s % 5)}
        for s in (3001, 3002, 3003, 3004)
        for k, v in zip((1, 2, 3, 4), (0.8, 0.6, 0.4, 0.1), strict=True)
    ]
    res = FS.family_s(
        contrasts,
        gr,
        trunk_delta_p=[0.8] * 12,
        d5net={"sham": [0.20] * 8, "L4a": [0.05] * 8, "L9": [0.19] * 8},
        recovery={"R1": [5, 8, 12, 30, 7, 9, 10, 6]},
        sting={"collusive": [(8, 10)] * 6, "competitive": [(1, 10)] * 8, "punished_share": 0.67},
        n_perm=500,
    )
    h = res["hypotheses"]
    assert res["m"] == 8 and set(h) == set(FS.FAMILY)
    assert h["H2"]["status"] == "tested" and h["H2"]["p"] == max(
        h["H2"]["spearman"]["p_perm"], h["H2"]["welch_s1_minus_s4"]["p"]
    )
    assert h["H4"]["reject"] and h["H4"]["test"]["mei"] == 0.15
    assert h["H5"]["status"] == "tested" and not h["H5"]["reject"]  # L5 flat: |mu| <= 0.15
    assert h["H6-carrier"]["reject"] and h["H6-carrier"]["co_condition_dependence_fall_ge_50pct"]
    assert h["H6-carrier"]["d5net_change"] == pytest.approx(-0.75)
    assert (
        h["H7-same"]["test"]["equivalent"] and h["H7-same"]["recovery_sign_test"]["successes"] == 7
    )
    assert not h["H7-other"]["reject"]  # mean 0.10 < MEI 0.15
    assert h["H8"]["reject"] and h["H8"]["co_condition_punished_share_ge_50pct"] is True
    assert (
        h["H10"]["test"]["equivalent"] and h["H10"]["co_condition_dependence_within_25pct"] is True
    )
    assert all(not v["placeholder"] for v in h.values())
    assert res["n_rejected"] >= 4
    # H7-same enters Holm with max(p_TOST, p_sign): 7 of 8 recoveries <= 20 -> 9 / 256
    assert h["H7-same"]["p"] == pytest.approx(9 / 256)
    assert res["n_supported"] <= res["n_rejected"]
    # Holm never rejects more than the raw tests do
    assert all(v["p_holm"] >= v["p"] - 1e-12 for v in h.values() if np.isfinite(v["p"]))


def test_family_s_missing_levers_and_fallback() -> None:
    res = FS.family_s(_contrasts(L3=BIG), [], [], n_perm=50)
    h = res["hypotheses"]
    assert h["H2"]["status"].startswith("not testable") and h["H2"]["p"] == 1.0
    assert h["H5"]["placeholder"] and h["H5"]["p_holm"] == 1.0
    assert h["H4"]["status"] == "tested" and h["H4"]["p_holm"] == pytest.approx(
        min(1.0, 8 * h["H4"]["p"])
    )
    fb = FS.family_s(_contrasts(L3=BIG, L5=BIG), [], [], fallback=True, n_perm=50)
    assert all(fb["hypotheses"][k]["status"].startswith("not testable") for k in FS.LEVER_OF)
    assert fb["hypotheses"]["H8"]["status"].startswith("not testable")
    assert fb["n_rejected"] == 0


def test_family_s_from_stats_and_cli(tmp_path: Path) -> None:
    runs = [
        {
            "run_id": f"tr{k}",
            "registered_arm": "trunk",
            "trunk_id": f"T{k}",
            "seed": k,
            "delta_p": 0.8,
        }
        for k in range(12)
    ]
    runs += [
        {
            "run_id": f"sh{k}",
            "registered_arm": "sham",
            "trunk_id": f"T{k}",
            "seed": k,
            "delta_p": 0.8,
        }
        for k in range(8)
    ]
    runs += [
        {
            "run_id": f"l4{k}",
            "registered_arm": "L4a",
            "trunk_id": f"T{k}",
            "seed": k,
            "delta_p": 0.4,
        }
        for k in range(8)
    ]
    st = {
        "runs": runs,
        "contrasts": _contrasts(L3=BIG, L4a=BIG, L9=FLAT),
        "inference": {
            "qualification": {
                "by_lane": {
                    "H": {
                        "available": True,
                        "qualified": [f"T{k}" for k in range(8)],
                        "outcome": {"h1_tested": True},
                    }
                }
            }
        },
    }
    (tmp_path / "stats.json").write_text(json.dumps(st), encoding="utf-8")
    gs = {
        "runs": [
            {"run_id": f"g{s}{k}", "registered_arm": f"GR-s{k}", "seed": s, "delta_p": v}
            for s in (3001, 3002, 3003, 3004)
            for k, v in zip((1, 2, 3, 4), (0.8, 0.6, 0.4, 0.1), strict=True)
        ]
    }
    (tmp_path / "gr.json").write_text(json.dumps(gs), encoding="utf-8")
    import polars as pl

    corpus = pl.DataFrame(
        {
            "run_id": [f"sh{k}" for k in range(8)] + [f"l4{k}" for k in range(8)],
            "D5net": [0.2] * 8 + [0.05] * 8,
        }
    )
    corpus.write_parquet(tmp_path / "corpus.parquet")
    res = FS.family_s_from_stats(
        tmp_path / "stats.json", tmp_path / "gr.json", tmp_path / "corpus.parquet", n_perm=200
    )
    h = res["hypotheses"]
    assert h["H2"]["status"] == "tested" and h["H2"]["n_pooled_s1"] == 16
    assert h["H6-carrier"]["co_condition_dependence_fall_ge_50pct"] is True
    assert h["H8"]["status"].startswith("not testable") and res["inputs"]["n_qualified"] == 8
    analyze = _load_analyze()
    rc = analyze.main(
        [
            "--family-s",
            "--stats",
            str(tmp_path / "stats.json"),
            "--gradient-stats",
            str(tmp_path / "gr.json"),
            "--out-root",
            str(tmp_path / "out"),
            "--h2-perm",
            "100",
        ]
    )
    assert rc == 0
    doc = json.loads((tmp_path / "out" / "family_s" / "family_s.json").read_text(encoding="utf-8"))
    assert doc["m"] == 8 and "H2" in doc["hypotheses"]
    assert "Family S" in (tmp_path / "out" / "family_s" / "family_s.md").read_text(encoding="utf-8")


# --------------------------------------------------------------------------- B3: H6 guard

D5 = {"sham": [0.20] * 8, "L4a": [0.05] * 8, "L9": [0.19] * 8}
LOW_NULL = list(np.linspace(0.0, 0.10, 101))  # 95th percentile 0.095 < sham D5-net 0.20
HIGH_NULL = list(np.linspace(0.0, 0.40, 101))  # 95th percentile 0.38 > sham D5-net 0.20


def test_dependence_guard_known_quantiles() -> None:
    g = FS.dependence_guard([0.20] * 8, LOW_NULL)
    assert g["passed"] is True and g["n_null_draws"] == 101
    assert g["null_q"] == pytest.approx(0.095) and g["sham_d5net"] == pytest.approx(0.20)
    assert FS.dependence_guard([0.20] * 8, HIGH_NULL)["passed"] is False
    assert FS.dependence_guard([0.20] * 8, 0.10)["passed"] is True  # the quantile itself
    assert FS.dependence_guard([0.20] * 8, 0.20)["passed"] is False  # strictly above
    assert FS.dependence_guard([0.20] * 8, None)["passed"] is None
    assert FS.dependence_guard([], LOW_NULL)["passed"] is None
    assert FS.dependence_guard([0.20] * 8, [])["passed"] is None


def test_h6_guard_gates_the_test_and_supported() -> None:
    contrasts = _contrasts(L4a=BIG)
    ok = FS.family_s(contrasts, d5net=D5, sham_d5net_ghost_null=LOW_NULL, n_perm=50)
    h6 = ok["hypotheses"]["H6-carrier"]
    assert h6["status"] == "tested" and h6["reject"] and h6["supported"]
    assert h6["co_conditions"] == {
        "sham_d5net_above_ghost_null_q95": True,
        "dependence_fall_ge_50pct": True,
    }
    # no dependence to remove in the sham: p = 1, no decision, the Delta-C_p test reported
    res = FS.family_s(contrasts, d5net=D5, sham_d5net_ghost_null=HIGH_NULL, n_perm=50)
    h6 = res["hypotheses"]["H6-carrier"]
    assert h6["status"] == "not testable: no dependence to remove in the sham"
    assert h6["p"] == 1.0 and h6["p_holm"] == 1.0 and h6["placeholder"]
    assert not h6["reject"] and not h6["supported"]
    assert h6["test"]["p"] < 1e-6 and h6["co_condition_dependence_fall_ge_50pct"] is None
    assert res["n_rejected"] == 0 and res["n_supported"] == 0
    # the quantile given directly behaves like the draws
    q = FS.family_s(contrasts, d5net=D5, sham_d5net_ghost_null=0.38, n_perm=50)
    assert q["hypotheses"]["H6-carrier"]["status"] == FS.H6_NOT_TESTABLE
    # D5-net falls by only 25 %: Holm rejects, the co-condition fails, not supported
    weak = FS.family_s(
        contrasts,
        d5net={"sham": [0.20] * 8, "L4a": [0.15] * 8},
        sham_d5net_ghost_null=LOW_NULL,
        n_perm=50,
    )["hypotheses"]["H6-carrier"]
    assert weak["reject"] and weak["d5net_change"] == pytest.approx(-0.25)
    assert weak["co_conditions"]["dependence_fall_ge_50pct"] is False and not weak["supported"]
    # no null input (backward compatible): the t-test decides reject, the guard is not
    # evaluated and H6 is not supported without it
    legacy = FS.family_s(contrasts, d5net=D5, n_perm=50)["hypotheses"]["H6-carrier"]
    assert legacy["reject"] and legacy["status"].startswith("tested; dependence guard not")
    assert legacy["co_conditions"]["sham_d5net_above_ghost_null_q95"] is None
    assert not legacy["supported"]


def test_n_supported_counts_co_conditions_not_just_rejections() -> None:
    res = FS.family_s(
        _contrasts(L3=BIG, L4a=BIG, L9=FLAT),
        d5net={"sham": [0.20] * 8, "L4a": [0.05] * 8, "L9": [0.40] * 8},  # L9 doubles D5-net
        sham_d5net_ghost_null=LOW_NULL,
        sting={"collusive": [(8, 10)] * 6, "competitive": [(1, 10)] * 8, "punished_share": 0.30},
        n_perm=50,
    )
    h = res["hypotheses"]
    assert h["H4"]["supported"] and h["H6-carrier"]["supported"]
    assert h["H8"]["reject"] and not h["H8"]["supported"]  # punished share 30 % < 50 %
    assert h["H8"]["co_conditions"] == {"punished_share_ge_50pct": False}
    assert h["H10"]["reject"] and not h["H10"]["supported"]  # D5-net +100 %, outside 25 %
    assert h["H10"]["co_conditions"] == {"dependence_within_25pct": False}
    assert res["n_rejected"] == 4 and res["n_supported"] == 2
    assert res["summary"].startswith("2 of 8 supported")
    md = FS.family_s_md(res)
    assert "| supported |" in md
    assert "Supported (Holm reject and every co-condition met): 2 of 8" in md
    assert "punished_share_ge_50pct: no" in md


# --------------------------------------------------------------------------- M4: H7 recovery


def _r1_run(pre_level: float, path: list[float], t0: int = 100) -> dict[int, float]:
    """20 baseline periods at ``pre_level`` (t0 - 20 .. t0 - 1), then ``path`` from t0."""
    s = {t: pre_level for t in range(t0 - 20, t0)}
    s.update({t0 + i: v for i, v in enumerate(path)})
    return s


DIP = [0.2, 0.4, 0.6, 0.75, 0.8] + [0.8] * 45  # back within 0.10 of 0.80 at period 103


def test_recovery_periods_known_answers() -> None:
    s = _r1_run(0.8, DIP)
    assert FS.recovery_periods(s, 100) == 3.0
    arr = [0.0] * 80 + [0.8] * 20 + DIP  # array form: element i is period i
    assert FS.recovery_periods(arr, 100) == 3.0
    assert FS.recovery_periods(arr[50:], 100, start=50) == 3.0
    assert FS.recovery_periods({str(k): v for k, v in s.items()}, 100) == 3.0  # JSON keys
    # the baseline is the mean of periods 80-99 only (periods 0-79 at 0 are ignored)
    assert FS.recovery_periods({t: 0.0 for t in range(80)} | s, 100) == 3.0
    alt = {t: (0.7 if t % 2 else 0.9) for t in range(80, 100)}
    alt.update({100 + i: v for i, v in enumerate(DIP)})
    assert FS.recovery_periods(alt, 100) == 3.0  # baseline mean 0.80
    assert FS.recovery_periods(_r1_run(0.8, [0.8] * 50), 100) == 0.0  # never left the band
    assert FS.recovery_periods(_r1_run(0.8, [0.7] * 50), 100) == 0.0  # 0.10 is inclusive
    assert np.isnan(FS.recovery_periods(_r1_run(0.8, [0.3] * 50), 100))  # right-censored
    assert FS.recovery_periods(s, 100, tol=0.25) == 2.0  # |0.6 - 0.8| = 0.2 <= 0.25
    assert FS.recovery_periods(s, 100, pre=5) == 3.0
    with pytest.raises(ValueError, match="not all observed"):
        FS.recovery_periods({t: 0.8 for t in range(85, 100)} | {100: 0.8}, 100)


def test_h7_same_is_max_of_tost_and_recovery_sign_test() -> None:
    contrasts = _contrasts(R1=FLAT)
    # series in registered coordinates: the replacement plays from T0 = 101 (A-008)
    fast = _r1_run(0.8, [0.2, 0.5, 0.75] + [0.8] * 47, t0=FS.T0)  # recovers in 2 periods
    slow = _r1_run(0.8, [0.2] * 30 + [0.8] * 20, t0=FS.T0)  # recovers in 30 periods (> 20)
    series = {f"r{k}": fast for k in range(7)} | {"r7": slow}
    h7 = FS.family_s(contrasts, r1_series=series, n_perm=50)["hypotheses"]["H7-same"]
    st = h7["recovery_sign_test"]
    assert st["successes"] == 7 and st["n"] == 8 and st["p"] == pytest.approx(9 / 256)
    assert h7["p_tost"] == h7["test"]["p"] < 1e-4
    assert h7["p"] == pytest.approx(max(h7["p_tost"], st["p"]))
    assert h7["p"] == pytest.approx(9 / 256)
    assert h7["recovery_periods"] == [2.0] * 7 + [30.0]
    assert "max(p_TOST, p_sign)" in h7["rule"] and h7["status"] == "tested"
    assert h7["co_conditions"] == {"recovery_sign_test_computed": True}
    # Holm m = 8 multiplies the smallest p by 8: 8 x 9 / 256 > 0.05, no rejection
    assert h7["p_holm"] == pytest.approx(8 * 9 / 256) and not h7["reject"]
    allfast = {f"r{k}": fast for k in range(8)}  # p_sign = 1 / 256, Holm 8 / 256 < 0.05
    h7a = FS.family_s(contrasts, r1_series=allfast, n_perm=50)["hypotheses"]["H7-same"]
    assert h7a["p"] == pytest.approx(1 / 256) and h7a["reject"] and h7a["supported"]
    # half the runs recover late: TOST equivalence alone no longer carries H7-same
    half = {f"r{k}": fast for k in range(4)} | {f"s{k}": slow for k in range(4)}
    h7h = FS.family_s(contrasts, r1_series=half, n_perm=50)["hypotheses"]["H7-same"]
    assert h7h["test"]["equivalent"] and h7h["p"] == pytest.approx(163 / 256)
    assert not h7h["reject"] and not h7h["supported"]
    # the default replacement period follows the window shift (T0 + 20 = 121 under EXTEND)
    shifted = {k: {t + 20: v for t, v in s.items()} for k, s in series.items()}
    h7s = FS.family_s(contrasts, r1_series=shifted, shift=20, n_perm=50)["hypotheses"]["H7-same"]
    assert h7s["recovery_sign_test"]["successes"] == 7
    assert h7s["recovery_runs"][0]["replacement_period"] == 121
    per_run = FS.family_s(
        contrasts, r1_series=shifted, r1_replacement_period=dict.fromkeys(shifted, 121), n_perm=50
    )["hypotheses"]["H7-same"]
    assert per_run["recovery_sign_test"]["successes"] == 7
    # no recovery input: the TOST p alone, flagged, and not supported
    alone = FS.family_s(contrasts, n_perm=50)["hypotheses"]["H7-same"]
    assert alone["p"] == alone["test"]["p"]
    assert alone["status"] == "tested; sign test not computed: no R1 recovery input"
    assert alone["reject"] and not alone["supported"]
    with pytest.raises(ValueError, match="not both"):
        FS.family_s(contrasts, recovery={"R1": [5.0]}, r1_series=series, n_perm=50)
    with pytest.raises(ValueError, match="undecidable"):
        FS.family_s(contrasts, r1_series={"r0": _r1_run(0.8, [0.2] * 10)}, n_perm=50)
    # precomputed recovery periods: a censored run (NaN) counts as a failure, never dropped
    rec = {"R1": [5, 8, 12, float("nan"), 7, 9, 10, 6]}
    pre = FS.family_s(contrasts, recovery=rec, n_perm=50)["hypotheses"]["H7-same"]
    assert pre["recovery_sign_test"]["n"] == 8 and pre["recovery_sign_test"]["successes"] == 7
    assert pre["n_censored"] == 1


# --------------------------------------------------------------------------- m12: windows, H2


def _gates(status: str, lane: str = "H", label: str = "Gate 2 (TR-H)") -> dict:
    return {
        "gates": {
            label: {
                "campaign": label[8:-1],
                "lane": lane,
                "status": status,
                "trend_rule": {"shift": 20, "extend_t0_to_120": status == "EXTEND"},
            }
        }
    }


def test_windows_registered_sensitivities_and_extend(tmp_path: Path) -> None:
    assert FS.family_s_windows() == {
        "primary": (111, 150),
        "sensitivity_30": (121, 150),
        "sensitivity_50": (101, 150),
    }
    assert FS.family_s_windows(20) == {
        "primary": (131, 170),
        "sensitivity_30": (141, 170),
        "sensitivity_50": (121, 170),
    }
    res = FS.family_s(_contrasts(L3=BIG), n_perm=50)
    assert res["windows"] == {
        "primary": [111, 150],
        "sensitivity_30": [121, 150],
        "sensitivity_50": [101, 150],
    }
    assert "primary 111-150; sensitivity 121-150 and 101-150" in FS.family_s_md(res)
    ext = _gates("EXTEND")
    with pytest.raises(ValueError, match="EXTEND"):
        FS.family_s(_contrasts(L3=BIG), gates=ext, n_perm=50)
    with pytest.raises(ValueError, match="EXTEND"):
        FS.family_s(_contrasts(L3=BIG), gates=ext, windows=FS.BRANCH_WINDOWS, n_perm=50)
    ok = FS.family_s(_contrasts(L3=BIG), gates=ext, shift=20, n_perm=50)
    assert ok["windows"]["primary"] == [131, 170] and ok["windows"]["sensitivity_50"] == [121, 170]
    assert ok["window_info"]["shift"] == 20 and ok["r1_replacement_period_default"] == 121
    pre = FS.family_s(_contrasts(L3=BIG), gates=ext, windows=FS.family_s_windows(20), n_perm=50)
    assert pre["windows"]["primary"] == [131, 170] and pre["window_info"]["shift"] == 20
    with pytest.raises(ValueError, match="do not match"):
        FS.family_s(_contrasts(L3=BIG), gates=_gates("PASS"), shift=20, n_perm=50)
    with pytest.raises(ValueError, match="do not match"):  # double shift
        FS.family_s(
            _contrasts(L3=BIG), gates=ext, windows=FS.family_s_windows(20), shift=20, n_perm=50
        )
    # legacy key, another lane's EXTEND, a gates.json path
    assert FS.gate2_trend_shift({"gates": {"gate2": ext["gates"]["Gate 2 (TR-H)"]}})["shift"] == 20
    s_only = _gates("EXTEND", lane="S", label="Gate 2 (TR-S)")
    assert FS.gate2_trend_shift(s_only)["extend"] is False
    assert FS.family_s(_contrasts(L3=BIG), gates=s_only, n_perm=50)["windows"]["primary"] == [
        111,
        150,
    ]
    (tmp_path / "gates.json").write_text(json.dumps(ext), encoding="utf-8")
    with pytest.raises(ValueError, match="EXTEND"):
        FS.family_s(_contrasts(L3=BIG), gates=tmp_path / "gates.json", n_perm=50)
    with pytest.raises(ValueError, match="exactly the keys"):
        FS.family_s(_contrasts(L3=BIG), windows={"primary": (111, 150)}, n_perm=50)


def test_h2_missing_seed_raises_and_components() -> None:
    gr = [
        {"arm": f"GR-s{k}", "seed": s, "delta_p": v}
        for s in (3001, 3002)
        for k, v in zip((1, 2, 3, 4), (0.8, 0.6, 0.4, 0.1), strict=True)
    ]
    with pytest.raises(ValueError, match="without a seed"):
        FS.family_s({}, [*gr, {"arm": "GR-s2", "seed": None, "delta_p": 0.5}], n_perm=50)
    with pytest.raises(ValueError, match="without a seed"):
        FS.family_s({}, [*gr, {"arm": "GR-s3", "delta_p": 0.5}], n_perm=50)
    with pytest.raises(ValueError, match="without a seed group"):
        FS.spearman_permutation(
            [1, 2, 3, 4, 1], [0.8, 0.6, 0.4, 0.1, 0.7], [1, 1, 1, 1, float("nan")], n_perm=10
        )
    with pytest.raises(ValueError, match="differ in length"):
        FS.spearman_permutation([1, 2, 3, 4], [0.8, 0.6, 0.4, 0.1], [1, 1, 1], n_perm=10)
    # a non-finite value is left out and counted, with the incomplete group reported
    sp = FS.spearman_permutation(
        [1, 2, 3, 4, 1, 2, 3, 4],
        [0.8, 0.6, 0.4, 0.1, 0.8, np.nan, 0.4, 0.1],
        ["a"] * 4 + ["b"] * 4,
        n_perm=50,
    )
    assert sp["n"] == 7 and sp["n_dropped_nonfinite"] == 1 and sp["dropped_groups"] == ["b"]
    assert sp["incomplete_groups"] == {"b": [1.0, 3.0, 4.0]}
    # a run without Delta_p is listed; a missing Welch component makes H2 untestable (IUT)
    h2 = FS.family_s({}, [*gr, {"arm": "GR-s4", "seed": 3003, "delta_p": None}], n_perm=50)
    assert h2["hypotheses"]["H2"]["excluded"] == [
        {"arm": "GR-s4", "seed": 3003, "reason": "no Delta_p"}
    ]
    low = [r for r in gr if r["arm"] in ("GR-s1", "GR-s2")]
    h2 = FS.family_s({}, low, n_perm=50)["hypotheses"]["H2"]
    assert np.isfinite(h2["spearman"]["p_perm"]) and np.isnan(h2["welch_s1_minus_s4"]["p"])
    assert h2["p"] == 1.0 and h2["status"].startswith("not testable") and h2["placeholder"]


def _write_ap(d: Path, periods: range, prices: float | list[float]) -> None:
    import polars as pl

    d.mkdir(parents=True, exist_ok=True)
    rows: dict[str, list] = {"period": [], "slot": [], "price": []}
    for i, t in enumerate(periods):  # ``periods`` are STORE periods (0-based, A-008)
        p = prices if isinstance(prices, float) else prices[i]
        for s in range(4):
            rows["period"].append(t)
            rows["slot"].append(s)
            rows["price"].append(p)
    pl.DataFrame(rows).write_parquet(d / "agent_period.parquet")


def test_family_s_from_stats_r1_recovery_guard_and_windows(tmp_path: Path) -> None:
    import polars as pl

    runs = []
    for k in range(8):
        tdir, rdir = tmp_path / "camp" / f"tr{k}", tmp_path / "camp" / f"r1{k}"
        _write_ap(tdir, range(0, 100), 1.8)  # Delta_p = 0.8 on p_N = 1, p_M = 2
        post = [1.2, 1.5, 1.75] + [1.8] * 47 if k < 7 else [1.2] * 30 + [1.8] * 20
        _write_ap(rdir, range(100, 150), post)
        base = {"trunk_id": f"T{k}", "seed": k, "delta_p": 0.8}
        runs.append({"run_id": f"tr{k}", "registered_arm": "trunk", "path": str(tdir), **base})
        runs.append(
            {
                "run_id": f"r1{k}",
                "registered_arm": "R1",
                "path": str(rdir),
                "branch_period": 100,
                "members": [0, 1, 2, 3],
                **base,
            }
        )
        runs.append({"run_id": f"sh{k}", "registered_arm": "sham", **base})
        runs.append({"run_id": f"l4{k}", "registered_arm": "L4a", **base})
    runs.append(
        {
            "run_id": "r1x",
            "registered_arm": "R1",
            "trunk_id": "T8",
            "seed": 8,
            "path": str(tmp_path / "missing"),
            "branch_period": 100,
        }
    )
    st = {
        "runs": runs,
        "windows": {"branch": [111, 150], "trunk": [71, 100]},
        "contrasts": _contrasts(R1=FLAT, L4a=BIG),
        "benchmarks": {"R1": {"primary": {"p_nash": 1.0, "p_mono": 2.0}}},
        "inference": {
            "qualification": {
                "by_lane": {
                    "H": {
                        "available": True,
                        "qualified": [f"T{k}" for k in range(9)],
                        "outcome": {"h1_tested": True},
                    }
                }
            }
        },
    }
    (tmp_path / "stats.json").write_text(json.dumps(st), encoding="utf-8")
    pl.DataFrame(
        {
            "run_id": [f"sh{k}" for k in range(8)] + [f"l4{k}" for k in range(8)],
            "D5net": [0.2] * 8 + [0.05] * 8,
        }
    ).write_parquet(tmp_path / "corpus.parquet")
    (tmp_path / "null.json").write_text(json.dumps({"q95": 0.5}), encoding="utf-8")
    res = FS.family_s_from_stats(
        tmp_path / "stats.json",
        corpus_path=tmp_path / "corpus.parquet",
        n_perm=50,
        ghost_null_path=tmp_path / "null.json",
    )
    h7 = res["hypotheses"]["H7-same"]
    assert res["inputs"]["r1_series"] == "R1 run directories"
    assert h7["recovery_sign_test"]["successes"] == 7 and h7["recovery_sign_test"]["n"] == 8
    assert h7["p"] == pytest.approx(9 / 256)
    assert sorted(r["recovery_periods"] for r in h7["recovery_runs"]) == [2.0] * 7 + [30.0]
    assert h7["recovery_excluded"] == [
        {"run_id": "r1x", "trunk_id": "T8", "reason": "R1 run directory missing"}
    ]
    h6 = res["hypotheses"]["H6-carrier"]  # sham D5-net 0.2 <= ghost-null q95 0.5
    assert h6["status"] == FS.H6_NOT_TESTABLE and h6["dependence_guard"]["null_q"] == 0.5
    assert h6["p"] == 1.0 and not h6["supported"]
    passed = FS.family_s_from_stats(
        tmp_path / "stats.json",
        corpus_path=tmp_path / "corpus.parquet",
        n_perm=50,
        sham_d5net_ghost_null=[0.0, 0.01, 0.02],
        load_r1_series=False,
    )
    assert passed["hypotheses"]["H6-carrier"]["supported"]
    assert passed["hypotheses"]["H7-same"]["status"].endswith("no R1 recovery input")
    # the stats branch window must equal the family-S primary window
    with pytest.raises(ValueError, match="contrasts were computed"):
        FS.family_s_from_stats(tmp_path / "stats.json", n_perm=50, shift=20)
    (tmp_path / "gates.json").write_text(json.dumps(_gates("EXTEND")), encoding="utf-8")
    with pytest.raises(ValueError, match="EXTEND"):
        FS.family_s_from_stats(tmp_path / "stats.json", n_perm=50, gates=tmp_path / "gates.json")
    with pytest.raises(ValueError, match="not both"):
        FS.family_s_from_stats(
            tmp_path / "stats.json",
            n_perm=50,
            sham_d5net_ghost_null=0.1,
            ghost_null_path=tmp_path / "null.json",
        )
