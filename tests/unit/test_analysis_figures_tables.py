import json

import numpy as np

from finswarm.analysis import figures as F
from finswarm.analysis import tables as T


def _check(paths, run_ids) -> dict:
    png, pdf, side = paths
    assert png.suffix == ".png" and png.stat().st_size > 1000
    assert pdf.suffix == ".pdf" and pdf.read_bytes()[:4] == b"%PDF"
    meta = json.loads(side.read_text(encoding="utf-8"))
    assert meta["run_ids"] == sorted(set(run_ids))
    return meta


def test_all_figures_write_png_pdf_and_sidecar(tmp_path) -> None:
    ids = ["r1", "r2", "r2"]
    rng = np.random.default_rng(0)
    paths = {"H": 1.5 + rng.normal(0, 0.02, (3, 50)), "M": 1.6 + rng.normal(0, 0.02, (4, 50))}
    m = _check(
        F.plot_positive_control(
            paths,
            {"H": [0.5, 0.6, 0.7], "M": [0.4] * 4},
            1.47,
            1.93,
            tmp_path / "fig1",
            ids,
            periods=np.arange(1, 61),  # longer than the paths: sliced to each lane's width
            delta_pi_by_seed={"H": [0.4, 0.5, 0.6], "M": [0.3] * 4},
            anchoring=0.95,
        ),
        ids,
    )
    assert m["figure"] == "positive_control" and len(m["data"]["H"]["mean"]) == 50
    assert m["data"]["thresholds"] == [0.25, 0.10] and m["data"]["anchoring_median_delta_p"] == 0.95
    assert m["data"]["delta_p_H"] == [0.5, 0.6, 0.7] and m["data"]["delta_pi_M"] == [0.3] * 4
    # the two-panel form (no Delta_pi) still works
    _check(
        F.plot_positive_control(paths, {"H": [0.5], "M": [0.4]}, 1.47, 1.93, tmp_path / "f1b", ids),
        ids,
    )
    m2 = _check(
        F.plot_structural_gradient(
            ["trunk (s1)", "GR-s2"],
            [[0.5, 0.6], [0.1, 0.2, 0.0]],
            tmp_path / "fig2",
            ids,
            references={"Fish et al. (P1)": (0.7, 0.7), "Agrawal": (0.0, 0.0)},
            markers=["s", "o"],
            notes={"Fish et al. (P1) Delta_pi": 0.43},
        ),
        ids,
    )
    assert m2["data"]["ref_Fish et al. (P1)"] == [0.7, 0.7]  # a mark, not a band
    assert m2["data"]["trunk (s1)"]["marker"] == "s" and m2["data"]["GR-s2"]["marker"] == "o"
    assert m2["data"]["notes"] == {"Fish et al. (P1) Delta_pi": 0.43}
    ca = "price reduction consistent with competitive adjustment"
    pts = [
        {"label": "L1", "mqc": 1.0, "dc": 0.3, "lo": 0.2, "hi": 0.4, "mechanical": 0.05},
        {"label": "L2 (M)", "mqc": 6.0, "dc": 0.1, "lo": -0.1, "hi": 0.3, "replication": True},
        {
            "label": "L3",
            "mqc": 0.5,
            "dc": 0.12,
            "lo": -0.02,
            "hi": 0.26,
            "own": -0.01,
            "own_lo": -0.1,
            "own_hi": 0.1,
            "dual_label": ca,
        },
    ]
    m3 = _check(
        F.plot_frontier(pts, tmp_path / "fig3", ids, sham_band=0.2, not_run=["L4a", "L5"]), ids
    )
    assert m3["data"]["sham_band"] == 0.2 and m3["data"]["y"].startswith("total capped")
    assert "sham2" in m3["data"]["sham_band_definition"] and m3["data"]["not_run"] == ["L4a", "L5"]
    assert m3["data"]["mei"] == 0.25 and len(m3["data"]["points"]) == 3
    assert m3["data"]["points"][2]["dual_label"] == ca
    t = np.arange(10)
    _check(
        F.plot_action_carrier_ablation(
            {"sham": (t, t * 0 + 0.5, t * 0 + 0.4, t * 0 + 0.6)},
            5,
            tmp_path / "fig4",
            ids,
            dependence={"sham": (0.1, 0.05, 0.15)},
        ),
        ids,
    )
    roc = {"composite": (np.array([0, 0.1, 1]), np.array([0, 0.9, 1]), 0.95)}
    _check(F.plot_detector_roc(roc, tmp_path / "fig5", ids), ids)


def test_mean_ci_and_tables() -> None:
    m, lo, hi = F.mean_ci(np.array([[1.0, 2.0], [3.0, 2.0]]))
    assert m.tolist() == [2.0, 2.0] and lo[0] < 2.0 < hi[0] and lo[1] == hi[1] == 2.0
    assert T.fmt_ci(0.4321, 0.31, 0.55) == "0.43 [0.31, 0.55]"
    assert T.fmt_p(0.0001) == "<0.001" and T.fmt(float("nan")) == "–"
    md = T.markdown_table([{"a": 1, "b": 0.5}], align={"a": "r"})
    assert md.splitlines()[1] == "| ---: | :--- |"
    arms = T.arm_table({"sham": {"n": 8, "mean": 0.5, "lo": 0.4, "hi": 0.6}})
    assert "0.50 [0.40, 0.60]" in arms
    rep = {
        "composite": {
            "auc": 0.9,
            "auc_ci": [0.8, 0.97],
            "tpr": 0.7,
            "fpr": 0.04,
            "fpr_by_type": {"ghost": 0.05},
        },
        "scorecard": {"tpr": 0.6, "fpr": 0.0, "fpr_by_type": {"ghost": 0.0}},
        "feature_auc": {"D1": 0.7},
    }
    out = T.detector_table(rep)
    assert "0.90 [0.80, 0.97]" in out and "ghost" in out


def test_frontier_from_stats_total_trunk_scoring_sham2_band_and_not_run() -> None:
    def bundle(mean, lo, hi, sd=0.05):
        return {"mean": mean, "bca_lo": lo, "bca_hi": hi, "sd": sd}

    st = {
        "runs": [{"run_id": "a"}, {"run_id": "b"}],
        "h1": {"lever": "L2", "n": 8},
        "h1r": {"family": ["L1", "L2", "L5"]},
        "mqc": {"levers": {"L1": {"mean": 1.0}, "L2": {"mean": 2.0}, "L5": {"mean": 3.0}}},
        "inference": {"sigma_d": {"estimate": 0.4}},
        "contrasts": {
            "sham2": {"n_pairs": 8, "delta_p": bundle(0.0, -0.05, 0.05, sd=0.037)},
            "L1": {
                "n_pairs": 8,
                "delta_p": bundle(0.30, 0.20, 0.40),
                "mechanical": {"delta_p": bundle(0.10, 0.08, 0.12)},
                "behavioural": {"delta_p": bundle(0.20, 0.10, 0.30)},
                "delta_p_own": bundle(-0.01, -0.10, 0.08),
                "dual_benchmark": {
                    "label": "price reduction consistent with competitive adjustment"
                },
            },
            "L2": {
                "n_pairs": 8,
                "delta_p": bundle(0.40, 0.30, 0.50),
                "delta_p_own": bundle(0.38, 0.25, 0.50),
                "dual_benchmark": {"label": "collusion reduction"},
            },
        },
    }
    pts, band, ids, not_run = F.frontier_from_stats(st)
    by = {p["label"]: p for p in pts}
    assert set(by) == {"L1", "L2"} and not_run == ["L5"] and ids == ["a", "b"]
    assert by["L1"]["dc"] == 0.30 and by["L1"]["mechanical"] == 0.10  # total, not behavioural
    assert by["L1"]["behavioural"] == 0.20 and by["L1"]["own"] == -0.01
    assert by["L1"]["dual_label"].startswith("price reduction") and by["L2"]["primary"]
    assert band == 0.037  # SD over trunks of Delta_p(sham) - Delta_p(sham2), not 1.96 sigma/sqrt(n)
    del st["contrasts"]["sham2"]
    assert F.frontier_from_stats(st)[1] == 0.0
