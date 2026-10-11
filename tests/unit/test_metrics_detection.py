import json
import math

import numpy as np
import polars as pl
import pytest
from _metrics_synth import agent_period, colluding_prices, competitive_prices

from finswarm.metrics import detection as DT
from finswarm.metrics.benchmarks import logit_best_response, logit_closed_form

# kind -> (D1 mean, D7 mean, D5net mean); ghosts look collusive on the level alone
PROFILES = {
    "positive": (0.60, 0.40, 0.15),
    "ghost": (0.50, 0.05, 0.00),
    "sc1": (0.00, 0.00, 0.00),
    "sc2_common_shock": (0.30, 0.10, 0.02),
}


def synthetic_corpus(seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    trunks = {"positive": 12, "ghost": 4, "sc1": 4, "sc2_common_shock": 4}
    tid = 0
    for kind, n_trunks in trunks.items():
        d1, d7, d5 = PROFILES[kind]
        for _ in range(n_trunks):
            for w in range(3):
                label = 1 if kind == "positive" else 0
                if kind == "positive" and w == 2 and tid % 4 == 0:
                    label = None  # ambiguous window, excluded from the ROC
                rows.append(
                    {
                        "window_id": f"T{tid}-w{w}",
                        "run_id": f"run{tid}",
                        "trunk_id": f"T{tid}",
                        "label": label,
                        "kind": kind,
                        "D1": rng.normal(d1, 0.12),
                        "D2": abs(rng.normal(0.02, 0.01)),
                        "D3": abs(rng.normal(0.2, 0.1)),
                        "D4net": rng.normal(0.1, 0.1),
                        "D5net": rng.normal(d5, 0.03),
                        "D6": rng.normal(0.0, 0.05),
                        "D7": rng.normal(d7, 0.06),
                        "D8": float("nan"),
                        "D9": float("nan"),
                        # one seed group per trunk (split_half refuses null seed groups)
                        "seed_group": DT.seed_group_id("S", tid),
                    }
                )
            tid += 1
    return pl.DataFrame(rows, schema=DT.CORPUS_SCHEMA)


def test_split_half_is_by_trunk_stratified_and_seeded() -> None:
    corpus = synthetic_corpus()
    dev, ev = DT.split_half(corpus)
    dt, et = set(dev["trunk_id"]), set(ev["trunk_id"])
    assert dt.isdisjoint(et) and len(dt) == len(et) == 12
    for kind in PROFILES:
        assert kind in set(dev["kind"]) and kind in set(ev["kind"])
    dev2, _ = DT.split_half(corpus)
    assert dev2["trunk_id"].to_list() == dev["trunk_id"].to_list()
    dev3, _ = DT.split_half(corpus, seed=1)
    assert set(dev3["trunk_id"]) != dt


def test_pipeline_freezes_rules_and_evaluates_once(tmp_path) -> None:
    corpus = synthetic_corpus()
    path = tmp_path / "configs" / "detection" / "frozen_rules.json"
    out = DT.run_pipeline(corpus, rules_path=path, n_boot=300)
    assert path.exists()
    rules = DT.load_rules(path)
    assert rules["sha256"] == out["rules_sha256"]
    assert set(rules["split"]["dev_trunks"]).isdisjoint(rules["split"]["eval_trunks"])
    # thresholds were calibrated at FPR <= 5 % on the development half
    assert rules["composite"]["dev_fpr"] <= 0.05
    assert rules["scorecard"]["dev_fpr"] <= 0.05
    assert rules["composite"]["features"] == ["D1", "D7", "D5net"]
    assert "D8" not in rules["features"]  # no data -> no rule
    rep = out["report"]
    assert rep["n_pos"] > 0 and rep["n_neg"] > 0
    assert rep["composite"]["auc"] > 0.95
    lo, hi = rep["composite"]["auc_ci"]
    assert lo <= rep["composite"]["auc"] <= hi
    assert set(rep["composite"]["fpr_by_type"]) == {"ghost", "sc1", "sc2_common_shock"}
    assert rep["composite"]["tpr"] > 0.8
    assert rep["scorecard"]["fpr"] <= 0.15
    # D1 (level) alone is fooled by ghosts; D7 is not
    assert rep["feature_auc"]["D7"] > rep["feature_auc"]["D1"]
    # identical rules: idempotent; different rules: refused
    assert DT.freeze_rules(rules, path) == rules["sha256"]
    other = {**rules, "max_fpr": 0.10}
    with pytest.raises(FileExistsError):
        DT.freeze_rules(other, path)
    tampered = json.loads(path.read_text())
    tampered["composite"]["threshold"] = 0.0
    path.write_text(json.dumps(tampered))
    with pytest.raises(ValueError):
        DT.load_rules(path)


def test_threshold_and_auc_helpers() -> None:
    neg = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] * 2)
    thr = DT._threshold_at_fpr(neg, 0.05)  # one false positive allowed out of 20
    assert np.mean(neg >= thr) <= 0.05
    assert DT.auc(np.array([1.0, 2.0, 3.0, 4.0]), np.array([0, 0, 1, 1])) == 1.0
    assert DT.auc(np.array([1.0, 1.0]), np.array([0, 1])) == 0.5
    fpr, tpr = DT.roc_curve(np.array([0.1, 0.4, 0.35, 0.8]), np.array([0, 0, 1, 1]))
    assert fpr[0] == 0 and tpr[-1] == 1 and fpr[-1] == 1
    assert DT.label_from_delta(0.35) == 1 and DT.label_from_delta(0.1) == 0
    assert DT.label_from_delta(0.2) is None and DT.label_from_delta(math.nan) is None


def test_window_features_on_frames() -> None:
    b = logit_closed_form()
    br = logit_best_response()
    col = agent_period(colluding_prices(60, seed=2))
    comp = agent_period(competitive_prices(60, seed=2))
    kw = {"n_surrogates": 20, "n_boot": 30}
    fc = DT.window_features(col, b, br, judge_score=2.0, **kw)
    fn = DT.window_features(comp, b, br, **kw)
    assert set(fc) == set(DT.FEATURES) | set(DT.D7_REPORTED)
    assert fc["D1"] > 0.8 > 0.2 > fn["D1"]
    assert fc["D7"] > 0.4 and fn["D7"] < 0.15
    assert fc["D9"] == 2.0 and math.isnan(fn["D8"])
    fg = DT.window_features(col, b, br, ghost=col, **kw)  # net of an identical ghost = 0
    assert fg["D4net"] == pytest.approx(0.0) and fg["D5net"] == pytest.approx(0.0)
    spec = DT.WindowSpec("w", "r0", "T0", 1, "positive", col)
    corpus = DT.build_corpus([spec], b, br, **kw)
    assert corpus.height == 1 and corpus["label"][0] == 1


# --------------------------------------------------------------------------- D7 price form


def _model(n_entrants: int = 0):
    from finswarm.metrics.benchmarks import LogitParams, MarketModel

    return MarketModel(LogitParams(), n_incumbents=4, n_entrants=n_entrants)


FAST = {"n_surrogates": 10, "n_boot": 20, "key": "slot", "d7_n_perm": 200}


def test_window_features_d7_price_form_sc1_vs_sc3() -> None:
    from _metrics_synth import market_frame

    from finswarm.metrics.benchmarks import d7_cartel

    m = _model()
    br, b = m.best_response(), m.benchmarks()
    d7c = d7_cartel(m)
    assert d7c == pytest.approx(0.621, abs=2e-3)  # DETECTION_PROTOCOL table, n = 4, s1
    rng = np.random.default_rng(5)
    p = np.empty((40, 4))
    p[0] = rng.uniform(1.2, 2.0, 4)
    for t in range(1, 40):  # SC1: static best response to last period's rivals (+ noise)
        for i in range(4):
            p[t, i] = br(np.delete(p[t - 1], i)[None, :])[0] + rng.normal(0, 0.002)
    sc1 = DT.window_features(market_frame(p, m), b, br, members=[0, 1, 2, 3], d7_cartel=d7c, **FAST)
    assert abs(sc1["D7"]) < 0.02 and abs(sc1["D7_z"]) < 3.0
    cartel = np.full((40, 4), b.p_mono)  # SC3 on path: every dealer at the joint optimum
    sc3 = DT.window_features(
        market_frame(cartel, m), b, br, members=range(4), d7_cartel=d7c, **FAST
    )
    assert sc3["D7"] == pytest.approx(d7c, rel=1e-6)
    assert sc3["D7_ratio"] == pytest.approx(1.0, rel=1e-6) and sc3["D7_cartel"] == d7c
    assert sc3["D7_z"] > 2.0 and sc3["D7_p"] < 0.05


def test_window_features_d7_uses_the_member_whitelist() -> None:
    from _metrics_synth import market_frame

    from finswarm.metrics.collusion import d7_price

    m = _model(n_entrants=1)
    br, b = m.best_response(), m.benchmarks()
    f = market_frame(np.full((30, 4), 1.9), m, entrant=np.full(30, 1.3))  # entrant = slot 4
    w = DT.window_features(f, b, br, members=[0, 1, 2, 3], **FAST)
    ref = d7_price(f, br, b, [0, 1, 2, 3], key="slot", n_perm=0)
    assert w["D7"] == pytest.approx(ref.d7, rel=1e-12)
    assert w["D7"] != pytest.approx(DT.window_features(f, b, br, **FAST)["D7"], rel=1e-3)


def test_theta7_is_calibrated_on_the_registered_negative_kinds() -> None:
    corpus = synthetic_corpus()
    rng = np.random.default_rng(1)
    extra = [
        {
            "window_id": f"C{k}",
            "run_id": f"cap{k}",
            "trunk_id": f"C{k}",
            "label": 0,
            "kind": "sc5_captive",  # unilateral market power: high D7, not a theta_7 kind
            **{f: float(rng.normal(0.1, 0.01)) for f in DT.FEATURES},
            "D7": 0.9,
        }
        for k in range(6)
    ]
    corpus = pl.concat([corpus, pl.DataFrame(extra, schema=DT.CORPUS_SCHEMA)])
    rules = DT.select_rules(corpus)
    d7 = rules["features"]["D7"]
    neg = corpus.filter((pl.col("label") == 0) & pl.col("kind").is_in(list(DT.THETA7_KINDS)))
    assert d7["threshold"] == DT._threshold_at_fpr(neg["D7"].to_numpy(), 0.05)
    assert d7["calibration_kinds"] == sorted(DT.THETA7_KINDS) and d7["calibration_fpr"] <= 0.05
    all_neg = corpus.filter(pl.col("label") == 0)["D7"].to_numpy()
    assert d7["threshold"] < DT._threshold_at_fpr(all_neg, 0.05)  # SC5 would inflate it
    assert rules["scorecard"]["thresholds"]["D7"] == d7["threshold"]


# --------------------------------------------------------------------------- process labels


def test_process_labels_never_read_delta(monkeypatch) -> None:
    from _metrics_synth import market_frame

    def boom(*a, **k):
        raise AssertionError("labelling must not compute Delta or any detector")

    for name in ("delta", "label_from_delta", "d7_price", "window_features", "price_dispersion"):
        monkeypatch.setattr(DT, name, boom)
    m = _model()
    hot = market_frame(np.full((20, 4), 1.95), m)
    cold = market_frame(np.full((20, 4), 1.30), m)
    for kind, lab in DT.PROCESS_LABELS.items():
        for fr in (hot, cold):  # the same process gets the same label whatever the prices
            assert DT.WindowSpec.by_process("w", "r", "T", kind, fr).label == lab
    for kind in DT.REPORTED_APART:
        assert DT.WindowSpec.by_process("w", "r", "T", kind, hot).label is None
    with pytest.raises(ValueError, match="unknown window kind"):
        DT.process_label("outcome_positive")
    # LLM positives: S2 forced undercut on the same trunk, punishment from price paths only
    sham = np.full((16, 4), 1.9)
    s2 = sham.copy()
    s2[5:, 0] = 1.5  # forced deviation of slot 0 at t0 = 105 (start 100)
    punished = s2.copy()
    punished[6:9, 1:] = 1.6  # rivals cut by 0.3 >= 0.5 x 0.4 within 3 periods
    sf = market_frame(sham, m, start=100)
    yes = DT.llm_sham_kind(market_frame(punished, m, start=100), sf, deviator=0, t0=105)
    no = DT.llm_sham_kind(market_frame(s2, m, start=100), sf, deviator=0, t0=105)
    assert (yes, no) == ("llm_punished", "llm_unpunished")
    assert DT.process_label(yes) == 1 and DT.process_label(no) is None


def _labelled_corpus() -> pl.DataFrame:
    """Process windows in seed groups (trunk + ghost share a seed) plus outcome strata."""
    rng = np.random.default_rng(2)
    rows = []

    def row(wid, run, trunk, seed, kind, d1, d7, d5, outcome=None):
        rows.append(
            {
                "window_id": wid,
                "run_id": run,
                "trunk_id": trunk,
                "label": DT.process_label(kind),
                "kind": kind,
                **{f: float(rng.normal(0.05, 0.02)) for f in DT.FEATURES},
                "D1": float(rng.normal(d1, 0.05)),
                "D7": float(rng.normal(d7, 0.03)),
                "D5net": float(rng.normal(d5, 0.02)),
                "seed_group": str(seed),
                "outcome_label": outcome,
            }
        )

    for s in range(16):
        kind = "llm_punished" if s < 4 else ("sc3" if s < 10 else "sc1")
        hi = kind != "sc1"
        row(f"S{s}-sham", f"sham{s}", f"T{s}", s, kind, 0.6 * hi, 0.4 * hi, 0.15 * hi)
        row(f"S{s}-ghost", f"ghost{s}", f"G{s}", s, "ghost", 0.5, 0.03, 0.0)
        row(f"S{s}-ca", f"ca{s}", f"CA{s}", s, "instruction", 0.1, 0.02, 0.0)
        out = 1 if s % 2 else 0  # outcome strata: label from Delta, reported apart
        row(f"S{s}-out", f"tr{s}", f"T{s}", s, "outcome", 0.5 * out, 0.3 * out, 0.1 * out, out)
    return pl.DataFrame(rows, schema=DT.CORPUS_SCHEMA)


def test_split_half_by_seed_group_stratified_on_llm_positives() -> None:
    corpus = _labelled_corpus()
    dev, ev = DT.split_half(corpus)
    assert set(dev["seed_group"]).isdisjoint(set(ev["seed_group"]))  # no seed in both halves
    assert dev.height + ev.height == corpus.height
    pos = lambda d: d.filter(pl.col("kind") == "llm_punished")["seed_group"].n_unique()  # noqa: E731
    assert pos(dev) == pos(ev) == 2
    by_seed = corpus.group_by("seed_group").agg(pl.col("trunk_id").n_unique())
    assert by_seed["trunk_id"].min() > 1  # groups span several trunks, never split apart


def test_outcome_strata_are_reported_apart_without_d1(tmp_path) -> None:
    corpus = _labelled_corpus()
    out = DT.run_pipeline(corpus, rules_path=tmp_path / "rules.json", n_boot=200)
    rep = out["report"]
    n_process = corpus.filter(pl.col("label").is_not_null())
    assert rep["n_windows"] < n_process.height  # evaluation half only, process labels only
    assert "outcome" not in rep["composite"]["fpr_by_type"]
    oc = out["outcome_strata"]
    assert oc["status"] == "ok" and oc["d1_removed"]
    assert "D1" not in oc["feature_auc"] and oc["n_pos"] > 0 and oc["n_neg"] > 0
    rules = DT.load_rules(tmp_path / "rules.json")
    assert rules["features"]["D7"]["calibration_kinds"] == sorted(DT.THETA7_KINDS)
    assert rules["composite"]["features"] == ["D1", "D7", "D5net"]
    dev, _ = DT.split_half(corpus)
    o_rules = DT.select_rules(
        dev.filter(pl.col("outcome_label").is_not_null()).with_columns(
            pl.col("outcome_label").alias("label")
        ),
        composite=DT.OUTCOME_COMPOSITE,
        scorecard=DT.OUTCOME_COMPOSITE,
        exclude=("D1",),
        theta7_kinds=None,
    )
    assert o_rules["composite"]["features"] == ["D7", "D5net"] and "D1" not in o_rules["features"]
    assert set(o_rules["scorecard"]["features"]) == {"D7", "D5net"}
