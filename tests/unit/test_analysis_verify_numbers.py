"""verify.py (numpy only) against the finswarm analysis on the same synthetic runs, and the
numbers.json producer (claims.yaml -> expected values from the analysis outputs)."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import yaml
from _arms_fixture import ROOT, temp_registry
from _metrics_synth import market_frame, write_frame_run

from finswarm.analysis import gates
from finswarm.analysis.arms import load_registered_arms, resolve_arm_benchmarks
from finswarm.analysis.campaign import analyze_campaign
from finswarm.analysis.numbers import build_numbers, dotted_get, write_numbers
from finswarm.config.load import load_campaign_config
from finswarm.metrics.benchmarks import MarketModel
from finswarm.store.schemas import EVENTS

PC_MARKET = load_campaign_config(ROOT / "configs/sprint/tier0/PC-H.yaml").base.model_dump(
    mode="json"
)["market"]
TRH_MARKET = load_campaign_config(ROOT / "configs/sprint/tier0/TR-H.yaml").base.model_dump(
    mode="json"
)["market"]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        f"_script_{name}_vn", ROOT / "scripts" / f"{name}.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def man(arm, label, trunk, seed, branch=True, market=None, **extra) -> dict:
    m = {
        "arm": arm,
        "label": label,
        "trunk_id": trunk,
        "seed": seed,
        "status": "ok",
        "branch_period": 100 if branch else None,
        "resolved_config": {"tags": {"arm": arm, "label": label}},
    }
    if market is not None:
        m["resolved_config"]["market"] = market
    m.update(extra)
    return m


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory):
    arms = temp_registry(tmp_path_factory.mktemp("arms"))
    spec = load_registered_arms(arms)
    ab = resolve_arm_benchmarks(spec, ROOT)
    tm = ab["sham"].primary_model
    croot = tmp_path_factory.mktemp("campaigns")
    rng = np.random.default_rng(20261010)
    # PC-H: duopoly, alpha strata, 3 colluding + 1 Nash per alpha
    pc = croot / "PC-H"
    k = 0
    for alpha in (1.0, 3.2):
        market = json.loads(json.dumps(PC_MARKET))
        market["alpha"] = alpha
        market["price_grid"] = {"min": alpha, "max": 4 * alpha, "tick": 0.01 * alpha}
        model = MarketModel.from_config(market, 2)
        for level in (1.90, 1.88, 1.92, 1.4729):
            frame = market_frame(
                np.full((50, 2), level * alpha) + rng.normal(0, 0.002, (50, 2)), model, start=251
            )
            m = man("fish_exact", "PC-H", f"P{k}", k, branch=False, market=market)
            m["alpha"] = alpha
            m["resolved_config"]["llm"] = {"slots": [{"family": "anthropic"}] * 2}
            write_frame_run(pc, f"pc-{k}", frame, m)
            k += 1
    # TR-H: 8 trunks, sham / L1 / L2 / S1 (sting events)
    tr = croot / "TR-H"
    for t in range(8):
        tid = f"T{t}"
        write_frame_run(
            tr,
            f"trunk-{tid}",
            market_frame(np.full((30, 4), 1.90), tm, start=71),
            man("trunk", "TR-H", tid, t, branch=False),
        )
        sham = 1.85 + 0.02 * (t % 4) + rng.normal(0, 0.01, (40, 4))
        write_frame_run(
            tr,
            f"sham-{tid}",
            market_frame(sham, tm),
            man("sham", "SH-H", tid, t, market=TRH_MARKET),
        )
        write_frame_run(
            tr,
            f"L1-{tid}",
            market_frame(sham - 0.20 - 0.01 * t, ab["L1"].window_model),
            man("L1", "L1-H", tid, t),
        )
        write_frame_run(
            tr,
            f"L2-{tid}",
            market_frame(sham - 0.30 - 0.01 * t, ab["L2"].window_model),
            man("L2", "L2-H", tid, t),
        )
        # sting S1: slot 4 forces a rise of 0.20 at periods 111 and 121; rivals follow in 6 of 8 trunks
        p = np.column_stack([sham, np.full((40, 1), 1.85)])
        starts = [111, 121]
        for s in starts:
            r = s - 111
            p[r : r + 6, 4] += 0.20
            if t < 6:
                p[r + 1 : r + 4, :4] += 0.15
        d = write_frame_run(
            tr, f"S1-{tid}", market_frame(p[:, :4], tm, entrant=p[:, 4]), man("S1", "S1-H", tid, t)
        )
        ev = {c: [None] * 2 for c in EVENTS}
        ev.update(
            run_id=[f"S1-{tid}"] * 2,
            event_id=[0, 1],
            tick=[s * 100 for s in starts],
            period=np.array(starts, dtype=np.int32) - 1,  # store: 0-based (A-008)
            event_type=["sting"] * 2,
            payload=[json.dumps({"slot": 4, "delta": 0.20, "hold_epochs": 5})] * 2,
            source=["branch"] * 2,
        )
        (d / "events").mkdir(exist_ok=True)
        pl.DataFrame(ev, schema=EVENTS).write_parquet(d / "events" / "part-0000.parquet")
    # S1c-SC: competitive stings, nobody follows
    sc = croot / "CA-H"  # the competitive stings live in the CA-H dir (A-016)
    for t in range(4):
        p = np.column_stack([np.full((40, 4), 1.40), np.full((40, 1), 1.40)])
        p[10:16, 4] += 0.20  # sting at period 121: a baseline exists in 118-120
        d = write_frame_run(
            sc,
            f"s1c-{t}",
            market_frame(p[:, :4], tm, entrant=p[:, 4]),
            man("S1c", "S1c-LLM-H", f"C{t}", 700 + t, branch=False),
        )
        ev = {c: [None] for c in EVENTS}
        ev.update(
            run_id=[f"s1c-{t}"],
            event_id=[0],
            tick=[12100],
            period=np.array([120], dtype=np.int32),  # store 120 = registered 121 (A-008)
            event_type=["sting"],
            payload=[json.dumps({"slot": 4, "delta": 0.20, "hold_epochs": 5})],
            source=["config"],
        )
        (d / "events").mkdir(exist_ok=True)
        pl.DataFrame(ev, schema=EVENTS).write_parquet(d / "events" / "part-0000.parquet")
    out = tmp_path_factory.mktemp("analysis")
    analyze_campaign(tr, arms, ROOT, out, mqc={"L1": 1.0, "L2": 3.0}, B=400, n_perm=0)
    rep = gates.evaluate_gates(croot, ["PC-H"], [], "TR-H", None, arms, ROOT, B=400)
    gates.write_report(rep, out / "gates")
    return {"arms": arms, "spec": spec, "croot": croot, "out": out, "ab": ab}


def _claims(world, tmp_path: Path) -> Path:
    cfg = yaml.safe_load((ROOT / "configs/verify/claims.yaml").read_text(encoding="utf-8"))
    keep = [
        "C1_delta_p_mean",
        "C1_delta_p_ci_lo",
        "C1_share_runs_above_0_10",
        "C1_delta_pi_mean",
        "C1_parse_fail_rate",
        "G2_n_established",
        "G2_trend_slope",
        "C3a_delta_c_p",
        "C3a_p_trunk_scoring",
        "C3a_p_own_scoring",
        "C3a_t_lower_bound_95",
        "C3b_holm_p_L2",
        "C3b_n_rejected_trunk_scoring",
        "C4_tost_p_L9",
        "C7_follow_rate_collusive",
        "C7_follow_rate_competitive",
        "C10_share_above_pm_sham",
        "C10_trade_weighted_price_L2",
        "C10_delta_c_pi_window_L2",
    ]
    cfg["claims"] = {k: v for k, v in cfg["claims"].items() if k in keep}
    path = tmp_path / "claims.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def test_numbers_producer_fills_expected_from_outputs(world, tmp_path: Path) -> None:
    doc = build_numbers(_claims(world, tmp_path), world["out"], world["croot"], world["arms"], ROOT)
    meta = doc.pop("_meta")
    assert meta["filled"] >= 12 and meta["pending"] >= 1  # C4 (L9 not run) and follow rates pending
    c1 = doc["C1_delta_p_mean"]
    assert c1["params"]["window"] == [251, 300] and c1["params"]["bench"] == "run_closed_form"
    assert str(c1["runs"]).endswith("PC-H/*") and c1["expected"] is not None
    g2 = doc["G2_n_established"]
    assert g2["expected"] == 8 and set(g2["params"]["bench"]) == {
        "p_nash",
        "p_mono",
        "pi_nash",
        "pi_mono",
    }
    assert g2["params"]["bench"]["p_nash"] == pytest.approx(world["ab"]["trunk"].primary.p_nash)
    assert doc["C3a_p_own_scoring"]["params"]["bench"]["p_nash"] == pytest.approx(
        world["ab"]["L2"].window.p_nash
    )
    assert doc["C4_tost_p_L9"]["expected"] is None
    assert dotted_get({"a": [{"b": 2}]}, "a.0.b") == 2 and dotted_get({"a": 1}, "a.b") is None
    p = write_numbers(doc | {"_meta": meta}, tmp_path / "numbers.json")
    assert json.loads(p.read_text(encoding="utf-8"))["_meta"]["filled"] == meta["filled"]


def test_verify_matches_finswarm_and_flags_mismatches(world, tmp_path: Path) -> None:
    verify = _load("verify")
    doc = build_numbers(_claims(world, tmp_path), world["out"], world["croot"], world["arms"], ROOT)
    numbers = tmp_path / "numbers.json"
    write_numbers(doc, numbers)
    report = tmp_path / "report.json"
    rc = verify.main([str(numbers), "--root", str(ROOT), "--report", str(report)])
    rows = {r["claim"]: r for r in json.loads(report.read_text(encoding="utf-8"))}
    assert rc == 0, [r for r in rows.values() if r["verdict"] not in ("PASS", "pending")]
    for cid in (
        "C1_delta_p_mean",
        "C1_delta_pi_mean",
        "G2_n_established",
        "C3a_delta_c_p",
        "C3a_p_trunk_scoring",
        "C3a_p_own_scoring",
        "C3a_t_lower_bound_95",
        "C3b_holm_p_L2",
        "C3b_n_rejected_trunk_scoring",
        "C10_share_above_pm_sham",
        "C10_trade_weighted_price_L2",
        "C10_delta_c_pi_window_L2",
        "C1_delta_p_ci_lo",
    ):
        assert rows[cid]["verdict"] == "PASS", (cid, rows[cid])
    assert rows["C4_tost_p_L9"]["verdict"].startswith("pending")  # L9 not run: not computable
    # the sting follow rates are computed from the events (no analysis output yet -> pending)
    assert rows["C7_follow_rate_collusive"]["verdict"].startswith("pending")
    assert rows["C7_follow_rate_collusive"]["computed"] == pytest.approx(6 / 8, abs=1e-9)
    assert rows["C7_follow_rate_competitive"]["computed"] == pytest.approx(0.0)
    # a wrong expected value fails, a missing glob errors
    doc["C3a_delta_c_p"]["expected"] += 0.05
    write_numbers(doc, numbers)
    assert verify.main([str(numbers), "--root", str(ROOT)]) == 1
    doc["C3a_delta_c_p"]["runs"] = str(world["croot"] / "nowhere" / "*")
    write_numbers(doc, numbers)
    assert verify.main([str(numbers), "--root", str(ROOT)]) == 2


def test_verify_distributions_and_detection_helpers(tmp_path: Path) -> None:
    verify = _load("verify")
    from scipy import stats as sps

    for p in (0.001, 0.025, 0.3, 0.5, 0.9, 0.975, 0.999):
        assert float(verify.norm_ppf(np.array([p]))[0]) == pytest.approx(sps.norm.ppf(p), abs=1e-7)
    for t, df in ((0.5, 3), (2.1, 7), (-1.3, 11), (4.0, 2)):
        assert verify.t_sf(t, df) == pytest.approx(sps.t.sf(t, df), abs=1e-9)
    assert verify.t_ppf(0.95, 7) == pytest.approx(sps.t.ppf(0.95, 7), abs=1e-6)
    d = np.array([0.3, 0.35, 0.28, 0.32, 0.31, 0.29, 0.33, 0.30])
    assert verify.paired_t_one_sided(d) == pytest.approx(
        sps.ttest_1samp(d, 0, alternative="greater").pvalue, abs=1e-9
    )
    assert verify.tost_p(np.array([0.0, 0.01, -0.01, 0.02, 0.0, -0.02]), 0.2) < 0.001
    adj, rej = verify.holm([0.01, 0.04, 0.2, float("nan")])
    assert adj[0] == pytest.approx(0.04) and rej[0] and not rej[2] and adj[3] == 1.0
    assert verify.spearman(np.array([1, 2, 3, 4]), np.array([4, 3, 2, 1])) == pytest.approx(-1.0)
    assert verify.auc(np.array([0.1, 0.4, 0.35, 0.8]), np.array([0, 0, 1, 1])) == pytest.approx(
        0.75
    )
    # detection AUC from a corpus parquet and frozen-style rules
    rules = {
        "composite": {"features": ["D1", "D7"], "threshold": 0.5},
        "directions": {"D1": 1, "D7": 1},
        "features": {
            "D1": {"reference": [0.0, 0.2, 0.4, 0.6, 0.8]},
            "D7": {"reference": [0.0, 0.1, 0.3, 0.5]},
        },
        "split": {"eval_groups": ["1", "2"]},
    }
    (tmp_path / "rules.json").write_text(json.dumps(rules), encoding="utf-8")
    corpus = pl.DataFrame(
        {
            "run_id": ["a", "b", "c", "d", "e"],
            "kind": ["llm_punished", "llm_punished", "ghost", "ghost", "sc1"],
            "label": [1, 1, 0, 0, 0],
            "lane": ["H"] * 5,
            "seed_group": ["1", "2", "1", "2", "1"],
            "D1": [0.9, 0.7, 0.3, 0.5, 0.0],
            "D7": [0.6, 0.4, 0.05, 0.2, 0.0],
        }
    )
    corpus.write_parquet(tmp_path / "corpus.parquet")
    params = {
        "corpus": "corpus.parquet",
        "rules": "rules.json",
        "positives": ["llm_punished"],
        "negatives": ["ghost"],
        "lane": "H",
        "stat": "auc",
    }
    assert verify.c_detection_auc([], params, tmp_path) == pytest.approx(1.0)
    assert verify.c_detection_auc([], {**params, "stat": "fpr:ghost"}, tmp_path) in (0.0, 0.5)
