"""A-010: verify.py recomputes the SAP exclusions and the qualification from raw tables, compares
p-values with a tight relative tolerance and the decision p < alpha, and H8 uses t with K-1 df."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from _metrics_synth import market_frame, write_frame_run

from finswarm.analysis import family_s as FS
from finswarm.metrics.benchmarks import MarketModel

ROOT = Path(__file__).resolve().parents[2]
MODEL = MarketModel(n_incumbents=4)
B = MODEL.benchmarks()
BENCH = {k: float(getattr(B, k)) for k in ("p_nash", "p_mono", "pi_nash", "pi_mono")}


def _verify():
    spec = importlib.util.spec_from_file_location("_verify_a010", ROOT / "scripts" / "verify.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _trunk(camp: Path, name: str, level: float, n: int = 30, status: str = "ok") -> None:
    p = np.full((n, 4), level)
    man = {
        "arm": "trunk",
        "label": "TR-H",
        "trunk_id": name,
        "status": status,
        "branch_period": None,
    }
    write_frame_run(camp, name, market_frame(p, MODEL, start=71), man)


def test_qualification_and_sap_exclusions_recomputed(tmp_path: Path) -> None:
    V = _verify()
    camp = tmp_path / "TR-H"
    hi = BENCH["p_nash"] + 0.85 * (BENCH["p_mono"] - BENCH["p_nash"])
    lo = BENCH["p_nash"] + 0.05 * (BENCH["p_mono"] - BENCH["p_nash"])
    _trunk(camp, "A", hi)
    _trunk(camp, "B", hi)
    _trunk(camp, "C", lo)  # competitive: does not qualify
    _trunk(camp, "D", hi, n=25)  # incomplete window: excluded by the SAP filter
    _trunk(camp, "E", hi, status="failed")  # manifest status: excluded
    params = {"window": [71, 100], "members": [0, 1, 2, 3], "bench": BENCH}
    runs = V.match_runs(tmp_path, "TR-H/*", {**params, "arm": "trunk"})
    assert sorted(r.name for r in runs) == ["A", "B", "C"]
    assert V.sap_excluded(camp / "D", params).startswith("incomplete window (25 of 30")
    spec = {"runs": "TR-H/*", **params}
    assert V.recompute_qualified(tmp_path, spec) == {"A", "B"}
    # claims filtered by the recomputed list; a disagreeing analysis list raises
    stats = tmp_path / "stats.json"
    stats.write_text(
        json.dumps({"inference": {"qualification": {"qualified": ["A", "B"]}}}), encoding="utf-8"
    )
    q = {**params, "arm": "trunk", "qualified_recompute": spec, "qualified_from": "stats.json"}
    assert sorted(r.name for r in V.match_runs(tmp_path, "TR-H/*", q)) == ["A", "B"]
    stats.write_text(
        json.dumps({"inference": {"qualification": {"qualified": ["A", "B", "C"]}}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="qualification mismatch"):
        V.match_runs(tmp_path, "TR-H/*", q)


def test_tight_relative_tolerance_and_alpha_decision(tmp_path: Path) -> None:
    V = _verify()
    camp = tmp_path / "TR-H"
    _trunk(camp, "A", BENCH["p_nash"] + 0.5 * (BENCH["p_mono"] - BENCH["p_nash"]))
    base = {
        "metric": "n_runs",
        "runs": "TR-H/*",
        "params": {"window": [71, 100]},
    }
    numbers = {
        "ok": {**base, "expected": 1.0, "tolerance_rel": 1e-6},
        "off": {**base, "expected": 1.0000100, "tolerance_rel": 1e-6},
    }
    f = tmp_path / "numbers.json"
    f.write_text(json.dumps(numbers), encoding="utf-8")
    rep = tmp_path / "rep.json"
    assert V.main([str(f), "--root", str(tmp_path), "--report", str(rep)]) == 1
    verdicts = {r["claim"]: r["verdict"] for r in json.loads(rep.read_text(encoding="utf-8"))}
    assert verdicts == {"ok": "PASS", "off": "FAIL"}
    # the decision p < alpha must agree even when the numbers are close
    numbers = {"flip": {**base, "expected": 1.0, "tolerance": 5.0, "alpha": 1.5}}
    f.write_text(json.dumps(numbers), encoding="utf-8")
    assert V.main([str(f), "--root", str(tmp_path)]) == 0  # 1.0 < 1.5 on both sides
    numbers = {"flip": {**base, "expected": 2.0, "tolerance": 5.0, "alpha": 1.5}}
    f.write_text(json.dumps(numbers), encoding="utf-8")
    assert V.main([str(f), "--root", str(tmp_path)]) == 1  # within tolerance, decision differs


def test_h8_uses_t_with_k_minus_one_df_in_analysis_and_verify() -> None:
    coll = [(8, 10), (9, 10), (7, 10), (8, 10), (9, 10), (8, 10)]
    comp = [(1, 10), (0, 10), (2, 10), (1, 10), (1, 10), (0, 10)]
    pd = FS.proportion_difference(coll, comp, 0.30)
    z = (pd["diff"] - 0.30) / pd["se"]
    from scipy import stats as sps

    assert pd["df"] == 5 and pd["p"] == pytest.approx(float(sps.t.sf(z, 5)))
    assert pd["p"] > float(sps.norm.sf(z))  # the normal was anti-conservative
    V = _verify()
    assert 1.0 - V.t_cdf(z, 5) == pytest.approx(pd["p"], rel=1e-6)
