"""MQC per lever from synthetic scripted MQC-Lx campaigns (sham vs lever arms, own trades)."""

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from _metrics_synth import to_store

from finswarm.analysis.arms import load_registered_arms
from finswarm.analysis.mqc import analyze_mqc, load_mqc, mqc_point_estimates
from finswarm.metrics.quality import true_value_surplus

ROOT = Path(__file__).resolve().parents[2]
SPEC = load_registered_arms(ROOT / "configs" / "analysis" / "registered_arms.yaml")
S, E = SPEC.windows[SPEC.mqc.window]
N_SEEDS = 6


def trades(values: np.ndarray, price: float = 1.8, cost: float = 1.0, skip=()) -> pl.DataFrame:
    """One unit trade per period of the window (plus one outside it), ``buyer_value`` per period."""
    periods = [p for p in range(S, E + 1) if p not in set(skip)]
    rows = {
        "period": [S - 5, *periods],
        "price": [price] * (len(periods) + 1),
        "size": [1.0] * (len(periods) + 1),
        "buyer_value": [100.0, *[float(values[p - S]) for p in periods]],  # S-5: outside
        "seller_cost": [cost] * (len(periods) + 1),
        "buyer_id": [7] * (len(periods) + 1),
    }
    df = pl.DataFrame(rows)
    # the closed-form aggregate buyer carries no reservation value: never counted
    agg = pl.DataFrame(
        {
            "period": [S],
            "price": [price],
            "size": [50.0],
            "buyer_value": [100.0],
            "seller_cost": [cost],
            "buyer_id": [-1],
        }
    )
    return pl.concat([df, agg])


def write(camp: Path, run_id: str, arm: str, label: str, seed: int, tr: pl.DataFrame) -> None:
    d = camp / run_id
    (d / "trades").mkdir(parents=True, exist_ok=True)
    to_store(tr).write_parquet(d / "trades" / "part-0000.parquet")  # store: 0-based
    man = {"run_id": run_id, "arm": arm, "label": label, "trunk_id": f"S{seed}", "seed": seed}
    (d / "manifest.json").write_text(json.dumps(man), encoding="utf-8")


@pytest.fixture()
def mqc_camps(tmp_path: Path) -> list[Path]:
    rng = np.random.default_rng(3)
    n = E - S + 1
    c1, c2 = tmp_path / "MQC-L1", tmp_path / "MQC-L6"
    for k in range(N_SEEDS):
        base = 3.0 + rng.normal(0, 0.01, n)  # sham TS about 2.0 per period
        write(c1, f"sham-{k}", "sham", "MQC-L1", k, trades(base))
        # L1 lowers the buyers' realised value by 0.1 per unit (TS 5 % lower)
        write(c1, f"L1-{k}", "L1", "MQC-L1", k, trades(base - 0.1))
        # L6: same per-trade surplus but two periods without any trade
        write(c2, f"sham-{k}", "sham", "MQC-L6", k, trades(base))
        write(c2, f"L6-{k}", "L6", "MQC-L6", k, trades(base, skip=(S, S + 1)))
    # a single paired seed for L2, and a foreign label that must not be mixed in
    write(c2, "sham-L2", "sham", "MQC-L2", 0, trades(np.full(n, 3.0)))
    write(c2, "L2-0", "L2", "MQC-L2", 0, trades(np.full(n, 2.0)))
    write(c2, "L1-foreign", "L1", "MQC-X", 99, trades(np.full(n, 1.0)))
    return [c1, c2]


def test_true_value_surplus_ignores_outside_window_and_aggregate_buyer() -> None:
    tr = trades(np.full(E - S + 1, 3.0))
    assert true_value_surplus(tr, S, E) == pytest.approx(2.0)
    gap = trades(np.full(E - S + 1, 3.0), skip=(S,))
    assert true_value_surplus(gap, S, E) == pytest.approx(2.0 * (E - S) / (E - S + 1))


def test_mqc_per_lever_from_sham_and_lever_arms(mqc_camps: list[Path]) -> None:
    res = analyze_mqc(mqc_camps, SPEC, B=500, seed=1)
    assert res["window"] == [S, E] and res["reference_arm"] == "sham"
    assert set(res["levers"]) == set(SPEC.lever_family)
    l1 = res["levers"]["L1"]
    assert l1["status"] == "ok" and l1["n"] == N_SEEDS and l1["label"] == "MQC-L1"
    assert l1["mean"] == pytest.approx(5.0, abs=0.05)  # 100 * 0.1 / 2.0
    assert l1["bca_lo"] <= l1["mean"] <= l1["bca_hi"]
    assert l1["ratio_of_means"] == pytest.approx(5.0, abs=0.05)
    assert sorted(l1["per_seed"]) == [str(k) for k in range(N_SEEDS)]
    l6 = res["levers"]["L6"]
    assert l6["mean"] == pytest.approx(100 * 2 / (E - S + 1), abs=0.05)  # empty periods = 0
    assert res["levers"]["L2"] == {"status": "too few paired seeds", "n": 1, "label": "MQC-L2"}
    assert res["levers"]["L3"]["status"] == "too few paired seeds"  # registered, no data
    assert res["levers"]["L8"]["status"] == "no MQC campaign registered"
    pts = mqc_point_estimates(res)
    assert pts["L1"] == pytest.approx(l1["mean"]) and pts["L2"] is None and pts["L8"] is None


def test_mqc_sham_vs_sham_is_zero_and_round_trips(mqc_camps: list[Path], tmp_path: Path) -> None:
    c = tmp_path / "MQC-L1-null"
    for k in range(4):
        v = np.full(E - S + 1, 3.0 + 0.01 * k)
        write(c, f"sham-{k}", "sham", "MQC-L1", k, trades(v))
        write(c, f"L1-{k}", "L1", "MQC-L1", k, trades(v))
    res = analyze_mqc(c, SPEC, B=200, seed=1)
    assert res["levers"]["L1"]["mean"] == pytest.approx(0.0, abs=1e-12)
    p = tmp_path / "mqc.json"
    p.write_text(json.dumps(res), encoding="utf-8")
    assert mqc_point_estimates(load_mqc(p))["L1"] == pytest.approx(0.0, abs=1e-12)
    assert mqc_point_estimates({"L1": 2.5, "L2": None}) == {"L1": 2.5}
    assert mqc_point_estimates(None) == {}
