"""Science's synthetic cases A-F on the registered-arm analysis (the readout re-test, with the
science session's exact constants: 12 trunks, noise sd 0.02, rng 20261010, MQC L1 1 / L2 3 /
L3 0.2 / L4a 0.5 / L6 2 pp). Each case pins one pre-registered reading:

  A effect    trunks 0.85, sham 0.80, L2 cut 0.25 (both scorings)  -> L2 "collusion reduction"
  B anchor    trunks and sham raw 1.45, L2 cut 0.35 (stays > 1)    -> capped index, H1 null (D-16)
  C qualif    6 trunks at 0.25, 6 at 0.85                          -> 6 qualified trunks (Gate 2)
  D F1        4 trunks at 0.85, 8 at 0.05                          -> F1 fallback, H1 not read
  E adjust    L1 prices 1.77: trunk d ~0.19, own d ~-0.02          -> competitive adjustment (D-17)
  F trend     trunk Delta_p rising 0.01/period over 71-100         -> trend rule fires (EXTEND)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from _metrics_synth import market_frame, write_frame_run

from finswarm.analysis.arms import load_registered_arms, resolve_arm_benchmarks
from finswarm.analysis.campaign import analyze_campaign
from finswarm.analysis.gates import gate2
from finswarm.config.load import load_campaign_config

pytestmark = pytest.mark.slow  # six synthetic campaigns, about a minute each

ROOT = Path(__file__).resolve().parents[2]
ARMS = ROOT / "configs" / "analysis" / "registered_arms.yaml"
SPEC = load_registered_arms(ARMS)
AB = resolve_arm_benchmarks(SPEC, ROOT)
TRUNK_M = AB["sham"].primary_model
WIN = {k: AB[k].window_model for k in ("L1", "L2", "L3", "L4a", "L6")}
TB = AB["sham"].primary
PN, PM = TB.p_nash, TB.p_mono
TRH_MARKET = load_campaign_config(ROOT / "configs/sprint/tier0/TR-H.yaml").base.model_dump(
    mode="json"
)["market"]
MQC = {  # science's MQC means; the upper bound (m11) and the SC1 ask fall (A-003) added
    "levers": {
        k: {"status": "ok", "mean": v, "bca_hi": v + 1.0, "sc1_price_response": -0.02}
        for k, v in {"L1": 1.0, "L2": 3.0, "L3": 0.2, "L4a": 0.5, "L6": 2.0}.items()
    }
}
N_TRUNKS = 12
RNG_SEED = 20261010
NOISE = 0.02
CA_LABEL = "price reduction consistent with competitive adjustment"

CASES: dict[str, dict[str, Any]] = {
    "A_effect": dict(trunk_delta=[0.85] * 12, sham_delta=0.80, cuts={"L2": 0.25, "L3": 0.0}),
    "B_anchor": dict(trunk_delta=[1.45] * 12, sham_delta=1.45, cuts={"L2": 0.35}),
    "C_qualif": dict(trunk_delta=[0.25] * 6 + [0.85] * 6, sham_delta=0.80, cuts={"L2": 0.25}),
    "D_F1": dict(trunk_delta=[0.85] * 4 + [0.05] * 8, sham_delta=0.80, cuts={"L2": 0.25}),
    "E_adjust": dict(
        trunk_delta=[0.85] * 12, sham_delta=0.80, cuts={"L1": 0.0, "L2": 0.25}, l1_price=1.77
    ),
    "F_trend": dict(trunk_delta=[0.85] * 12, sham_delta=0.80, cuts={"L2": 0.25}, slope=0.01),
}


def price_at(delta: float) -> float:
    return PN + delta * (PM - PN)


def manifest(arm, label, trunk, seed, branch=True, market=None) -> dict:
    m = {
        "arm": arm,
        "label": label,
        "trunk_id": trunk,
        "seed": seed,
        "branch_period": 100 if branch else None,
        "resolved_config": {"tags": {"arm": arm, "label": label}},
    }
    if market is not None:
        m["resolved_config"]["market"] = market
    return m


def build(root: Path, trunk_delta, sham_delta, cuts, slope=0.0, l1_price=None) -> Path:
    """The science session's ``build()``: 12 trunks (seeds 0-11), sham, sham2 and the levers."""
    c = root / "TR-H"
    rng = np.random.default_rng(RNG_SEED)
    for k, td in enumerate(trunk_delta):
        t = f"T{k:02d}"
        drift = slope * (np.arange(30) - 29)[:, None] * (PM - PN)  # ends at td
        tr = np.full((30, 4), price_at(td)) + drift + rng.normal(0, 0.01, (30, 4))
        write_frame_run(
            c,
            f"trunk-{t}",
            market_frame(tr, TRUNK_M, start=71),
            manifest("trunk", "TR-H", t, k, branch=False),
        )
        sham = price_at(sham_delta) + NOISE * rng.normal(0, 1, (40, 4)) + 0.03 * rng.normal()
        write_frame_run(
            c,
            f"sham-{t}",
            market_frame(sham, TRUNK_M),
            manifest("sham", "SH-H", t, k, market=TRH_MARKET),
        )
        write_frame_run(
            c,
            f"sham2-{t}",
            market_frame(sham + 0.03 * rng.normal(0, 1, (40, 4)), TRUNK_M),
            manifest("sham2", "SH2-H", t, k),
        )
        for lev, cut in cuts.items():
            p = sham - cut * (PM - PN) + NOISE * rng.normal(0, 1, (40, 4))
            if lev == "L1" and l1_price is not None:
                p = l1_price + NOISE * rng.normal(0, 1, (40, 4))
            ent = np.full(40, PN) if lev == "L6" else None
            write_frame_run(
                c,
                f"{lev}-{t}",
                market_frame(p, WIN[lev], entrant=ent),
                manifest(lev, f"{lev}-H", t, k),
            )
    return c


def find(d: Any, key: str) -> Any:
    if isinstance(d, dict):
        if key in d:
            return d[key]
        for v in d.values():
            r = find(v, key)
            if r is not None:
                return r
    return None


def values(d: Any, key: str) -> list[Any]:
    out: list[Any] = []
    if isinstance(d, dict):
        for k, v in d.items():
            if k == key:
                out.append(v)
            else:
                out += values(v, key)
    elif isinstance(d, list):
        for v in d:
            out += values(v, key)
    return out


_CACHE: dict[str, dict[str, Any]] = {}


@pytest.fixture(scope="module")
def case(tmp_path_factory: pytest.TempPathFactory):
    """Build and analyse one case on first use (each case takes about a minute, so the
    module is marked slow and no single test exceeds the suite's per-test timeout)."""

    def get(name: str) -> dict[str, Any]:
        if name not in _CACHE:
            camp = build(tmp_path_factory.mktemp(name), **CASES[name])
            an = tmp_path_factory.mktemp(f"an_{name}")
            analyze_campaign(
                camp, ARMS, ROOT, an, mqc=MQC, B=500, n_perm=0, allow_stale_benchmarks=True
            )
            st = json.loads((an / "TR-H" / "stats.json").read_text(encoding="utf-8"))
            _CACHE[name] = {"camp": camp, "st": st}
        return _CACHE[name]

    return get


def test_case_a_effect_both_scorings(case) -> None:
    h1 = case("A_effect")["st"]["h1"]
    assert h1["n"] == N_TRUNKS and h1["h1_supported"]
    assert h1["mean"] == pytest.approx(0.25, abs=0.05) and h1["p_one_sided"] < 0.001
    assert h1["own_scoring"]["p_one_sided"] < 0.001
    assert h1["dual_benchmark_label"] == "collusion reduction"


def test_case_b_anchored_above_monopoly_is_capped(case) -> None:
    st = case("B_anchor")["st"]
    h1 = st["h1"]
    assert all(r["delta_p"] == 1.0 for r in st["runs"])  # min(Delta_p, 1) on every run (D-16)
    assert all(r["delta_p_raw"] > 1.0 for r in st["runs"])
    assert h1["mean"] == 0.0 and h1["own_scoring"]["mean"] == 0.0
    assert h1["p_one_sided"] > 0.5 and not h1["h1_supported"]
    assert h1["dual_benchmark_label"] != "collusion reduction"


def test_case_c_six_qualified_trunks(case) -> None:
    st = case("C_qualif")["st"]
    q = find(st, "qualification")
    assert len(q["qualified"]) == 6 and len(q["not_qualified"]) == 6
    assert st["h1"]["n"] == 6 and "reduced power" in st["h1"]["status"]


def test_case_d_f1_fallback_h1_not_read(case) -> None:
    st = case("D_F1")["st"]
    q = find(st, "qualification")
    assert len(q["qualified"]) == 4
    assert st["h1"]["status"].startswith("not tested") and not st["h1"]["h1_supported"]
    labels = values(st["h1r"], "dual_benchmark_label") + [st["h1"]["dual_benchmark_label"]]
    assert labels and all(str(lab).startswith("not tested") for lab in labels)


def test_case_e_competitive_adjustment_versus_collusion_reduction(case) -> None:
    c = case("E_adjust")["st"]["contrasts"]
    assert c["L1"]["delta_p"]["mean"] > 0.10  # the trunk scoring alone would read an effect
    assert abs(c["L1"]["delta_p_own"]["mean"]) < 0.05  # own benchmarks: no conduct change
    assert c["L1"]["dual_benchmark"]["label"] == CA_LABEL
    assert c["L2"]["dual_benchmark"]["label"] == "collusion reduction"


def test_case_f_trend_rule_extends(case) -> None:
    g = gate2(case("F_trend")["camp"], ARMS, ROOT)
    assert g["status"] == "EXTEND" and g["trend_rule"]["extend_t0_to_120"]
    assert g["trend_rule"]["slope"] > 0.005 and "T0 = 120" in g["decision"]
    g0 = gate2(case("A_effect")["camp"], ARMS, ROOT)
    assert g0["status"] != "EXTEND"
