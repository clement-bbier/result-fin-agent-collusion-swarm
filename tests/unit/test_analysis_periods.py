"""A-008 period convention: the store numbers periods from 0, the registration from 1.
``analysis.io.read_table`` (and, independently, ``scripts/verify.py``'s loader) map store
periods to registered periods in one place, so the registered windows (trunk 71-100, branch
111-150, fish 251-300) cover the store's 70..99, 110..149 and 250..299 in full."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from _metrics_synth import market_frame, write_frame_run

from finswarm.analysis import io as IO
from finswarm.analysis.arms import load_registered_arms
from finswarm.metrics._frames import window
from finswarm.metrics.benchmarks import MarketModel
from finswarm.store.schemas import AGENT_PERIOD

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT.parents[2] if ROOT.parent.name == "worktrees" else ROOT
REAL = next((p for p in (ROOT, REPO) if (p / "results" / "campaigns" / "TR-H").is_dir()), None)
SPEC = load_registered_arms(ROOT / "configs" / "analysis" / "registered_arms.yaml")


def _verify():
    spec = importlib.util.spec_from_file_location("_verify_periods", ROOT / "scripts" / "verify.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _raw_run(root: Path, name: str, n_periods: int, manifest: dict) -> Path:
    """A run written the way the engine writes it: periods 0..n_periods-1."""
    d = root / name
    (d / "agent_period").mkdir(parents=True)
    rows = {c: [] for c in ("period", "slot", "price")}
    for t in range(n_periods):
        for s in range(4):
            rows["period"].append(t)
            rows["slot"].append(s)
            rows["price"].append(1.8)
    pl.DataFrame(rows).write_parquet(d / "agent_period" / "part-0000.parquet")
    (d / "manifest.json").write_text(json.dumps({"run_id": name, **manifest}), encoding="utf-8")
    return d


def test_registered_windows_cover_the_store_periods_in_full(tmp_path: Path) -> None:
    trunk = _raw_run(tmp_path, "trunk", 100, {"arm": "trunk", "branch_period": None})
    branch = _raw_run(tmp_path, "sham", 150, {"arm": "sham", "branch_period": 100})
    pc = _raw_run(tmp_path, "pc", 300, {"arm": "fish_exact"})
    ap = IO.read_table(trunk, "agent_period")
    assert ap["period"].min() == 1 and ap["period"].max() == 100  # registered coordinates
    assert ap["period"].dtype == AGENT_PERIOD["period"]
    s, e = SPEC.window_of("trunk")
    assert (s, e) == (71, 100) and window(ap, s, e)["period"].n_unique() == 30
    assert window(ap, s, e)["period"].min() == 71
    bp = IO.read_table(branch, "agent_period")
    s, e = SPEC.window_of("sham")
    assert (s, e) == (111, 150) and window(bp, s, e)["period"].n_unique() == 40
    assert bp.filter(pl.col("period") > 100)["period"].n_unique() == 50  # post-branch = 101-150
    pp = IO.read_table(pc, "agent_period")
    s, e = SPEC.window_of("PC-H")
    assert (s, e) == (251, 300) and window(pp, s, e)["period"].n_unique() == 50
    assert IO.registered_period(0) == 1 and IO.store_period(71) == 70
    # the independent verifier applies the same offset
    V = _verify()
    d = V.load(trunk, "agent_period", ["period", "slot", "price"])
    assert d["period"].min() == 1 and d["period"].max() == 100
    assert np.unique(d["period"][(d["period"] >= 71) & (d["period"] <= 100)]).size == 30
    # the synthetic writer stores 0-based and reads back as built
    model = MarketModel(n_incumbents=4)
    r = write_frame_run(
        tmp_path / "syn",
        "r",
        market_frame(np.full((30, 4), 1.8), model, start=71),
        {"arm": "trunk", "branch_period": None},
    )
    raw = pl.read_parquet(r / "agent_period" / "part-0000.parquet")
    assert raw["period"].min() == 70 and raw["period"].max() == 99
    back = IO.read_table(r, "agent_period")
    assert back["period"].min() == 71 and back["period"].max() == 100


@pytest.mark.skipif(REAL is None, reason="no results/campaigns/TR-H on this machine")
def test_real_store_trunk_and_sham_windows_are_complete() -> None:
    camp = REAL / "results" / "campaigns" / "TR-H"
    runs = [p for p in camp.iterdir() if p.is_dir() and (p / "manifest.json").exists()]
    by_arm: dict[str, Path] = {}
    for p in runs:
        man = json.loads((p / "manifest.json").read_text(encoding="utf-8"))
        if man.get("status") in (None, "ok"):
            by_arm.setdefault(str(man.get("arm")), p)
    if "trunk" not in by_arm or "sham" not in by_arm:
        pytest.skip("TR-H holds no finished trunk and sham run yet")
    parts = sorted((by_arm["trunk"] / "agent_period").rglob("*.parquet"))  # one part per chunk
    raw = pl.concat([pl.read_parquet(f) for f in parts], how="diagonal_relaxed")
    assert raw["period"].min() == 0 and raw["period"].max() == 99  # the store is 0-based
    ap = IO.read_table(by_arm["trunk"], "agent_period")
    s, e = SPEC.window_of("trunk")
    assert window(ap, s, e)["period"].n_unique() == e - s + 1 == 30
    sh = IO.read_table(by_arm["sham"], "agent_period")
    s, e = SPEC.window_of("sham")
    assert sh["period"].max() == 150 and window(sh, s, e)["period"].n_unique() == 40
    ev = IO.read_table(by_arm["sham"], "events")
    if ev.height:
        assert ev["period"].min() == 101  # the branch event sits on the first branch period
    dec = IO.read_table(by_arm["sham"], "llm_decisions")
    assert dec.filter(pl.col("period") > 100)["period"].n_unique() == 50
