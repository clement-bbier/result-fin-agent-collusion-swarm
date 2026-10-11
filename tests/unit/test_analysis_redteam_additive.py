"""Red-team additive rows: sham cache reuse (analysis script and the independent verify metric)
and the pinned benchmarks re-solved by verify's own logit solver."""

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"_rt_{name}", ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run(camp: Path, name: str, arm: str, rows: list[tuple[int, int, str, bool, str]]) -> Path:
    """A run directory with a manifest and an llm_decisions table; ``rows`` hold registered
    periods (written 0-based, A-008)."""
    d = camp / name
    (d / "llm_decisions").mkdir(parents=True)
    man = {"run_id": name, "arm": arm, "trunk_id": "T", "status": "ok", "branch_period": 100}
    (d / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
    pl.DataFrame(
        {
            "period": [r[0] - 1 for r in rows],
            "slot": [r[1] for r in rows],
            "prompt_hash": [r[2] for r in rows],
            "cache_hit": [r[3] for r in rows],
            "raw_output": [r[4] for r in rows],
        },
        schema={
            "period": pl.Int32,
            "slot": pl.Int8,
            "prompt_hash": pl.Utf8,
            "cache_hit": pl.Boolean,
            "raw_output": pl.Utf8,
        },
    ).write_parquet(d / "llm_decisions" / "part-0000.parquet")
    return d


@pytest.fixture
def camp(tmp_path: Path) -> Path:
    c = tmp_path / "TR-H"
    _run(
        c,
        "sham",
        "sham",
        [(101, 0, "h1", False, "s"), (101, 1, "h2", False, "s"), (102, 0, "h3", False, "s")],
    )
    _run(
        c,
        "lever",
        "L3",
        [
            (100, 0, "h0", True, "s"),  # pre-branch: outside the window
            (101, 0, "h1", True, "s"),  # served from the sham's entry
            (101, 1, "h2", True, "other"),  # a cache hit, not the sham's answer
            (102, 0, "h3", False, "x"),  # same prompt, fresh draw
            (102, 1, "h9", False, "y"),  # different prompt
            (102, 4, "h1", True, "s"),  # entrant slot: not an incumbent
        ],
    )
    return c


def test_verify_cache_reuse_counts_only_sham_answers(camp: Path) -> None:
    V = _load("verify")
    base = {
        "arm": "L3",
        "reference_runs": "TR-H/*",
        "reference_arm": "sham",
        "window": [101, 150],
        "members": [0, 1, 2, 3],
        "sap_filter": False,
    }
    runs = V.match_runs(camp.parent, "TR-H/*", base)
    assert [r.name for r in runs] == ["lever"]
    f = V.c_cache_reuse
    assert f(runs, {**base, "stat": "n"}, camp.parent) == 4
    assert f(runs, {**base, "stat": "n_from_ref"}, camp.parent) == 1
    assert f(runs, {**base, "stat": "share"}, camp.parent) == pytest.approx(0.25)
    assert f(runs, {**base, "stat": "share_hits"}, camp.parent) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="unknown stat"):
        f(runs, {**base, "stat": "mean"}, camp.parent)


def test_analysis_cache_reuse_matches_verify(camp: Path) -> None:
    R = _load("redteam_additive")
    st = {
        "runs": [
            {
                "registered_arm": "sham",
                "trunk_id": "T",
                "run_id": "sham",
                "path": str(camp / "sham"),
            },
            {
                "registered_arm": "L3",
                "trunk_id": "T",
                "run_id": "lever",
                "path": str(camp / "lever"),
            },
            {"registered_arm": "ghost", "trunk_id": "T", "run_id": "g", "path": "unused"},
        ]
    }
    runs = R.index_runs(st)
    assert set(runs) == {"sham", "L3"}
    out = R.cache_reuse(runs, ["T"])["L3"]
    assert out["n_decisions"] == 4
    assert out["n_from_sham"] == 1
    assert out["n_cache_hits"] == 2
    assert out["n_same_prompt_as_sham"] == 3
    assert out["n_same_prompt_and_output"] == 1
    assert out["share_from_sham"] == pytest.approx(0.25)
    assert out["all_trunks"]["n_from_sham"] == 1
    assert out["per_trunk"]["T"]["run_id"] == "lever"
    assert R.cache_reuse(runs, [])["L3"]["share_from_sham"] is None  # no qualified trunk
    st["runs"].append(dict(st["runs"][1], run_id="lever2"))
    with pytest.raises(ValueError, match="two L3 runs"):
        R.index_runs(st)


@pytest.mark.parametrize("market", ["trunk", "L1"])
def test_logit_closed_form_reproduces_pinned_benchmarks(market: str) -> None:
    V = _load("verify")
    path = "configs/analysis/benchmarks/TR-H.json"
    pinned = json.loads((ROOT / path).read_text(encoding="utf-8"))["markets"][market]
    for key in ("p_nash", "p_mono"):
        got = V.c_logit_closed_form([], {"market_file": path, "market": market, "key": key}, ROOT)
        assert got == pytest.approx(pinned["benchmarks"][key], abs=1e-6)
