"""Swarm throughput bench (F8 provenance): modes, Tier 0 profile, ``<sha>.json`` document with
a flat ``ticks_per_s``; the throughput target itself is the ``slow`` test."""

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load():
    spec = importlib.util.spec_from_file_location(
        "bench_swarm", ROOT / "scripts" / "bench_swarm.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


SMALL = ["--agents", "300", "--ticks", "300", "--ticks-per-period", "100", "--min-ticks-per-s", "1"]


def test_tier0_profile_is_the_registered_trunk_market() -> None:
    mod = _load()
    p = mod.tier0_profile()
    assert p.name == "tier0" and p.source == "configs/sprint/tier0/TR-H.yaml"
    assert p.agents == 10_000 and p.arrivals_per_tick == pytest.approx(1.0)  # lam_mean 1e-4
    assert p.ticks_per_period == 100 and p.tick == 0.05 and len(p.asks) == 4
    small = mod.tier0_profile(agents=500)
    assert small.arrivals_per_tick == pytest.approx(1.0)  # the rate per tick is kept
    assert small.lam_mean == pytest.approx(1.0 / 500)
    legacy = mod.legacy_profile(10_000, 20.0, 100)
    assert legacy.asks == (1.50, 1.70) and legacy.tick == 0.01


def test_both_modes_write_one_provenance_document(tmp_path: Path, capsys) -> None:
    mod = _load()
    rc = mod.main([*SMALL, "--mode", "both", "--tier0", "--out-dir", str(tmp_path)])
    assert rc == 0
    files = sorted(tmp_path.glob("*.json"))
    assert [f.name for f in files if f.name != "latest.json"] and (
        tmp_path / "latest.json"
    ).exists()
    (named,) = [f for f in files if f.name != "latest.json"]
    assert re.fullmatch(r"([0-9a-f]{12}(-dirty)?|nogit)\.json", named.name), named.name
    doc = json.loads(named.read_text(encoding="utf-8"))
    assert doc == json.loads((tmp_path / "latest.json").read_text(encoding="utf-8"))
    assert doc["schema_version"] == 1 and doc["sha"] and "generated_at" in doc
    assert doc["dirty"] in (True, False, None) and isinstance(doc["numba"], bool)
    assert set(doc["modes"]) == {"dealer", "fringe"} and doc["mode"] == "both"
    assert doc["profile"]["name"] == "tier0" and doc["profile"]["agents"] == 300
    assert doc["modes"]["dealer"]["fringe_share"] == 0.0
    assert doc["modes"]["fringe"]["fringe_share"] == 0.3
    for b in doc["modes"].values():
        assert b["ticks"] == 300 and b["ticks_per_s"] > 0 and b["warmup_seconds"] >= 0
        assert {"trades", "trades_per_s", "fringe_trades", "seconds"} <= set(b)
    # the flat figure is the slowest mode's, so a single threshold guards both mechanisms
    assert doc["ticks_per_s"] == min(b["ticks_per_s"] for b in doc["modes"].values())
    assert doc["pass"] is True and doc["threshold_ticks_per_s"] == 1.0
    out = capsys.readouterr().out
    assert "dealer (tier0: 300 agents" in out and "fringe (tier0" in out and "PASS" in out


def test_single_mode_keeps_the_flat_layout_and_threshold(tmp_path: Path) -> None:
    mod = _load()
    out = tmp_path / "fringe.json"
    rc = mod.main([*SMALL, "--mode", "fringe", "--out", str(out), "--no-latest"])
    assert rc == 0 and not (tmp_path / "latest.json").exists()
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["mode"] == "fringe" and list(doc["modes"]) == ["fringe"]
    assert doc["profile"]["name"] == "legacy" and doc["agents"] == 300
    assert doc["ticks_per_s"] == doc["modes"]["fringe"]["ticks_per_s"]
    for key in ("agents", "ticks", "seconds", "ticks_per_s", "trades", "trades_per_s", "pass"):
        assert key in doc  # the historical flat keys
    # an unreachable threshold fails the exit code but still writes the document
    rc = mod.main([*SMALL[:-2], "--min-ticks-per-s", "1e12", "--mode", "dealer", "--out", str(out)])
    assert rc == 1 and json.loads(out.read_text(encoding="utf-8"))["pass"] is False


@pytest.mark.slow
def test_bench_swarm_meets_throughput_target(tmp_path: Path) -> None:
    mod = _load()
    out = tmp_path / "latest.json"
    code = mod.main(
        [
            "--agents",
            "10000",
            "--ticks",
            "100000",
            "--arrivals",
            "20",
            "--mode",
            "dealer",
            "--out",
            str(out),
        ]
    )
    res = json.loads(out.read_text(encoding="utf-8"))
    assert code == 0, res
    assert res["ticks_per_s"] >= 10_000


@pytest.mark.slow
def test_tier0_profile_meets_the_f6_floor(tmp_path: Path) -> None:
    """The registered market, both mechanisms, must stay above 25k ticks/s (F6 condition)."""
    mod = _load()
    out = tmp_path / "tier0.json"
    code = mod.main(
        [
            "--tier0",
            "--mode",
            "both",
            "--ticks",
            "100000",
            "--min-ticks-per-s",
            "25000",
            "--out",
            str(out),
        ]
    )
    res = json.loads(out.read_text(encoding="utf-8"))
    assert code == 0, {m: b["ticks_per_s"] for m, b in res["modes"].items()}
