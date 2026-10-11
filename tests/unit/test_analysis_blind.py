"""Blind mode (PREREG section 5): letters for the levers, key apart, leak check, unblind."""

import importlib.util
import json
import re
import string
from pathlib import Path

import numpy as np
import pytest
from _arms_fixture import ROOT, temp_registry
from _metrics_synth import market_frame, write_frame_run

from finswarm.analysis import blind
from finswarm.analysis.arms import load_registered_arms, resolve_arm_benchmarks
from finswarm.config.load import load_campaign_config

LEVERS = ("L1", "L2")
TOKEN = r"(?<![A-Za-z0-9_]){}(?![A-Za-z0-9_])"
TRH_MARKET = load_campaign_config(ROOT / "configs/sprint/tier0/TR-H.yaml").base.model_dump(
    mode="json"
)["market"]
MQC = {"levers": {"L1": {"status": "ok", "mean": 1.0}, "L2": {"status": "ok", "mean": 3.0}}}


def _load_analyze():
    spec = importlib.util.spec_from_file_location(
        "_script_analyze_blind", ROOT / "scripts" / "analyze.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _manifest(arm: str, label: str, trunk: str, seed: int, branch: bool = True) -> dict:
    m = {
        "arm": arm,
        "label": label,
        "trunk_id": trunk,
        "seed": seed,
        "branch_period": 100 if branch else None,
        "resolved_config": {"tags": {"arm": arm, "label": label}},
    }
    if arm == "sham":
        m["resolved_config"]["market"] = TRH_MARKET
    return m


@pytest.fixture(scope="module")
def reg(tmp_path_factory: pytest.TempPathFactory):
    """(registry path, spec, per-arm benchmarks) of a fresh closed-form pin."""
    arms = temp_registry(tmp_path_factory.mktemp("arms"))
    spec = load_registered_arms(arms)
    return arms, spec, resolve_arm_benchmarks(spec, ROOT)


@pytest.fixture(scope="module")
def campaign(tmp_path_factory: pytest.TempPathFactory, reg) -> Path:
    """Four qualified trunks with trunk, sham, sham2, L1 and L2 runs; the run ids carry the
    lever names on purpose (the synthetic layout is the worst case for a leak)."""
    _, _, ab = reg
    trunk_m = ab["sham"].primary_model
    c = tmp_path_factory.mktemp("camp") / "TR-H"
    rng = np.random.default_rng(0)
    for k in range(4):
        t = f"T{k}"
        tr = market_frame(np.full((30, 4), 1.90), trunk_m, start=71)
        write_frame_run(c, f"trunk-{t}", tr, _manifest("trunk", "TR-H", t, k, branch=False))
        sham = 1.85 + 0.02 * k + rng.normal(0, 0.01, (40, 4))
        write_frame_run(
            c, f"sham-{t}", market_frame(sham, trunk_m), _manifest("sham", "SH-H", t, k)
        )
        write_frame_run(
            c,
            f"sham2-{t}",
            market_frame(sham + 0.005, trunk_m),
            _manifest("sham2", "SH2-H", t, k),
        )
        for i, lev in enumerate(LEVERS):
            p = market_frame(sham - (0.25 - 0.1 * i + 0.01 * k), ab[lev].window_model)
            write_frame_run(c, f"{lev}-{t}", p, _manifest(lev, f"{lev}-H", t, k))
    bad = market_frame(np.full((40, 4), 1.6), ab["L2"].window_model)
    write_frame_run(c, "L2-bad", bad, _manifest("L2", "L2-H", "T9", 9), fail_every=5)
    return c


def test_letter_key_is_seeded_stable_and_reuses_existing() -> None:
    arms = ["sham", "L1", "L2", "L4a", "trunk"]
    k1 = blind.letter_key(arms)
    assert set(k1.values()) == {"L1", "L2", "L4a"} and set(k1) <= set(string.ascii_uppercase)
    assert blind.letter_key(arms) == k1
    k2 = blind.letter_key([*arms, "L6"], existing=k1)
    assert all(k2[letter] == arm for letter, arm in k1.items())
    assert "L6" in k2.values() and len(k2) == 4


@pytest.mark.timeout(400)  # two full analyses (blind + unblind) of a 9-trunk campaign
def test_blind_analysis_hides_levers_and_unblinds(campaign: Path, reg, tmp_path: Path) -> None:
    arms_path, spec, _ = reg
    analyze = _load_analyze()
    out_root = tmp_path / "analysis"
    mqc = tmp_path / "mqc.json"
    mqc.write_text(json.dumps(MQC), encoding="utf-8")
    common = [
        str(campaign),
        "--out-root",
        str(out_root),
        "--mqc",
        str(mqc),
        "--arms",
        str(arms_path),
        "--root",
        str(ROOT),
        "--B",
        "300",
        "--n-perm",
        "0",
    ]
    assert analyze.main([*common, "--blind"]) == 0
    out = out_root / "TR-H"
    stats_text = (out / "stats.json").read_text(encoding="utf-8")
    md = (out / "summary.md").read_text(encoding="utf-8")
    exc = json.loads((out / "exclusions.json").read_text(encoding="utf-8"))
    for lever in LEVERS:
        assert re.search(TOKEN.format(lever), stats_text) is None
        assert re.search(TOKEN.format(lever), md) is None
    st = json.loads(stats_text)
    letters = st["blind"]["letters"]
    assert len(letters) == 8  # the whole H1-R family is lettered, run or not
    assert set(st["arms"]) <= {"trunk", "sham", "sham2", *letters} and len(st["arms"]) == 5
    assert set(st["contrasts"]) - {"sham2"} <= set(letters) and len(st["contrasts"]) == 3
    assert st["h1"]["lever"] in letters and set(st["h1r"]["levers"]) == set(letters)
    assert set(st["inference"]["power"]["h1r_holm_worst_case"]) == set(letters)
    assert st["benchmarks"] == {
        "blinded": True,
        "n_arms": sum(a.benchmarks is not None for a in spec.arms.values()),
    }
    assert st["inference"]["primary_lever"] in letters
    lever_runs = [r for r in st["runs"] if r["registered_arm"] in letters]
    assert lever_runs and all(r["run_id"].startswith("blind:") for r in lever_runs)
    assert all(r["path"] is None and r["label"] in letters for r in lever_runs)
    assert all(not r["run_id"].startswith("blind:") for r in st["runs"] if r["arm"] == "sham")
    assert exc[0]["arm"] in letters and exc[0]["run_id"].startswith("blind:")
    assert "BLIND" in md and st["blind"]["key_sha256"] in md and "Mechanical" in md
    assert sorted(p.name for p in out.iterdir()) == ["exclusions.json", "stats.json", "summary.md"]

    key_file = Path(st["blind"]["key_path"])
    assert key_file == out_root / "_blind_keys" / "TR-H.key.json"
    key = blind.load_key(key_file)
    assert key["sha256"] == st["blind"]["key_sha256"]
    assert set(key["mapping"].values()) == set(spec.lever_family)
    assert {v["run_id"] for v in key["runs"].values()} >= {"L1-T0", "L2-T3", "L2-bad"}

    # the open analysis gives the same contrasts under the real names
    assert analyze.main([*common, "--out-root", str(tmp_path / "open")]) == 0
    open_st = json.loads((tmp_path / "open" / "TR-H" / "stats.json").read_text(encoding="utf-8"))
    for letter, arm in key["mapping"].items():
        if arm not in open_st["contrasts"]:
            assert letter not in st["contrasts"]
            continue
        assert st["contrasts"][letter]["delta_p"]["mean"] == pytest.approx(
            open_st["contrasts"][arm]["delta_p"]["mean"]
        )
    assert st["h1"]["mean"] == pytest.approx(open_st["h1"]["mean"])

    # re-running keeps the letters; unblind restores names, run ids and labels
    assert analyze.main([*common, "--blind"]) == 0
    again = json.loads((out / "stats.json").read_text(encoding="utf-8"))
    assert again["blind"] == st["blind"]
    rc = analyze.main(["--unblind", str(out / "stats.json"), "--key", str(key_file)])
    assert rc == 0
    opened = json.loads((out / "stats.unblinded.json").read_text(encoding="utf-8"))
    assert set(opened["arms"]) == {"trunk", "sham", "sham2", *LEVERS}
    assert opened["h1"]["lever"] == "L2" and set(opened["contrasts"]) == {"sham2", *LEVERS}
    assert {r["run_id"] for r in opened["runs"]} == {r["run_id"] for r in open_st["runs"]}
    assert opened["exclusions"][0] == {"run_id": "L2-bad", "arm": "L2", "reason": exc[0]["reason"]}
    swapped = {
        **key,
        "mapping": dict(zip(key["mapping"], reversed(list(key["mapping"].values())), strict=True)),
    }
    with pytest.raises(ValueError):
        blind.unblind_stats(out / "stats.json", swapped)


def test_leak_check_and_tampered_key(tmp_path: Path) -> None:
    with pytest.raises(blind.BlindLeak):
        blind.check_no_leak('{"arm": "L2"}', ["L2"])
    with pytest.raises(blind.BlindLeak):
        blind.check_no_leak('{"label": "L2-H"}', ["L2"])  # a hyphen is a token boundary
    blind.check_no_leak('{"path": "results/campaigns/L2x"}', ["L2"])
    blind.check_no_leak("B = 10000", ["CA"])
    kp = tmp_path / "k.json"
    blind.write_key(kp, "c", {"A": "L2"}, 1, ["sham"], {"blind:abc": {"run_id": "L2-T0"}})
    body = json.loads(kp.read_text(encoding="utf-8"))
    body["mapping"]["A"] = "L1"
    kp.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(ValueError):
        blind.load_key(kp)


def test_unblind_requires_key(tmp_path: Path) -> None:
    analyze = _load_analyze()
    assert analyze.main(["--unblind", str(tmp_path / "stats.json")]) == 2
