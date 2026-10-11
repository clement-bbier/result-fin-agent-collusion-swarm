import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from _metrics_synth import write_run

from finswarm.metrics.benchmarks import logit_closed_form, logit_profits

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        f"_script_{name}", ROOT / "scripts" / f"{name}.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _delta(p: float) -> float:
    b = logit_closed_form()
    return float(b.delta(logit_profits([p, p])[0]))


@pytest.fixture()
def campaign(tmp_path: Path) -> Path:
    camp = tmp_path / "runs" / "A2-test"
    for k in range(4):
        sham = np.full((40, 2), 1.70 + 0.01 * k)
        lever = np.full((40, 2), 1.55 + 0.01 * k)
        write_run(camp / "sham", f"sham-T{k}", sham, {"arm": "sham", "trunk_id": f"T{k}"})
        write_run(camp / "L1", f"L1-T{k}", lever, {"arm": "L1", "trunk_id": f"T{k}"})
    # a run with 20 % failed decisions is excluded by the SAP rule (> 10 %)
    write_run(
        camp / "L1", "L1-bad", np.full((40, 2), 1.5), {"arm": "L1", "trunk_id": "T9"}, fail_every=5
    )
    return camp


def test_verify_passes_and_fails(campaign: Path, tmp_path: Path) -> None:
    verify = _load("verify")
    bench = {"a": 2.0, "a0": 0.0, "mu": 0.25, "c": 1.0, "alpha": 1.0, "beta": 1.0}
    expect_dc = np.mean([_delta(1.70 + 0.01 * k) - _delta(1.55 + 0.01 * k) for k in range(4)])
    claims = {
        "_comment": "ignored",
        "sham_delta": {
            "metric": "delta",
            "runs": "runs/A2-test/sham/*",
            "params": {"window": [21, 40], "bench": bench},
            "expected": float(np.mean([_delta(1.70 + 0.01 * k) for k in range(4)])),
            "tolerance": 1e-6,
        },
        "dc": {
            "metric": "paired_contrast",
            "runs": "runs/A2-test/sham/*",
            "params": {
                "treatment_runs": "runs/A2-test/L1/L1-T*",
                "window": [21, 40],
                "bench": bench,
            },
            "expected": expect_dc,
            "tolerance": 1e-6,
        },
        "n_sham": {
            "metric": "n_runs",
            "runs": "runs/A2-test/sham/*",
            "expected": 4,
            "tolerance": 0,
        },
        "undercut": {
            "metric": "undercut_rate",
            "runs": "runs/A2-test/sham/*",
            "expected": 0.0,
            "tolerance": 0.0,
        },
        "fails": {
            "metric": "parse_fail_rate",
            "runs": "runs/A2-test/L1/L1-bad",
            "expected": 0.2,
            "tolerance": 1e-9,
        },
    }
    numbers = tmp_path / "numbers.json"
    numbers.write_text(json.dumps(claims), encoding="utf-8")
    assert verify.main([str(numbers), "--root", str(tmp_path)]) == 0

    claims["sham_delta"]["expected"] += 0.01
    numbers.write_text(json.dumps(claims), encoding="utf-8")
    assert verify.main([str(numbers), "--root", str(tmp_path)]) == 1

    claims["sham_delta"]["runs"] = "runs/none/*"
    numbers.write_text(json.dumps(claims), encoding="utf-8")
    assert verify.main([str(numbers), "--root", str(tmp_path)]) == 2


def test_verify_is_independent_and_exits_nonzero(campaign: Path, tmp_path: Path) -> None:
    src = (ROOT / "scripts" / "verify.py").read_text(encoding="utf-8")
    assert "finswarm" not in src.replace("``finswarm``", "")
    numbers = tmp_path / "bad.json"
    numbers.write_text(
        json.dumps(
            {
                "x": {
                    "metric": "n_runs",
                    "runs": "runs/A2-test/sham/*",
                    "expected": 5,
                    "tolerance": 0,
                }
            }
        ),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "verify.py"),
            str(numbers),
            "--root",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 1 and "FAIL" in proc.stdout


@pytest.mark.slow
def test_make_figures_demo(tmp_path: Path) -> None:
    mk = _load("make_figures")
    assert mk.main(["--demo", str(tmp_path / "demo")]) == 0
    assert len(list((tmp_path / "demo").glob("*.png"))) == 5
    assert len(list((tmp_path / "demo").glob("*.pdf"))) == 5
