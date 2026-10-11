"""A self-contained registry for the analysis tests.

The real ``configs/analysis/registered_arms.yaml`` pins ``benchmarks/<campaign>.json`` by
SHA-256 and the lead re-pins it when configurations move; tests must not depend on that
state. :func:`temp_registry` copies the registry into a temporary directory, solves every
market in closed form (the fringe market with the flow-diversion closed form, fast) and
re-pins the copy, so ``analyze_campaign`` / the gates run against a consistent, fresh pin.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from finswarm.analysis.arms import RegisteredArms, file_sha256
from finswarm.analysis.benchmarks_file import build_benchmark_file, dumps

ROOT = Path(__file__).resolve().parents[2]
ARMS = ROOT / "configs" / "analysis" / "registered_arms.yaml"


def temp_registry(tmp_dir: Path, root: Path = ROOT) -> Path:
    """Write ``<tmp_dir>/registered_arms.yaml`` with closed-form pins; returns its path."""
    data = yaml.safe_load(ARMS.read_text(encoding="utf-8"))
    for bf in data["benchmark_files"].values():
        for m in bf["markets"].values():
            m["method"] = "closed_form"
    spec = RegisteredArms.model_validate(data)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for pin in spec.benchmark_files:
        doc = build_benchmark_file(spec, pin, root)
        jpath = tmp_dir / f"{pin}.json"
        with open(jpath, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(dumps(doc))
        data["benchmark_files"][pin]["file"] = str(jpath)
        data["benchmark_files"][pin]["sha256"] = file_sha256(jpath)
    out = tmp_dir / "registered_arms.yaml"
    out.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return out
