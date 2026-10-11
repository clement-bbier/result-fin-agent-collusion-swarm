"""``paper/numbers.json`` producer (SAP section 12).

``configs/verify/claims.yaml`` lists, per paper claim, the verify rows: the metric
``scripts/verify.py`` recomputes from the raw logs, the run glob and parameters, and the
``source`` (analysis output file + dotted path) that gives the EXPECTED value. This module
expands the placeholders (``{out}`` analysis out-root, ``{camp}`` campaigns root, named
windows, ``{bench:<pin>:<market>}`` pinned benchmark values from the registry), reads the
expected values from the analysis outputs that exist and writes the JSON ``verify.py``
consumes. A row whose source is not available yet is written with ``expected: null`` and
verifies as ``pending``; nothing is ever typed by hand.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml

from finswarm.analysis.arms import (
    BENCH_KEYS,
    DEFAULT_ARMS_PATH,
    RegisteredArms,
    load_pinned,
    load_registered_arms,
)

DEFAULT_CLAIMS_PATH = Path("configs/verify/claims.yaml")
DEFAULT_NUMBERS_PATH = Path("paper/numbers.json")
_BENCH_RE = re.compile(r"^\{bench:([^:}]+):([^}]+)\}$")


def dotted_get(doc: Any, path: str) -> Any:
    """``a.b.0.c`` into dicts and lists; ``None`` when any step is missing."""
    cur = doc
    for k in path.split("."):
        if isinstance(cur, dict):
            if k not in cur:
                return None
            cur = cur[k]
        elif isinstance(cur, list) and k.lstrip("-").isdigit() and -len(cur) <= int(k) < len(cur):
            cur = cur[int(k)]
        else:
            return None
    return cur


class _Ctx:
    def __init__(
        self,
        out_root: Path,
        campaigns_root: Path,
        spec: RegisteredArms | None,
        root: Path,
        windows: dict[str, Any],
    ) -> None:
        self.out = out_root
        self.camp = campaigns_root
        self.spec = spec
        self.root = root
        self.windows = windows
        self._pins: dict[str, dict[str, Any]] = {}
        self._docs: dict[str, Any] = {}

    def expand(self, s: str) -> str:
        return s.replace("{out}", str(self.out).replace("\\", "/")).replace(
            "{camp}", str(self.camp).replace("\\", "/")
        )

    def bench(self, pin: str, market: str) -> dict[str, float]:
        if self.spec is None:
            raise ValueError("benchmark placeholders need the registered arms")
        if pin not in self._pins:
            self._pins[pin] = load_pinned(self.spec, pin, self.root)
        entry = self._pins[pin]["markets"][market]["benchmarks"]
        return {k: float(entry[k]) for k in BENCH_KEYS}

    def doc(self, path: str) -> Any:
        p = Path(self.expand(path))
        if not p.is_absolute():
            p = self.root / p
        key = str(p)
        if key not in self._docs:
            self._docs[key] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
        return self._docs[key]


def resolve_params(params: dict[str, Any], ctx: _Ctx) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in params.items():
        if isinstance(v, str):
            m = _BENCH_RE.match(v)
            if m:
                out[k] = ctx.bench(m.group(1), m.group(2))
                continue
            if k == "window" and v in ctx.windows:
                out[k] = list(ctx.windows[v])
                continue
            out[k] = ctx.expand(v)
        elif isinstance(v, dict):
            out[k] = resolve_params(v, ctx)
        else:
            out[k] = v
    return out


def _number(v: Any) -> float | None:
    if v is None or isinstance(v, bool):
        return None if v is None else float(v)
    if isinstance(v, int | float):
        return float(v)
    return None


def build_numbers(
    claims_path: str | Path = DEFAULT_CLAIMS_PATH,
    out_root: str | Path = "results/analysis",
    campaigns_root: str | Path = "results/campaigns",
    arms_path: str | Path | None = DEFAULT_ARMS_PATH,
    root: str | Path = ".",
) -> dict[str, Any]:
    """The ``numbers.json`` document: ``claim_id -> {claim, metric, runs, params, expected,
    tolerance, source}`` plus a ``_meta`` entry (counts of filled and pending rows)."""
    cfg = yaml.safe_load(Path(claims_path).read_text(encoding="utf-8"))
    spec = load_registered_arms(arms_path) if arms_path and Path(arms_path).exists() else None
    ctx = _Ctx(Path(out_root), Path(campaigns_root), spec, Path(root), cfg.get("windows", {}))
    tol_default = float(cfg.get("tolerance_default", 0.005))
    out: dict[str, Any] = {}
    filled = pending = 0
    for cid, row in cfg["claims"].items():
        params = resolve_params(dict(row.get("params", {})), ctx)
        expected: float | None = None
        src = row.get("source")
        if "expected" in row:
            expected = _number(row["expected"])
        elif src:
            doc = ctx.doc(src["file"])
            expected = _number(dotted_get(doc, src["path"])) if doc is not None else None
        if expected is None:
            pending += 1
        else:
            filled += 1
        out[cid] = {
            "claim": row.get("claim"),
            "metric": row["metric"],
            "runs": ctx.expand(row["runs"]),
            "params": params,
            "expected": expected,
            "tolerance": float(row.get("tolerance", tol_default)),
            "source": {"file": ctx.expand(src["file"]), "path": src["path"]} if src else None,
        }
    out["_meta"] = {
        "claims": str(claims_path),
        "out_root": str(out_root),
        "campaigns_root": str(campaigns_root),
        "filled": filled,
        "pending": pending,
        "note": "expected values copied from the registered analysis outputs; verify.py "
        "re-derives them from the raw logs without finswarm code",
    }
    return out


def write_numbers(doc: dict[str, Any], path: str | Path = DEFAULT_NUMBERS_PATH) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(doc, indent=2) + "\n")
    return p
