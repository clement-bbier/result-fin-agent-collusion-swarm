"""Read run directories written by the store (Parquet parts per table + manifest).

Assumed layout (tolerant): a run directory holds ``manifest.json`` and, per table, either
``<table>/<part>.parquet`` (any depth below the table directory) or ``<table>*.parquet``
files directly inside the run directory. Runs are the direct children of a campaign
directory; ghost sub-runs (``<run>/ghost/<k>/``) are discovered separately and are never
read as part of, or pooled with, their parent.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from finswarm.store.schemas import TABLES


def table_files(run_dir: str | Path, table: str) -> list[Path]:
    d = Path(run_dir)
    files: list[Path] = []
    tdir = d / table
    if tdir.is_dir():
        files.extend(sorted(tdir.rglob("*.parquet")))
    files.extend(sorted(p for p in d.glob(f"{table}*.parquet") if p.is_file()))
    return files


#: A-008 period convention. The store (engine) numbers periods from 0: a trunk holds
#: 0..99, a branch 0..149, a positive control 0..299, and ``branch_period: 100`` is the
#: first period played by the branch. The pre-registration, the registry windows (trunk
#: [71, 100], branch [111, 150], fish [251, 300]), the trend window 81-100 and every report
#: count periods from 1. ``read_table`` is the ONE place where store periods become
#: registered periods (``period + STORE_PERIOD_OFFSET``); everything downstream (windows,
#: ``branch_period`` arithmetic, labels) is in registered coordinates, so the trunk window
#: 71-100 covers the store's 70..99 (30 periods) and the branch window 111-150 its
#: 110..149 (40). ``scripts/verify.py`` applies the same offset independently.
STORE_PERIOD_OFFSET = 1


def registered_period(store_period: int) -> int:
    """Registered (1-based) period of a store (0-based) period."""
    return int(store_period) + STORE_PERIOD_OFFSET


def store_period(registered: int) -> int:
    """Store (0-based) period of a registered (1-based) period."""
    return int(registered) - STORE_PERIOD_OFFSET


def read_table(run_dir: str | Path, table: str) -> pl.DataFrame:
    """Concatenate a table's parts and cast to the frozen schema (missing columns -> null);
    the ``period`` column is returned in REGISTERED coordinates (store period + 1, A-008)."""
    schema = TABLES[table]
    parts = [pl.read_parquet(p) for p in table_files(run_dir, table)]
    if not parts:
        return pl.DataFrame(schema=schema)
    df = pl.concat(parts, how="diagonal_relaxed")
    cols = []
    for name, dtype in schema.items():
        if name in df.columns:
            col = pl.col(name).cast(dtype, strict=False)
            if name == "period":
                col = (col + STORE_PERIOD_OFFSET).cast(dtype)
            cols.append(col)
        else:
            cols.append(pl.lit(None, dtype=dtype).alias(name))
    extra = [c for c in df.columns if c not in schema]
    return df.select(cols + extra)


def read_manifest(run_dir: str | Path) -> dict[str, Any]:
    p = Path(run_dir) / "manifest.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def find_key(d: Any, key: str) -> Any:
    """First value stored under ``key`` anywhere in a nested dict/list (depth-first)."""
    if isinstance(d, dict):
        if key in d:
            return d[key]
        for v in d.values():
            r = find_key(v, key)
            if r is not None:
                return r
    elif isinstance(d, list):
        for v in d:
            r = find_key(v, key)
            if r is not None:
                return r
    return None


def run_info(run_dir: str | Path) -> dict[str, Any]:
    """Run id, arm, label, trunk (pairing key), seed, lane and ghost tags of a run."""
    d = Path(run_dir)
    m = read_manifest(d)
    run_id = m.get("run_id") or d.name
    arm = (
        m.get("arm")
        or m.get("branch_name")
        or m.get("condition")
        or m.get("label")
        or find_key(m, "arm")
        or "unknown"
    )
    trunk = (
        m.get("trunk_id") or m.get("trunk") or m.get("parent_run_id") or m.get("parent") or run_id
    )
    tags = (m.get("resolved_config") or {}).get("tags") or m.get("tags") or {}
    return {
        "run_id": str(run_id),
        "arm": str(arm),
        "label": str(m.get("label") or tags.get("label") or ""),
        "trunk_id": str(trunk),
        "seed": m.get("seed", find_key(m, "seed")),
        "lane": m.get("lane", find_key(m, "lane")),
        "alpha": find_key(m, "alpha"),
        "branch_period": m.get("branch_period"),
        "ghost_of": tags.get("ghost_of"),
        "ghost_slot": tags.get("ghost_slot"),
        "path": str(d),
    }


def is_run_dir(d: Path) -> bool:
    """A directory holding a manifest or an ``agent_period`` table."""
    if not d.is_dir():
        return False
    if (d / "manifest.json").exists() or (d / "agent_period").is_dir():
        return True
    return any(p.is_file() for p in d.glob("agent_period*.parquet"))


def discover_runs(campaign_dir: str | Path) -> list[Path]:
    """Top-level run directories of a campaign (direct children only).

    Nested directories (ghost sub-runs under ``<run>/ghost/<k>/``, checkpoints) are never
    returned as runs of their own: ghosts are collected explicitly by
    :func:`discover_ghosts`, so they can never be pooled with their parent's arm.
    """
    root = Path(campaign_dir)
    if is_run_dir(root) and not any(is_run_dir(c) for c in root.iterdir() if c.is_dir()):
        return [root]
    return sorted(c for c in root.iterdir() if c.name != "ghost" and is_run_dir(c))


def discover_ghosts(run_dir: str | Path) -> list[Path]:
    """Ghost sub-runs of one run (``<run>/ghost/<k>/``, written by ``sim.ghost``)."""
    g = Path(run_dir) / "ghost"
    return sorted(c for c in g.iterdir() if is_run_dir(c)) if g.is_dir() else []
