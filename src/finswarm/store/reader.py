"""Read run and campaign outputs as lazy polars frames, validating dtypes against the schemas."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl

from finswarm.store.schemas import TABLES
from finswarm.store.writer import PART_GLOB, part_index

MANIFEST_FILE = "manifest.json"


class SchemaMismatchError(ValueError):
    pass


@dataclass
class RunData:
    run_dir: Path
    manifest: dict[str, Any]
    tables: dict[str, pl.LazyFrame] = field(default_factory=dict)

    @property
    def run_id(self) -> str:
        return str(self.manifest.get("run_id", self.run_dir.name))

    def frame(self, table: str) -> pl.DataFrame:
        return self.tables[table].collect()


def table_parts(run_dir: Path, table: str) -> list[Path]:
    tdir = Path(run_dir) / table
    if not tdir.exists():
        return []
    return sorted(tdir.glob(PART_GLOB), key=part_index)


def _empty(table: str) -> pl.LazyFrame:
    return pl.DataFrame(schema=TABLES[table]).lazy()


def validate_schema(table: str, schema: pl.Schema | dict[str, Any], where: str = "") -> None:
    expected = TABLES[table]
    got = dict(schema)
    if list(got) != list(expected) or any(got[k] != v for k, v in expected.items()):
        raise SchemaMismatchError(f"{where}{table}: expected {expected}, got {got}")


def scan_table(run_dir: Path, table: str, *, validate: bool = True) -> pl.LazyFrame:
    parts = table_parts(run_dir, table)
    if not parts:
        return _empty(table)
    if validate:
        for p in parts:
            validate_schema(table, pl.read_parquet_schema(p), where=f"{p}: ")
    return pl.scan_parquet([str(p) for p in parts])


def read_manifest(run_dir: Path) -> dict[str, Any]:
    path = Path(run_dir) / MANIFEST_FILE
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_run(run_dir: str | Path, *, validate: bool = True) -> RunData:
    """Lazy frames for every table of one run (empty frames with the schema if absent)."""
    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)
    tables = {t: scan_table(run_dir, t, validate=validate) for t in TABLES}
    return RunData(run_dir=run_dir, manifest=read_manifest(run_dir), tables=tables)


@dataclass
class CampaignData:
    root: Path
    runs: list[RunData]
    tables: dict[str, pl.LazyFrame]
    manifests: pl.DataFrame


def find_run_dirs(root: str | Path, *, include_ghosts: bool = False) -> list[Path]:
    root = Path(root)
    out = []
    for m in sorted(root.rglob(MANIFEST_FILE)):
        rel = m.parent.relative_to(root).parts
        if not include_ghosts and "ghost" in rel:
            continue
        out.append(m.parent)
    return out


def load_campaign(root: str | Path, *, include_ghosts: bool = False) -> CampaignData:
    """Concatenate every run under ``root`` (one lazy frame per table) plus a manifest index."""
    root = Path(root)
    runs = [load_run(d) for d in find_run_dirs(root, include_ghosts=include_ghosts)]
    tables: dict[str, pl.LazyFrame] = {}
    for t in TABLES:
        frames = [r.tables[t] for r in runs]
        tables[t] = pl.concat(frames, how="vertical") if frames else _empty(t)
    rows = [
        {
            "run_id": r.run_id,
            "name": str(r.manifest.get("name", "")),
            "seed": int(r.manifest.get("seed", -1)),
            "status": str(r.manifest.get("status", "")),
            "parent_run_id": r.manifest.get("parent_run_id"),
            "branch_period": r.manifest.get("branch_period"),
            "run_dir": str(r.run_dir),
        }
        for r in runs
    ]
    manifests = pl.DataFrame(
        rows,
        schema={
            "run_id": pl.Utf8,
            "name": pl.Utf8,
            "seed": pl.Int64,
            "status": pl.Utf8,
            "parent_run_id": pl.Utf8,
            "branch_period": pl.Int64,
            "run_dir": pl.Utf8,
        },
    )
    return CampaignData(root=root, runs=runs, tables=tables, manifests=manifests)
