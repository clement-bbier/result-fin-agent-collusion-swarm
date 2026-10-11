"""Buffered, atomic Parquet writer: one part per table per flushed period range.

Layout: ``run_dir/<table>/part-NNNNN.parquet`` (dtypes from ``finswarm.store.schemas``) and an
optional ``run_dir/prompts.jsonl`` sidecar. Every file is written to a temporary name in the
same directory and renamed into place, so a crash never leaves a partial part behind.

The runner flushes exactly at checkpoint boundaries, so part boundaries line up with
checkpoints; a resumed run deletes parts numbered at or above the checkpoint's counters and
re-writes them byte-identically.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import polars as pl

from finswarm.store.schemas import TABLES

PART_GLOB = "part-*.parquet"
PROMPTS_FILE = "prompts.jsonl"


def part_name(index: int) -> str:
    return f"part-{index:05d}.parquet"


def part_index(path: Path) -> int:
    return int(path.stem.split("-")[1])


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with open(tmp, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_parquet(df: pl.DataFrame, path: Path) -> None:
    """Write ``df`` to ``path`` through a temporary file + rename (no partial files)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        df.write_parquet(tmp, compression="zstd", statistics=True)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _coerce(value: Any, dtype: pl.DataType) -> Any:
    if value is None:
        return None
    if dtype == pl.Float64 or dtype == pl.Float32:
        return float(value)
    if dtype == pl.Boolean:
        return bool(value)
    if dtype == pl.Utf8:
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return None
        return int(value)
    return int(value)


def frame_from_rows(table: str, rows: Iterable[dict[str, Any]]) -> pl.DataFrame:
    """Build a frame with the exact frozen schema; missing keys become nulls."""
    schema = TABLES[table]
    rows = list(rows)
    unknown = {k for r in rows for k in r} - set(schema)
    if unknown:
        raise KeyError(f"{table}: unknown columns {sorted(unknown)}")
    cols = {name: [_coerce(r.get(name), dt) for r in rows] for name, dt in schema.items()}
    return pl.DataFrame(cols, schema=schema)


def frame_from_columns(table: str, cols: dict[str, Any]) -> pl.DataFrame:
    """Build a frame with the exact frozen schema from columns (vectorised fast path).

    Values are numpy arrays / sequences of one common length, or scalars broadcast to it;
    missing columns become nulls. Integer columns are cast strictly (no silent wrap-around).
    """
    schema = TABLES[table]
    unknown = set(cols) - set(schema)
    if unknown:
        raise KeyError(f"{table}: unknown columns {sorted(unknown)}")
    lengths = {len(v) for v in cols.values() if not _is_scalar(v)}
    if len(lengths) > 1:
        raise ValueError(f"{table}: columns of different lengths {sorted(lengths)}")
    n = lengths.pop() if lengths else 1
    out: dict[str, pl.Series] = {}
    for name, dt in schema.items():
        v = cols.get(name)
        if v is None:
            out[name] = pl.Series(name, [None] * n, dtype=dt)
        elif _is_scalar(v):
            out[name] = pl.Series(name, [_coerce(v, dt)] * n, dtype=dt)
        else:
            out[name] = pl.Series(name, v).cast(dt, strict=True)
    return pl.DataFrame(out)


def _is_scalar(v: Any) -> bool:
    return isinstance(v, str | bytes | int | float | bool) or (
        hasattr(v, "shape") and getattr(v, "shape", None) == ()
    )


class EpochBuffers:
    """In-memory rows accumulated between two flushes.

    Each table holds an ordered list of segments: lists of row dicts (``add``) and frames built
    from columns (``add_columns``, the vectorised path for trades and per-tick flow). Segment
    order is insertion order, so a flushed part is a deterministic function of the calls.
    """

    def __init__(self) -> None:
        self.segments: dict[str, list[list[dict[str, Any]] | pl.DataFrame]] = {
            t: [] for t in TABLES
        }
        self.prompts: list[dict[str, Any]] = []

    def _rows(self, table: str) -> list[dict[str, Any]]:
        segs = self.segments[table]
        if not segs or not isinstance(segs[-1], list):
            segs.append([])
        last = segs[-1]
        assert isinstance(last, list)
        return last

    def add(self, table: str, row: dict[str, Any]) -> None:
        self._rows(table).append(row)

    def extend(self, table: str, rows: Iterable[dict[str, Any]]) -> None:
        self._rows(table).extend(rows)

    def add_columns(self, table: str, cols: dict[str, Any]) -> None:
        df = frame_from_columns(table, cols)
        if df.height:
            self.segments[table].append(df)

    def add_prompt(self, record: dict[str, Any]) -> None:
        self.prompts.append(record)

    def n_rows(self, table: str | None = None) -> int:
        tables = [table] if table is not None else list(self.segments)
        return sum(
            len(seg) if isinstance(seg, list) else seg.height
            for t in tables
            for seg in self.segments[t]
        )

    def to_frames(self) -> dict[str, pl.DataFrame]:
        out: dict[str, pl.DataFrame] = {}
        for t, segs in self.segments.items():
            frames = [
                frame_from_rows(t, seg) if isinstance(seg, list) else seg
                for seg in segs
                if (len(seg) if isinstance(seg, list) else seg.height)
            ]
            if frames:
                out[t] = frames[0] if len(frames) == 1 else pl.concat(frames, rechunk=True)
        return out

    def clear(self) -> None:
        for v in self.segments.values():
            v.clear()
        self.prompts.clear()


class PartWriter:
    """Flushes ``EpochBuffers`` into numbered parts; its counters live in checkpoints."""

    def __init__(self, run_dir: Path, *, store_prompts: bool = True) -> None:
        self.run_dir = Path(run_dir)
        self.store_prompts = store_prompts
        self.next_index: dict[str, int] = {t: 0 for t in TABLES}
        self.prompts_bytes = 0

    def flush(self, buffers: EpochBuffers) -> dict[str, Path]:
        written: dict[str, Path] = {}
        for table, df in buffers.to_frames().items():
            path = self.run_dir / table / part_name(self.next_index[table])
            atomic_write_parquet(df, path)
            self.next_index[table] += 1
            written[table] = path
        if self.store_prompts and buffers.prompts:
            self._append_prompts(buffers.prompts)
        buffers.clear()
        return written

    def _append_prompts(self, records: list[dict[str, Any]]) -> None:
        path = self.run_dir / PROMPTS_FILE
        payload = "".join(
            json.dumps(r, sort_keys=True, ensure_ascii=False, default=str) + "\n" for r in records
        ).encode("utf-8")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "ab") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        self.prompts_bytes += len(payload)

    def state(self) -> dict[str, Any]:
        return {"next_index": dict(self.next_index), "prompts_bytes": self.prompts_bytes}

    def restore(self, state: dict[str, Any]) -> None:
        """Adopt checkpoint counters and delete anything written after that checkpoint."""
        self.next_index = {t: int(state["next_index"].get(t, 0)) for t in TABLES}
        self.prompts_bytes = int(state.get("prompts_bytes", 0))
        for table in TABLES:
            tdir = self.run_dir / table
            if not tdir.exists():
                continue
            for p in tdir.glob(PART_GLOB):
                if part_index(p) >= self.next_index[table]:
                    p.unlink()
            for p in tdir.glob(".part-*.tmp-*"):
                p.unlink()
        prompts = self.run_dir / PROMPTS_FILE
        if prompts.exists():
            if self.prompts_bytes == 0:
                prompts.unlink()
            elif prompts.stat().st_size > self.prompts_bytes:
                with open(prompts, "r+b") as fh:
                    fh.truncate(self.prompts_bytes)


def copy_parts_rewriting_run_id(
    src_dir: Path, dst_dir: Path, next_index: dict[str, int], run_id: str
) -> None:
    """Copy parts ``< next_index`` of every table from ``src_dir`` with a new ``run_id`` column.

    Used by ``branch()``: the child holds the parent's history before the restore checkpoint,
    with identical rows except for ``run_id``.
    """
    for table, upto in next_index.items():
        for i in range(int(upto)):
            src = Path(src_dir) / table / part_name(i)
            if not src.exists():
                continue
            df = pl.read_parquet(src).with_columns(pl.lit(run_id, dtype=pl.Utf8).alias("run_id"))
            df = df.select(list(TABLES[table])).cast(TABLES[table])
            atomic_write_parquet(df, Path(dst_dir) / table / part_name(i))
