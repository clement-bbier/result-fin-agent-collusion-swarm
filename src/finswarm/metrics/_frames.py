"""Shared helpers for metric functions: column checks, windows, pivots, per-run mapping.

Every metric accepts polars frames with the columns defined in
``finswarm.store.schemas`` (a subset is enough). Functions that work on a single run
raise if the frame mixes several ``run_id`` values; use :func:`by_run` to map them.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import polars as pl

ArrayOrFrame = np.ndarray | pl.DataFrame


def require(df: pl.DataFrame, cols: Sequence[str], where: str) -> None:
    """Raise ``KeyError`` naming the missing columns."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{where}: missing columns {missing}; got {df.columns}")


def single_run(df: pl.DataFrame, where: str) -> None:
    """Raise if ``df`` contains more than one ``run_id``."""
    if "run_id" in df.columns and df.height and df["run_id"].n_unique() > 1:
        raise ValueError(f"{where}: frame mixes several run_id values; use by_run()")


def window(
    df: pl.DataFrame, start: int | None = None, end: int | None = None, col: str = "period"
) -> pl.DataFrame:
    """Rows with ``start <= col <= end`` (bounds inclusive, ``None`` = open)."""
    if start is None and end is None:
        return df
    expr = pl.lit(True)
    if start is not None:
        expr = expr & (pl.col(col) >= start)
    if end is not None:
        expr = expr & (pl.col(col) <= end)
    return df.filter(expr)


def pivot_matrix(
    df: pl.DataFrame,
    value: str = "price",
    index: str = "period",
    key: str = "agent_id",
    keys: Sequence[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pivot a long frame to ``(index values, key values, matrix[index, key])``.

    Duplicate ``(index, key)`` rows keep the last one. Missing cells are NaN.
    """
    require(df, [value, index, key], "pivot_matrix")
    single_run(df, "pivot_matrix")
    d = (
        df.select(pl.col(index), pl.col(key), pl.col(value).cast(pl.Float64))
        .group_by([index, key], maintain_order=True)
        .agg(pl.col(value).last())
    )
    idx = np.sort(d[index].unique().to_numpy())
    ks = np.asarray(sorted(d[key].unique().to_list()) if keys is None else list(keys))
    mat = np.full((len(idx), len(ks)), np.nan)
    if d.height:
        ii = np.searchsorted(idx, d[index].to_numpy())
        pos = {int(k): j for j, k in enumerate(ks)}
        jj = np.array([pos.get(int(k), -1) for k in d[key].to_list()])
        ok = jj >= 0
        mat[ii[ok], jj[ok]] = d[value].to_numpy()[ok]
    return idx, ks, mat


def as_matrix(
    x: ArrayOrFrame, value: str = "price", key: str = "agent_id", diff: bool = False
) -> np.ndarray:
    """Return a ``periods x agents`` float matrix from an array or an ``agent_period`` frame.

    With ``diff=True`` the first difference along time is returned (quote changes).
    """
    if isinstance(x, pl.DataFrame):
        _, _, mat = pivot_matrix(x, value=value, key=key)
    else:
        mat = np.asarray(x, dtype=float)
        if mat.ndim == 1:
            mat = mat[:, None]
    if diff:
        mat = np.diff(mat, axis=0)
    return mat


def by_run(fn: Callable[..., Any], df: pl.DataFrame, *args: Any, **kwargs: Any) -> dict[str, Any]:
    """Apply a single-run metric to every ``run_id`` of ``df``."""
    require(df, ["run_id"], "by_run")
    out: dict[str, Any] = {}
    for (rid,), part in df.partition_by("run_id", as_dict=True, maintain_order=True).items():
        out[str(rid)] = fn(part, *args, **kwargs)
    return out


def nan_mean(x: np.ndarray) -> float:
    """Mean of finite values, NaN when there are none (no warning)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(x.mean()) if x.size else float("nan")
