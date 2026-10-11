"""Synthetic traces with known properties for the metrics/analysis tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from finswarm.metrics.benchmarks import MarketModel, logit_profits
from finswarm.store.schemas import AGENT_PERIOD, LLM_DECISIONS

N_SWARM = 100  # tagged agents start at this id; slot = agent_id - N_SWARM


def market_frame(
    prices: np.ndarray,
    model: MarketModel,
    start: int = 111,
    entrant: np.ndarray | float | None = None,
    run_id: str = "r",
) -> pl.DataFrame:
    """``agent_period`` of incumbents (slots 0..n-1) priced by ``prices`` with expected
    quantities and profits of ``model``, plus the entrants' rows (slots n..) when
    ``entrant`` prices are given."""
    p = np.atleast_2d(np.asarray(prices, dtype=float))
    t, _ = p.shape
    q_inc, q_ent = model.all_quantities(p, entrant)
    cols, qs = [p], [q_inc]
    if entrant is not None:
        e = np.broadcast_to(np.asarray(entrant, dtype=float).reshape(t, -1), q_ent.shape)
        cols.append(e)
        qs.append(q_ent)
    allp, allq = np.concatenate(cols, axis=1), np.concatenate(qs, axis=1)
    k = allp.shape[1]
    prof = (allp - model.params.marginal_cost) * allq
    return pl.DataFrame(
        {
            "run_id": [run_id] * (t * k),
            "period": np.repeat(np.arange(start, start + t), k).astype(np.int32),
            "agent_id": np.tile(np.arange(N_SWARM, N_SWARM + k), t).astype(np.int32),
            "slot": np.tile(np.arange(k), t).astype(np.int8),
            "price": allp.ravel(),
            "quantity": allq.ravel(),
            "profit": prof.ravel(),
            "inventory": np.zeros(t * k),
            "cash": np.zeros(t * k),
            "undercut": np.zeros(t * k, dtype=bool),
        },
        schema=AGENT_PERIOD,
    )


def to_store(df: pl.DataFrame) -> pl.DataFrame:
    """Frames are built in registered (1-based) periods; the store numbers periods from 0
    (A-008), so every synthetic parquet is written with ``period - 1`` and read back through
    ``analysis.io.read_table`` (which adds 1) exactly as built."""
    if "period" not in df.columns:
        return df
    return df.with_columns((pl.col("period") - 1).cast(df.schema["period"]))


def write_frame_run(
    camp: Path,
    run_id: str,
    ap: pl.DataFrame,
    manifest: dict,
    fail_every: int = 0,
    subdir: str | None = None,
) -> Path:
    """Write a store-layout run directory ``camp/<run_id>`` (or ``camp/<subdir>``) from an
    ``agent_period`` frame, with matching ``llm_decisions`` and a manifest."""
    d = camp / (subdir or run_id)
    (d / "agent_period").mkdir(parents=True, exist_ok=True)
    (d / "llm_decisions").mkdir(parents=True, exist_ok=True)
    to_store(ap.with_columns(pl.lit(run_id).alias("run_id"))).write_parquet(
        d / "agent_period" / "part-0000.parquet"
    )
    periods = ap["period"].unique().sort().to_numpy()
    to_store(llm_decisions(run_id, periods, fail_every)).write_parquet(
        d / "llm_decisions" / "part-0000.parquet"
    )
    (d / "manifest.json").write_text(json.dumps({"run_id": run_id, **manifest}), encoding="utf-8")
    return d


def agent_period(prices: np.ndarray, run_id: str = "r0", start: int = 1) -> pl.DataFrame:
    """``agent_period`` frame for a ``periods x agents`` price matrix in the golden logit
    market (profits from the closed-form demand)."""
    p = np.asarray(prices, dtype=float)
    t, n = p.shape
    prof = logit_profits(p)
    rows = {
        "run_id": [run_id] * (t * n),
        "period": np.repeat(np.arange(start, start + t), n).astype(np.int32),
        "agent_id": np.tile(np.arange(N_SWARM, N_SWARM + n), t).astype(np.int32),
        "slot": np.tile(np.arange(n), t).astype(np.int8),
        "price": p.ravel(),
        "quantity": (prof / np.maximum(p - 1.0, 1e-9)).ravel(),
        "profit": prof.ravel(),
        "inventory": np.zeros(t * n),
        "cash": np.cumsum(prof, axis=0).ravel(),
        "undercut": np.zeros(t * n, dtype=bool),
    }
    return pl.DataFrame(rows, schema=AGENT_PERIOD)


def colluding_prices(t: int = 200, seed: int = 0) -> np.ndarray:
    """Leader random-walks on a 0.01 grid near the monopoly price (mostly up or flat,
    reflected in [1.85, 1.95]); the follower matches the leader's previous price."""
    rng = np.random.default_rng(seed)
    lead = np.empty(t)
    lead[0] = 1.90
    for k in range(1, t):
        step = rng.choice([-0.01, 0.0, 0.01], p=[0.2, 0.4, 0.4])
        nxt = lead[k - 1] + step
        lead[k] = nxt if 1.85 <= nxt <= 1.95 else lead[k - 1] - step
    lead = np.round(lead, 2)
    follow = np.r_[lead[0], lead[:-1]]
    return np.column_stack([lead, follow])


def competitive_prices(t: int = 200, n: int = 2, seed: int = 0, sd: float = 0.05) -> np.ndarray:
    """Independent noisy play around the static Nash price."""
    rng = np.random.default_rng(seed)
    return 1.4729 + rng.normal(0.0, sd, size=(t, n))


def llm_decisions(run_id: str, periods: np.ndarray, fail_every: int = 0) -> pl.DataFrame:
    n = len(periods)
    ok = np.ones(n, dtype=bool)
    if fail_every:
        ok[::fail_every] = False
    data = {c: [None] * n for c in LLM_DECISIONS}
    data.update(
        run_id=[run_id] * n,
        decision_id=np.arange(n),
        period=np.asarray(periods, dtype=np.int32),
        agent_id=np.full(n, N_SWARM, dtype=np.int32),
        parse_ok=ok,
        plans=["I will set my price to 1.80"] * n,
        price=np.full(n, 1.80),
    )
    return pl.DataFrame(data, schema=LLM_DECISIONS)


def write_run(
    root: Path,
    name: str,
    prices: np.ndarray,
    manifest: dict,
    fail_every: int = 0,
    start: int = 1,
) -> Path:
    """Write a run directory in the store layout ``<run>/<table>/part-0000.parquet``."""
    d = root / name
    (d / "agent_period").mkdir(parents=True, exist_ok=True)
    (d / "llm_decisions").mkdir(parents=True, exist_ok=True)
    ap = agent_period(prices, run_id=manifest.get("run_id", name), start=start)
    to_store(ap).write_parquet(d / "agent_period" / "part-0000.parquet")
    periods = ap["period"].unique().sort().to_numpy()
    to_store(llm_decisions(manifest.get("run_id", name), periods, fail_every)).write_parquet(
        d / "llm_decisions" / "part-0000.parquet"
    )
    (d / "manifest.json").write_text(json.dumps({"run_id": name, **manifest}), encoding="utf-8")
    return d
