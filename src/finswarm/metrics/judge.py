"""LLM-judge interface for scratchpads (detector D9).

The LLM call itself belongs to the llm layer: callers inject an async ``score_fn(text)
-> score`` (e.g. a rubric score 0-4 from an independent judge model). This module only
samples scratchpads, runs the injected scorer with bounded concurrency, aggregates scores
per window and measures agreement with blind human labels (quadratic-weighted kappa and
Spearman).
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy import stats

from finswarm.metrics._frames import require
from finswarm.metrics.registry import register_metric

ScoreFn = Callable[[str], Awaitable[float | int | None]]

_SCORE_SCHEMA = {
    "run_id": pl.Utf8,
    "agent_id": pl.Int64,
    "period": pl.Int64,
    "decision_id": pl.Int64,
    "score": pl.Float64,
    "error": pl.Utf8,
}


@dataclass(frozen=True)
class JudgeItem:
    run_id: str
    agent_id: int
    period: int
    decision_id: int
    text: str


def scratchpad_text(row: dict, fields: Sequence[str] = ("plans", "insights")) -> str:
    return "\n\n".join(f"[{f.upper()}]\n{row[f]}" for f in fields if row.get(f))


def sample_items(
    llm_decisions: pl.DataFrame,
    window: int = 40,
    k: int = 8,
    seed: int = 0,
    start_period: int = 0,
    fields: Sequence[str] = ("plans", "insights"),
) -> list[JudgeItem]:
    """Draw up to ``k`` non-empty scratchpads per ``(run_id, window)`` (seeded)."""
    require(llm_decisions, ["run_id", "agent_id", "period", "decision_id"], "sample_items")
    rng = np.random.default_rng(seed)
    d = llm_decisions.with_columns(
        ((pl.col("period") - start_period) // window).alias("_win")
    ).sort(["run_id", "_win", "decision_id"])
    items: list[JudgeItem] = []
    for _, part in d.partition_by(["run_id", "_win"], as_dict=True, maintain_order=True).items():
        rows = [r for r in part.iter_rows(named=True) if scratchpad_text(r, fields).strip()]
        if not rows:
            continue
        pick = rng.choice(len(rows), size=min(k, len(rows)), replace=False)
        for i in sorted(pick):
            r = rows[int(i)]
            items.append(
                JudgeItem(
                    run_id=str(r["run_id"]),
                    agent_id=int(r["agent_id"]),
                    period=int(r["period"]),
                    decision_id=int(r["decision_id"]),
                    text=scratchpad_text(r, fields),
                )
            )
    return items


async def score_items(
    items: Sequence[JudgeItem], score_fn: ScoreFn, concurrency: int = 8
) -> pl.DataFrame:
    """Score every item with the injected async ``score_fn`` (bounded concurrency).

    Exceptions are captured per item (``score`` NaN, ``error`` set) so one failed call
    never sinks a batch.
    """
    sem = asyncio.Semaphore(max(1, concurrency))

    async def one(it: JudgeItem) -> dict:
        async with sem:
            try:
                s = await score_fn(it.text)
                return {"score": float("nan") if s is None else float(s), "error": None}
            except Exception as e:  # recorded per item, not swallowed
                return {"score": float("nan"), "error": f"{type(e).__name__}: {e}"}

    res = await asyncio.gather(*(one(it) for it in items))
    rows = [
        {
            "run_id": it.run_id,
            "agent_id": it.agent_id,
            "period": it.period,
            "decision_id": it.decision_id,
            **r,
        }
        for it, r in zip(items, res, strict=True)
    ]
    return pl.DataFrame(rows, schema=_SCORE_SCHEMA)


def score_items_sync(
    items: Sequence[JudgeItem], score_fn: ScoreFn, concurrency: int = 8
) -> pl.DataFrame:
    """Blocking wrapper around :func:`score_items` for scripts."""
    return asyncio.run(score_items(items, score_fn, concurrency))


@register_metric("judge_score", "rubric points", "window", True, source="agrawal2025")
def aggregate_windows(
    scores: pl.DataFrame, window: int = 40, start_period: int = 0
) -> pl.DataFrame:
    """Per ``(run_id, window)``: mean and max judge score and number of scored items."""
    require(scores, ["run_id", "period", "score"], "aggregate_windows")
    return (
        scores.filter(pl.col("score").is_finite())
        .with_columns(((pl.col("period") - start_period) // window).alias("window"))
        .group_by(["run_id", "window"])
        .agg(
            pl.col("score").mean().alias("mean_score"),
            pl.col("score").max().alias("max_score"),
            pl.len().alias("n"),
        )
        .sort(["run_id", "window"])
    )


def quadratic_weighted_kappa(
    a: Sequence[int], b: Sequence[int], min_rating: int | None = None, max_rating: int | None = None
) -> float:
    """Cohen's kappa with quadratic weights for two integer ratings vectors."""
    x = np.asarray(a, dtype=int)
    y = np.asarray(b, dtype=int)
    if x.shape != y.shape or x.size == 0:
        raise ValueError("ratings must be non-empty and of equal length")
    lo = int(min(x.min(), y.min())) if min_rating is None else min_rating
    hi = int(max(x.max(), y.max())) if max_rating is None else max_rating
    k = hi - lo + 1
    obs = np.zeros((k, k))
    np.add.at(obs, (x - lo, y - lo), 1.0)
    exp = np.outer(obs.sum(1), obs.sum(0)) / obs.sum()
    i, j = np.indices((k, k))
    w = (i - j) ** 2 / max((k - 1) ** 2, 1)
    den = float((w * exp).sum())
    return 1.0 - float((w * obs).sum()) / den if den > 0 else 1.0


def agreement(judge: Sequence[float], human: Sequence[float]) -> dict[str, float]:
    """Judge-versus-human agreement: quadratic-weighted kappa (on rounded scores) and
    Spearman's rho."""
    j = np.asarray(judge, dtype=float)
    h = np.asarray(human, dtype=float)
    ok = np.isfinite(j) & np.isfinite(h)
    j, h = j[ok], h[ok]
    if j.size < 2:
        return {
            "kappa_qw": float("nan"),
            "spearman_rho": float("nan"),
            "spearman_p": float("nan"),
            "n": float(j.size),
        }
    rho, p = stats.spearmanr(j, h)
    return {
        "kappa_qw": quadratic_weighted_kappa(np.rint(j).astype(int), np.rint(h).astype(int)),
        "spearman_rho": float(rho),
        "spearman_p": float(p),
        "n": float(j.size),
    }
