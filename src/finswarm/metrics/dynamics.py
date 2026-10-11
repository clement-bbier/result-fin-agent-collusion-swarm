"""Dynamics of collusion: onset, break, recovery, hysteresis and mean time to recover.

All functions take a per-period series (numpy array, list or polars Series), typically
the Calvano Delta from :func:`finswarm.metrics.collusion.delta_series`. Indices are
periods relative to the first element of the series. Censored outcomes (the event never
happens inside the series) return NaN; the caller reports censoring.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from finswarm.metrics.registry import register_metric


def _arr(x: Any) -> np.ndarray:
    return np.asarray(x.to_numpy() if hasattr(x, "to_numpy") else x, dtype=float)


def _first_run(cond: np.ndarray, persist: int, start: int = 0) -> float:
    """First index ``t >= start`` such that ``cond[t:t+persist]`` is all True."""
    if persist < 1:
        raise ValueError("persist must be >= 1")
    c = cond.astype(int)
    if len(c) - start < persist:
        return float("nan")
    run = np.convolve(c, np.ones(persist, dtype=int), mode="valid")  # run[t] = sum c[t:t+p]
    hits = np.where(run[start:] == persist)[0]
    return float(hits[0] + start) if hits.size else float("nan")


@register_metric("time_to_collude", "periods", "run", False, source="calvano2020")
def time_to_collude(delta_series: Any, thr: float = 0.3, persist: int = 20) -> float:
    """First period from which Delta stays ``>= thr`` for ``persist`` periods (NaN =
    censored)."""
    return _first_run(_arr(delta_series) >= thr, persist)


@register_metric("time_to_break", "periods", "run", False, source="novel")
def time_to_break(after_event: Any, baseline: float, frac: float = 0.5, persist: int = 1) -> float:
    """Periods after an intervention until Delta falls to ``<= frac * baseline`` (the
    trunk's pre-event level) for ``persist`` periods. ``after_event[0]`` is the first
    post-event period."""
    return _first_run(_arr(after_event) <= frac * baseline, persist)


@register_metric("time_to_recover", "periods", "run", True, source="novel")
def time_to_recover(
    after_event: Any,
    target: float | None = None,
    baseline: float | None = None,
    frac: float = 0.9,
    persist: int = 5,
    after_break: bool = True,
    break_frac: float = 0.5,
) -> float:
    """Periods after an intervention until Delta is back ``>= target`` for ``persist``
    periods (``target`` defaults to ``frac * baseline``). With ``after_break`` the search
    starts at the break (:func:`time_to_break` with ``break_frac``); NaN if no break or
    no recovery."""
    x = _arr(after_event)
    if target is None:
        if baseline is None:
            raise ValueError("time_to_recover needs target or baseline")
        target = frac * baseline
    start = 0
    if after_break:
        if baseline is None:
            raise ValueError("after_break=True needs baseline")
        b = time_to_break(x, baseline, break_frac)
        if not np.isfinite(b):
            return float("nan")
        start = int(b)
    return _first_run(x >= target, persist, start)


@register_metric("hysteresis", "ratio", "run", True, source="novel")
def hysteresis(on: Any, off: Any, baseline: float | None = None, tail: int = 10) -> float:
    """Persistence of a lever's effect after it is removed.

    ``on``: Delta while the lever is active; ``off``: Delta after removal. With a
    pre-lever ``baseline``: ``(baseline - mean(off_tail)) / (baseline - mean(on_tail))``
    -- 1 = the effect fully persists, 0 = full recovery. Without a baseline:
    ``mean(off_tail) - mean(on_tail)`` (recovered amount).
    """
    on_t = _arr(on)[-tail:]
    off_t = _arr(off)[-tail:]
    m_on, m_off = float(np.nanmean(on_t)), float(np.nanmean(off_t))
    if baseline is None:
        return m_off - m_on
    eff = baseline - m_on
    return (baseline - m_off) / eff if eff != 0 else float("nan")


@register_metric("mttr", "periods", "run", True, source="novel")
def mttr(series: Any, thr: float = 0.3, persist: int = 1) -> float:
    """Mean time to recover: average length of completed episodes where Delta drops
    below ``thr`` after having been at or above it, until it is back ``>= thr`` for
    ``persist`` periods. NaN when no episode completes."""
    x = _arr(series)
    above = x >= thr
    durations: list[int] = []
    t = 0
    established = False
    while t < len(x):
        if above[t]:
            established = True
            t += 1
            continue
        if not established:
            t += 1
            continue
        rec = _first_run(above, persist, t)
        if not np.isfinite(rec):
            break
        durations.append(int(rec) - t)
        t = int(rec)
    return float(np.mean(durations)) if durations else float("nan")
