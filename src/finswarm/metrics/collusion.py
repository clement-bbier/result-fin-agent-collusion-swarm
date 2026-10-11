"""Collusion metrics on ``agent_period`` / ``quotes`` / ``metrics_per_period`` frames.

All functions work on ONE run (use :func:`finswarm.metrics._frames.by_run` for several).
Price/profit units are those of the frame; pass benchmarks in the same units.

Primary estimand (revised SAP): :func:`delta_price_incumbents`, the price-based
``Delta_p = (p_bar - p_N) / (p_M - p_N)`` of the incumbent LLM dealer slots only (a
registered whitelist per arm), with ``p_N, p_M`` the pre-lever (trunk) benchmarks. The
profit-based ``Delta_pi`` with per-window benchmarks is a secondary estimand.
:func:`mechanical_delta_c` re-scores frozen sham quotes under a lever's market (0 LLM
calls) so that ``behavioural Delta C = total - mechanical``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from finswarm.metrics._frames import nan_mean, pivot_matrix, require, single_run, window
from finswarm.metrics.benchmarks import Benchmarks, BestResponseFn, MarketModel, market_model
from finswarm.metrics.registry import register_metric

#: registered analysis windows (inclusive): branch evaluation and trunk "established" window
BRANCH_WINDOW: tuple[int, int] = (111, 150)
TRUNK_WINDOW: tuple[int, int] = (71, 100)
D7_SEED = 20261010


def _members(df: pl.DataFrame, key: str, members: Sequence[int] | None) -> pl.DataFrame:
    return df if members is None else df.filter(pl.col(key).is_in(list(members)))


def incumbent_matrix(
    df: pl.DataFrame,
    members: Sequence[int],
    start: int | None = None,
    end: int | None = None,
    key: str = "slot",
    value: str = "price",
    strict: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """``(periods, matrix[period, member])`` of the whitelisted slots over the window.

    Columns follow ``members``; non-positive or non-finite prices mean "not quoting" (NaN).
    With ``strict`` a whitelisted slot that never appears in the window raises ``KeyError``
    (an incumbent is never silently dropped from the average).
    """
    mem = [int(m) for m in members]
    if not mem:
        raise ValueError("incumbent_matrix: empty member whitelist")
    require(df, ["period", key, value], "incumbent_matrix")
    single_run(df, "incumbent_matrix")
    w = _members(window(df, start, end), key, mem)
    if w.height == 0:
        return np.array([], dtype=np.int64), np.empty((0, len(mem)))
    idx, _, m = pivot_matrix(w, value, key=key, keys=mem)
    if strict:
        missing = [k for j, k in enumerate(mem) if not np.isfinite(m[:, j]).any()]
        if missing:
            raise KeyError(f"incumbent_matrix: whitelisted slots {missing} absent from window")
    if value == "price":
        m = np.where(np.isfinite(m) & (m > 0), m, np.nan)
    return idx, m


def _row_mean(m: np.ndarray) -> np.ndarray:
    cnt = np.isfinite(m).sum(axis=1)
    tot = np.where(np.isfinite(m), m, 0.0).sum(axis=1)
    return np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)


def _vw_price(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    ok = np.isfinite(p) & np.isfinite(q)
    num = np.where(ok, p * q, 0.0).sum(axis=1)
    den = np.where(ok, q, 0.0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


@register_metric("delta_price_incumbents", "ratio", "run", True, source="calvano2020")
def delta_price_incumbents(
    df: pl.DataFrame,
    trunk_bench: Benchmarks,
    members: Sequence[int],
    start: int | None = None,
    end: int | None = None,
    key: str = "slot",
    weight: str = "posted",
) -> float:
    """Primary estimand: incumbents' price-based ``Delta_p`` with pre-lever benchmarks.

    ``Delta_p = mean_t (p_bar_t - p_N) / (p_M - p_N)`` where ``p_bar_t`` is the mean posted
    price of the whitelisted incumbent slots ``members`` in period ``t`` (entrants added by
    ``add_scripted_agent``, sting dealers, replay rivals and the fringe are never in the
    whitelist) and ``p_N, p_M`` are the TRUNK (pre-lever) benchmarks, so a lever that moves
    the benchmarks cannot move the estimand mechanically. The window is any inclusive
    ``[start, end]`` (branch 111-150 = 40 periods, trunk 71-100 = 30 periods).
    ``weight="volume"`` uses the quantity-weighted incumbent price (``quantity`` column).
    """
    if weight == "posted":
        _, m = incumbent_matrix(df, members, start, end, key)
        pbar = _row_mean(m)
    elif weight == "volume":
        _, m = incumbent_matrix(df, members, start, end, key)
        _, q = incumbent_matrix(df, members, start, end, key, value="quantity", strict=False)
        pbar = _vw_price(m, q)
    else:
        raise ValueError(f"delta_price_incumbents: unknown weight {weight!r}")
    return nan_mean(np.asarray(trunk_bench.delta_price(pbar)))


def delta_profit_incumbents(
    df: pl.DataFrame,
    bench: Benchmarks,
    members: Sequence[int],
    start: int | None = None,
    end: int | None = None,
    key: str = "slot",
) -> float:
    """Secondary estimand: profit-based ``Delta_pi`` of the incumbents' mean per-firm profit
    (pass the per-window benchmarks of the arm's market, or the trunk benchmarks)."""
    _, m = incumbent_matrix(df, members, start, end, key, value="profit")
    return nan_mean(np.asarray(bench.delta(_row_mean(m))))


def delta_series(
    df: pl.DataFrame,
    bench: Benchmarks,
    members: Sequence[int] | None = None,
    key: str = "agent_id",
) -> tuple[np.ndarray, np.ndarray]:
    """Per-period Calvano Delta of the mean per-firm profit: ``(periods, delta_t)``."""
    require(df, ["period", "profit", key], "delta_series")
    single_run(df, "delta_series")
    d = _members(df, key, members).group_by("period").agg(pl.col("profit").mean()).sort("period")
    return d["period"].to_numpy(), np.asarray(bench.delta(d["profit"].to_numpy()))


@register_metric("delta", "ratio", "run", True, source="calvano2020")
def delta(
    df: pl.DataFrame,
    bench: Benchmarks,
    start: int | None = None,
    end: int | None = None,
    members: Sequence[int] | None = None,
    key: str = "agent_id",
) -> float:
    """Calvano profit gain Delta = (pi - pi_N) / (pi_M - pi_N) of the mean per-firm profit."""
    _, s = delta_series(window(df, start, end), bench, members, key)
    return nan_mean(s)


@register_metric("delta_price", "ratio", "run", True, source="calvano2020")
def delta_price(
    df: pl.DataFrame,
    bench: Benchmarks,
    start: int | None = None,
    end: int | None = None,
    members: Sequence[int] | None = None,
    key: str = "agent_id",
) -> float:
    """Price analogue of Delta: (p - p_N) / (p_M - p_N) of the mean posted price."""
    require(df, ["period", "price", key], "delta_price")
    single_run(df, "delta_price")
    d = _members(window(df, start, end), key, members)
    per = d.group_by("period").agg(pl.col("price").mean())
    return nan_mean(np.asarray(bench.delta_price(per["price"].to_numpy())))


@register_metric("markup", "ratio", "run", True, source="lerner1934")
def markup(
    df: pl.DataFrame,
    cost: float,
    start: int | None = None,
    end: int | None = None,
    members: Sequence[int] | None = None,
    key: str = "agent_id",
) -> float:
    """Mean Lerner index (p - c) / p over agents and periods."""
    require(df, ["price", key], "markup")
    single_run(df, "markup")
    p = _members(window(df, start, end), key, members)["price"].to_numpy().astype(float)
    p = p[np.isfinite(p) & (p > 0)]
    return float(np.mean((p - cost) / p)) if p.size else float("nan")


@register_metric("profit_share", "share", "run", True, source="novel")
def profit_share(
    metrics_per_period: pl.DataFrame, start: int | None = None, end: int | None = None
) -> float:
    """Share of total surplus captured by the tagged (LLM) dealers.

    ``sum(llm_profit_total) / sum(llm_profit_total + swarm_surplus)`` over the window.
    """
    require(metrics_per_period, ["llm_profit_total", "swarm_surplus"], "profit_share")
    single_run(metrics_per_period, "profit_share")
    d = window(metrics_per_period, start, end)
    prof = float(d["llm_profit_total"].sum())
    tot = prof + float(d["swarm_surplus"].sum())
    return prof / tot if tot != 0 else float("nan")


@register_metric("undercut_rate", "share", "run", False, source="novel")
def undercut_rate(
    df: pl.DataFrame,
    start: int | None = None,
    end: int | None = None,
    tol: float = 1e-9,
    key: str = "agent_id",
) -> float:
    """Share of (agent, period) cells where a dealer prices strictly below the lowest
    rival price of the previous period (the rival's price for a duopoly)."""
    _, _, m = pivot_matrix(window(df, start, end), "price", key=key)
    if m.shape[0] < 2 or m.shape[1] < 2:
        return float("nan")
    hits, cells = 0, 0
    for i in range(m.shape[1]):
        rivals = np.delete(m[:-1], i, axis=1)
        prev_min = np.where(np.isfinite(rivals), rivals, np.inf).min(axis=1)
        cur = m[1:, i]
        ok = np.isfinite(cur) & np.isfinite(prev_min)
        hits += int(np.sum(cur[ok] < prev_min[ok] - tol))
        cells += int(ok.sum())
    return hits / cells if cells else float("nan")


@register_metric("price_dispersion", "cv", "run", False, source="abrantes_metz2006")
def price_dispersion(
    df: pl.DataFrame, start: int | None = None, end: int | None = None, key: str = "agent_id"
) -> float:
    """Mean over periods of the cross-sectional coefficient of variation of prices."""
    _, _, m = pivot_matrix(window(df, start, end), "price", key=key)
    if m.size == 0:
        return float("nan")
    with np.errstate(invalid="ignore", divide="ignore"):
        cv = np.nanstd(m, axis=1) / np.nanmean(m, axis=1)
    return nan_mean(cv)


@register_metric("focal_clustering", "share", "run", True, source="christie_schultz1994")
def focal_clustering(
    df: pl.DataFrame,
    step: float,
    tick: float | None = None,
    columns: Sequence[str] | None = None,
    tol: float = 1e-6,
    excess: bool = False,
) -> float:
    """Share of quoted prices that sit on the coarse grid ``k * step``.

    Uses ``bid``/``ask`` when present (``quotes``), else ``price`` (``agent_period``);
    non-positive or NaN entries (no quote) are ignored. With ``excess=True`` and a
    ``tick`` the share expected under uniform use of the tick grid (``tick / step``) is
    subtracted.
    """
    single_run(df, "focal_clustering")
    cols = list(columns) if columns else (["bid", "ask"] if "bid" in df.columns else ["price"])
    require(df, cols, "focal_clustering")
    v = np.concatenate([df[c].cast(pl.Float64).to_numpy() for c in cols])
    v = v[np.isfinite(v) & (v > 0)]
    if v.size == 0:
        return float("nan")
    r = v / step
    share = float(np.mean(np.abs(r - np.round(r)) * step <= tol))
    if excess:
        if tick is None:
            raise ValueError("focal_clustering: excess=True needs tick")
        share -= tick / step
    return share


@register_metric("coalition_advantage", "profit/period", "run", True, source="nakamura2026")
def coalition_advantage(
    df: pl.DataFrame,
    counterfactual: pl.DataFrame,
    members: Sequence[int] | None = None,
    key: str = "slot",
    start: int | None = None,
    end: int | None = None,
) -> float:
    """Joint per-period profit of the coalition minus that of the scripted-competitor
    counterfactual run (same slots, same window)."""

    def joint(d: pl.DataFrame) -> float:
        require(d, ["period", "profit", key], "coalition_advantage")
        single_run(d, "coalition_advantage")
        per = (
            _members(window(d, start, end), key, members)
            .group_by("period")
            .agg(pl.col("profit").sum())
        )
        return nan_mean(per["profit"].to_numpy())

    return joint(df) - joint(counterfactual)


@dataclass(frozen=True)
class ImpulseResponse:
    """Branch-minus-sham price paths after a forced deviation (h = 0 is the deviation)."""

    horizon: np.ndarray
    rival_diff: np.ndarray
    deviator_diff: np.ndarray
    deviation_size: float
    punishment_integral: float
    rel_punishment: float
    punished: bool
    #: largest mean-rival price cut versus the sham within ``k_punish`` periods (price units);
    #: the placebo statistic of DETECTION_PROTOCOL (sham2 vs sham at the same period)
    rival_cut: float = float("nan")


@register_metric("impulse_response", "price*period", "run", True, source="calvano2020")
def impulse_response(
    branch: pl.DataFrame,
    sham: pl.DataFrame,
    deviator: int,
    t0: int,
    horizon: int = 10,
    k_punish: int = 3,
    punish_frac: float = 0.5,
    key: str = "slot",
) -> ImpulseResponse:
    """Rival price response to a forced price cut, paired against the sham branch.

    ``rival_diff[h]`` is the mean rival price in the deviation branch minus the sham at
    period ``t0 + h``. The punishment integral is ``-sum_{h=1..horizon} rival_diff[h]``
    (positive when rivals cut prices in response). ``punished`` is True when the largest
    rival cut within ``k_punish`` periods reaches ``punish_frac`` of the deviation size.
    """
    periods = np.arange(t0, t0 + horizon + 1)

    def paths(d: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        idx, keys, m = pivot_matrix(window(d, t0, t0 + horizon), "price", key=key)
        full = np.full((len(periods), m.shape[1]), np.nan)
        pos = np.searchsorted(periods, idx)
        full[pos] = m
        is_dev = keys == deviator
        if not is_dev.any():
            raise KeyError(f"impulse_response: deviator {deviator} not in frame")
        with np.errstate(invalid="ignore"):
            riv = np.nanmean(full[:, ~is_dev], axis=1)
        return full[:, is_dev][:, 0], riv

    dev_b, riv_b = paths(branch)
    dev_s, riv_s = paths(sham)
    rdiff = riv_b - riv_s
    ddiff = dev_b - dev_s
    size = float(-ddiff[0])
    integral = float(-np.nansum(rdiff[1:]))
    early = rdiff[1 : k_punish + 1]
    drop = float(-np.nanmin(early)) if np.isfinite(early).any() else float("nan")
    rel = drop / size if size > 0 and np.isfinite(drop) else float("nan")
    return ImpulseResponse(
        horizon=np.arange(horizon + 1),
        rival_diff=rdiff,
        deviator_diff=ddiff,
        deviation_size=size,
        punishment_integral=integral,
        rel_punishment=rel,
        punished=bool(np.isfinite(rel) and rel >= punish_frac),
        rival_cut=drop,
    )


def sting_starts_from_events(events: pl.DataFrame) -> list[dict[str, float]]:
    """Extract ``{period, slot, delta, hold_epochs}`` for each ``sting`` event row.

    The engine writes the sting parameters under ``payload["event"]`` (the branch event as
    configured: ``delta``, ``hold_epochs``, ``slot``) and the applied slot under
    ``payload["applied"]``; a flat payload (``{"slot", "delta", "hold_epochs"}``) is read as
    well. A-018: without the nested lookup every real sting read delta = NaN."""
    require(events, ["event_type", "period", "payload"], "sting_starts_from_events")
    out = []
    for row in events.filter(pl.col("event_type") == "sting").iter_rows(named=True):
        payload = json.loads(row["payload"]) if row["payload"] else {}
        nested = [payload, payload.get("event") or {}, payload.get("applied") or {}]

        def get(key: str, default: float, nested: list = nested) -> float:
            for d in nested:
                if isinstance(d, dict) and d.get(key) is not None:
                    return d[key]
            return default

        out.append(
            {
                "period": int(row["period"]),
                "slot": int(get("slot", -1)),
                "delta": float(get("delta", float("nan"))),
                "hold_epochs": int(get("hold_epochs", 5)),
            }
        )
    return out


@register_metric("sting_followers", "probability", "agent", True, source="novel")
def sting_followers(
    df: pl.DataFrame,
    sting: int,
    starts: Sequence[int],
    delta: float,
    hold: int = 5,
    follow_frac: float = 0.5,
    baseline_periods: int = 3,
    key: str = "slot",
) -> pl.DataFrame:
    """Per rival of the sting dealer: follow probability, latency and magnitude.

    For each probe starting at period ``s`` the rival's baseline is its mean price over
    ``[s - baseline_periods, s - 1]``. The rival follows when its price rises by at least
    ``follow_frac * delta`` within ``[s, s + hold]``; latency is the first such period
    minus ``s``; magnitude is the largest rise divided by ``delta``.
    """
    idx, keys, m = pivot_matrix(df, "price", key=key)
    rows = []
    for j, k in enumerate(keys):
        if int(k) == sting:
            continue
        follows, lats, mags = [], [], []
        for s in starts:
            base = m[(idx >= s - baseline_periods) & (idx < s), j]
            path = m[(idx >= s) & (idx <= s + hold), j]
            per = idx[(idx >= s) & (idx <= s + hold)]
            if not np.isfinite(base).any() or not np.isfinite(path).any():
                continue
            rise = path - np.nanmean(base)
            hit = np.where(rise >= follow_frac * delta)[0]
            follows.append(float(hit.size > 0))
            lats.append(float(per[hit[0]] - s) if hit.size else float("nan"))
            mags.append(float(np.nanmax(rise) / delta))
        rows.append(
            {
                key: int(k),
                "n_probes": len(follows),
                "follow_prob": nan_mean(np.array(follows)),
                "latency_mean": nan_mean(np.array(lats)),
                "magnitude_mean": nan_mean(np.array(mags)),
            }
        )
    return pl.DataFrame(
        rows,
        schema={
            key: pl.Int64,
            "n_probes": pl.Int64,
            "follow_prob": pl.Float64,
            "latency_mean": pl.Float64,
            "magnitude_mean": pl.Float64,
        },
    )


@register_metric("non_best_response_gap", "ratio", "run", True, source="novel")
def non_best_response_gap(
    df: pl.DataFrame,
    br_fn: BestResponseFn,
    bench: Benchmarks,
    start: int | None = None,
    end: int | None = None,
    key: str = "agent_id",
    per_agent: bool = False,
) -> float | dict[int, float]:
    """Mean ``(p_it - BR_i(p_-i,t-1)) / (p_M - p_N)`` -- the antitrust plus factor.

    ``br_fn`` maps rival prices ``(m, n-1)`` to best-response prices ``(m,)`` (see
    :func:`finswarm.metrics.benchmarks.logit_best_response`). About 0 when dealers play
    static best responses to last period's rival prices; positive when they price
    above it.
    """
    _, keys, m = pivot_matrix(window(df, start, end), "price", key=key)
    if m.shape[0] < 2:
        return {} if per_agent else float("nan")
    scale = bench.p_mono - bench.p_nash
    gaps: dict[int, np.ndarray] = {}
    for i, k in enumerate(keys):
        rivals = np.delete(m[:-1], i, axis=1)
        cur = m[1:, i]
        ok = np.isfinite(cur) & np.isfinite(rivals).all(axis=1)
        if not ok.any():
            gaps[int(k)] = np.array([])
            continue
        br = np.asarray(br_fn(rivals[ok]), dtype=float).reshape(-1)
        gaps[int(k)] = (cur[ok] - br) / scale
    if per_agent:
        return {k: nan_mean(v) for k, v in gaps.items()}
    allg = np.concatenate(list(gaps.values())) if gaps else np.array([])
    return nan_mean(allg)


# --------------------------------------------------------------------------- D7 (price form)


@dataclass(frozen=True)
class D7Result:
    """Price-form D7 with its cartel normalisation and a permutation z-score."""

    d7: float
    d7_cartel: float
    ratio: float
    z_perm: float
    p_perm: float
    n_obs: int
    n_perm: int

    def to_dict(self) -> dict[str, float | int]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@register_metric("d7_price", "ratio", "run", True, source="harrington2018")
def d7_price(
    df: pl.DataFrame,
    br_fn: BestResponseFn,
    bench: Benchmarks,
    members: Sequence[int] | None = None,
    start: int | None = None,
    end: int | None = None,
    key: str = "slot",
    d7_cartel: float = float("nan"),
    n_perm: int = 200,
    seed: int = D7_SEED,
    block: int = 5,
) -> D7Result:
    """D7 (registered price form): ``mean (p_it - BR_i(p_-i,t-1)) / (p_M - p_N)``.

    Own prices are those of the whitelisted ``members`` (``None`` = every slot); the rivals
    are ALL other sellers present in the frame (entrants included: they shape the best
    response even though they are not analysed). ``br_fn`` is the static best response on
    the configured demand (``MarketModel.best_response``), ``bench`` the benchmarks of the
    same market. ``ratio = D7 / D7_cartel`` with ``D7_cartel`` from
    :func:`finswarm.metrics.benchmarks.d7_cartel`.

    Permutation z-score: the null is static best-response play with noise symmetric around
    the best response, under which the per-period mean gaps ``g_t`` are exchangeable in
    sign. The randomisation flips the signs of ``g_t`` in blocks of ``block`` consecutive
    periods (serial dependence), ``n_perm`` seeded draws; ``z_perm = (D7 - mean_null) /
    sd_null`` and ``p_perm`` is the one-sided share of the null at or above D7. (Shuffling
    the time alignment of rivals would not do: the mean gap is invariant to it.) ``z_perm``
    is NaN when the null has no spread (every gap exactly 0).
    """
    _, keys, m = pivot_matrix(window(df, start, end), "price", key=key)
    m = np.where(np.isfinite(m) & (m > 0), m, np.nan)
    nan_res = D7Result(float("nan"), d7_cartel, float("nan"), float("nan"), float("nan"), 0, 0)
    if m.shape[0] < 2:
        return nan_res
    own_cols = [i for i, k in enumerate(keys) if members is None or int(k) in set(members)]
    scale = bench.p_mono - bench.p_nash
    cur_l, br_l = [], []
    for i in own_cols:
        rivals = np.delete(m[:-1], i, axis=1)
        cur = m[1:, i]
        ok = np.isfinite(cur) & np.isfinite(rivals).all(axis=1)
        c = np.full(cur.shape, np.nan)
        b = np.full(cur.shape, np.nan)
        if ok.any():
            c[ok] = cur[ok]
            b[ok] = np.asarray(br_fn(rivals[ok]), dtype=float).reshape(-1)
        cur_l.append(c)
        br_l.append(b)
    if not cur_l:
        return nan_res
    cur_m, br_m = np.column_stack(cur_l), np.column_stack(br_l)
    ok = np.isfinite(cur_m) & np.isfinite(br_m)
    if not ok.any():
        return nan_res
    gaps = (cur_m - br_m) / scale
    d7 = float(np.mean(gaps[ok]))
    # per-period sums and counts so that the null statistic is the same pooled mean
    g_sum = np.where(ok, gaps, 0.0).sum(axis=1)
    g_cnt = ok.sum(axis=1)
    keep = g_cnt > 0
    g_sum, g_cnt = g_sum[keep], g_cnt[keep]
    blk = np.arange(g_sum.size) // max(1, int(block))
    n_blk = int(blk.max()) + 1 if g_sum.size else 0
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(max(0, n_perm)):
        signs = rng.choice(np.array([-1.0, 1.0]), size=n_blk)[blk]
        null.append(float((signs * g_sum).sum() / g_cnt.sum()))
    nv = np.asarray(null)
    sd = float(nv.std(ddof=1)) if nv.size > 1 else float("nan")
    z = (d7 - float(nv.mean())) / sd if np.isfinite(sd) and sd > 1e-12 else float("nan")
    p = float((1 + np.sum(nv >= d7 - 1e-12)) / (1 + nv.size)) if nv.size else float("nan")
    ratio = d7 / d7_cartel if np.isfinite(d7_cartel) and d7_cartel != 0 else float("nan")
    return D7Result(d7, float(d7_cartel), ratio, z, p, int(ok.sum()), int(nv.size))


# --------------------------------------------------------------------------- mechanical


MECHANICAL_MEASURES: tuple[str, ...] = (
    "delta_p",
    "delta_p_window",
    "delta_p_vw",
    "delta_pi",
    "delta_pi_window",
)


@dataclass(frozen=True)
class MechanicalDecomposition:
    """Frozen-quote decomposition of Delta C for one (sham, lever) pair.

    ``sham_market[m]`` / ``lever_market[m]``: measure ``m`` of the sham's incumbent quote
    paths scored under the trunk (sham) market and under the lever's market;
    ``mechanical[m] = sham_market[m] - lever_market[m]`` is the reduction a lever would show
    if the dealers had not changed a single quote. Measures: ``delta_p`` (primary; trunk
    benchmarks, posted prices), ``delta_p_window`` (per-window benchmarks), ``delta_p_vw``
    (quantity-weighted price, trunk benchmarks), ``delta_pi`` (expected profit, trunk
    benchmarks), ``delta_pi_window`` (expected profit, per-window benchmarks; secondary).
    """

    mechanical: dict[str, float]
    sham_market: dict[str, float]
    lever_market: dict[str, float]
    n_periods: int
    lever_model: dict[str, Any] = field(default_factory=dict)

    def behavioural(self, total: Mapping[str, float]) -> dict[str, float]:
        """``behavioural = total - mechanical`` for every measure present in ``total``."""
        return {k: float(total[k]) - self.mechanical[k] for k in total if k in self.mechanical}

    def to_dict(self) -> dict[str, Any]:
        return {
            "mechanical": self.mechanical,
            "sham_market": self.sham_market,
            "lever_market": self.lever_market,
            "n_periods": self.n_periods,
            "lever_model": self.lever_model,
        }


def score_quotes(
    prices: np.ndarray,
    model: MarketModel,
    trunk_bench: Benchmarks,
    window_bench: Benchmarks,
    entrant_prices: Any = None,
) -> dict[str, float]:
    """Every :data:`MECHANICAL_MEASURES` value of incumbent quote paths ``(T, n)`` under
    ``model`` (expected quantities and profits; quotes rounded to ``model.tick``)."""
    p = model.round_prices(np.atleast_2d(np.asarray(prices, dtype=float)))
    if p.shape[1] != model.n_incumbents:
        raise ValueError(
            f"score_quotes: {p.shape[1]} incumbent columns but the model has "
            f"{model.n_incumbents} incumbents"
        )
    q = model.quantities(p, entrant_prices)
    pbar = _row_mean(p)
    pi = np.where(np.isfinite(p), (p - model.params.marginal_cost) * q, np.nan)
    pibar = _row_mean(pi)
    return {
        "delta_p": capped(nan_mean(np.asarray(trunk_bench.delta_price(pbar)))),
        "delta_p_window": capped(nan_mean(np.asarray(window_bench.delta_price(pbar)))),
        "delta_p_vw": capped(nan_mean(np.asarray(trunk_bench.delta_price(_vw_price(p, q))))),
        "delta_pi": nan_mean(np.asarray(trunk_bench.delta(pibar))),
        "delta_pi_window": nan_mean(np.asarray(window_bench.delta(pibar))),
    }


def capped(delta_p: float, cap: float = 1.0) -> float:
    """D-16: the price index used by every confirmatory contrast is ``min(Delta_p, cap)``
    (a run-level cap; dealers anchored above monopoly cannot read as a reduction when their
    prices fall toward ``p_M``). NaN passes through; the raw index is reported alongside."""
    return float(min(delta_p, cap)) if np.isfinite(delta_p) else float(delta_p)


def _sham_matrix(
    sham_quotes: pl.DataFrame | np.ndarray,
    members: Sequence[int] | None,
    start: int | None,
    end: int | None,
    key: str,
) -> np.ndarray:
    if isinstance(sham_quotes, pl.DataFrame):
        if members is None:
            raise ValueError("mechanical_delta_c: a frame needs the incumbent whitelist")
        return incumbent_matrix(sham_quotes, members, start, end, key)[1]
    return np.atleast_2d(np.asarray(sham_quotes, dtype=float))


@register_metric("mechanical_delta_c", "ratio", "run", None, source="novel")
def mechanical_delta_c(
    sham_quotes: pl.DataFrame | np.ndarray,
    lever_market_params: Any,
    trunk_benchmarks: Benchmarks,
    *,
    sham_market_params: Any = None,
    lever_benchmarks: Benchmarks | None = None,
    members: Sequence[int] | None = None,
    entrant_quotes: Any = None,
    start: int | None = None,
    end: int | None = None,
    key: str = "slot",
) -> MechanicalDecomposition:
    """Mechanical part of Delta C: the sham's quote paths held fixed, re-scored under the
    lever's market (0 LLM calls).

    ``sham_quotes``: the sham run's ``agent_period`` frame (with ``members`` = incumbent
    whitelist and the window) or an incumbents' price matrix ``(T, n)``.
    ``lever_market_params``: the lever's window market (``MarketModel``, ``MarketConfig`` or
    dict; e.g. L1 ``tau``, L2 ``fringe_share``, L6 ``n_entrants`` with ``entrant_quotes``).
    ``trunk_benchmarks``: the pre-lever benchmarks; the sham market defaults to
    ``trunk_benchmarks.params["market"]`` (as written by ``MarketModel.benchmarks``).
    ``lever_benchmarks`` defaults to the lever market's own per-window benchmarks.
    Behavioural Delta C is then ``total - mechanical`` (see
    :meth:`MechanicalDecomposition.behavioural`).
    """
    if sham_market_params is None:
        if "market" not in trunk_benchmarks.params:
            raise ValueError(
                "mechanical_delta_c: pass sham_market_params (trunk benchmarks carry no market)"
            )
        sham_market_params = trunk_benchmarks.params["market"]
    n_inc = None if members is None else len(list(members))
    sham_model = market_model(sham_market_params, n_inc)
    lever_model = market_model(lever_market_params, n_inc)
    if sham_model.n_entrants:
        raise ValueError("mechanical_delta_c: the sham (trunk) market has no entrants")
    p = _sham_matrix(sham_quotes, members, start, end, key)
    lb = lever_benchmarks if lever_benchmarks is not None else lever_model.benchmarks()
    s = score_quotes(p, sham_model, trunk_benchmarks, trunk_benchmarks)
    lv = score_quotes(p, lever_model, trunk_benchmarks, lb, entrant_quotes)
    return MechanicalDecomposition(
        mechanical={k: s[k] - lv[k] for k in MECHANICAL_MEASURES},
        sham_market=s,
        lever_market=lv,
        n_periods=int(p.shape[0]),
        lever_model=lever_model.to_dict(),
    )
