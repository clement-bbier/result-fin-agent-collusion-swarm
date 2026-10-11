"""Market-quality metrics: spreads, depth, price impact, volatility, efficiency, surplus.

Inputs are the ``quotes``, ``trades``, ``book_snapshots`` and ``metrics_per_period``
tables of ONE run. A dealer's missing side is logged as NaN or a negative sentinel; both
are treated as "no quote". The market-quality cost (MQC) of a lever is the percentage-
point loss of total surplus in the scripted competitive market (see
:func:`market_quality_cost`), measured from TRUE valuations without taste shocks in one
unit for every lever (:func:`true_value_surplus`: buyers' value minus price for executed
trades plus seller profit, the fringe through its trades). The logit inclusive value
(:func:`logit_surplus`, ``total_surplus(mode="logit_inclusive")``) is a separate metric and
is not part of the MQC axis.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl

from finswarm.metrics._frames import nan_mean, pivot_matrix, require, single_run, window
from finswarm.metrics.benchmarks import LogitParams
from finswarm.metrics.registry import register_metric

_INSIDE_SCHEMA = {
    "tick": pl.Int64,
    "best_bid": pl.Float64,
    "best_ask": pl.Float64,
    "bid_depth": pl.Float64,
    "ask_depth": pl.Float64,
    "mid": pl.Float64,
    "spread": pl.Float64,
}


def _clean(col: str) -> pl.Expr:
    c = pl.col(col).cast(pl.Float64)
    return pl.when(c.is_finite() & (c > 0)).then(c).otherwise(float("nan")).alias(col)


def inside_quotes(quotes: pl.DataFrame, llm_only: bool = False) -> pl.DataFrame:
    """As-of inside market at every tick where some dealer updated its quote.

    Each dealer's latest quote stays live until its next update (a NaN/negative side
    withdraws that side). Returns ``tick, best_bid, best_ask, bid_depth, ask_depth, mid,
    spread`` (NaN when a side is empty).
    """
    require(quotes, ["tick", "agent_id", "bid", "ask", "bid_size", "ask_size"], "inside_quotes")
    single_run(quotes, "inside_quotes")
    q = quotes
    if llm_only and "is_llm" in q.columns:
        q = q.filter(pl.col("is_llm"))
    if q.height == 0:
        return pl.DataFrame(schema=_INSIDE_SCHEMA)
    q = (
        q.select("tick", "agent_id", _clean("bid"), _clean("ask"), "bid_size", "ask_size")
        .group_by(["tick", "agent_id"], maintain_order=True)
        .agg(pl.all().last())
    )
    ticks = np.sort(q["tick"].unique().to_numpy())
    agents = sorted(q["agent_id"].unique().to_list())
    pos = {a: j for j, a in enumerate(agents)}
    shape = (len(ticks), len(agents))
    mats = {c: np.full(shape, np.nan) for c in ("bid", "ask", "bid_size", "ask_size")}
    seen = np.zeros(shape, dtype=bool)
    ii = np.searchsorted(ticks, q["tick"].to_numpy())
    jj = np.array([pos[a] for a in q["agent_id"].to_list()])
    seen[ii, jj] = True
    for c in mats:
        mats[c][ii, jj] = q[c].cast(pl.Float64).to_numpy()
    # forward-fill: each cell takes the dealer's latest update at or before that tick
    last = np.maximum.accumulate(np.where(seen, np.arange(shape[0])[:, None], -1), axis=0)
    cols = np.arange(shape[1])[None, :]
    for c in mats:
        mats[c] = np.where(last >= 0, mats[c][np.clip(last, 0, None), cols], np.nan)
    bid, ask = mats["bid"], mats["ask"]
    with np.errstate(invalid="ignore"):
        any_b = np.isfinite(bid).any(axis=1)
        any_a = np.isfinite(ask).any(axis=1)
        bb = np.where(any_b, np.nanmax(np.where(np.isfinite(bid), bid, -np.inf), axis=1), np.nan)
        ba = np.where(any_a, np.nanmin(np.where(np.isfinite(ask), ask, np.inf), axis=1), np.nan)
        bdep = np.nansum(np.where(np.isclose(bid, bb[:, None]), mats["bid_size"], 0.0), axis=1)
        adep = np.nansum(np.where(np.isclose(ask, ba[:, None]), mats["ask_size"], 0.0), axis=1)
    return pl.DataFrame(
        {
            "tick": ticks.astype(np.int64),
            "best_bid": bb,
            "best_ask": ba,
            "bid_depth": np.where(any_b, bdep, np.nan),
            "ask_depth": np.where(any_a, adep, np.nan),
            "mid": 0.5 * (bb + ba),
            "spread": ba - bb,
        },
        schema=_INSIDE_SCHEMA,
    )


def inside_from_book(book_snapshots: pl.DataFrame) -> pl.DataFrame:
    """Inside market from ``book_snapshots`` level 0 (side 1 = bid, -1 = ask)."""
    require(book_snapshots, ["tick", "side", "level", "price", "depth"], "inside_from_book")
    top = book_snapshots.filter(pl.col("level") == 0)
    b = top.filter(pl.col("side") == 1).select(
        "tick", pl.col("price").alias("best_bid"), pl.col("depth").alias("bid_depth")
    )
    a = top.filter(pl.col("side") == -1).select(
        "tick", pl.col("price").alias("best_ask"), pl.col("depth").alias("ask_depth")
    )
    d = b.join(a, on="tick", how="full", coalesce=True).sort("tick")
    return d.with_columns(
        ((pl.col("best_bid") + pl.col("best_ask")) / 2).alias("mid"),
        (pl.col("best_ask") - pl.col("best_bid")).alias("spread"),
    ).select(list(_INSIDE_SCHEMA))


def _mids_at(inside: pl.DataFrame, ticks: np.ndarray) -> np.ndarray:
    """Prevailing (as-of, backward) mid at each tick."""
    t = inside["tick"].to_numpy()
    m = inside["mid"].to_numpy()
    pos = np.searchsorted(t, ticks, side="right") - 1
    out = np.full(len(ticks), np.nan)
    ok = pos >= 0
    out[ok] = m[pos[ok]]
    return out


@register_metric("quoted_spread", "price", "run", True, source="hasbrouck2007")
def quoted_spread(quotes: pl.DataFrame, relative: bool = False, llm_only: bool = False) -> float:
    """Mean inside spread (best ask - best bid) over quote-update ticks."""
    ins = inside_quotes(quotes, llm_only=llm_only)
    s = ins["spread"].to_numpy()
    if relative:
        s = s / ins["mid"].to_numpy()
    return nan_mean(s)


@register_metric("depth", "size", "run", False, source="hasbrouck2007")
def depth(quotes: pl.DataFrame, llm_only: bool = False) -> float:
    """Mean size quoted at the inside (bid depth + ask depth)."""
    ins = inside_quotes(quotes, llm_only=llm_only)
    return nan_mean(ins["bid_depth"].to_numpy() + ins["ask_depth"].to_numpy())


def _signs(trades: pl.DataFrame, mids: np.ndarray) -> np.ndarray:
    agg = trades["aggressor"].cast(pl.Float64).to_numpy()
    price = trades["price"].to_numpy()
    return np.where(agg != 0, np.sign(agg), np.sign(price - mids))


@register_metric("effective_spread", "price", "run", True, source="hasbrouck2007")
def effective_spread(trades: pl.DataFrame, inside: pl.DataFrame, relative: bool = False) -> float:
    """Size-weighted mean of ``2 d (p - m)`` with ``m`` the prevailing mid at the trade.

    ``d`` is the aggressor sign (+1 buy, -1 sell); for batch trades (aggressor 0)
    ``d = sign(p - m)``, i.e. the absolute deviation.
    """
    require(trades, ["tick", "price", "size", "aggressor"], "effective_spread")
    if trades.height == 0:
        return float("nan")
    m = _mids_at(inside, trades["tick"].to_numpy())
    es = 2.0 * _signs(trades, m) * (trades["price"].to_numpy() - m)
    if relative:
        es = es / m
    w = trades["size"].to_numpy()
    ok = np.isfinite(es)
    return float(np.sum(es[ok] * w[ok]) / np.sum(w[ok])) if ok.any() else float("nan")


@register_metric("realised_spread", "price", "run", True, source="hasbrouck2007")
def realised_spread(trades: pl.DataFrame, inside: pl.DataFrame, horizon: int = 10) -> float:
    """Size-weighted mean of ``2 d (p - m_{t+horizon})`` (liquidity providers' revenue)."""
    require(trades, ["tick", "price", "size", "aggressor"], "realised_spread")
    if trades.height == 0:
        return float("nan")
    ticks = trades["tick"].to_numpy()
    d = _signs(trades, _mids_at(inside, ticks))
    rs = 2.0 * d * (trades["price"].to_numpy() - _mids_at(inside, ticks + horizon))
    w = trades["size"].to_numpy()
    ok = np.isfinite(rs)
    return float(np.sum(rs[ok] * w[ok]) / np.sum(w[ok])) if ok.any() else float("nan")


def kyle_lambda_arrays(dmid: np.ndarray, signed_volume: np.ndarray) -> tuple[float, float, float]:
    """OLS of mid changes on signed volume with intercept: ``(lambda, se, r2)``."""
    y = np.asarray(dmid, dtype=float)
    x = np.asarray(signed_volume, dtype=float)
    ok = np.isfinite(y) & np.isfinite(x)
    y, x = y[ok], x[ok]
    if len(y) < 3 or np.var(x) == 0:
        return float("nan"), float("nan"), float("nan")
    X = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    dof = len(y) - 2
    s2 = float(resid @ resid) / dof
    cov = s2 * np.linalg.inv(X.T @ X)
    r2 = 1.0 - float(resid @ resid) / float(((y - y.mean()) ** 2).sum()) if np.var(y) else 0.0
    return float(coef[1]), float(np.sqrt(cov[1, 1])), r2


@register_metric("kyle_lambda", "price/size", "run", True, source="kyle1985")
def kyle_lambda(trades: pl.DataFrame, mids: pl.DataFrame, bucket: str = "period") -> float:
    """Kyle's lambda: OLS slope of per-bucket mid changes on per-bucket signed volume.

    ``mids`` has ``bucket`` and ``mid`` (or ``mid_mean`` as in ``metrics_per_period``).
    Signed volume is ``sum(aggressor * size)``; buckets without trades count as zero.
    """
    require(trades, [bucket, "size", "aggressor"], "kyle_lambda")
    mcol = "mid" if "mid" in mids.columns else "mid_mean"
    require(mids, [bucket, mcol], "kyle_lambda")
    m = mids.select(bucket, pl.col(mcol).alias("mid")).sort(bucket)
    sv = trades.group_by(bucket).agg(
        (pl.col("aggressor").cast(pl.Float64) * pl.col("size")).sum().alias("sv")
    )
    d = m.join(sv, on=bucket, how="left").with_columns(pl.col("sv").fill_null(0.0))
    mid = d["mid"].to_numpy()
    return kyle_lambda_arrays(np.diff(mid), d["sv"].to_numpy()[1:])[0]


def _mid_series(x: pl.DataFrame | np.ndarray) -> np.ndarray:
    if isinstance(x, pl.DataFrame):
        col = "mid_mean" if "mid_mean" in x.columns else "mid"
        sort = "period" if "period" in x.columns else "tick"
        return x.sort(sort)[col].cast(pl.Float64).to_numpy()
    return np.asarray(x, dtype=float)


@register_metric("realised_volatility", "log-return/period", "run", None, source="andersen2001")
def realised_volatility(mids: pl.DataFrame | np.ndarray) -> float:
    """Root mean squared log return of the mid between consecutive observations."""
    m = _mid_series(mids)
    m = m[np.isfinite(m) & (m > 0)]
    if len(m) < 2:
        return float("nan")
    r = np.diff(np.log(m))
    return float(np.sqrt(np.mean(r**2)))


@register_metric("pricing_error", "ratio", "run", True, source="hasbrouck1993")
def pricing_error(
    metrics_per_period: pl.DataFrame,
    relative: bool = True,
    start: int | None = None,
    end: int | None = None,
) -> float:
    """Mean absolute deviation of the mid from the fundamental (relative by default)."""
    require(metrics_per_period, ["mid_mean", "fundamental"], "pricing_error")
    d = (
        window(metrics_per_period, start, end)
        if "period" in metrics_per_period.columns
        else (metrics_per_period)
    )
    m = d["mid_mean"].to_numpy()
    f = d["fundamental"].to_numpy()
    err = np.abs(m - f)
    if relative:
        err = err / np.abs(f)
    return nan_mean(err)


@register_metric("volume", "size/period", "run", None, source="novel")
def volume(trades: pl.DataFrame, n_periods: int | None = None) -> float:
    """Traded size per period (``n_periods`` defaults to the periods seen in ``trades``)."""
    require(trades, ["size", "period"], "volume")
    if trades.height == 0:
        return 0.0
    n = n_periods if n_periods is not None else trades["period"].n_unique()
    return float(trades["size"].sum()) / n


def _known(trades: pl.DataFrame, cols: Sequence[str]) -> pl.DataFrame:
    """Trades with finite values in ``cols``, excluding the closed-form aggregate buyer
    account (``buyer_id = -1``), which carries no individual reservation value."""
    require(trades, ["size", *cols], "surplus")
    d = trades.filter(pl.all_horizontal([pl.col(c).is_finite() for c in cols]))
    if "buyer_id" in d.columns:
        d = d.filter(pl.col("buyer_id") >= 0)
    return d


@register_metric("buyer_surplus", "currency", "run", False, source="novel")
def buyer_surplus(trades: pl.DataFrame) -> float:
    """``sum(size * (buyer_value - price))`` over trades with a known buyer value."""
    d = _known(trades, ["buyer_value", "price"])
    return float((d["size"] * (d["buyer_value"] - d["price"])).sum())


@register_metric("seller_surplus", "currency", "run", True, source="novel")
def seller_surplus(trades: pl.DataFrame) -> float:
    """``sum(size * (price - seller_cost))`` over trades with a known seller cost."""
    d = _known(trades, ["seller_cost", "price"])
    return float((d["size"] * (d["price"] - d["seller_cost"])).sum())


# --------------------------------------------------------------------------- logit surplus


def _price_matrix(data: pl.DataFrame | np.ndarray, key: str = "agent_id") -> np.ndarray:
    if isinstance(data, pl.DataFrame):
        _, _, m = pivot_matrix(data, "price", key=key)
        return m
    m = np.asarray(data, dtype=float)
    return m[:, None] if m.ndim == 1 else m


def logit_surplus(
    prices: pl.DataFrame | np.ndarray,
    params: LogitParams | object | None = None,
    values: np.ndarray | None = None,
    key: str = "agent_id",
) -> dict[str, np.ndarray]:
    """Per-period expected surplus of the logit routing at the posted asks.

    With ``tau`` the routing temperature, ``u0`` the outside utility and buyer values ``v``
    (default ``alpha * a``; pass ``values`` of shape ``(T,)`` or ``(T, m)`` for a sample of
    reservation values per period), per unit of buyer mass ``beta``::

        CS_t = beta * mean_v tau * log(1 + sum_i exp((v - p_it - u0) / tau))
        PS_t = beta * mean_v sum_i (p_it - c alpha) * share_i(v)
        TS_t = CS_t + PS_t

    ``CS_t`` is the logit inclusive value (expected best utility over the outside option),
    which includes the taste shocks and is never negative, unlike trade-based surplus from
    ``buyer_value``. NaN prices mean "not quoting".
    """
    prm = LogitParams.coerce(params)
    p = _price_matrix(prices, key)
    t = p.shape[0]
    v = np.full((t, 1), prm.alpha * prm.a) if values is None else np.asarray(values, float)
    if v.ndim == 1:
        v = v[:, None]
    tau, u0, mc = prm.tau_eff, prm.u0, prm.marginal_cost
    x = (v[:, :, None] - p[:, None, :] - u0) / tau  # (T, m, n)
    x = np.where(np.isfinite(x), x, -np.inf)
    mx = np.maximum(x.max(axis=-1), 0.0)  # the outside option has x = 0
    ex = np.exp(x - mx[..., None])
    denom = ex.sum(axis=-1) + np.exp(-mx)
    cs = tau * (mx + np.log(denom))  # tau * log(1 + sum exp(x)), overflow-safe
    share = ex / denom[..., None]
    margin = np.where(np.isfinite(p), p - mc, 0.0)[:, None, :]
    ps = (share * margin).sum(axis=-1)
    cs_t, ps_t = prm.beta * cs.mean(axis=1), prm.beta * ps.mean(axis=1)
    return {"cs": cs_t, "ps": ps_t, "ts": cs_t + ps_t}


def _logit_efficiency(
    prices: pl.DataFrame | np.ndarray, params: object, values: np.ndarray | None, key: str
) -> float:
    p = _price_matrix(prices, key)
    ts = logit_surplus(p, params, values)["ts"]
    mc = LogitParams.coerce(params).marginal_cost
    best = logit_surplus(np.where(np.isfinite(p), mc, np.nan), params, values)["ts"]
    den = float(np.nansum(best))
    return float(np.nansum(ts)) / den if den > 0 else float("nan")


def _trades_efficiency(
    trades: pl.DataFrame, buyer_values: np.ndarray | None, seller_costs: np.ndarray | None
) -> float:
    d = _known(trades, ["buyer_value", "seller_cost"])
    realised = float((d["size"] * (d["buyer_value"] - d["seller_cost"])).sum())
    if buyer_values is not None and seller_costs is not None:
        best = max_surplus(buyer_values, seller_costs)
    else:
        sz = d["size"].to_numpy()
        best = max_surplus(d["buyer_value"].to_numpy(), d["seller_cost"].to_numpy(), sz, sz)
    return realised / best if best > 0 else float("nan")


@register_metric("total_surplus", "currency/period", "run", False, source="smith1962")
def total_surplus(
    data: pl.DataFrame | np.ndarray,
    mode: str = "logit_inclusive",
    params: LogitParams | object | None = None,
    values: np.ndarray | None = None,
    n_periods: int | None = None,
    key: str = "agent_id",
) -> float:
    """Total surplus per period.

    ``mode="logit_inclusive"`` (default; ``data`` = ``agent_period`` or a periods x dealers
    price matrix): mean of :func:`logit_surplus` ``TS_t``. ``mode="trades"`` (``data`` =
    ``trades``): realised ``sum(size * (buyer_value - seller_cost))`` per period; in agents
    mode it can be negative because ``buyer_value`` omits the logit taste shock, and the
    closed-form aggregate account (``buyer_id = -1``) is excluded.
    """
    if mode == "logit_inclusive":
        return nan_mean(logit_surplus(data, params, values, key)["ts"])
    if mode == "trades":
        d = _known(data, ["buyer_value", "seller_cost"])
        tot = float((d["size"] * (d["buyer_value"] - d["seller_cost"])).sum())
        n = n_periods or (data["period"].n_unique() if "period" in data.columns else 1)
        return tot / max(n, 1)
    raise ValueError(f"unknown surplus mode {mode!r}")


def max_surplus(
    buyer_values: np.ndarray,
    seller_costs: np.ndarray,
    buyer_sizes: np.ndarray | None = None,
    seller_sizes: np.ndarray | None = None,
) -> float:
    """Surplus of the competitive allocation: match units from the highest value and the
    lowest cost while ``value > cost`` (unit-step demand and supply schedules)."""
    v = np.asarray(buyer_values, dtype=float)
    c = np.asarray(seller_costs, dtype=float)
    vs = np.ones_like(v) if buyer_sizes is None else np.asarray(buyer_sizes, dtype=float)
    cs = np.ones_like(c) if seller_sizes is None else np.asarray(seller_sizes, dtype=float)
    ov, oc = np.argsort(-v), np.argsort(c)
    v, vs, c, cs = v[ov], vs[ov].copy(), c[oc], cs[oc].copy()
    i = j = 0
    total = 0.0
    while i < len(v) and j < len(c) and v[i] > c[j]:
        q = min(vs[i], cs[j])
        total += q * (v[i] - c[j])
        vs[i] -= q
        cs[j] -= q
        if vs[i] <= 1e-12:
            i += 1
        if cs[j] <= 1e-12:
            j += 1
    return total


@register_metric("gode_sunder_efficiency", "ratio", "run", False, source="gode_sunder1993")
def gode_sunder_efficiency(
    data: pl.DataFrame | np.ndarray,
    mode: str = "logit_inclusive",
    params: LogitParams | object | None = None,
    values: np.ndarray | None = None,
    buyer_values: np.ndarray | None = None,
    seller_costs: np.ndarray | None = None,
    key: str = "agent_id",
) -> float:
    """Allocative efficiency: realised surplus / surplus of the competitive allocation.

    ``mode="logit_inclusive"`` (default; ``data`` = ``agent_period`` or a price matrix):
    expected total surplus at the posted asks over the total surplus with every quoting
    dealer at marginal cost (the welfare optimum of logit routing), both from
    :func:`logit_surplus`. ``mode="trades"`` (``data`` = ``trades``): realised
    ``sum(size * (buyer_value - seller_cost))`` over the surplus of the competitive
    allocation of the traders in ``trades`` (or of the full population when
    ``buyer_values`` and ``seller_costs`` are given). Trades mode equals 1 for the
    competitive allocation; in agents mode it can be negative because ``buyer_value`` omits
    the logit taste shock.
    """
    if mode == "logit_inclusive":
        return _logit_efficiency(data, params, values, key)
    if mode == "trades":
        return _trades_efficiency(data, buyer_values, seller_costs)
    raise ValueError(f"unknown efficiency mode {mode!r}")


def mqc_pp(ts_reference: float, ts_treated: float) -> float:
    """``100 * (TS_reference - TS_treated) / TS_reference`` (percentage points)."""
    if ts_reference == 0 or not np.isfinite(ts_reference):
        return float("nan")
    return 100.0 * (ts_reference - ts_treated) / ts_reference


# --------------------------------------------------------------------------- true-value surplus


def true_value_surplus_series(
    trades: pl.DataFrame,
    start: int | None = None,
    end: int | None = None,
    periods: Sequence[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-period surplus from TRUE valuations (no taste shock), in currency per period.

    ``TS_t = sum size * (buyer_value - price) + sum size * (price - seller_cost)`` over the
    trades executed in period ``t``: the buyers' surplus at their reservation value (the
    swarm's ``buyer_value`` omits the logit taste shock by construction) plus the sellers'
    profit (dealers, entrants and fringe sellers alike, each through its own trades; for
    a one-sided dealer market the seller part equals ``agent_period.profit``). Periods
    without trades count as 0 (``periods`` defaults to ``start..end`` when both are given,
    else to the periods seen). Trades with an unknown value or cost and the closed-form
    aggregate buyer (``buyer_id = -1``) are excluded.
    """
    require(trades, ["period", "price", "size", "buyer_value", "seller_cost"], "true_value")
    d = _known(window(trades, start, end), ["buyer_value", "seller_cost", "price"])
    d = d.with_columns(
        (
            pl.col("size") * (pl.col("buyer_value") - pl.col("price"))
            + pl.col("size") * (pl.col("price") - pl.col("seller_cost"))
        ).alias("_ts")
    )
    per = d.group_by("period").agg(pl.col("_ts").sum()).sort("period")
    if periods is None:
        if start is not None and end is not None:
            periods = range(int(start), int(end) + 1)
        else:
            periods = per["period"].to_list()
    idx = np.asarray(list(periods), dtype=np.int64)
    ts = np.zeros(idx.size)
    pos = {int(p): i for i, p in enumerate(idx)}
    for p, v in zip(per["period"].to_list(), per["_ts"].to_list(), strict=True):
        if int(p) in pos:
            ts[pos[int(p)]] = float(v)
    return idx, ts


@register_metric("true_value_surplus", "currency/period", "run", False, source="smith1962")
def true_value_surplus(
    trades: pl.DataFrame,
    start: int | None = None,
    end: int | None = None,
    periods: Sequence[int] | None = None,
) -> float:
    """Mean per-period true-value surplus (see :func:`true_value_surplus_series`); the MQC
    axis unit for every lever."""
    _, ts = true_value_surplus_series(trades, start, end, periods)
    return float(ts.mean()) if ts.size else float("nan")


def expected_true_value_surplus(
    prices: pl.DataFrame | np.ndarray,
    params: object,
    entrant_prices: object = None,
    values: np.ndarray | None = None,
    key: str = "agent_id",
) -> np.ndarray:
    """Expected per-period true-value surplus at posted asks under one arm's market model.

    ``TS_t = sum_sellers q_it (v_t - c alpha)`` with ``q`` the expected routed quantities of
    ``params`` (coerced with :func:`finswarm.metrics.benchmarks.market_model`; the number of
    incumbents is the number of price columns) and ``v`` the buyers' value without taste
    shock (default ``alpha * a``; ``values`` of shape ``(T,)`` overrides it).
    """
    from finswarm.metrics.benchmarks import market_model

    p = _price_matrix(prices, key)
    m = market_model(params, p.shape[1])
    q_inc, q_ent = m.all_quantities(p, entrant_prices)
    qtot = q_inc.sum(axis=1) + (q_ent.sum(axis=1) if q_ent.size else 0.0)
    prm = m.params
    v = np.full(p.shape[0], prm.alpha * prm.a) if values is None else np.asarray(values, float)
    return (v - prm.marginal_cost) * qtot


@register_metric("mqc", "pp", "run", None, source="novel")
def market_quality_cost(
    reference: pl.DataFrame | np.ndarray,
    treated: pl.DataFrame | np.ndarray,
    mode: str = "true_value",
    *,
    reference_params: object = None,
    treated_params: object = None,
    reference_entrants: object = None,
    treated_entrants: object = None,
    values: np.ndarray | None = None,
    start: int | None = None,
    end: int | None = None,
) -> float:
    """Market-quality cost of a lever (x axis of the frontier), in pp of total surplus.

    ``MQC = 100 * (TS_reference - TS_treated) / TS_reference`` with ``TS`` the true-value
    surplus per period in ONE unit for every lever (buyers' value minus price for executed
    trades plus seller profit, the fringe through its trades; no logit taste shock and no
    inclusive-value variety term). ``reference`` is the scripted SC1 sham, ``treated`` the
    same market under the lever. With ``trades`` frames the realised surplus is used and no
    parameters are needed; with price matrices or ``agent_period`` frames the expected
    surplus is computed under EACH ARM'S OWN market: ``reference_params`` (sham) and
    ``treated_params`` (lever) are both required and never shared implicitly. The logit
    inclusive value stays available as :func:`total_surplus` ``mode="logit_inclusive"``
    (a separate metric) but is not part of the MQC axis.
    """
    if mode != "true_value":
        raise ValueError(
            f"market_quality_cost: mode {mode!r} is not part of the MQC axis; the logit "
            "inclusive variety term is reported separately (total_surplus / logit_surplus)"
        )

    def ts(data: pl.DataFrame | np.ndarray, prm: object, ent: object, arm: str) -> float:
        if isinstance(data, pl.DataFrame) and "buyer_value" in data.columns:
            return true_value_surplus(data, start, end)
        if prm is None:
            raise ValueError(f"market_quality_cost: {arm}_params required for price inputs")
        if isinstance(data, pl.DataFrame):
            data = window(data, start, end)
        return nan_mean(expected_true_value_surplus(data, prm, ent, values))

    return mqc_pp(
        ts(reference, reference_params, reference_entrants, "reference"),
        ts(treated, treated_params, treated_entrants, "treated"),
    )
