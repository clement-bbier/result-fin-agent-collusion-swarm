"""Dependence statistics between dealers on a ``periods x agents`` matrix.

Input is a numpy matrix of quote changes (or an ``agent_period`` frame, differenced
with ``diff=True``). Statistics:

* :func:`lagged_xcorr` -- max absolute lagged cross-correlation, i leads j;
* :func:`granger_pairwise` -- bivariate VAR(p) F-tests (statsmodels);
* :func:`transfer_entropy` -- 3-bin plug-in TE, bias-corrected with circular-shift
  surrogates;
* :func:`net_influence` -- Bonjour et al. (2022) net influence on discrete actions
  conditioned on a discretised market state, with a within-state permutation test;
* :func:`net_of_ghost` -- any statistic net of the ghost baseline (rivals replayed
  open-loop), with a permutation p-value.

Validation fixtures (:func:`simulate_rps`, :func:`validate_net_influence_rps`) reproduce a
scripted 3-player rock-paper-scissors game with one colluding pair.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl
from scipy import stats

from finswarm.metrics._frames import as_matrix
from finswarm.metrics.registry import register_metric

Matrix = np.ndarray | pl.DataFrame


def _off_diag(n: int) -> np.ndarray:
    return ~np.eye(n, dtype=bool)


# --------------------------------------------------------------------------- xcorr


@dataclass(frozen=True)
class XCorrResult:
    matrix: np.ndarray  # [i, j] = max_l |corr(x_i[t-l], x_j[t])|
    by_lag: np.ndarray  # (max_lag, n, n), signed correlations
    summary: float  # mean of the off-diagonal entries of ``matrix``


@register_metric("lagged_xcorr", "abs corr", "run", True, source="box_jenkins1970")
def lagged_xcorr(x: Matrix, max_lag: int = 3, diff: bool = False) -> XCorrResult:
    """Max over lags ``1..max_lag`` of ``|corr(x_i[t-l], x_j[t])|`` for every ordered pair."""
    m = as_matrix(x, diff=diff)
    t, n = m.shape
    by_lag = np.zeros((max_lag, n, n))
    for lag in range(1, max_lag + 1):
        if t - lag < 3:
            break
        a, b = m[:-lag], m[lag:]
        ok = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
        a, b = a[ok], b[ok]
        with np.errstate(invalid="ignore", divide="ignore"):
            za = (a - a.mean(0)) / a.std(0)
            zb = (b - b.mean(0)) / b.std(0)
            c = (za.T @ zb) / len(a)
        by_lag[lag - 1] = np.nan_to_num(c, nan=0.0)
    mat = np.abs(by_lag).max(axis=0)
    np.fill_diagonal(mat, np.nan)
    summary = float(np.nanmean(mat[_off_diag(n)])) if n > 1 else float("nan")
    return XCorrResult(matrix=mat, by_lag=by_lag, summary=summary)


# --------------------------------------------------------------------------- Granger


@dataclass(frozen=True)
class GrangerResult:
    F: np.ndarray  # [i, j] = F statistic for "i Granger-causes j"
    p: np.ndarray
    share_significant: float  # share of ordered pairs with p < alpha
    lags: int


@register_metric("granger_pairwise", "share", "run", True, source="granger1969")
def granger_pairwise(
    x: Matrix, lags: int = 2, alpha: float = 0.05, diff: bool = False
) -> GrangerResult:
    """Bivariate VAR(``lags``) Granger F-test for every ordered pair (statsmodels).

    Pairs where either series is (numerically) constant get ``F = 0, p = 1``.
    """
    from statsmodels.tsa.api import VAR  # lazy: heavy import

    m = as_matrix(x, diff=diff)
    m = m[np.isfinite(m).all(axis=1)]
    n = m.shape[1]
    F = np.zeros((n, n))
    P = np.ones((n, n))
    np.fill_diagonal(F, np.nan)
    np.fill_diagonal(P, np.nan)
    sd = m.std(axis=0) if len(m) else np.zeros(n)
    for i in range(n):
        for j in range(n):
            if i == j or len(m) < 3 * lags + 5 or sd[i] < 1e-12 or sd[j] < 1e-12:
                continue
            data = np.column_stack([m[:, j], m[:, i]])
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = VAR(data).fit(lags, trend="c")
                    tc = res.test_causality(caused=0, causing=1, kind="f")
                F[i, j] = float(tc.test_statistic)
                P[i, j] = float(tc.pvalue)
            except (np.linalg.LinAlgError, ValueError):
                # singular VAR covariance (e.g. an exact copy of a lagged rival):
                # single-equation F-test with rank-aware least squares
                F[i, j], P[i, j] = granger_f_ols(m[:, j], m[:, i], lags)
    off = _off_diag(n)
    share = float(np.mean(P[off] < alpha)) if n > 1 else float("nan")
    return GrangerResult(F=F, p=P, share_significant=share, lags=lags)


def granger_f_ols(y: np.ndarray, x: np.ndarray, lags: int) -> tuple[float, float]:
    """Single-equation Granger F-test of ``x -> y`` (restricted: own lags + constant)."""
    t = len(y)
    yy = y[lags:]
    own = np.column_stack([y[lags - k : t - k] for k in range(1, lags + 1)])
    oth = np.column_stack([x[lags - k : t - k] for k in range(1, lags + 1)])
    xr = np.column_stack([np.ones(len(yy)), own])
    xu = np.column_stack([xr, oth])

    def rss(z: np.ndarray) -> float:
        coef, *_ = np.linalg.lstsq(z, yy, rcond=None)
        r = yy - z @ coef
        return float(r @ r)

    rr, ru = rss(xr), rss(xu)
    q = np.linalg.matrix_rank(xu) - np.linalg.matrix_rank(xr)
    dfd = len(yy) - np.linalg.matrix_rank(xu)
    if q <= 0 or dfd <= 0 or rr - ru <= 1e-12 * max(rr, 1e-300):
        return 0.0, 1.0
    if ru <= 1e-12 * rr:
        return float("inf"), 0.0
    f = ((rr - ru) / q) / (ru / dfd)
    return float(f), float(stats.f.sf(f, q, dfd))


# --------------------------------------------------------------------------- transfer entropy


def discretise(
    x: np.ndarray, bins: int = 3, method: str = "sign", eps: float | None = None
) -> np.ndarray:
    """Map a real series to integer codes ``0..bins-1``.

    ``sign`` (bins=3): down / flat (``|x| <= eps``) / up -- natural for quote changes.
    ``quantile``: equal-frequency bins.
    """
    v = np.asarray(x, dtype=float)
    if method == "sign":
        if bins != 3:
            raise ValueError("sign discretisation has 3 bins")
        if eps is None:
            scale = np.nanmax(np.abs(v)) if np.isfinite(v).any() else 1.0
            eps = 1e-9 * max(1.0, float(scale))
        return np.where(v > eps, 2, np.where(v < -eps, 0, 1)).astype(np.int64)
    if method == "quantile":
        edges = np.nanquantile(v, np.linspace(0, 1, bins + 1)[1:-1])
        return np.searchsorted(edges, v, side="right").astype(np.int64)
    raise ValueError(f"unknown discretisation {method!r}")


def _te_counts(c: np.ndarray) -> np.ndarray:
    """Plug-in TE (nats) from count cubes ``c[..., y_next, y_past, x_past]``."""
    c = c.astype(float)
    n = c.sum(axis=(-3, -2, -1), keepdims=True)
    c_yp_xp = c.sum(axis=-3, keepdims=True)
    c_yn_yp = c.sum(axis=-1, keepdims=True)
    c_yp = c.sum(axis=(-3, -1), keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(c > 0, c / n * np.log(c * c_yp / (c_yn_yp * c_yp_xp)), 0.0)
    return term.sum(axis=(-3, -2, -1))


def _te_pair(y: np.ndarray, x: np.ndarray, k: int, lag: int, shifts: np.ndarray) -> tuple:
    y_next, y_past = y[lag:], y[lag - 1 : -1]
    x_past = x[:-lag]
    base = (y_next * k + y_past) * k
    te = float(_te_counts(np.bincount(base + x_past, minlength=k**3).reshape(k, k, k)))
    if shifts.size == 0:
        return te, np.array([])
    xs = np.stack([np.roll(x, int(s))[:-lag] for s in shifts])
    idx = base[None, :] + xs + (np.arange(len(shifts)) * k**3)[:, None]
    cubes = np.bincount(idx.ravel(), minlength=len(shifts) * k**3).reshape(-1, k, k, k)
    return te, _te_counts(cubes)


@dataclass(frozen=True)
class TEResult:
    te: np.ndarray  # [i, j] = TE_{i -> j} (nats), plug-in
    te_net: np.ndarray  # te minus the mean of circular-shift surrogates
    p: np.ndarray  # surrogate p-values
    summary: float  # mean off-diagonal te_net
    max_net: float


@register_metric("transfer_entropy", "nats", "run", True, source="schreiber2000")
def transfer_entropy(
    x: Matrix,
    lag: int = 1,
    bins: int = 3,
    method: str = "sign",
    n_surrogates: int = 100,
    seed: int = 0,
    diff: bool = False,
) -> TEResult:
    """Pairwise transfer entropy with 3-bin discretisation and lag 1 (default).

    Bias correction: subtract the mean TE over ``n_surrogates`` circular shifts of the
    source series (which keep its marginal and autocorrelation but break the coupling).
    """
    m = as_matrix(x, diff=diff)
    m = m[np.isfinite(m).all(axis=1)]
    t, n = m.shape
    codes = np.column_stack([discretise(m[:, i], bins, method) for i in range(n)])
    rng = np.random.default_rng(seed)
    te = np.full((n, n), np.nan)
    net = np.full((n, n), np.nan)
    p = np.full((n, n), np.nan)
    if t <= lag + 2:
        return TEResult(te, net, p, float("nan"), float("nan"))
    lo, hi = min(lag + 1, t - 1), max(lag + 2, t - lag)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            shifts = rng.integers(lo, hi, size=n_surrogates)
            obs, surr = _te_pair(codes[:, j], codes[:, i], bins, lag, shifts)
            te[i, j] = obs
            if surr.size:
                net[i, j] = obs - surr.mean()
                p[i, j] = (1 + np.sum(surr >= obs - 1e-12)) / (1 + surr.size)
            else:
                net[i, j] = obs
    off = _off_diag(n)
    return TEResult(
        te=te,
        te_net=net,
        p=p,
        summary=float(np.nanmean(net[off])) if n > 1 else float("nan"),
        max_net=float(np.nanmax(net[off])) if n > 1 else float("nan"),
    )


# --------------------------------------------------------------------------- net influence


def _codes(a: np.ndarray) -> tuple[np.ndarray, int]:
    v = np.asarray(a)
    if np.issubdtype(v.dtype, np.integer) or np.issubdtype(v.dtype, np.bool_):
        u, inv = np.unique(v, return_inverse=True)
        return inv.astype(np.int64), max(len(u), 1)
    return discretise(v.astype(float), 3, "sign"), 3


def cond_mutual_info(
    x: np.ndarray, y: np.ndarray, s: np.ndarray, kx: int, ky: int, ks: int
) -> float:
    """Plug-in ``I(X; Y | S)`` in nats for integer codes."""
    c = np.bincount((s * kx + x) * ky + y, minlength=ks * kx * ky).reshape(ks, kx, ky)
    return float(_cmi_counts(c))


def _cmi_counts(c: np.ndarray) -> np.ndarray:
    c = c.astype(float)
    n = c.sum(axis=(-3, -2, -1), keepdims=True)
    c_s = c.sum(axis=(-2, -1), keepdims=True)
    c_sx = c.sum(axis=-1, keepdims=True)
    c_sy = c.sum(axis=-2, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(c > 0, c / n * np.log(c * c_s / (c_sx * c_sy)), 0.0)
    return term.sum(axis=(-3, -2, -1))


@dataclass(frozen=True)
class NetInfluenceResult:
    gamma: np.ndarray  # gamma[i, j] = I(A_i[t-lag]; A_j[t] | S_t)
    Gamma: np.ndarray  # gamma[i, j] - max_{k != i, j} gamma[k, j]
    pair_stat: np.ndarray  # symmetric: min(Gamma[i, j], Gamma[j, i])
    p_values: np.ndarray  # symmetric: bootstrap p-value of H0 "pair_stat <= 0"
    flagged: list[tuple[int, int]]
    statistic: float  # max over pairs of pair_stat
    max_pair: tuple[int, int]
    alpha: float = 0.05
    extra: dict[str, Any] = field(default_factory=dict)


def _gamma_batch(
    src: list[np.ndarray], dst: list[np.ndarray], ks: list[int], s: np.ndarray, kS: int
) -> np.ndarray:
    """``gamma[..., i, j] = I(src_i; dst_j | s)`` for code arrays of shape ``(..., T)``."""
    n = len(src)
    lead = s.shape[:-1]
    g = np.full((*lead, n, n), np.nan)
    nb = int(np.prod(lead)) if lead else 1
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            size = kS * ks[i] * ks[j]
            idx = (s * ks[i] + src[i]) * ks[j] + dst[j]
            off = (np.arange(nb) * size).reshape(*lead, 1) if lead else 0
            cnt = np.bincount((idx + off).ravel(), minlength=nb * size)
            g[..., i, j] = _cmi_counts(cnt.reshape(*lead, kS, ks[i], ks[j]))
    return g


def _Gamma(g: np.ndarray) -> np.ndarray:
    """``Gamma[..., i, j] = g[..., i, j] - max_{k != i, j} g[..., k, j]`` (0 if no k)."""
    n = g.shape[-1]
    G = np.full(g.shape, np.nan)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            others = [k for k in range(n) if k not in (i, j)]
            sub = g[..., others, j].max(axis=-1) if others else 0.0
            G[..., i, j] = g[..., i, j] - sub
    return G


def _boot_indices(t: int, n_boot: int, block: int, rng: np.random.Generator) -> np.ndarray:
    if block <= 1:
        return rng.integers(0, t, size=(n_boot, t))
    nblk = -(-t // block)
    starts = rng.integers(0, max(t - block + 1, 1), size=(n_boot, nblk))
    return (starts[..., None] + np.arange(block)).reshape(n_boot, -1)[:, :t]


@register_metric("net_influence", "nats", "pair", True, source="bonjour2022")
def net_influence(
    actions: Matrix,
    state: np.ndarray | None = None,
    lag: int = 0,
    n_boot: int = 200,
    alpha: float = 0.05,
    seed: int = 0,
    block: int = 1,
) -> NetInfluenceResult:
    """Bonjour et al. (2022) net influence between every pair of agents.

    ``gamma(i;j) = I(A_i; A_j | S)`` (with ``A_i`` lagged by ``lag``); ``Gamma(i;j) =
    gamma(i;j) - max_{k != i,j} gamma(k;j)``; pair statistic ``min(Gamma(i;j),
    Gamma(j;i))`` -- positive only when i and j inform each other more than any third
    agent informs either (a symmetric cartel of everyone scores 0). Significance: one-sided
    bootstrap test of ``pair_stat <= 0`` resampling periods (blocks of ``block`` periods
    for serially dependent data); a pair is flagged when ``p < alpha``. ``actions`` holds
    integer actions, or real quote changes (3-bin sign discretisation); ``state`` holds
    integer market states (real values are cut in tertiles), ``None`` = one state.
    """
    a = as_matrix(actions) if isinstance(actions, pl.DataFrame) else np.asarray(actions)
    if a.ndim != 2:
        raise ValueError("net_influence: actions must be periods x agents")
    t, n = a.shape
    cols = [_codes(a[:, i]) for i in range(n)]
    ks = [k for _, k in cols]
    if state is None:
        s, kS = np.zeros(t, dtype=np.int64), 1
    else:
        sv = np.asarray(state)
        if not np.issubdtype(sv.dtype, np.integer):
            sv = discretise(sv.astype(float), 3, "quantile")
        u, s = np.unique(sv, return_inverse=True)
        s, kS = s.astype(np.int64), len(u)
    # align (A_i[t - lag], A_j[t], S_t)
    src = [c[: t - lag] if lag else c for c, _ in cols]
    dst = [c[lag:] for c, _ in cols]
    s = s[lag:]
    g = _gamma_batch(src, dst, ks, s, kS)
    G = _Gamma(g)
    pair = np.minimum(G, np.swapaxes(G, -1, -2))
    rng = np.random.default_rng(seed)
    idx = _boot_indices(len(s), n_boot, block, rng)
    Gb = _Gamma(_gamma_batch([x[idx] for x in src], [y[idx] for y in dst], ks, s[idx], kS))
    pb = np.minimum(Gb, np.swapaxes(Gb, -1, -2))
    pv = (1 + np.sum(pb <= 1e-12, axis=0)) / (1 + n_boot)
    np.fill_diagonal(pv, np.nan)
    flagged = [(i, j) for i in range(n) for j in range(i + 1, n) if pv[i, j] < alpha]
    iu = np.triu_indices(n, 1)
    if len(iu[0]):
        k = int(np.nanargmax(pair[iu]))
        stat, best = float(pair[iu][k]), (int(iu[0][k]), int(iu[1][k]))
    else:
        stat, best = float("nan"), (-1, -1)
    return NetInfluenceResult(
        gamma=g,
        Gamma=G,
        pair_stat=pair,
        p_values=pv,
        flagged=flagged,
        statistic=stat,
        max_pair=best,
        alpha=alpha,
        extra={"n_boot": n_boot, "block": block, "lag": lag, "n_states": kS},
    )


# --------------------------------------------------------------------------- ghost baseline


@dataclass(frozen=True)
class GhostNetResult:
    stat: float  # mean(treatment) - mean(ghost)  (mean paired difference when paired)
    baseline: np.ndarray  # statistic on each ghost run
    treatment: np.ndarray  # statistic on each treatment run
    p_value: float
    paired: bool


@register_metric("net_of_ghost", "stat units", "run", True, source="novel")
def net_of_ghost(
    stat_fn: Callable[[Any], float],
    treatment_runs: Sequence[Any],
    ghost_runs: Sequence[Any],
    n_perm: int = 10_000,
    seed: int = 0,
    paired: bool = False,
    alternative: str = "greater",
) -> GhostNetResult:
    """A dependence statistic net of the ghost baseline, with a permutation p-value.

    Ghost runs replay the rivals open-loop, so they keep every common input (shocks,
    shared prompt, same model) but remove strategic interaction. Unpaired: label
    permutation of the difference in means. Paired (ghost k replays treatment k):
    sign-flip test on the paired differences.
    """
    tr = np.array([float(stat_fn(r)) for r in treatment_runs])
    gh = np.array([float(stat_fn(r)) for r in ghost_runs])
    rng = np.random.default_rng(seed)
    if paired:
        if len(tr) != len(gh):
            raise ValueError("paired net_of_ghost needs as many ghost as treatment runs")
        d = tr - gh
        obs = float(d.mean())
        signs = rng.choice([-1.0, 1.0], size=(n_perm, len(d)))
        null = (signs * d).mean(axis=1)
    else:
        obs = float(tr.mean() - gh.mean())
        pool = np.concatenate([tr, gh])
        order = np.argsort(rng.random((n_perm, len(pool))), axis=1)
        perm = pool[order]
        null = perm[:, : len(tr)].mean(axis=1) - perm[:, len(tr) :].mean(axis=1)
    if alternative == "greater":
        p = (1 + np.sum(null >= obs - 1e-12)) / (1 + n_perm)
    elif alternative == "two-sided":
        p = (1 + np.sum(np.abs(null) >= abs(obs) - 1e-12)) / (1 + n_perm)
    else:
        raise ValueError("alternative must be 'greater' or 'two-sided'")
    return GhostNetResult(stat=obs, baseline=gh, treatment=tr, p_value=float(p), paired=paired)


# --------------------------------------------------------------------------- RPS validation


def _rps_outcome(actions: np.ndarray) -> int:
    """0 = tie; ``1 + w`` when action ``w`` wins (R=0, P=1, S=2; ``(a+1)%3`` beats a)."""
    u = np.unique(actions)
    if len(u) != 2:
        return 0
    a, b = int(u[0]), int(u[1])
    return 1 + (b if b == (a + 1) % 3 else a)


def simulate_rps(
    n_games: int,
    collusion_prob: float = 1.0,
    pair: tuple[int, int] = (0, 1),
    n_players: int = 3,
    shift: int = 0,
    common_input: float = 0.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Scripted multi-player rock-paper-scissors.

    Every player draws uniformly; with probability ``common_input`` a player instead plays
    the action that beats last game's winning action (a shared public input). With
    probability ``collusion_prob`` the second member of ``pair`` plays ``(a_first +
    shift) % 3``. Returns ``(actions[n_games, n_players], state[n_games])`` where the
    state is last game's outcome (0 tie, 1 + winning action).
    """
    rng = np.random.default_rng(seed)
    acts = np.zeros((n_games, n_players), dtype=np.int64)
    states = np.zeros(n_games, dtype=np.int64)
    st = 0
    for t in range(n_games):
        states[t] = st
        a = rng.integers(0, 3, size=n_players)
        if st > 0 and common_input > 0:
            follow = rng.random(n_players) < common_input
            a[follow] = (st - 1 + 1) % 3  # (winner + 1) % 3 beats last winner
        if rng.random() < collusion_prob:
            a[pair[1]] = (a[pair[0]] + shift) % 3
        acts[t] = a
        st = _rps_outcome(a)
    return acts, states


def validate_net_influence_rps(
    n_games: int = 60,
    collusion_prob: float = 1.0,
    n_trials: int = 50,
    n_boot: int = 200,
    alpha: float = 0.05,
    common_input: float = 0.0,
    seed: int = 0,
) -> dict[str, float]:
    """Detection rate of the colluding pair (0, 1) over ``n_trials`` seeded games.

    Returns ``detection_rate`` (pair flagged), ``top_pair_rate`` (pair has the largest
    statistic) and ``false_flag_rate`` (share of non-colluding pairs flagged).
    """
    det = top = false = 0
    n_false_pairs = 0
    for k in range(n_trials):
        a, s = simulate_rps(
            n_games, collusion_prob, common_input=common_input, seed=seed * 100_003 + k
        )
        r = net_influence(a, s, n_boot=n_boot, alpha=alpha, seed=seed + k)
        det += (0, 1) in r.flagged
        top += r.max_pair == (0, 1)
        others = [p for p in r.flagged if p != (0, 1)]
        false += len(others)
        n_false_pairs += 2
    return {
        "n_games": float(n_games),
        "collusion_prob": float(collusion_prob),
        "n_trials": float(n_trials),
        "detection_rate": det / n_trials,
        "top_pair_rate": top / n_trials,
        "false_flag_rate": false / n_false_pairs,
    }
