"""Statistical procedures of the analysis plan (SAP, D.3', revised).

Unit of analysis = run / trunk. Paired contrasts use the BCa bootstrap (B = 10,000, fixed
seed), the paired t-test and the Wilcoxon signed-rank test, ONE-SIDED in the direction of a
reduction (``Delta(sham) - Delta(lever) > 0``).

* H1 (:func:`primary_h1_test`): one pre-specified lever, one-sided paired t-test on all
  qualified trunks at alpha = 0.05 with no multiplicity, Wilcoxon as sensitivity, BCa CI,
  one-sided 95 % lower bound; supported iff p < alpha and MQC <= 5 pp, practically
  significant iff also the lower bound exceeds 0.10; TOST when not supported.
* H1-R: the 8 levers, one-sided tests, Holm over the fixed family of m = 8 with missing
  levers counted at p = 1 (:func:`holm_fixed`), plus a descriptive split-sample estimate of
  the empirical best lever (:func:`split_sample_best`, odd seeds select, even seeds
  estimate).

sigma_d is the pooled SD of the realised lever-minus-sham differences (:func:`pooled_sd`);
nulls are claimed only through TOST equivalence (+/- 0.20 by default); rates use
cluster-robust (cluster = run) confidence intervals.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy import stats

DEFAULT_SEED = 20261010
DEFAULT_B = 10_000


@dataclass(frozen=True)
class BootstrapCI:
    estimate: float
    lo: float
    hi: float
    level: float
    B: int
    method: str
    n: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _stat_batch(stat: Callable[..., Any], samples: np.ndarray) -> np.ndarray:
    try:
        out = np.asarray(stat(samples, axis=1), dtype=float)
        if out.shape == (samples.shape[0],):
            return out
    except TypeError:
        pass
    return np.array([float(stat(row)) for row in samples])


def bca_ci(
    x: Sequence[float] | np.ndarray,
    stat: Callable[..., Any] = np.mean,
    B: int = DEFAULT_B,
    level: float = 0.95,
    seed: int = DEFAULT_SEED,
) -> BootstrapCI:
    """Bias-corrected and accelerated bootstrap CI (Efron 1987) for a one-sample statistic.

    ``stat`` should accept ``axis=`` for speed (numpy reductions do). Degenerate inputs
    (n < 2 or constant data) return a zero-width interval.
    """
    v = np.asarray(x, dtype=float)
    v = v[np.isfinite(v)]
    n = v.size
    if n == 0:
        return BootstrapCI(float("nan"), float("nan"), float("nan"), level, B, "bca", 0)
    theta = float(stat(v))
    if n < 2 or np.all(v == v[0]):
        return BootstrapCI(theta, theta, theta, level, B, "bca", n)
    rng = np.random.default_rng(seed)
    boot = _stat_batch(stat, v[rng.integers(0, n, size=(B, n))])
    prop = (np.sum(boot < theta) + 0.5 * np.sum(boot == theta)) / B
    z0 = stats.norm.ppf(np.clip(prop, 1.0 / (2 * B), 1 - 1.0 / (2 * B)))
    jack = _stat_batch(stat, np.array([np.delete(v, i) for i in range(n)]))
    d = jack.mean() - jack
    den = 6.0 * (np.sum(d**2) ** 1.5)
    a = float(np.sum(d**3) / den) if den > 0 else 0.0
    alpha = (1.0 - level) / 2.0
    zs = stats.norm.ppf([alpha, 1.0 - alpha])
    adj = stats.norm.cdf(z0 + (z0 + zs) / (1.0 - a * (z0 + zs)))
    lo, hi = np.quantile(boot, adj)
    return BootstrapCI(theta, float(lo), float(hi), level, B, "bca", n)


def paired_bca(
    a: Sequence[float],
    b: Sequence[float],
    stat: Callable[..., Any] = np.mean,
    B: int = DEFAULT_B,
    level: float = 0.95,
    seed: int = DEFAULT_SEED,
) -> BootstrapCI:
    """BCa CI of ``stat(a - b)`` for paired observations."""
    return bca_ci(_diff(a, b), stat, B, level, seed)


def _diff(a: Sequence[float], b: Sequence[float]) -> np.ndarray:
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if x.shape != y.shape:
        raise ValueError("paired samples must have equal length")
    d = x - y
    return d[np.isfinite(d)]


ALTERNATIVES = ("two-sided", "greater", "less")


def _check_alt(alternative: str) -> None:
    if alternative not in ALTERNATIVES:
        raise ValueError(f"alternative must be one of {ALTERNATIVES}, got {alternative!r}")


def paired_t(
    a: Sequence[float],
    b: Sequence[float],
    level: float = 0.95,
    alternative: str = "two-sided",
) -> dict[str, Any]:
    """Paired t-test on ``d = a - b``: mean, t, df, p and the two-sided t-based CI.

    ``alternative="greater"`` tests H1: mean(d) > 0 (one-sided; for ``d = Delta(sham) -
    Delta(lever)`` this is "the lever reduces collusion"), ``"less"`` the reverse.
    """
    _check_alt(alternative)
    d = _diff(a, b)
    n = d.size
    if n < 2:
        return {
            "mean": float(d.mean()) if n else float("nan"),
            "t": float("nan"),
            "df": float(n - 1),
            "p": float("nan"),
            "lo": float("nan"),
            "hi": float("nan"),
            "alternative": alternative,
        }
    se = d.std(ddof=1) / np.sqrt(n)
    m = float(d.mean())
    if se == 0:
        if alternative == "two-sided":
            p = 1.0 if m == 0 else 0.0
        elif alternative == "greater":
            p = 0.0 if m > 0 else 1.0
        else:
            p = 0.0 if m < 0 else 1.0
        t = 0.0 if m == 0 else float(np.sign(m) * np.inf)
        out = {"mean": m, "t": t, "df": float(n - 1), "p": p, "lo": m, "hi": m}
        return out | {"alternative": alternative}
    t = m / se
    if alternative == "two-sided":
        p = float(2 * stats.t.sf(abs(t), n - 1))
    elif alternative == "greater":
        p = float(stats.t.sf(t, n - 1))
    else:
        p = float(stats.t.cdf(t, n - 1))
    q = stats.t.ppf(0.5 + level / 2, n - 1)
    return {
        "mean": m,
        "t": float(t),
        "df": float(n - 1),
        "p": p,
        "lo": float(m - q * se),
        "hi": float(m + q * se),
        "alternative": alternative,
    }


def wilcoxon(
    a: Sequence[float], b: Sequence[float], alternative: str = "two-sided"
) -> dict[str, Any]:
    """Wilcoxon signed-rank test on ``a - b`` (zero differences dropped)."""
    _check_alt(alternative)
    d = _diff(a, b)
    nz = d[d != 0]
    if nz.size == 0:
        return {"statistic": 0.0, "p": 1.0, "n": float(d.size), "alternative": alternative}
    res = stats.wilcoxon(nz, alternative=alternative)
    return {
        "statistic": float(res.statistic),
        "p": float(res.pvalue),
        "n": float(d.size),
        "alternative": alternative,
    }


def holm(pvalues: Sequence[float], alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Holm step-down adjusted p-values and rejections (NaN p-values stay NaN, never
    rejected)."""
    p = np.asarray(pvalues, dtype=float)
    adj = np.full_like(p, np.nan)
    ok = np.where(np.isfinite(p))[0]
    m = ok.size
    if m:
        order = ok[np.argsort(p[ok], kind="stable")]
        running = 0.0
        for rank, idx in enumerate(order):
            running = max(running, min(1.0, (m - rank) * p[idx]))
            adj[idx] = running
    reject = np.where(np.isfinite(adj), adj <= alpha, False)
    return adj, reject


def holm_fixed(
    pvalues: Mapping[str, float], family: Sequence[str], alpha: float = 0.05
) -> dict[str, dict[str, Any]]:
    """Holm over a FIXED, pre-registered family (m = ``len(family)``).

    Every member of ``family`` enters the correction; a member without a finite p-value
    (lever not run, too few pairs, excluded) is counted with the placeholder ``p = 1`` so the
    family size never shrinks after the data are seen. Names outside ``family`` are ignored.
    Returns ``{name: {p, p_holm, reject, placeholder}}`` in family order.
    """
    names = list(family)
    if len(set(names)) != len(names):
        raise ValueError("holm_fixed: duplicate names in the family")
    raw, ph = [], []
    for n in names:
        v = pvalues.get(n)
        ok = v is not None and np.isfinite(float(v))
        raw.append(float(v) if ok else 1.0)
        ph.append(not ok)
    adj, rej = holm(raw, alpha)
    return {
        n: {"p": raw[i], "p_holm": float(adj[i]), "reject": bool(rej[i]), "placeholder": ph[i]}
        for i, n in enumerate(names)
    }


def one_sample(
    d: Sequence[float],
    alternative: str = "greater",
    B: int = DEFAULT_B,
    seed: int = DEFAULT_SEED,
    tost_margin: float = 0.20,
) -> dict[str, Any]:
    """The SAP bundle for paired differences ``d`` already formed (``sham - lever``)."""
    x = np.asarray(d, dtype=float)
    x = x[np.isfinite(x)]
    return paired_contrast(x, np.zeros_like(x), B, seed, tost_margin, alternative)


def pooled_sd(diffs: Mapping[str, Sequence[float]]) -> dict[str, Any]:
    """Realised paired-difference SD: pooled within-group SD of ``diffs[group]``.

    ``sigma_d = sqrt(sum_g sum_t (d_gt - mean_g)^2 / sum_g (n_g - 1))`` over groups with at
    least two finite differences (here: lever-minus-sham differences across trunks, one
    group per lever). Also returns the per-group SDs and the pooled degrees of freedom.
    """
    ss, dof, per = 0.0, 0, {}
    for g, v in diffs.items():
        x = np.asarray(v, dtype=float)
        x = x[np.isfinite(x)]
        if x.size < 2:
            per[g] = float("nan")
            continue
        ss += float(np.sum((x - x.mean()) ** 2))
        dof += x.size - 1
        per[g] = float(x.std(ddof=1))
    return {"sigma_d": float(np.sqrt(ss / dof)) if dof else float("nan"), "df": dof, "by": per}


def power_one_sided(n: int, sigma: float, effect: float, alpha: float = 0.05) -> float:
    """Power of the one-sided paired t-test (noncentral t) for ``n`` pairs."""
    if n < 2 or not np.isfinite(sigma) or sigma <= 0:
        return float("nan")
    df = n - 1
    nc = effect / (sigma / np.sqrt(n))
    crit = stats.t.ppf(1 - alpha, df)
    return float(stats.nct.sf(crit, df, nc))


def split_by_seed_parity(seeds: Mapping[str, int]) -> tuple[list[str], list[str]]:
    """``(selection, estimation)`` trunk ids: odd-seed trunks select, even-seed estimate."""
    sel = sorted(t for t, s in seeds.items() if int(s) % 2 == 1)
    est = sorted(t for t, s in seeds.items() if int(s) % 2 == 0)
    return sel, est


def select_best(
    effects: Mapping[str, float],
    family: Sequence[str],
    mqc: Mapping[str, float | None] | None = None,
    mqc_max_pp: float | None = None,
) -> str | None:
    """Arg-max of ``effects`` over ``family`` (ties: earlier in ``family``); with
    ``mqc_max_pp`` only levers with a known ``MQC <= mqc_max_pp`` are eligible."""
    best, best_v = None, -np.inf
    for lev in family:
        v = effects.get(lev)
        if v is None or not np.isfinite(v):
            continue
        if mqc_max_pp is not None:
            q = (mqc or {}).get(lev)
            if q is None or not np.isfinite(float(q)) or float(q) > mqc_max_pp:
                continue
        q_lev = (mqc or {}).get(lev)
        q_best = (mqc or {}).get(best) if best is not None else None
        tie_lower_mqc = (
            best is not None
            and v == best_v
            and q_lev is not None
            and q_best is not None
            and float(q_lev) < float(q_best)
        )
        if v > best_v or tie_lower_mqc:  # m11: ties broken by the lower MQC
            best, best_v = lev, float(v)
    return best


def _sig(p: float | None, alpha: float) -> bool:
    return p is not None and bool(np.isfinite(p)) and float(p) < alpha


def dual_benchmark_label(
    p_trunk: float | None, p_own: float | None, alpha: float = 0.05, mqc_ok: bool = True
) -> str:
    """D-17 label of a lever contrast from its one-sided p-values under the trunk scoring
    (pre-lever benchmarks, primary) and the own scoring (the lever market's own benchmarks).

    "collusion reduction" needs significance under BOTH scorings (and, for H1, an MQC
    inside the cap); trunk-only significance is a "price reduction consistent with
    competitive adjustment" (the lever moved the competitive price, the dealers followed).
    ``p_own = None`` means the own scoring is unavailable (reported as such).
    """
    st, so = _sig(p_trunk, alpha), _sig(p_own, alpha)
    return label_from_decisions(st, None if p_own is None else so, mqc_ok)


def label_from_decisions(reject_trunk: bool, reject_own: bool | None, mqc_ok: bool = True) -> str:
    """D-17 label from DECISIONS: raw-p significance for a single contrast (H1, H1-M, H-S)
    or the Holm rejections of the H1-R family (B1: never raw p inside the family);
    ``reject_own = None`` means the own scoring is unavailable."""
    if reject_trunk and reject_own is None:
        return "reduction under the trunk scoring (own scoring unavailable)"
    if reject_trunk and reject_own:
        return "collusion reduction" if mqc_ok else "collusion reduction (MQC above the cap)"
    if reject_trunk:
        return "price reduction consistent with competitive adjustment"
    if reject_own:
        return "reduction under the own scoring only"
    return "no significant reduction"


def t_interval(d: Sequence[float], level: float = 0.90) -> tuple[float, float]:
    """Two-sided ``level`` paired-t interval of the mean difference (NaN when n < 2)."""
    x = np.asarray(d, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 2:
        return float("nan"), float("nan")
    se = float(x.std(ddof=1) / np.sqrt(n))
    q = float(stats.t.ppf(0.5 + level / 2, n - 1))
    return float(x.mean() - q * se), float(x.mean() + q * se)


def practical_significance(
    d: Sequence[float],
    lower_bound_gt: float = 0.10,
    band: tuple[float, float] = (0.10, 0.20),
    B: int = DEFAULT_B,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """D-19: practical significance of a paired contrast on the trunk scoring.

    "practically significant" iff the one-sided 95 % paired-t lower bound exceeds
    ``lower_bound_gt``; "inconclusive" iff the two-sided 90 % paired-t interval straddles the
    band ``[0.10, 0.20]`` (lower limit <= 0.10 and upper limit >= 0.20); otherwise "below
    practical relevance". The BCa 90 % interval (trunks resampled) is reported alongside.
    """
    x = np.asarray(d, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 2:
        return {
            "verdict": "not applicable",
            "lower_bound_one_sided_95": float("nan"),
            "t_interval_90": [float("nan"), float("nan")],
            "bca_90": [float("nan"), float("nan")],
            "band": list(band),
            "lower_bound_gt": lower_bound_gt,
        }
    se = float(x.std(ddof=1) / np.sqrt(n))
    lb = float(x.mean() - stats.t.ppf(0.95, n - 1) * se)
    lo90, hi90 = t_interval(x, 0.90)
    bca = bca_ci(x, B=B, level=0.90, seed=seed)
    if lb > lower_bound_gt:
        verdict = "practically significant"
    elif lo90 <= band[0] and hi90 >= band[1]:
        verdict = "inconclusive"
    else:
        verdict = "below practical relevance"
    return {
        "verdict": verdict,
        "lower_bound_one_sided_95": lb,
        "t_interval_90": [lo90, hi90],
        "bca_90": [bca.lo, bca.hi],
        "band": list(band),
        "lower_bound_gt": lower_bound_gt,
    }


def primary_h1_test(
    d: Sequence[float],
    mqc: float | None,
    alpha: float = 0.05,
    mqc_max_pp: float = 5.0,
    practical_lower_bound_gt: float = 0.10,
    tost_margin: float = 0.20,
    B: int = DEFAULT_B,
    seed: int = DEFAULT_SEED,
    d_own: Sequence[float] | None = None,
    inconclusive_band: tuple[float, float] = (0.10, 0.20),
) -> dict[str, Any]:
    """Primary H1 (D-3', D-16, D-17, D-19): ONE pre-specified lever, no multiplicity.

    ``d`` are the paired differences of the CAPPED index ``min(Delta_p, 1)`` under the trunk
    scoring (``Delta_p(sham) - Delta_p(lever)`` on all qualified trunks), ``d_own`` the same
    differences under the lever market's own benchmarks (own scoring). One-sided paired
    t-tests (H1: mean > 0) at ``alpha`` under each scoring, Wilcoxon signed-rank as a
    sensitivity analysis, BCa 95 % CI (trunks resampled).

    * ``h1_supported`` ("collusion reduction") iff ``p < alpha`` under BOTH scorings and the
      MQC point estimate is ``<= mqc_max_pp``; trunk-only significance is labelled "price
      reduction consistent with competitive adjustment" (:func:`dual_benchmark_label`);
    * ``practical_significance`` (trunk scoring, :func:`practical_significance`): the
      one-sided 95 % paired-t lower bound against ``practical_lower_bound_gt``, with the
      "inconclusive" band and the BCa 90 % interval alongside; ``h1_practically_significant``
      iff supported and that bound exceeds the threshold;
    * ``tost`` +/- ``tost_margin`` when neither scoring is significant.

    Without ``d_own`` (own scoring unavailable) ``h1_supported`` is False and the label says
    so (m9); the D-19 verdict is reported in ``practical_verdict`` and never folded into
    ``h1_supported``.
    """
    x = np.asarray(d, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    bundle = one_sample(x, "greater", B, seed, tost_margin)
    p = bundle["t"]["p"]
    own: dict[str, Any] | None = None
    p_own: float | None = None
    if d_own is not None:
        xo = np.asarray(d_own, dtype=float)
        xo = xo[np.isfinite(xo)]
        ob = one_sample(xo, "greater", B, seed, tost_margin)
        p_own = ob["t"]["p"]
        own = {
            "available": True,
            "n": int(xo.size),
            "mean": ob["mean"],
            "bca_lo": ob["bca_lo"],
            "bca_hi": ob["bca_hi"],
            "p_one_sided": p_own,
            "t": ob["t"],
            "wilcoxon_sensitivity": ob["wilcoxon"],
        }
    prac = practical_significance(x, practical_lower_bound_gt, inconclusive_band, B, seed)
    lb = prac["lower_bound_one_sided_95"]
    mqc_ok = mqc is not None and np.isfinite(float(mqc)) and float(mqc) <= mqc_max_pp
    sig = _sig(p, alpha)
    sig_own = _sig(p_own, alpha) if own is not None else False  # m9: no own scoring, no support
    label = dual_benchmark_label(p, p_own if own is not None else None, alpha, bool(mqc_ok))
    supported = sig and sig_own and bool(mqc_ok)
    practical = supported and prac["verdict"] == "practically significant"
    neither = not sig and not sig_own
    return {
        "n": int(n),
        "mean": bundle["mean"],
        "bca_lo": bundle["bca_lo"],
        "bca_hi": bundle["bca_hi"],
        "p_one_sided": p,
        "t": bundle["t"],
        "wilcoxon_sensitivity": bundle["wilcoxon"],
        "lower_bound_one_sided_95": lb,
        "own_scoring": own if own is not None else {"available": False},
        "own_scoring_available": own is not None,
        "dual_benchmark_label": label,
        "practical_significance": prac,
        "practical_verdict": prac["verdict"],  # D-19 verdict, its own field (m9)
        "mqc": None if mqc is None else float(mqc),
        "checks": {
            "p_lt_alpha": sig,
            "p_own_lt_alpha": sig_own if own is not None else None,
            "mqc_ok": bool(mqc_ok),
            "lower_bound_gt": practical,
        },
        "h1_supported": supported,
        "h1_practically_significant": practical,
        "tost": bundle["tost"] if neither else None,
        "rule": {
            "alpha": alpha,
            "mqc_max_pp": mqc_max_pp,
            "practical_lower_bound_gt": practical_lower_bound_gt,
            "inconclusive_band": list(inconclusive_band),
            "tost_margin": tost_margin,
            "multiplicity": "none (single pre-specified lever)",
            "index": "capped min(Delta_p, 1) (D-16)",
            "scorings": "trunk (primary) and own benchmarks; collusion reduction needs both (D-17)",
        },
    }


def split_sample_best(
    diffs: Mapping[str, Mapping[str, float]],
    seeds: Mapping[str, int],
    family: Sequence[str],
    B: int = DEFAULT_B,
    seed: int = DEFAULT_SEED,
    min_per_half: int = 3,
    mqc: Mapping[str, float | None] | None = None,
    mqc_max_pp: float | None = None,
    min_total: int | None = None,
    n_qualified: int | None = None,
) -> dict[str, Any]:
    """DESCRIPTIVE split-sample estimate of the empirical best lever (H1-R).

    ``diffs[lever][trunk] = Delta(sham) - Delta(lever)`` (capped index, trunk scoring); the
    best lever is the arg-max mean on the odd-seed trunks (selection half) among the eligible
    levers (``MQC <= mqc_max_pp`` when ``mqc`` is given) and its effect is estimated, with a
    BCa 95 % CI, on the even-seed trunks only (estimation half), which removes the
    winner's-curse bias of selecting and estimating on the same trunks. D-22: reported only
    when each half holds at least ``min_per_half`` trunks (``status = "off"`` otherwise). No
    decision is attached to it.
    """
    sel_t, est_t = split_by_seed_parity(seeds)
    out: dict[str, Any] = {
        "label": "descriptive (no decision; selection on odd-seed, estimation on even-seed trunks)",
        "selection_trunks": sel_t,
        "estimation_trunks": est_t,
        "min_per_half": int(min_per_half),
        "selection_effects": {},
        "best_lever": None,
        "estimate": {},
    }
    n_q = len(seeds) if n_qualified is None else int(n_qualified)  # m11: the Gate 2 count
    if min_total is not None and n_q < min_total:
        out["status"] = (
            f"off: {n_q} qualified trunks, fewer than {min_total} (Gate 2 outcome table, D-25)"
        )
        return out
    if len(sel_t) < min_per_half or len(est_t) < min_per_half:
        out["status"] = (
            f"off: {len(sel_t)} selection and {len(est_t)} estimation trunks, "
            f"fewer than {min_per_half} per half (D-22)"
        )
        return out
    sel_eff = {}
    for lev in family:
        dl = diffs.get(lev, {})
        xs = [dl[t] for t in sel_t if t in dl and np.isfinite(dl[t])]
        sel_eff[lev] = float(np.mean(xs)) if xs else float("nan")
    best = select_best(sel_eff, family, mqc, mqc_max_pp)
    est: dict[str, Any] = {}
    if best is not None:
        dl = diffs[best]
        xe = [dl[t] for t in est_t if t in dl and np.isfinite(dl[t])]
        ci = bca_ci(xe, B=B, seed=seed)
        est = {"n": ci.n, "mean": ci.estimate, "bca_lo": ci.lo, "bca_hi": ci.hi}
    out.update(status="ok", selection_effects=sel_eff, best_lever=best, estimate=est)
    return out


def tost(
    d: Sequence[float], low: float = -0.20, high: float = 0.20, alpha: float = 0.05
) -> dict[str, float | bool]:
    """Two one-sided t-tests for equivalence of a paired difference to ``[low, high]``.

    ``p = max(p_lower, p_upper)``; equivalence is claimed when ``p < alpha`` (the
    ``1 - 2 alpha`` CI lies inside the margins).
    """
    x = np.asarray(d, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 2:
        return {
            "mean": float("nan"),
            "p_lower": float("nan"),
            "p_upper": float("nan"),
            "p": float("nan"),
            "equivalent": False,
            "lo": float("nan"),
            "hi": float("nan"),
        }
    m = float(x.mean())
    se = float(x.std(ddof=1) / np.sqrt(n))
    df = n - 1
    if se == 0:
        inside = low < m < high
        return {
            "mean": m,
            "p_lower": 0.0 if m > low else 1.0,
            "p_upper": 0.0 if m < high else 1.0,
            "p": 0.0 if inside else 1.0,
            "equivalent": inside,
            "lo": m,
            "hi": m,
        }
    p_lower = float(stats.t.sf((m - low) / se, df))
    p_upper = float(stats.t.cdf((m - high) / se, df))
    q = stats.t.ppf(1 - alpha, df)
    p = max(p_lower, p_upper)
    return {
        "mean": m,
        "p_lower": p_lower,
        "p_upper": p_upper,
        "p": p,
        "equivalent": bool(p < alpha),
        "lo": m - q * se,
        "hi": m + q * se,
    }


def cluster_proportion_ci(
    successes: Sequence[float], trials: Sequence[float], level: float = 0.95
) -> dict[str, float]:
    """Proportion with a cluster-robust (ratio-estimator, CR1) CI; one entry per cluster.

    ``p = sum(y) / sum(n)``; ``Var = K / (K - 1) * sum((y_c - p n_c)^2) / (sum n)^2``.
    """
    y = np.asarray(successes, dtype=float)
    n = np.asarray(trials, dtype=float)
    k = y.size
    tot = n.sum()
    if k == 0 or tot == 0:
        return {
            "p": float("nan"),
            "lo": float("nan"),
            "hi": float("nan"),
            "se": float("nan"),
            "clusters": float(k),
        }
    p = float(y.sum() / tot)
    var = k / (k - 1) * float(np.sum((y - p * n) ** 2)) / tot**2 if k > 1 else 0.0
    se = float(np.sqrt(var))
    z = stats.norm.ppf(0.5 + level / 2)
    return {
        "p": p,
        "lo": max(0.0, p - z * se),
        "hi": min(1.0, p + z * se),
        "se": se,
        "clusters": float(k),
    }


def paired_contrast(
    a: Sequence[float],
    b: Sequence[float],
    B: int = DEFAULT_B,
    seed: int = DEFAULT_SEED,
    tost_margin: float = 0.20,
    alternative: str = "greater",
) -> dict[str, Any]:
    """Everything the SAP reports for one paired contrast ``d = a - b``.

    Tests are one-sided by default (``alternative="greater"``: H1 mean(d) > 0, i.e. a
    reduction when ``a`` is the sham and ``b`` the lever); the BCa interval is the two-sided
    95 % interval and TOST is two one-sided tests by construction.
    """
    ci = paired_bca(a, b, B=B, seed=seed)
    d = _diff(a, b)
    return {
        "n": int(d.size),
        "mean": ci.estimate,
        "bca_lo": ci.lo,
        "bca_hi": ci.hi,
        "alternative": alternative,
        "t": paired_t(a, b, alternative=alternative),
        "wilcoxon": wilcoxon(a, b, alternative=alternative),
        "tost": tost(d, -tost_margin, tost_margin),
        "sd": float(d.std(ddof=1)) if d.size > 1 else float("nan"),
    }
