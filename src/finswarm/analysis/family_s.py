"""Family S (PREREG section 3, SAP sections 2-4 and 7, decision D-20).

Family S = {H2, H4, H5, H6-carrier, H7-same, H7-other, H8, H10}, m = 8 fixed, Holm at
family-wise alpha = 0.05; a hypothesis that cannot be tested enters with p = 1. Lever
contrasts use the CAPPED price index under the trunk scoring on the qualified trunks
(``contrasts[arm]["delta_p"]["diffs"]`` of the campaign analysis). Threshold rules test
``H0: mu <= MEI`` (one-sided) for "at least MEI" statements and TOST with margin = MEI for
"less than MEI" statements; for a joint rule the p-value entering Holm is the larger of its
component p-values (intersection-union; a component that cannot be computed makes the joint
rule untestable, p = 1) and point co-conditions are checked on the estimate.

Every hypothesis reports ``reject`` (the pure Holm decision), ``co_conditions`` (name ->
True / False / None, None = could not be evaluated) and ``supported`` = Holm reject AND every
co-condition True. The family summary counts ``n_supported``; ``n_rejected`` is kept for
reference only.

* H2: run-level Spearman rho(level, Delta_p) over the GR-H runs (4 levels x 4 CRN seeds),
  one-sided (decrease), permutation p obtained by permuting the level labels WITHIN each seed
  group (a GR-H run without a seed raises: it cannot be placed in a group); second component:
  one-sided Welch t of the pooled s1 runs (TR-H trunks + GR-H s1) minus s4 against 0.30.
  Holm p = max of the two.
* H4 (L3), H7-other (R3): one-sided paired t against 0.15.
* H6-carrier (L4a): one-sided paired t against 0.15, testable only when the sham's D5-net
  exceeds the 95th percentile of its ghost cyclic-shift null (``sham_d5net_ghost_null``);
  otherwise p = 1 and "not testable: no dependence to remove in the sham" (no decision; the
  Delta-C_p test is still reported, the carrier claim rests on it only). Co-condition:
  D5-net (net of the sham's ghosts) falls by >= 50 % from the sham to L4a.
* H5 (L5): two one-sided paired t against +0.15 and -0.15 (alpha / 2 each), H0 |mu| <= 0.15.
* H7-same (R1): Holm p = max(p_TOST, p_sign): TOST margin 0.15 and the one-sided sign test of
  "recovery <= 20 periods" over the R1 runs (:func:`recovery_periods`, H0 P <= 0.5); without a
  recovery input the p is the TOST p alone and the status says so.
* H10 (L9): TOST, margin 0.15; co-condition "dependence within +/- 25 %" (D5-net of L9 vs sham).
* H8: one-sided difference of follow proportions (collusive minus competitive markets)
  against 0.30, cluster = run; co-condition: punished share >= 50 % (placebo flag).

Windows (SAP section 7): primary 111-150, sensitivities 121-150 and 101-150 (reported, not
selected); every window shifts by the trend rule's +20 when the Gate 2 report says EXTEND,
and a gates report saying EXTEND together with unshifted windows raises ``ValueError``.

Inputs come from the campaign ``stats.json`` files (TR-H, GR-H), the detection corpus (D5-net
per run), the ghost cyclic-shift null of the sham's D5-net, the R1 runs and the sting runs;
what is missing is reported as "not testable" and enters Holm at p = 1, never silently
dropped.
"""

from __future__ import annotations

import json
import numbers
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from scipy import stats as sps

from finswarm.analysis import stats
from finswarm.analysis.io import discover_runs, read_manifest, read_table, run_info
from finswarm.metrics.collusion import sting_followers, sting_starts_from_events

FAMILY: tuple[str, ...] = ("H2", "H4", "H5", "H6-carrier", "H7-same", "H7-other", "H8", "H10")
MEI: dict[str, float] = {
    "H2": 0.30,
    "H4": 0.15,
    "H5": 0.15,
    "H6-carrier": 0.15,
    "H7-same": 0.15,
    "H7-other": 0.15,
    "H8": 0.30,
    "H10": 0.15,
}
LEVER_OF: dict[str, str] = {
    "H4": "L3",
    "H5": "L5",
    "H6-carrier": "L4a",
    "H7-same": "R1",
    "H7-other": "R3",
    "H10": "L9",
}
GRADIENT_LEVELS: dict[str, int] = {"GR-s1": 1, "GR-s2": 2, "GR-s3": 3, "GR-s4": 4}
DEFAULT_SEED = 20261010
N_PERM = 10_000

#: SAP section 7: the registered branch windows (inclusive periods) before any trend shift;
#: ``primary`` is the confirmatory window, the two sensitivities are reported, not selected
BRANCH_WINDOWS: dict[str, tuple[int, int]] = {
    "primary": (111, 150),
    "sensitivity_30": (121, 150),
    "sensitivity_50": (101, 150),
}
#: PREREG section 1: every window shifts by +20 when the trend rule fires (Gate 2 EXTEND)
TREND_SHIFT = 20
#: R1-H branches (and replaces slot 0) after 100 periods: the first period played by the
#: replacement is the registered period 101 (store period 100; A-008: ``read_table`` returns
#: registered 1-based periods, ``branch_period: 100`` is the last trunk period)
T0 = 101
#: H7-same recovery: tolerance on Delta_p, baseline length and the "<= 20 periods" bound
RECOVERY_TOL = 0.10
RECOVERY_PRE = 20
RECOVERY_MAX = 20
#: H6 guard: the sham's D5-net must exceed this quantile of its ghost cyclic-shift null
GHOST_NULL_QUANTILE = 0.95
H6_NOT_TESTABLE = "not testable: no dependence to remove in the sham"
H6_FALL = 0.50
DEPENDENCE_BAND = 0.25
PUNISHED_SHARE_MIN = 0.50
_EPS = 1e-9

RECOVERY_DEFINITION = (
    "baseline = mean Delta_p over the 20 periods before the replacement period "
    "(replacement_period - 20 .. replacement_period - 1); recovery = the first period "
    "t >= replacement_period with |Delta_p(t) - baseline| <= 0.10; recovery_periods = "
    "t - replacement_period (0 = never left the band); no such period = right-censored "
    "(counted as a failure of 'recovery <= 20'); sign test: successes = runs with "
    "recovery_periods <= 20, one-sided H0 P(success) <= 0.5"
)


# --------------------------------------------------------------------------- tests


def _clean_arr(d: Sequence[float]) -> np.ndarray:
    x = np.asarray(list(d), dtype=float)
    return x[np.isfinite(x)]


def _finite(x: Any) -> bool:
    return x is not None and isinstance(x, numbers.Real) and bool(np.isfinite(float(x)))


def _iut(*ps: Any) -> float:
    """Intersection-union p: the largest component p; 1 when a component is not finite."""
    if not ps or not all(_finite(p) for p in ps):
        return 1.0
    return float(max(float(p) for p in ps))


def t_vs_mei(d: Sequence[float], mei: float, level: float = 0.95) -> dict[str, Any]:
    """One-sided paired t of ``H0: mean(d) <= mei`` (a reduction of at least ``mei``)."""
    x = _clean_arr(d)
    n = x.size
    if n < 2:
        return {"n": int(n), "mean": float(x.mean()) if n else float("nan"), "p": float("nan")}
    se = float(x.std(ddof=1) / np.sqrt(n))
    m = float(x.mean())
    if se == 0:
        p = 0.0 if m > mei else 1.0
        return {
            "n": n,
            "mean": m,
            "se": 0.0,
            "t": float("nan"),
            "p": p,
            "lo": m,
            "hi": m,
            "mei": mei,
        }
    t = (m - mei) / se
    q = sps.t.ppf(0.5 + level / 2, n - 1)
    return {
        "n": int(n),
        "mean": m,
        "se": se,
        "t": float(t),
        "p": float(sps.t.sf(t, n - 1)),
        "lo": m - q * se,
        "hi": m + q * se,
        "mei": mei,
        "h0": f"mean <= {mei:g}",
    }


def two_one_sided_abs(d: Sequence[float], mei: float, alpha: float = 0.05) -> dict[str, Any]:
    """``H0: |mean(d)| <= mei`` by two one-sided paired t-tests at alpha / 2 each; the
    reported p is ``min(1, 2 min(p_gt, p_lt))`` (Bonferroni of the two directions)."""
    x = _clean_arr(d)
    n = x.size
    if n < 2:
        return {"n": int(n), "mean": float(x.mean()) if n else float("nan"), "p": float("nan")}
    se = float(x.std(ddof=1) / np.sqrt(n))
    m = float(x.mean())
    if se == 0:
        p = 0.0 if abs(m) > mei else 1.0
        return {"n": n, "mean": m, "p": p, "p_gt": p, "p_lt": p, "mei": mei}
    p_gt = float(sps.t.sf((m - mei) / se, n - 1))
    p_lt = float(sps.t.cdf((m + mei) / se, n - 1))
    return {
        "n": int(n),
        "mean": m,
        "se": se,
        "p_gt": p_gt,
        "p_lt": p_lt,
        "p": float(min(1.0, 2.0 * min(p_gt, p_lt))),
        "alpha_each": alpha / 2,
        "mei": mei,
        "h0": f"|mean| <= {mei:g}",
    }


def tost_mei(d: Sequence[float], mei: float, alpha: float = 0.05) -> dict[str, Any]:
    """TOST equivalence of a paired difference to ``[-mei, mei]`` (margin = the MEI)."""
    out = dict(stats.tost(_clean_arr(d), -mei, mei, alpha))
    out["n"] = int(_clean_arr(d).size)
    out["mei"] = mei
    out["h0"] = f"|mean| >= {mei:g} (equivalence claimed when p < alpha)"
    return out


def _missing_group(g: Any) -> bool:
    """True for a missing seed label (None, NaN, empty, "none", "nan", "null")."""
    if g is None:
        return True
    if isinstance(g, numbers.Real) and not isinstance(g, bool) and not np.isfinite(float(g)):
        return True
    return isinstance(g, str) and g.strip().lower() in ("", "none", "nan", "null")


def spearman_permutation(
    levels: Sequence[float],
    values: Sequence[float],
    groups: Sequence[Any],
    n_perm: int = N_PERM,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Run-level Spearman rho(level, value), one-sided (rho < 0), with a permutation p that
    shuffles the level labels WITHIN each group (the CRN seed): the null keeps every seed's
    set of levels and breaks only the level-value pairing inside the seed.

    A run without a group label (missing seed) raises ``ValueError``: it cannot be placed in
    a seed group, and pooling or dropping it would change the permutation null. Runs with a
    non-finite level or value are left out and counted (``n_dropped_nonfinite``,
    ``dropped_groups``); ``incomplete_groups`` lists the groups lacking some level.
    """
    gl = list(groups)
    lv = np.asarray(list(levels), dtype=float)
    va = np.asarray(list(values), dtype=float)
    if not (len(gl) == lv.size == va.size):
        raise ValueError("spearman_permutation: levels, values and groups differ in length")
    missing = [i for i, g in enumerate(gl) if _missing_group(g)]
    if missing:
        raise ValueError(
            f"spearman_permutation: {len(missing)} run(s) without a seed group (positions "
            f"{missing}); the within-seed permutation cannot place them (fix the inputs, do "
            "not drop them)"
        )
    gr = np.asarray([str(g) for g in gl])
    ok = np.isfinite(lv) & np.isfinite(va)
    dropped = sorted(set(gr[~ok].tolist()))
    n_dropped = int((~ok).sum())
    lv, va, gr = lv[ok], va[ok], gr[ok]
    n = lv.size
    all_levels = sorted(set(lv.tolist()))
    incomplete = {
        g: sorted(set(lv[gr == g].tolist()))
        for g in sorted(set(gr.tolist()))
        if sorted(set(lv[gr == g].tolist())) != all_levels
    }
    if n < 4 or np.unique(lv).size < 2:
        return {
            "n": int(n),
            "rho": float("nan"),
            "p_perm": float("nan"),
            "p_asymptotic": float("nan"),
            "n_dropped_nonfinite": n_dropped,
            "dropped_groups": dropped,
        }
    rho = float(sps.spearmanr(lv, va).statistic)
    rng = np.random.default_rng(seed)
    idx_by_group = [np.where(gr == g)[0] for g in np.unique(gr)]
    count = 0
    for _ in range(n_perm):
        perm = lv.copy()
        for idx in idx_by_group:
            perm[idx] = lv[idx][rng.permutation(idx.size)]
        r = sps.spearmanr(perm, va).statistic
        if np.isfinite(r) and r <= rho + 1e-12:
            count += 1
    asym = float(sps.spearmanr(lv, va, alternative="less").pvalue)
    return {
        "n": int(n),
        "rho": rho,
        "p_perm": float((1 + count) / (1 + n_perm)),
        "p_asymptotic": asym,
        "n_perm": int(n_perm),
        "groups": sorted(np.unique(gr).tolist()),
        "incomplete_groups": incomplete,
        "n_dropped_nonfinite": n_dropped,
        "dropped_groups": dropped,
        "h0": "rho >= 0 (one-sided: Delta_p decreases with routing sensitivity)",
    }


def welch_one_sided(a: Sequence[float], b: Sequence[float], mei: float) -> dict[str, Any]:
    """One-sided Welch t of ``H0: mean(a) - mean(b) <= mei``."""
    x, y = _clean_arr(a), _clean_arr(b)
    if x.size < 2 or y.size < 2:
        return {"n_a": int(x.size), "n_b": int(y.size), "diff": float("nan"), "p": float("nan")}
    va, vb = x.var(ddof=1) / x.size, y.var(ddof=1) / y.size
    se = float(np.sqrt(va + vb))
    diff = float(x.mean() - y.mean())
    if se == 0:
        return {
            "n_a": int(x.size),
            "n_b": int(y.size),
            "diff": diff,
            "p": 0.0 if diff > mei else 1.0,
        }
    df = (va + vb) ** 2 / (va**2 / (x.size - 1) + vb**2 / (y.size - 1))
    t = (diff - mei) / se
    return {
        "n_a": int(x.size),
        "n_b": int(y.size),
        "diff": diff,
        "se": se,
        "df": float(df),
        "t": float(t),
        "p": float(sps.t.sf(t, df)),
        "mei": mei,
        "h0": f"mean(s1) - mean(s4) <= {mei:g}",
    }


def proportion_difference(
    collusive: Sequence[tuple[float, float]],
    competitive: Sequence[tuple[float, float]],
    mei: float,
    level: float = 0.95,
) -> dict[str, Any]:
    """One-sided difference of follow proportions (collusive minus competitive) against
    ``mei`` with cluster-robust standard errors (cluster = run; CR1 ratio estimator). The
    statistic is referred to a t distribution with ``min(K_collusive, K_competitive) - 1``
    degrees of freedom (K = number of clusters, A-010), not to the normal: with the 6 sting
    runs per side the normal is anti-conservative."""
    c = stats.cluster_proportion_ci([s for s, _ in collusive], [n for _, n in collusive], level)
    k = stats.cluster_proportion_ci([s for s, _ in competitive], [n for _, n in competitive], level)
    diff = c["p"] - k["p"]
    se = (
        float(np.sqrt(c["se"] ** 2 + k["se"] ** 2))
        if np.isfinite(c["se"]) and np.isfinite(k["se"])
        else float("nan")
    )
    if not np.isfinite(diff):
        return {"diff": float("nan"), "p": float("nan"), "collusive": c, "competitive": k}
    k_min = min(len(collusive), len(competitive))
    df = max(k_min - 1, 1)
    if se == 0:
        p = 0.0 if diff > mei else 1.0
    else:
        p = float(sps.t.sf((diff - mei) / se, df))
    return {
        "diff": float(diff),
        "se": se,
        "df": int(df),
        "p": p,
        "collusive": c,
        "competitive": k,
        "mei": mei,
        "h0": f"follow(collusive) - follow(competitive) <= {mei:g}",
    }


def sign_test(successes: int, n: int, p0: float = 0.5) -> dict[str, Any]:
    """One-sided exact binomial test of ``H0: P(success) <= p0``."""
    if n <= 0:
        return {"successes": int(successes), "n": 0, "p": float("nan")}
    p = float(sps.binom.sf(int(successes) - 1, int(n), p0))
    return {"successes": int(successes), "n": int(n), "p": p, "p0": p0}


# --------------------------------------------------------------------------- H7 recovery


def _series_map(series: Any, start: int = 0) -> dict[int, float]:
    """``period -> Delta_p`` from a mapping (keys may be strings) or an array whose element
    ``i`` is period ``start + i``."""
    if isinstance(series, Mapping):
        return {int(k): float(v) if v is not None else float("nan") for k, v in series.items()}
    raw = series.to_numpy() if hasattr(series, "to_numpy") else series
    arr = np.asarray(list(raw), dtype=float)
    return {int(start) + i: float(v) for i, v in enumerate(arr)}


def _recovery(
    series: Any,
    replacement_period: int,
    tol: float = RECOVERY_TOL,
    pre: int = RECOVERY_PRE,
    start: int = 0,
) -> dict[str, Any]:
    if pre < 1:
        raise ValueError("recovery_periods: pre must be >= 1")
    if tol < 0:
        raise ValueError("recovery_periods: tol must be >= 0")
    s = _series_map(series, start)
    rp = int(replacement_period)
    base = np.asarray([s.get(t, float("nan")) for t in range(rp - pre, rp)], dtype=float)
    if not np.all(np.isfinite(base)):
        raise ValueError(
            f"recovery_periods: the {pre} periods before the replacement period {rp} "
            f"({rp - pre}-{rp - 1}) are not all observed; the baseline is undefined"
        )
    baseline = float(base.mean())
    post = sorted(t for t in s if t >= rp)
    horizon = (post[-1] - rp) if post else -1
    rec = float("nan")
    for t in post:
        v = s[t]
        if np.isfinite(v) and abs(v - baseline) <= tol + _EPS:
            rec = float(t - rp)
            break
    return {
        "replacement_period": rp,
        "baseline": baseline,
        "recovery_periods": rec,
        "censored": not np.isfinite(rec),
        "horizon": int(horizon),
    }


def recovery_periods(
    series: Mapping[int, float] | Sequence[float] | np.ndarray,
    replacement_period: int,
    tol: float = RECOVERY_TOL,
    pre: int = RECOVERY_PRE,
    *,
    start: int = 0,
) -> float:
    """Periods until Delta_p recovers after the replacement (H7-same, R1 arm).

    Definition (PREREG H7-same, "recovery <= 20 periods"):

    * ``replacement_period`` is the first period played by the replacement agent (R1-H
      branches and replaces slot 0 at ``period:100``, so 100 before any trend shift);
    * baseline = the mean of Delta_p over the ``pre`` (20) periods before the replacement
      period, i.e. periods ``replacement_period - pre`` .. ``replacement_period - 1``;
    * recovery = the first period ``t >= replacement_period`` at which Delta_p is back
      within ``tol`` (0.10, inclusive) of the baseline: ``|Delta_p(t) - baseline| <= tol``;
    * ``recovery_periods = t - replacement_period`` (0 = Delta_p never left the band).

    The sign test of H7-same counts the runs with ``recovery_periods <= 20`` against the
    one-sided null P(success) <= 50 %. Returns NaN when no such period exists in the series
    (right-censored; the caller counts it as a failure). ``series`` is a mapping
    ``period -> Delta_p`` or an array whose element ``i`` is period ``start + i``. Raises
    ``ValueError`` when the baseline periods are not all observed.
    """
    return float(_recovery(series, replacement_period, tol, pre, start)["recovery_periods"])


def _r1_recovery(
    recovery: Mapping[str, Sequence[float]] | None,
    r1_series: Mapping[str, Any] | None,
    r1_replacement_period: int | Mapping[str, int] | None,
    default_rp: int,
) -> dict[str, Any] | None:
    """Recovery periods of the R1 runs and the sign test of "recovery <= 20 periods";
    ``None`` when no recovery input is available."""
    if recovery is not None and r1_series is not None:
        raise ValueError("family_s: pass either recovery (precomputed) or r1_series, not both")
    runs: list[dict[str, Any]] = []
    if r1_series:
        for run, ser in r1_series.items():
            if isinstance(r1_replacement_period, Mapping):
                if run not in r1_replacement_period:
                    raise ValueError(f"family_s: no replacement period for the R1 run {run!r}")
                rp = int(r1_replacement_period[run])
            else:
                rp = default_rp if r1_replacement_period is None else int(r1_replacement_period)
            det = _recovery(ser, rp)
            if det["censored"] and det["horizon"] < RECOVERY_MAX:
                raise ValueError(
                    f"family_s: R1 run {run!r} is censored {det['horizon']} periods after the "
                    f"replacement (< {RECOVERY_MAX}): 'recovery <= {RECOVERY_MAX}' is undecidable"
                )
            runs.append({"run": str(run), **det})
        values = [r["recovery_periods"] for r in runs]
        source = "r1_series"
    elif recovery is not None and recovery.get("R1") is not None and len(recovery["R1"]) > 0:
        values = [float("nan") if v is None else float(v) for v in recovery["R1"]]
        source = "recovery (precomputed recovery periods)"
    else:
        return None
    rec = np.asarray(values, dtype=float)
    ok = np.isfinite(rec)
    successes = int(np.sum(ok & (rec <= RECOVERY_MAX + _EPS)))
    st = sign_test(successes, int(rec.size), 0.5)
    st["threshold_periods"] = RECOVERY_MAX
    st["h0"] = f"P(recovery_periods <= {RECOVERY_MAX}) <= 0.5"
    out: dict[str, Any] = {
        "recovery_source": source,
        "recovery_definition": RECOVERY_DEFINITION,
        "recovery_periods": rec.tolist(),
        "n_censored": int(np.sum(~ok)),
        "recovery_sign_test": st,
    }
    if runs:
        out["recovery_runs"] = runs
    return out


# --------------------------------------------------------------------------- H6 guard


def dependence_guard(
    sham_d5net: Sequence[float],
    ghost_null: Sequence[float] | float | None,
    quantile: float = GHOST_NULL_QUANTILE,
) -> dict[str, Any]:
    """H6 guard: is the sham's D5-net above the ``quantile`` (0.95) of its ghost cyclic-shift
    null? ``sham_d5net`` holds the per-run D5-net of the sham (their mean is compared);
    ``ghost_null`` is either the null draws of that same statistic (D5-net obtained by
    cyclically shifting the ghost sub-runs) or directly its 95th percentile. ``passed`` is
    True / False, or None when the guard cannot be evaluated (no sham D5-net, no null)."""
    x = _clean_arr(sham_d5net)
    stat = float(x.mean()) if x.size else float("nan")
    out: dict[str, Any] = {
        "sham_d5net": stat,
        "n_sham_runs": int(x.size),
        "quantile": quantile,
        "null_q": float("nan"),
        "n_null_draws": None,
        "passed": None,
        "rule": f"sham D5-net > the {quantile:g} quantile of its ghost cyclic-shift null",
    }
    if ghost_null is None:
        out["reason"] = "no ghost cyclic-shift null of the sham's D5-net"
        return out
    if isinstance(ghost_null, numbers.Real):
        q = float(ghost_null)
        out["null_source"] = "quantile given"
    else:
        draws = _clean_arr(ghost_null)
        out["n_null_draws"] = int(draws.size)
        out["null_source"] = "null draws"
        q = float(np.quantile(draws, quantile)) if draws.size else float("nan")
    out["null_q"] = q
    if not np.isfinite(q):
        out["reason"] = "empty or non-finite ghost cyclic-shift null"
        return out
    if not np.isfinite(stat):
        out["reason"] = "no D5-net of the sham"
        return out
    out["passed"] = bool(stat > q)
    out["reason"] = (
        "sham dependence above its ghost null"
        if out["passed"]
        else "sham D5-net not above its ghost null quantile: no dependence to remove"
    )
    return out


# --------------------------------------------------------------------------- windows


def family_s_windows(
    shift: int = 0, base: Mapping[str, Sequence[int]] | None = None
) -> dict[str, tuple[int, int]]:
    """The registered branch windows (``BRANCH_WINDOWS`` or ``base``, same keys) shifted by
    ``shift`` periods: primary 111-150, sensitivities 121-150 and 101-150 at shift 0."""
    src = BRANCH_WINDOWS if base is None else base
    if set(src) != set(BRANCH_WINDOWS):
        raise ValueError(
            f"family-S windows need exactly the keys {sorted(BRANCH_WINDOWS)}, got {sorted(src)}"
        )
    out: dict[str, tuple[int, int]] = {}
    for k in BRANCH_WINDOWS:
        v = list(src[k])
        if len(v) != 2 or int(v[0]) > int(v[1]):
            raise ValueError(f"family-S window {k!r} must be [start, end] with start <= end")
        out[k] = (int(v[0]) + int(shift), int(v[1]) + int(shift))
    return out


def gate2_trend_shift(gates: Mapping[str, Any] | str | Path, lane: str = "H") -> dict[str, Any]:
    """Trend-rule shift of ``lane`` from a gates report (the ``gates.json`` of
    ``analyze.py --gates``, a path or the loaded dict, or a single Gate 2 entry): the Gate 2
    entries of that lane (labels ``Gate 2 (<campaign>)`` or the legacy key ``gate2``); when
    one says EXTEND the shift is its ``trend_rule.shift`` (default +20), otherwise 0."""
    if isinstance(gates, str | Path):
        p = Path(gates)
        if not p.is_file():
            raise ValueError(f"gates report {p} not found")
        rep: Mapping[str, Any] = json.loads(p.read_text(encoding="utf-8"))
        source = str(p)
    else:
        rep, source = gates, "mapping"
    if isinstance(rep.get("gates"), Mapping):
        entries = [
            (str(k), v)
            for k, v in rep["gates"].items()
            if isinstance(v, Mapping) and (k == "gate2" or str(k).startswith("Gate 2"))
        ]
    elif "status" in rep:
        entries = [("gate2", rep)]
    else:
        raise ValueError("not a gates report: no 'gates' mapping and no Gate 2 'status'")
    mine = [(k, v) for k, v in entries if v.get("lane") in (None, lane)]
    ext = [(k, v) for k, v in mine if v.get("status") == "EXTEND"]
    shifts = {int((v.get("trend_rule") or {}).get("shift") or TREND_SHIFT) for _, v in ext}
    if len(shifts) > 1:
        raise ValueError(f"gates report: conflicting trend shifts {sorted(shifts)} for lane {lane}")
    return {
        "source": source,
        "lane": lane,
        "entries": [k for k, _ in mine],
        "statuses": {k: v.get("status") for k, v in mine},
        "extend": bool(ext),
        "shift": shifts.pop() if shifts else 0,
    }


def resolve_windows(
    windows: Mapping[str, Sequence[int]] | None = None,
    shift: int | None = None,
    gates: Mapping[str, Any] | str | Path | None = None,
    lane: str = "H",
) -> tuple[dict[str, tuple[int, int]], dict[str, Any]]:
    """Family-S windows = ``windows`` (default the registered ones) shifted by ``shift``.

    With a ``gates`` report the result must equal the registered windows shifted by the
    report's trend shift: a report saying EXTEND together with unshifted windows (or any
    other mismatch, e.g. ``shift=20`` on a PASS report) raises ``ValueError``."""
    sh = 0 if shift is None else int(shift)
    resolved = family_s_windows(sh, windows)
    info: dict[str, Any] = {
        "source": "explicit windows" if windows is not None else "registered defaults",
        "requested_shift": sh,
        "shift": sh,
        "gate2": None,
    }
    if gates is None:
        return resolved, info
    g = gate2_trend_shift(gates, lane)
    info["gate2"] = g
    expected = family_s_windows(g["shift"])
    if resolved != expected:
        p0, p1 = resolved["primary"]
        e0, e1 = expected["primary"]
        if g["extend"] and resolved == family_s_windows(0):
            raise ValueError(
                f"Gate 2 says EXTEND (trend rule, +{g['shift']}) but the family-S windows are "
                f"the unshifted defaults (primary {p0}-{p1}); pass shift={g['shift']} or the "
                f"shifted windows (primary {e0}-{e1})"
            )
        raise ValueError(
            f"family-S windows (primary {p0}-{p1}) do not match the Gate 2 report (shift "
            f"{g['shift']}, primary {e0}-{e1})"
        )
    info["shift"] = g["shift"]
    return resolved, info


# --------------------------------------------------------------------------- the family


def _diffs(contrasts: Mapping[str, Any], arm: str, measure: str = "delta_p") -> list[float]:
    c = contrasts.get(arm)
    if not c:
        return []
    return [float(v) for v in c.get(measure, {}).get("diffs", {}).values()]


def _relative_change(before: Sequence[float], after: Sequence[float]) -> float:
    b, a = _clean_arr(before), _clean_arr(after)
    if b.size == 0 or a.size == 0 or abs(float(b.mean())) < 1e-12:
        return float("nan")
    return float((a.mean() - b.mean()) / abs(b.mean()))


def family_s(
    contrasts: Mapping[str, Any],
    gradient_runs: Sequence[Mapping[str, Any]] = (),
    trunk_delta_p: Sequence[float] = (),
    d5net: Mapping[str, Sequence[float]] | None = None,
    recovery: Mapping[str, Sequence[float]] | None = None,
    sting: Mapping[str, Any] | None = None,
    alpha: float = 0.05,
    n_perm: int = N_PERM,
    seed: int = DEFAULT_SEED,
    fallback: bool = False,
    *,
    sham_d5net_ghost_null: Sequence[float] | float | None = None,
    r1_series: Mapping[str, Mapping[int, float] | Sequence[float]] | None = None,
    r1_replacement_period: int | Mapping[str, int] | None = None,
    windows: Mapping[str, Sequence[int]] | None = None,
    shift: int | None = None,
    gates: Mapping[str, Any] | str | Path | None = None,
) -> dict[str, Any]:
    """Every family-S hypothesis with its component tests, the Holm p (m = 8), the status,
    the co-conditions and ``supported``.

    ``contrasts`` is the campaign analysis' contrasts dict (qualified trunks already applied);
    ``gradient_runs`` rows with ``arm`` (GR-s1..GR-s4), ``seed`` (required: a GR-H run without
    a seed raises ``ValueError``) and ``delta_p``; ``trunk_delta_p`` the TR-H trunks' Delta_p
    (pooled s1); ``d5net`` the per-run D5-net of the sham, L4a and L9 windows;
    ``sham_d5net_ghost_null`` the ghost cyclic-shift null of the sham's D5-net (null draws of
    the mean sham D5-net, or directly its 95th percentile) for the H6 guard; ``recovery``
    precomputed R1 recovery periods (``{"R1": [...]}``) OR ``r1_series`` the per-period
    Delta_p of each R1 run (``run -> {period: Delta_p}`` or array from period 0) with
    ``r1_replacement_period`` (an int, a per-run mapping, or ``None`` = ``T0 + shift``);
    ``sting`` = ``{"collusive": [(successes, trials), ...], "competitive": [...],
    "punished_share": float}``. With ``fallback`` (Gate 2 below the threshold) every
    hypothesis using trunks is "not tested". ``windows`` / ``shift`` / ``gates`` resolve the
    registered windows (:func:`resolve_windows`; EXTEND with unshifted windows raises).
    """
    win, winfo = resolve_windows(windows, shift, gates)
    eff_shift = int(winfo["shift"])
    hyp: dict[str, dict[str, Any]] = {}
    pvals: dict[str, float] = {}

    def not_testable(h: str, why: str) -> None:
        # enters Holm as a placeholder (p = 1, flagged), never silently dropped
        hyp[h] = {"status": f"not testable: {why}", "p": 1.0, "mei": MEI[h], "co_conditions": {}}
        pvals[h] = float("nan")

    # H2 -- gradient from scratch (does not depend on Gate 2)
    gr_all = [r for r in gradient_runs if r.get("arm") in GRADIENT_LEVELS]
    no_seed = [r for r in gr_all if _missing_group(r.get("seed"))]
    if no_seed:
        raise ValueError(
            f"H2: {len(no_seed)} GR-H run(s) without a seed "
            f"({sorted({str(r['arm']) for r in no_seed})}); the within-seed-group permutation "
            "cannot place them (fix the inputs, do not drop them)"
        )
    gr = [r for r in gr_all if r.get("delta_p") is not None]
    gr_excluded = [
        {"arm": r["arm"], "seed": r.get("seed"), "reason": "no Delta_p"}
        for r in gr_all
        if r.get("delta_p") is None
    ]
    if len(gr) >= 4 and len({r["arm"] for r in gr}) >= 2:
        sp = spearman_permutation(
            [GRADIENT_LEVELS[r["arm"]] for r in gr],
            [float(r["delta_p"]) for r in gr],
            [r.get("seed") for r in gr],
            n_perm,
            seed,
        )
        s1 = list(trunk_delta_p) + [float(r["delta_p"]) for r in gr if r["arm"] == "GR-s1"]
        s4 = [float(r["delta_p"]) for r in gr if r["arm"] == "GR-s4"]
        we = welch_one_sided(s1, s4, MEI["H2"])
        p = _iut(sp.get("p_perm"), we.get("p"))
        complete = _finite(sp.get("p_perm")) and _finite(we.get("p"))
        hyp["H2"] = {
            "status": "tested"
            if complete
            else "not testable: a joint-rule component (Spearman or Welch s1 - s4) cannot be "
            "computed",
            "spearman": sp,
            "welch_s1_minus_s4": we,
            "p": p,
            "rule": "intersection-union: Holm p = max(p_perm, p_welch); p = 1 when a component "
            "cannot be computed",
            "mei": MEI["H2"],
            "n_gradient_runs": len(gr),
            "n_pooled_s1": len(s1),
            "excluded": gr_excluded,
            "co_conditions": {},
        }
        pvals["H2"] = p if complete else float("nan")
    else:
        not_testable("H2", "fewer than 4 GR-H runs or a single level")
        hyp["H2"]["excluded"] = gr_excluded

    # lever contrasts on the qualified trunks
    d5 = d5net or {}
    for h, arm in LEVER_OF.items():
        d = _diffs(contrasts, arm)
        if fallback:
            not_testable(h, "Gate 2 fallback (fewer qualified trunks than the threshold)")
            continue
        if len(d) < 2:
            not_testable(h, f"no paired contrast for {arm} (lever not run or < 2 pairs)")
            continue
        mei = MEI[h]
        if h in ("H4", "H6-carrier", "H7-other"):
            res = t_vs_mei(d, mei)
        elif h == "H5":
            res = two_one_sided_abs(d, mei, alpha)
        else:  # H7-same, H10: equivalence
            res = tost_mei(d, mei, alpha)
        entry: dict[str, Any] = {"status": "tested", "test": res, "p": res["p"], "lever": arm}
        co: dict[str, bool | None] = {}
        ci = stats.bca_ci(d, B=2000, seed=seed)
        entry["estimate"] = {"mean": ci.estimate, "bca_lo": ci.lo, "bca_hi": ci.hi, "n": ci.n}
        entry["mei"] = mei
        holm_p = entry["p"] if _finite(entry["p"]) else 1.0
        # co-conditions on the estimate
        if h == "H6-carrier":
            guard = dependence_guard(d5.get("sham", []), sham_d5net_ghost_null)
            entry["dependence_guard"] = guard
            change = _relative_change(d5.get("sham", []), d5.get("L4a", []))
            if d5:
                entry["d5net_change"] = change
                alt = tost_mei(d, mei, alpha)
                entry["alternative_price_level_suffices"] = {
                    "tost_equivalent": bool(alt.get("equivalent")),
                    "dependence_within_25pct": bool(abs(change) <= DEPENDENCE_BAND)
                    if np.isfinite(change)
                    else None,
                }
            # the fall criterion is evaluated only when the guard does not fail
            fall = (
                bool(change <= -H6_FALL + _EPS)
                if np.isfinite(change) and guard["passed"] is not False
                else None
            )
            entry["co_condition_dependence_fall_ge_50pct"] = fall
            co["sham_d5net_above_ghost_null_q95"] = guard["passed"]
            co["dependence_fall_ge_50pct"] = fall
            if guard["passed"] is False:
                entry["status"] = H6_NOT_TESTABLE
                entry["p"] = 1.0
                entry["note"] = (
                    "no decision on the carrier (enters Holm at p = 1); the Delta-C_p test is "
                    "reported under 'test' and the carrier claim rests on it only"
                )
                holm_p = float("nan")
            elif guard["passed"] is None:
                entry["status"] = (
                    f"tested; dependence guard not evaluated ({guard['reason']}): "
                    "not supported without it"
                )
        if h == "H10":
            change = _relative_change(d5.get("sham", []), d5.get("L9", []))
            within = bool(abs(change) <= DEPENDENCE_BAND + _EPS) if np.isfinite(change) else None
            if d5:
                entry["d5net_change"] = change
            entry["co_condition_dependence_within_25pct"] = within
            co["dependence_within_25pct"] = within
        if h == "H7-same":
            entry["p_tost"] = res["p"]
            rec = _r1_recovery(recovery, r1_series, r1_replacement_period, T0 + eff_shift)
            if rec is None:
                entry["status"] = "tested; sign test not computed: no R1 recovery input"
                entry["rule"] = "TOST p alone (the recovery sign test is missing)"
                co["recovery_sign_test_computed"] = False
            else:
                entry.update(rec)
                entry["p_sign"] = rec["recovery_sign_test"]["p"]
                entry["p"] = _iut(res["p"], entry["p_sign"])
                entry["rule"] = (
                    f"intersection-union: Holm p = max(p_TOST, p_sign), sign test on "
                    f"recovery <= {RECOVERY_MAX} periods"
                )
                co["recovery_sign_test_computed"] = True
                holm_p = entry["p"]
        entry["co_conditions"] = co
        hyp[h] = entry
        pvals[h] = holm_p

    # H8 -- sting follow rates
    if fallback:
        not_testable("H8", "Gate 2 fallback")
    elif sting and sting.get("collusive") and sting.get("competitive"):
        pd = proportion_difference(sting["collusive"], sting["competitive"], MEI["H8"])
        share = sting.get("punished_share")
        ok_share = None if share is None else bool(share >= PUNISHED_SHARE_MIN - _EPS)
        hyp["H8"] = {
            "status": "tested",
            "test": pd,
            "p": pd["p"],
            "mei": MEI["H8"],
            "punished_share": share,
            "co_condition_punished_share_ge_50pct": ok_share,
            "co_conditions": {"punished_share_ge_50pct": ok_share},
        }
        pvals["H8"] = pd["p"] if np.isfinite(pd["p"]) else 1.0
    else:
        not_testable("H8", "no sting follow counts (S1 collusive and competitive sides)")

    holm = stats.holm_fixed(pvals, FAMILY, alpha)
    for h in FAMILY:
        co_h = hyp[h].setdefault("co_conditions", {})
        rej = bool(holm[h]["reject"])
        hyp[h].update(
            {
                "p_holm": holm[h]["p_holm"],
                "reject": rej,
                "placeholder": holm[h]["placeholder"],
                "supported": bool(rej and all(v is True for v in co_h.values())),
            }
        )
    n_rej = int(sum(1 for h in FAMILY if hyp[h]["reject"]))
    n_sup = int(sum(1 for h in FAMILY if hyp[h]["supported"]))
    return {
        "family": list(FAMILY),
        "m": len(FAMILY),
        "alpha": alpha,
        "rule": "Holm over the fixed family of 8; not testable -> p = 1; joint rules enter with "
        "the larger component p (intersection-union); supported = Holm reject and every point "
        "co-condition met (one not evaluable is not met)",
        "mei": MEI,
        "windows": {k: list(v) for k, v in win.items()},
        "window_info": winfo,
        "r1_replacement_period_default": T0 + eff_shift,
        "hypotheses": hyp,
        "n_rejected": n_rej,
        "n_supported": n_sup,
        "summary": f"{n_sup} of {len(FAMILY)} supported (Holm reject and co-conditions met); "
        f"{n_rej} rejected by Holm",
    }


# --------------------------------------------------------------------------- loaders


def follow_counts(run_dir: str | Path, sting_slot: int, hold: int = 5) -> tuple[float, float]:
    """``(successes, trials)`` of the forced-rise follow events of one sting run (H8): every
    rival x probe cell counts one trial, a follow (rise >= half the sting delta within
    ``hold`` periods) one success."""
    ev = read_table(run_dir, "events")
    starts = sting_starts_from_events(ev) if ev.height else []
    ap = read_table(run_dir, "agent_period")
    if not starts or ap.height == 0:
        return 0.0, 0.0
    delta = float(np.nanmean([s["delta"] for s in starts if np.isfinite(s["delta"])] or [np.nan]))
    if not np.isfinite(delta) or delta == 0:
        return 0.0, 0.0
    hold = int(max(s["hold_epochs"] for s in starts) or hold)
    # A-018: the forced rises are the "raise" phases of the sting (one per probe: shadow
    # epochs hold the price, then it rises by delta); the single ``sting`` row only marks the
    # branch, so counting from it measured the shadow price (zero rise, 4 trials per run)
    raises = []
    if "event_type" in ev.columns:
        for row in ev.filter(pl.col("event_type") == "sting_phase").iter_rows(named=True):
            ph = json.loads(row["payload"]) if row["payload"] else {}
            if ph.get("phase") == "raise":
                raises.append(int(row["period"]))
    periods = sorted(set(raises)) or [s["period"] for s in starts]
    tab = sting_followers(ap, sting_slot, periods, delta, hold=hold)
    succ = float((tab["follow_prob"].fill_nan(0.0) * tab["n_probes"]).sum())
    trials = float(tab["n_probes"].sum())
    return succ, trials


def sting_inputs(
    campaign_dirs: Sequence[str | Path],
    qualified: Sequence[str],
    sting_slot: int = 4,
    probe_flags: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Follow counts per run: collusive = sting runs (arm S1) on qualified trunks; competitive
    = S1c runs (scripted or LLM) and S1 runs on non-qualified trunks."""
    coll, comp = [], []
    for d in campaign_dirs:
        if not Path(d).is_dir():
            continue
        for rd in discover_runs(d):
            info = run_info(rd)
            if read_manifest(rd).get("status") not in (None, "ok"):
                continue
            if info["arm"] == "S1":
                (coll if info["trunk_id"] in set(qualified) else comp).append(
                    follow_counts(rd, sting_slot)
                )
            elif info["arm"] in ("S1c", "SC-1") and info["label"].startswith("S1c"):
                comp.append(follow_counts(rd, sting_slot))
    share = None
    if probe_flags:
        flags = [v for v in probe_flags.values() if v in ("llm_punished", "llm_unpunished")]
        if flags:
            share = float(np.mean([v == "llm_punished" for v in flags]))
    return {
        "collusive": [c for c in coll if c[1] > 0],
        "competitive": [c for c in comp if c[1] > 0],
        "punished_share": share,
    }


def d5net_by_arm(corpus_path: str | Path, run_arms: Mapping[str, str]) -> dict[str, list[float]]:
    """D5-net per arm from the detection corpus (``run_id -> registered arm`` mapping)."""
    df = pl.read_parquet(corpus_path)
    out: dict[str, list[float]] = {}
    for rid, v in zip(df["run_id"].to_list(), df["D5net"].to_list(), strict=True):
        arm = run_arms.get(str(rid))
        if arm is not None and v is not None and np.isfinite(v):
            out.setdefault(arm, []).append(float(v))
    return out


def load_ghost_null(path: str | Path) -> list[float] | float:
    """The ghost cyclic-shift null of the sham's D5-net from a JSON file: a list of null
    draws, or an object with ``draws`` (list) or ``q95`` (its 95th percentile)."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(doc, list):
        return [float(x) for x in doc]
    if isinstance(doc, Mapping):
        if "draws" in doc:
            return [float(x) for x in doc["draws"]]
        if "q95" in doc:
            return float(doc["q95"])
    raise ValueError(f"{path}: expected a list of null draws or an object with 'draws' or 'q95'")


def _period_delta(
    run_dir: str | Path, members: Sequence[int], bench: Mapping[str, Any]
) -> dict[int, float]:
    """Per-period Delta_p of the members' mean posted price (non-positive or non-finite
    prices = not quoting) against the trunk benchmarks ``bench`` (p_nash, p_mono)."""
    ap = read_table(run_dir, "agent_period")
    if ap.height == 0:
        return {}
    pn, pm = float(bench["p_nash"]), float(bench["p_mono"])
    g = (
        ap.filter(
            pl.col("slot").is_in([int(m) for m in members])
            & pl.col("price").is_finite()
            & (pl.col("price") > 0)
        )
        .group_by("period")
        .agg(pl.col("price").mean().alias("pbar"))
        .sort("period")
    )
    return {
        int(t): (float(p) - pn) / (pm - pn)
        for t, p in zip(g["period"].to_list(), g["pbar"].to_list(), strict=True)
    }


def r1_series_from_runs(
    runs: Sequence[Mapping[str, Any]],
    benchmarks: Mapping[str, Any],
    qualified: Sequence[str] = (),
    arm: str = "R1",
    trunk_arm: str = "trunk",
) -> dict[str, Any]:
    """Per-period Delta_p of every R1 run of a qualified trunk (the H7-same recovery input).

    ``runs`` are the run summaries of the campaign ``stats.json`` (``path``, ``trunk_id``,
    ``registered_arm``, ``branch_period``, ``members``) and ``benchmarks`` its pinned
    benchmarks (``benchmarks[arm]["primary"]``, the trunk's). Periods before the branch
    (the replacement period) come from the parent trunk run, periods from it on from the R1
    run. Returns ``{"series", "replacement_period", "excluded"}``; every R1 run that cannot
    be built is listed in ``excluded`` with its reason."""
    qset = {str(q) for q in qualified}
    trunks = {
        str(r.get("trunk_id")): r
        for r in runs
        if r.get("registered_arm") == trunk_arm and r.get("path")
    }
    bench = (benchmarks.get(arm) or {}).get("primary") or (benchmarks.get(trunk_arm) or {}).get(
        "primary"
    )
    series: dict[str, dict[int, float]] = {}
    rps: dict[str, int] = {}
    excluded: list[dict[str, Any]] = []
    for r in runs:
        if r.get("registered_arm") != arm:
            continue
        rid, tid = str(r.get("run_id")), str(r.get("trunk_id"))
        if qset and tid not in qset:
            continue  # not in the primary set (not an exclusion)
        bp = r.get("branch_period")
        tr = trunks.get(tid)
        why = None
        if r.get("error"):
            why = f"run error: {r['error']}"
        elif not bench or abs(float(bench["p_mono"]) - float(bench["p_nash"])) < 1e-12:
            why = "no usable trunk benchmarks for R1"
        elif bp is None:
            why = "no branch (replacement) period in the manifest"
        elif not r.get("path") or not Path(str(r["path"])).is_dir():
            why = "R1 run directory missing"
        elif tr is None or not Path(str(tr["path"])).is_dir():
            why = "parent trunk run missing (pre-replacement baseline)"
        if why is not None:
            excluded.append({"run_id": rid, "trunk_id": tid, "reason": why})
            continue
        members = [int(m) for m in (r.get("members") or [0, 1, 2, 3])]
        rp = int(bp) + 1  # registered coordinates: the first period played by the replacement
        pre = {t: v for t, v in _period_delta(tr["path"], members, bench).items() if t < rp}
        post = {t: v for t, v in _period_delta(r["path"], members, bench).items() if t >= rp}
        if not post:
            excluded.append({"run_id": rid, "trunk_id": tid, "reason": "no post-branch rows"})
            continue
        series[rid] = {**pre, **post}
        rps[rid] = rp
    return {"series": series, "replacement_period": rps, "excluded": excluded}


def family_s_from_stats(
    trunk_stats: str | Path,
    gradient_stats: str | Path | None = None,
    corpus_path: str | Path | None = None,
    sting_campaigns: Sequence[str | Path] = (),
    probe_flags_path: str | Path | None = None,
    alpha: float = 0.05,
    n_perm: int = N_PERM,
    seed: int = DEFAULT_SEED,
    *,
    sham_d5net_ghost_null: Sequence[float] | float | None = None,
    ghost_null_path: str | Path | None = None,
    r1_series: Mapping[str, Mapping[int, float] | Sequence[float]] | None = None,
    r1_replacement_period: int | Mapping[str, int] | None = None,
    load_r1_series: bool = True,
    windows: Mapping[str, Sequence[int]] | None = None,
    shift: int | None = None,
    gates: Mapping[str, Any] | str | Path | None = None,
) -> dict[str, Any]:
    """Assemble the inputs from the analysis outputs and run :func:`family_s`.

    The H6 guard null comes from ``sham_d5net_ghost_null`` or ``ghost_null_path``
    (:func:`load_ghost_null`), never both. Without ``r1_series`` the R1 recovery series are
    built from the R1 run directories listed in ``trunk_stats`` (``load_r1_series``). The
    windows resolve as in :func:`resolve_windows`; when ``trunk_stats`` records its branch
    window it must equal the family-S primary window (``ValueError`` otherwise)."""
    if sham_d5net_ghost_null is not None and ghost_null_path is not None:
        raise ValueError("pass either sham_d5net_ghost_null or ghost_null_path, not both")
    st = json.loads(Path(trunk_stats).read_text(encoding="utf-8"))
    win, _ = resolve_windows(windows, shift, gates)
    branch = (st.get("windows") or {}).get("branch")
    if branch is not None and (int(branch[0]), int(branch[1])) != win["primary"]:
        p0, p1 = win["primary"]
        raise ValueError(
            f"{trunk_stats}: the contrasts were computed on the branch window "
            f"{branch[0]}-{branch[1]} but the family-S primary window is {p0}-{p1}; re-run the "
            "campaign analysis on the registered (shifted) windows or pass the matching shift"
        )
    runs = st.get("runs", [])
    contrasts = st.get("contrasts", {})
    qual = st.get("inference", {}).get("qualification", {})
    lane_h = qual.get("by_lane", {}).get("H", {})
    qualified = lane_h.get("qualified") or qual.get("qualified") or []
    fallback = bool(lane_h.get("available")) and not lane_h.get("outcome", {}).get(
        "h1_tested", True
    )
    trunk_dp = [
        r["delta_p"]
        for r in runs
        if r.get("registered_arm") == "trunk" and r.get("delta_p") is not None
    ]
    gradient_runs: list[dict[str, Any]] = []
    if gradient_stats and Path(gradient_stats).exists():
        gs = json.loads(Path(gradient_stats).read_text(encoding="utf-8"))
        gradient_runs = [
            {"arm": r["registered_arm"], "seed": r.get("seed"), "delta_p": r.get("delta_p")}
            for r in gs.get("runs", [])
            if r.get("registered_arm") in GRADIENT_LEVELS
        ]
    gradient_runs += [
        {"arm": r["registered_arm"], "seed": r.get("seed"), "delta_p": r.get("delta_p")}
        for r in runs
        if r.get("registered_arm") in GRADIENT_LEVELS
    ]
    d5 = None
    if corpus_path and Path(corpus_path).exists():
        qset = set(qualified)
        d5 = d5net_by_arm(
            corpus_path,
            {
                r["run_id"]: r["registered_arm"]
                for r in runs
                if not qset or r.get("trunk_id") in qset
            },
        )
    ghost_null = (
        load_ghost_null(ghost_null_path) if ghost_null_path is not None else sham_d5net_ghost_null
    )
    r1_excluded: list[dict[str, Any]] = []
    r1_source = "given" if r1_series is not None else None
    if r1_series is None and load_r1_series and any(r.get("registered_arm") == "R1" for r in runs):
        r1 = r1_series_from_runs(runs, st.get("benchmarks", {}), qualified)
        r1_excluded = r1["excluded"]
        if r1["series"]:
            r1_series, r1_replacement_period = r1["series"], r1["replacement_period"]
            r1_source = "R1 run directories"
    flags = None
    if probe_flags_path and Path(probe_flags_path).exists():
        flags = json.loads(Path(probe_flags_path).read_text(encoding="utf-8")).get("probe_flags")
    sting = sting_inputs(sting_campaigns, qualified, probe_flags=flags) if sting_campaigns else None
    out = family_s(
        contrasts,
        gradient_runs,
        trunk_dp,
        d5,
        None,
        sting,
        alpha,
        n_perm,
        seed,
        fallback=fallback,
        sham_d5net_ghost_null=ghost_null,
        r1_series=r1_series,
        r1_replacement_period=r1_replacement_period,
        windows=windows,
        shift=shift,
        gates=gates,
    )
    if r1_excluded:
        out["hypotheses"]["H7-same"]["recovery_excluded"] = r1_excluded
    out["inputs"] = {
        "trunk_stats": str(trunk_stats),
        "gradient_stats": str(gradient_stats) if gradient_stats else None,
        "corpus": str(corpus_path) if corpus_path else None,
        "ghost_null": str(ghost_null_path)
        if ghost_null_path is not None
        else ("given" if sham_d5net_ghost_null is not None else None),
        "r1_series": r1_source,
        "r1_excluded": r1_excluded,
        "gates": str(gates) if isinstance(gates, str | Path) else ("given" if gates else None),
        "stats_branch_window": list(branch) if branch is not None else None,
        "sting_campaigns": [str(c) for c in sting_campaigns],
        "n_qualified": len(qualified),
        "fallback": fallback,
    }
    return out


def _fmt_co(co: Mapping[str, Any] | None) -> str:
    if not co:
        return "–"
    return "; ".join(
        f"{k}: {'yes' if v is True else 'no' if v is False else 'n/a'}" for k, v in co.items()
    )


def family_s_md(res: dict[str, Any]) -> str:
    from finswarm.analysis import tables

    rows = []
    for h in res["family"]:
        v = res["hypotheses"][h]
        est = v.get("estimate", {})
        rows.append(
            {
                "hypothesis": h,
                "lever / data": v.get(
                    "lever", "gradient" if h == "H2" else "sting" if h == "H8" else "–"
                ),
                "estimate [BCa 95 %]": tables.fmt_ci(
                    est.get("mean"), est.get("bca_lo"), est.get("bca_hi")
                )
                if est
                else tables.fmt(v.get("test", {}).get("diff", v.get("spearman", {}).get("rho")), 3),
                "MEI": tables.fmt(v.get("mei"), 2),
                "p": tables.fmt_p(v.get("p")),
                "p Holm (m = 8)": tables.fmt_p(v.get("p_holm")),
                "reject (Holm)": "yes" if v.get("reject") else "no",
                "co-conditions": _fmt_co(v.get("co_conditions")),
                "supported": "yes" if v.get("supported") else "no",
                "status": v.get("status", "–"),
            }
        )
    lines = ["# Family S (Holm, m = 8 fixed)", "", res["rule"] + ".", ""]
    w = res.get("windows")
    if w:
        sh = (res.get("window_info") or {}).get("shift", 0)
        lines += [
            f"Windows (shift +{sh}): primary {w['primary'][0]}-{w['primary'][1]}; sensitivity "
            f"{w['sensitivity_30'][0]}-{w['sensitivity_30'][1]} and "
            f"{w['sensitivity_50'][0]}-{w['sensitivity_50'][1]} (reported, not selected).",
            "",
        ]
    if "n_supported" in res:
        lines += [
            f"Supported (Holm reject and every co-condition met): {res['n_supported']} of "
            f"{res['m']}; rejected by Holm: {res['n_rejected']}.",
            "",
        ]
    lines.append(tables.markdown_table(rows))
    return "\n".join(lines)
