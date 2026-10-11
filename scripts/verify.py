"""Independently re-derive every headline number of the paper from the raw run logs.

Usage::

    uv run python scripts/verify.py [paper/numbers.json] [--root .] [--report out.json]

``numbers.json`` maps ``claim_id -> {metric, runs, params, expected, tolerance}`` where
``runs`` is a glob of run directories (relative to ``--root``). Exit code 0 when every
claim is within tolerance, 1 on any mismatch, 2 when a claim cannot be computed. A claim
whose ``expected`` is null is reported ``pending`` and does not fail the run.

Independence (SAP section 12): this file shares NO code with ``finswarm`` (not even the
benchmarks); numpy + pyarrow only; the normal, t and beta functions, the BCa bootstrap, the
Holm step, the Spearman permutation and the AUC are re-implemented here. Bootstraps and
permutations use a fixed seed (20261010).

Run filters (``params``): ``arm`` / ``label`` (manifest fields), ``filter`` (dotted manifest
path -> value), ``qualified_from`` (a ``stats.json`` whose qualified trunks restrict the runs),
``members`` (incumbent slots; default every seller), ``window`` ``[start, end]``, ``cap``
(``min(index, cap)``: 1.0 = the registered capped index), ``bench`` (explicit
``{p_nash, p_mono, pi_nash, pi_mono}``, logit parameters solved here by bisection, or
``"run_closed_form"`` = the closed form of the run's own resolved market, alpha-scaled).

Per-run metrics (aggregated with ``params.agg``: mean | median | min | max | count | sum |
share_above:<x> | pooled | bca_lo | bca_hi | bca90_lo | bca90_hi, the BCa ones stratified by
the manifest key ``params.strata``): ``delta_price`` (incumbents' price index), ``delta``
(profit index), ``share_above_pm``, ``mean_price``, ``trade_weighted_price``, ``mean_profit``,
``undercut_rate``, ``parse_fail_rate`` (NaN without decisions; ``pooled`` = failed / total),
``gs_efficiency``, ``established`` (1 iff the price and profit indices reach
``params.thresholds``).
Across runs: ``n_runs``; ``paired_contrast`` (reference ``runs`` vs ``treatment_runs`` by
trunk; ``stat``: mean | bca_lo | bca_hi | bca90_lo | bca90_hi | t_lower_95 | p_t_one_sided
[``mei``] | tost_p [``margin``] | sd | n); ``holm_family`` (``family`` of levers, ``stat``
p_holm of ``lever`` | n_rejected); ``split_sample_best`` (odd-seed selection, even-seed
estimate); ``spearman_level`` (``level_key``, ``level_map``, ``group_key``; rho | p_perm);
``welch_one_sided`` (``mei``; diff | p); ``trend_slope``; ``follow_rate`` (sting events);
``proportion_diff`` (collusive vs competitive follow rates, cluster-robust, ``mei``);
``mqc_pp`` (true-value surplus loss, trades); ``detection_auc`` (frozen rules on a corpus
parquet: auc | tpr | fpr | fpr:<kind>); ``corpus_feature_check`` (D1 of the corpus recomputed
from the logs: max abs difference).
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

try:  # pyarrow is the reader; polars is the fallback when the OS blocks pyarrow's Parquet DLL
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover - environment-dependent (Windows application control)
    pq = None


class _Col:
    def __init__(self, s: Any) -> None:
        self._s = s

    def to_numpy(self, zero_copy_only: bool = False) -> np.ndarray:
        return self._s.to_numpy()

    def to_pylist(self) -> list[Any]:
        return self._s.to_list()


class _Tbl:
    """The slice of the pyarrow Table API that this module uses, on a polars DataFrame."""

    def __init__(self, df: Any) -> None:
        self._df = df
        self.column_names = list(df.columns)
        self.num_rows = df.height

    def column(self, name: str) -> _Col:
        return _Col(self._df[name])


class _Schema:
    def __init__(self, names: list[str]) -> None:
        self.names = names


class _PolarsParquet:
    """Drop-in for ``pyarrow.parquet`` (read_table, read_schema) backed by polars."""

    @staticmethod
    def read_table(path: Any, columns: list[str] | None = None) -> _Tbl:
        import polars as pl

        return _Tbl(pl.read_parquet(path, columns=columns))

    @staticmethod
    def read_schema(path: Any) -> _Schema:
        import polars as pl

        return _Schema(list(pl.read_parquet_schema(path)))


if pq is None:
    pq = _PolarsParquet()  # type: ignore[assignment]

SEED = 20261010
DEFAULT_B = 10_000


# --------------------------------------------------------------------------- io


def table_files(run_dir: Path, table: str) -> list[Path]:
    files: list[Path] = []
    tdir = run_dir / table
    if tdir.is_dir():
        files.extend(sorted(tdir.rglob("*.parquet")))
    files.extend(sorted(p for p in run_dir.glob(f"{table}*.parquet") if p.is_file()))
    return files


# A-008 period convention: the store numbers periods from 0 (trunk 0..99, branch 0..149,
# positive control 0..299; branch_period 100 = the first branch period), the registered
# windows from 1 (trunk [71, 100], branch [111, 150], fish [251, 300]). ``load`` returns
# ``period`` in registered coordinates (store period + 1), independently of the analysis
# package's io.read_table, so the YAML windows apply unchanged.
STORE_PERIOD_OFFSET = 1


def load(run_dir: Path, table: str, columns: list[str]) -> dict[str, np.ndarray]:
    files = table_files(run_dir, table)
    if not files:
        raise FileNotFoundError(f"{run_dir}: no '{table}' parquet files")
    parts = []
    for f in files:
        names = pq.read_schema(f).names
        have = [c for c in columns if c in names]
        t = pq.read_table(f, columns=have)  # only the columns used: one bad column elsewhere
        parts.append((t, have))  # in a part never blocks a metric that does not read it
    out: dict[str, np.ndarray] = {}
    for c in columns:
        arrs = []
        for t, have in parts:
            if c in have:
                arr = t.column(c).to_numpy(zero_copy_only=False)
                if c == "period":
                    arr = arr.astype(np.int64) + STORE_PERIOD_OFFSET
                arrs.append(arr)
            else:
                arrs.append(np.full(t.num_rows, None, dtype=object))
        out[c] = np.concatenate(arrs) if arrs else np.array([])
    return out


def has_table(run_dir: Path, table: str) -> bool:
    return bool(table_files(run_dir, table))


def manifest(run_dir: Path) -> dict[str, Any]:
    p = run_dir / "manifest.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def dotted(d: Any, path: str) -> Any:
    cur = d
    for k in path.split("."):
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        elif isinstance(cur, list) and k.isdigit() and int(k) < len(cur):
            cur = cur[int(k)]
        else:
            return None
    return cur


def trunk_of(run_dir: Path) -> str:
    m = manifest(run_dir)
    for k in ("trunk_id", "trunk", "parent_run_id", "parent"):
        if m.get(k) is not None:
            return str(m[k])
    return str(m.get("run_id") or run_dir.name)


def arm_of(run_dir: Path) -> str:
    m = manifest(run_dir)
    return str(m.get("arm") or dotted(m, "resolved_config.tags.arm") or m.get("condition") or "")


def label_of(run_dir: Path) -> str:
    m = manifest(run_dir)
    return str(m.get("label") or dotted(m, "resolved_config.tags.label") or "")


def seed_of(run_dir: Path) -> Any:
    return manifest(run_dir).get("seed")


def _same(a: Any, b: Any) -> bool:
    try:
        return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-12)
    except (TypeError, ValueError):
        return str(a) == str(b)


def match_runs(root: Path, pattern: str, params: dict[str, Any], prefix: str = "") -> list[Path]:
    """Run directories matching the glob and the filters (``arm``, ``label``, ``filter``,
    ``qualified_from``); ``prefix`` selects ``treatment_*`` / ``extra_*`` filters."""
    runs = sorted(Path(p) for p in glob.glob(str(root / pattern)) if Path(p).is_dir())
    runs = [r for r in runs if (r / "manifest.json").exists() or has_table(r, "agent_period")]
    arm = params.get(f"{prefix}arm")
    label = params.get(f"{prefix}label")
    flt = params.get(f"{prefix}filter") or {}
    if arm is not None:
        runs = [r for r in runs if arm_of(r) == str(arm)]
    if label is not None:
        runs = [r for r in runs if label_of(r) == str(label)]
    for k, v in flt.items():
        runs = [r for r in runs if _same(dotted(manifest(r), k), v)]
    runs = [r for r in runs if manifest(r).get("status") in (None, "ok")]
    if params.get("window") and not prefix and params.get("sap_filter", True):
        runs = [r for r in runs if sap_excluded(r, params) is None]
    q = params.get("qualified_from")
    rec_spec = params.get("qualified_recompute")
    if rec_spec and not prefix:
        rec = recompute_qualified(root, rec_spec)
        if q:  # the analysis' own list must agree with the independent recomputation (A-010)
            theirs = _qualified(root / q)
            known = {trunk_of(r) for r in glob_runs(root, rec_spec["runs"])}
            if theirs is not None and (theirs & known) != rec:
                raise ValueError(
                    f"qualification mismatch: analysis {sorted(theirs & known)} vs "
                    f"recomputed {sorted(rec)}"
                )
        runs = [r for r in runs if trunk_of(r) in rec]
    elif q:
        qualified = _qualified(root / q)
        if qualified is not None:
            runs = [r for r in runs if trunk_of(r) in qualified]
    return runs


MAX_FAIL_RATE = 0.10  # SAP s10: a run with more than 10 % failed decisions is excluded


def glob_runs(root: Path, pattern: str) -> list[Path]:
    runs = sorted(Path(p) for p in glob.glob(str(root / pattern)) if Path(p).is_dir())
    return [r for r in runs if (r / "manifest.json").exists() or has_table(r, "agent_period")]


def _own_fail_rate(run: Path) -> float:
    """Share of failed decisions (parse failure, error or refusal) of the run's own periods:
    after the branch period for a branch run, all periods for a trunk (as the analysis)."""
    if not has_table(run, "llm_decisions"):
        return float("nan")
    bp = manifest(run).get("branch_period")
    cols = ["parse_ok", "error", "refusal"] + (["period"] if bp is not None else [])
    try:  # a from-scratch run needs no period column (all its decisions are its own)
        d = load(run, "llm_decisions", cols)
    except OSError as e:
        raise ValueError(
            f"{run}: unreadable llm_decisions ({e}); data integrity, report to ops"
        ) from e
    n = d["parse_ok"].size
    if bp is not None:
        keep = d["period"].astype(float) > float(bp)
    else:
        keep = np.ones(n, dtype=bool)
    if not keep.any():
        return float("nan")
    ok = np.array([bool(v) if v is not None else False for v in d["parse_ok"]])[keep]
    err = np.array([e is not None and e != "" for e in d["error"]])[keep]
    ref = np.array([bool(v) if v is not None else False for v in d["refusal"]])[keep]
    return float(np.mean(~ok | err | ref))


def sap_excluded(run: Path, params: dict[str, Any]) -> str | None:
    """The SAP s10 exclusions, recomputed from the raw tables (A-010): an incomplete analysis
    window (fewer distinct periods than the window length) or more than 10 % failed decisions.
    Returns the reason, or None when the run stays."""
    w = params.get("window")
    if w and has_table(run, "agent_period"):
        per = load(run, "agent_period", ["period"])["period"].astype(float)
        n = np.unique(per[(per >= w[0]) & (per <= w[1])]).size
        if n != int(w[1] - w[0] + 1):
            return f"incomplete window ({n} of {int(w[1] - w[0] + 1)} periods)"
    rate = _own_fail_rate(run)
    if math.isfinite(rate) and rate > MAX_FAIL_RATE:
        return f"failed decisions {rate:.1%}"
    return None


def recompute_qualified(root: Path, spec: dict[str, Any]) -> set[str]:
    """Qualified trunks from the raw trunk runs (Gate 2 definition, A-010): capped Delta_p AND
    Delta_pi both >= the thresholds over the trunk window on the pinned trunk benchmarks, after
    the SAP exclusions. ``spec``: runs (glob), arm, window, members, bench (+ thresholds)."""
    p = {k: v for k, v in spec.items() if k != "runs"}
    p.setdefault("arm", "trunk")
    p["cap"] = p.get("cap", 1.0)
    trunks = match_runs(root, spec["runs"], p)
    return {trunk_of(r) for r in trunks if m_established(r, p) == 1.0}


def _qualified(stats_path: Path) -> set[str] | None:
    if not stats_path.exists():
        return None
    st = json.loads(stats_path.read_text(encoding="utf-8"))
    q = st.get("inference", {}).get("qualification", {})
    lanes = q.get("by_lane", {})
    out: set[str] = set()
    found = False
    for v in lanes.values():
        if v.get("available"):
            found = True
            out |= set(v.get("qualified", []))
    if not found:
        if q.get("qualified") is None:
            return None
        return set(q["qualified"])
    return out


# --------------------------------------------------------------------------- distributions


def _erf(x: np.ndarray) -> np.ndarray:
    return np.vectorize(math.erf)(np.asarray(x, dtype=float))


def norm_cdf(x: Any) -> np.ndarray:
    return 0.5 * (1.0 + _erf(np.asarray(x, dtype=float) / math.sqrt(2.0)))


def norm_ppf(p: Any) -> np.ndarray:
    """Acklam's rational approximation of the normal quantile (relative error ~1e-9),
    refined by one Newton step."""
    p = np.asarray(p, dtype=float)
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00]
    plow, phigh = 0.02425, 1 - 0.02425
    out = np.empty_like(p)
    lo = p < plow
    hi = p > phigh
    mid = ~(lo | hi)
    q = np.sqrt(-2 * np.log(p[lo]))
    out[lo] = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
        (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
    )
    q = p[mid] - 0.5
    r = q * q
    out[mid] = (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
        * q
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    )
    q = np.sqrt(-2 * np.log(1 - p[hi]))
    out[hi] = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
        (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
    )
    e = norm_cdf(out) - p
    out -= e * math.sqrt(2 * math.pi) * np.exp(out**2 / 2)
    return out


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction of the incomplete beta function (Numerical Recipes)."""
    tiny, eps = 1e-300, 3e-16
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 400):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / (c if abs(c) > tiny else tiny)
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / (c if abs(c) > tiny else tiny)
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta ``I_x(a, b)``."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log(1 - x))
    if x < (a + 1) / (a + b + 2):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1 - x) / b


def t_sf(t: float, df: float) -> float:
    """Upper tail of Student's t."""
    if not math.isfinite(t):
        return 0.0 if t > 0 else 1.0
    x = df / (df + t * t)
    tail = 0.5 * betainc(df / 2.0, 0.5, x)
    return tail if t > 0 else 1.0 - tail


def t_cdf(t: float, df: float) -> float:
    return 1.0 - t_sf(t, df)


def t_ppf(p: float, df: float) -> float:
    lo, hi = -50.0, 50.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def rankdata(x: np.ndarray) -> np.ndarray:
    """Average ranks (ties share the mean rank), 1-based."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size, dtype=float)
    sx = x[order]
    i = 0
    while i < x.size:
        j = i
        while j + 1 < x.size and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


# --------------------------------------------------------------------------- statistics


def bca(
    values: np.ndarray, strata: np.ndarray | None, level: float, B: int, seed: int = SEED
) -> tuple[float, float]:
    """BCa percentile interval of the mean (Efron 1987), optionally stratified."""
    v = np.asarray(values, dtype=float)
    keep = np.isfinite(v)
    v = v[keep]
    n = v.size
    if n < 2 or np.all(v == v[0]):
        m = float(v.mean()) if n else float("nan")
        return m, m
    s = (
        np.asarray(strata)[keep].astype(str)
        if strata is not None
        else np.zeros(n, dtype=int).astype(str)
    )
    rng = np.random.default_rng(seed)
    theta = float(v.mean())
    boot = np.zeros(B)
    for g in np.unique(s):
        idx = np.where(s == g)[0]
        boot += v[idx[rng.integers(0, idx.size, size=(B, idx.size))]].sum(axis=1)
    boot /= n
    prop = (np.sum(boot < theta) + 0.5 * np.sum(boot == theta)) / B
    z0 = float(norm_ppf(np.clip(prop, 1.0 / (2 * B), 1 - 1.0 / (2 * B))))
    jack = np.array([np.delete(v, i).mean() for i in range(n)])
    d = jack.mean() - jack
    den = 6.0 * (np.sum(d**2) ** 1.5)
    acc = float(np.sum(d**3) / den) if den > 0 else 0.0
    alpha = (1.0 - level) / 2.0
    zs = norm_ppf(np.array([alpha, 1.0 - alpha]))
    adj = norm_cdf(z0 + (z0 + zs) / (1.0 - acc * (z0 + zs)))
    lo, hi = np.quantile(boot, adj)
    return float(lo), float(hi)


def paired_t_one_sided(d: np.ndarray, mei: float = 0.0) -> float:
    d = d[np.isfinite(d)]
    n = d.size
    if n < 2:
        return float("nan")
    se = d.std(ddof=1) / math.sqrt(n)
    if se == 0:
        return 0.0 if d.mean() > mei else 1.0
    return t_sf((d.mean() - mei) / se, n - 1)


def t_lower_bound(d: np.ndarray, level: float = 0.95) -> float:
    d = d[np.isfinite(d)]
    n = d.size
    if n < 2:
        return float("nan")
    return float(d.mean() - t_ppf(level, n - 1) * d.std(ddof=1) / math.sqrt(n))


def tost_p(d: np.ndarray, margin: float) -> float:
    d = d[np.isfinite(d)]
    n = d.size
    if n < 2:
        return float("nan")
    se = d.std(ddof=1) / math.sqrt(n)
    m = d.mean()
    if se == 0:
        return 0.0 if -margin < m < margin else 1.0
    return max(t_sf((m + margin) / se, n - 1), t_cdf((m - margin) / se, n - 1))


def holm(p: list[float], alpha: float = 0.05) -> tuple[list[float], list[bool]]:
    raw = np.array([1.0 if not math.isfinite(x) else x for x in p])
    order = np.argsort(raw, kind="stable")
    adj = np.full(raw.size, np.nan)
    running = 0.0
    m = raw.size
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * raw[i]))
        adj[i] = running
    return adj.tolist(), [bool(a <= alpha) for a in adj]


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = rankdata(x), rankdata(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def cluster_proportion(pairs: list[tuple[float, float]]) -> tuple[float, float]:
    """Pooled proportion and cluster-robust SE (CR1 ratio estimator)."""
    y = np.array([s for s, _ in pairs], dtype=float)
    n = np.array([t for _, t in pairs], dtype=float)
    k = y.size
    tot = n.sum()
    if k == 0 or tot == 0:
        return float("nan"), float("nan")
    p = y.sum() / tot
    var = k / (k - 1) * float(np.sum((y - p * n) ** 2)) / tot**2 if k > 1 else 0.0
    return float(p), float(math.sqrt(var))


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    s = np.nan_to_num(np.asarray(scores, dtype=float), nan=-np.inf)
    y = np.asarray(labels)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# --------------------------------------------------------------------------- benchmarks


def _bisect(f: Any, lo: float, hi: float) -> float:
    flo = f(lo)
    while np.sign(f(hi)) == np.sign(flo):
        hi = lo + 2.0 * (hi - lo)
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if np.sign(f(mid)) == np.sign(flo):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def logit_bench(
    n: int,
    a: float = 2.0,
    a0: float = 0.0,
    mu: float = 0.25,
    c: float = 1.0,
    alpha: float = 1.0,
    beta: float = 1.0,
    tau: float | None = None,
    outside_utility: float = 0.0,
) -> dict[str, float]:
    """Symmetric Nash and joint-monopoly prices and per-firm profits of the routed logit
    demand (``tau`` and ``outside_utility`` folded into ``mu`` and ``a0`` as the engine
    does; prices scale with ``alpha``, profits with ``alpha * beta``)."""
    mu_eff = (tau / alpha) if tau is not None else mu
    a0_eff = a0 + outside_utility / alpha

    # stable in x = exp((a0 - a + p) / mu): s = 1 / (n + x), 1 - s = (n - 1 + x) / (n + x) and
    # 1 - n s = x / (n + x); the naive forms overflow or hit 1 - n s = 0 at small tau (A-010)
    def x(p: float) -> float:
        return math.exp(min((a0_eff - a + p) / mu_eff, 700.0))

    def s(p: float) -> float:
        return 1.0 / (n + x(p))

    pn = _bisect(lambda p: p - c - mu_eff * (n + x(p)) / (n - 1 + x(p)), c, c + mu_eff + 1)
    pm = _bisect(lambda p: p - c - mu_eff * (n + x(p)) / x(p), c, c + mu_eff + 1)
    return {
        "p_nash": alpha * pn,
        "p_mono": alpha * pm,
        "pi_nash": alpha * beta * (pn - c) * s(pn),
        "pi_mono": alpha * beta * (pm - c) * s(pm),
    }


def bench_for(params: dict[str, Any], run: Path, n: int, key: str = "bench") -> dict[str, float]:
    b = params.get(key, {})
    if b == "run_closed_form":
        m = manifest(run)
        market = dotted(m, "resolved_config.market") or {}
        dem = market.get("demand", {})
        routing = market.get("routing", {}) or {}
        tau = routing.get("tau")
        return logit_bench(
            n,
            a=float(dem.get("a", 2.0)),
            a0=float(dem.get("a0", 0.0)),
            mu=float(dem.get("mu", 0.25)),
            c=float(market.get("cost", 1.0)),
            alpha=float(market.get("alpha", m.get("alpha", 1.0))),
            beta=float(dem.get("beta", 1.0)),
            tau=None if tau is None else float(tau),
            outside_utility=float(routing.get("outside_utility", 0.0) or 0.0),
        )
    if isinstance(b, dict) and all(k in b for k in ("p_nash", "p_mono", "pi_nash", "pi_mono")):
        return {k: float(b[k]) for k in ("p_nash", "p_mono", "pi_nash", "pi_mono")}
    return logit_bench(n, **{k: float(v) for k, v in (b or {}).items()})


# --------------------------------------------------------------------------- per-run metrics


def _window(d: dict[str, np.ndarray], params: dict[str, Any]) -> dict[str, np.ndarray]:
    w = params.get("window")
    if not w:
        return d
    lo = -np.inf if w[0] is None else w[0]
    hi = np.inf if w[1] is None else w[1]
    keep = (d["period"].astype(float) >= lo) & (d["period"].astype(float) <= hi)
    return {k: v[keep] for k, v in d.items()}


def _members(d: dict[str, np.ndarray], params: dict[str, Any]) -> dict[str, np.ndarray]:
    mem = params.get("members")
    if mem is None or "slot" not in d or d["slot"] is None:
        return d
    sl = np.array([(-1 if v is None else int(v)) for v in d["slot"]])
    keep = np.isin(sl, [int(m) for m in mem])
    return {k: v[keep] for k, v in d.items()}


def _frame(run: Path, params: dict[str, Any], cols: list[str]) -> dict[str, np.ndarray]:
    d = load(run, "agent_period", ["period", "slot", *cols])
    d = _members(_window(d, params), params)
    return d


def _period_means(period: np.ndarray, value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ps = np.unique(period)
    vals = np.array([np.nanmean(value[period == p].astype(float)) for p in ps])
    return ps, vals


def _n_members(params: dict[str, Any], d: dict[str, np.ndarray]) -> int:
    mem = params.get("members")
    if mem is not None:
        return len(mem)
    return len(np.unique(d["slot"] if "slot" in d else d["period"]))


def _cap(v: float, params: dict[str, Any]) -> float:
    cap = params.get("cap")
    return float(min(v, float(cap))) if cap is not None and math.isfinite(v) else v


def m_delta_price(run: Path, params: dict[str, Any]) -> float:
    d = _frame(run, params, ["price"])
    if d["period"].size == 0:
        return float("nan")
    b = bench_for(params, run, _n_members(params, d))
    price = d["price"].astype(float)
    ok = np.isfinite(price) & (price > 0)
    _, pbar = _period_means(d["period"][ok], price[ok])
    return _cap(float(np.mean((pbar - b["p_nash"]) / (b["p_mono"] - b["p_nash"]))), params)


def m_delta(run: Path, params: dict[str, Any]) -> float:
    d = _frame(run, params, ["profit"])
    if d["period"].size == 0:
        return float("nan")
    b = bench_for(params, run, _n_members(params, d))
    _, pi = _period_means(d["period"], d["profit"].astype(float))
    return float(np.mean((pi - b["pi_nash"]) / (b["pi_mono"] - b["pi_nash"])))


def m_share_above_pm(run: Path, params: dict[str, Any]) -> float:
    d = _frame(run, params, ["price"])
    if d["period"].size == 0:
        return float("nan")
    b = bench_for(params, run, _n_members(params, d))
    price = d["price"].astype(float)
    ok = np.isfinite(price) & (price > 0)
    return float(np.mean(price[ok] > b["p_mono"])) if ok.any() else float("nan")


def m_mean_price(run: Path, params: dict[str, Any]) -> float:
    d = _frame(run, params, ["price"])
    p = d["price"].astype(float)
    p = p[np.isfinite(p) & (p > 0)]
    return float(np.mean(p)) if p.size else float("nan")


def m_trade_weighted_price(run: Path, params: dict[str, Any]) -> float:
    d = _frame(run, params, ["price", "quantity"])
    p, q = d["price"].astype(float), d["quantity"].astype(float)
    ok = np.isfinite(p) & np.isfinite(q) & (q > 0) & (p > 0)
    return (
        float(np.sum(p[ok] * q[ok]) / np.sum(q[ok]))
        if ok.any() and q[ok].sum() > 0
        else float("nan")
    )


def m_mean_profit(run: Path, params: dict[str, Any]) -> float:
    d = _frame(run, params, ["profit"])
    return float(np.mean(d["profit"].astype(float))) if d["period"].size else float("nan")


def m_undercut_rate(run: Path, params: dict[str, Any]) -> float:
    d = _frame(run, params, ["price", "agent_id"])
    key = (
        d["slot"]
        if "slot" in d and d["slot"] is not None and all(v is not None for v in d["slot"])
        else d["agent_id"]
    )
    ps, ags = np.unique(d["period"]), np.unique(key)
    if ps.size < 2 or ags.size < 2:
        return float("nan")
    mat = np.full((len(ps), len(ags)), np.nan)
    mat[np.searchsorted(ps, d["period"]), np.searchsorted(ags, key)] = d["price"].astype(float)
    hits = cells = 0
    for i in range(len(ags)):
        rivals = np.delete(mat[:-1], i, axis=1)
        prev = np.where(np.isnan(rivals), np.inf, rivals).min(axis=1)
        cur = mat[1:, i]
        ok = ~np.isnan(cur) & np.isfinite(prev)
        hits += int(np.sum(cur[ok] < prev[ok] - 1e-9))
        cells += int(ok.sum())
    return hits / cells if cells else float("nan")


def _decisions(run: Path, params: dict[str, Any]) -> tuple[int, int]:
    if not has_table(run, "llm_decisions"):
        return 0, 0
    d = load(run, "llm_decisions", ["parse_ok", "error", "refusal", "period"])
    d = _window(d, params) if params.get("window") else d
    n = d["parse_ok"].size
    ok = np.array([bool(v) if v is not None else False for v in d["parse_ok"]])
    err = np.array([e is not None and e != "" for e in d["error"]])
    ref = np.array([bool(v) if v is not None else False for v in d["refusal"]])
    return int(np.sum(~ok | err | ref)), int(n)


def m_parse_fail_rate(run: Path, params: dict[str, Any]) -> float:
    bad, n = _decisions(run, params)
    return bad / n if n else float("nan")


def m_gs_efficiency(run: Path, params: dict[str, Any]) -> float:
    d = load(run, "trades", ["size", "buyer_value", "seller_cost"])
    bv = np.array([np.nan if v is None else float(v) for v in d["buyer_value"]])
    sc = np.array([np.nan if v is None else float(v) for v in d["seller_cost"]])
    keep = np.isfinite(bv) & np.isfinite(sc)
    q, v, c = d["size"][keep].astype(float), bv[keep], sc[keep]
    realised = float(np.sum(q * (v - c)))
    ov, oc = np.argsort(-v, kind="stable"), np.argsort(c, kind="stable")
    vv, qv = v[ov], q[ov].copy()
    cc, qc = c[oc], q[oc].copy()
    i = j = 0
    best = 0.0
    while i < len(vv) and j < len(cc) and vv[i] > cc[j]:
        x = min(qv[i], qc[j])
        best += x * (vv[i] - cc[j])
        qv[i] -= x
        qc[j] -= x
        i += int(qv[i] <= 1e-12)
        j += int(qc[j] <= 1e-12)
    return realised / best if best > 0 else float("nan")


def m_established(run: Path, params: dict[str, Any]) -> float:
    thr = params.get("thresholds", {"delta_price": 0.30, "delta": 0.30})
    dp = m_delta_price(run, {**params, "cap": params.get("cap", 1.0)})
    dpi = m_delta(run, params)
    ok = (
        math.isfinite(dp)
        and math.isfinite(dpi)
        and dp >= thr["delta_price"]
        and dpi >= thr["delta"]
    )
    return 1.0 if ok else 0.0


def m_true_value_surplus(run: Path, params: dict[str, Any]) -> float:
    """Mean per-period true-value surplus from the trades (window periods without trades = 0)."""
    d = load(run, "trades", ["period", "size", "price", "buyer_value", "seller_cost", "buyer_id"])
    w = params.get("window")
    per = d["period"].astype(float)
    bv = np.array([np.nan if v is None else float(v) for v in d["buyer_value"]])
    sc = np.array([np.nan if v is None else float(v) for v in d["seller_cost"]])
    bid = np.array([-1 if v is None else int(v) for v in d["buyer_id"]])
    keep = np.isfinite(bv) & np.isfinite(sc) & (bid != -1)
    if w:
        keep &= (per >= w[0]) & (per <= w[1])
    ts = d["size"][keep].astype(float) * (
        (bv[keep] - d["price"][keep].astype(float)) + (d["price"][keep].astype(float) - sc[keep])
    )
    n_periods = (w[1] - w[0] + 1) if w else max(len(np.unique(per)), 1)
    return float(ts.sum() / n_periods)


RUN_METRICS = {
    "delta": m_delta,
    "delta_pi": m_delta,
    "delta_price": m_delta_price,
    "share_above_pm": m_share_above_pm,
    "mean_price": m_mean_price,
    "trade_weighted_price": m_trade_weighted_price,
    "mean_profit": m_mean_profit,
    "undercut_rate": m_undercut_rate,
    "parse_fail_rate": m_parse_fail_rate,
    "gs_efficiency": m_gs_efficiency,
    "established": m_established,
    "true_value_surplus": m_true_value_surplus,
}


def aggregate(
    values: np.ndarray, how: str, strata: list[Any] | None = None, B: int = DEFAULT_B
) -> float:
    fin = np.isfinite(values)
    v = values[fin]
    if how == "count":
        return float(np.sum(v > 0.5)) if v.size else 0.0
    if v.size == 0:
        return float("nan")
    if how == "mean":
        return float(v.mean())
    if how == "median":
        return float(np.median(v))
    if how == "min":
        return float(v.min())
    if how == "max":
        return float(v.max())
    if how == "sum":
        return float(v.sum())
    if how.startswith("share_above:"):
        return float(np.mean(v > float(how.split(":", 1)[1])))
    if how in ("bca_lo", "bca_hi", "bca90_lo", "bca90_hi"):
        level = 0.90 if how.startswith("bca90") else 0.95
        s = np.asarray(strata)[fin] if strata is not None else None
        lo, hi = bca(v, s, level, B)
        return lo if how.endswith("lo") else hi
    raise ValueError(f"unknown aggregation {how!r}")


# --------------------------------------------------------------------------- across runs


def _by_trunk(runs: list[Path], fn: Any, params: dict[str, Any]) -> dict[str, float]:
    acc: dict[str, list[float]] = {}
    for r in runs:
        acc.setdefault(trunk_of(r), []).append(fn(r, params))
    return {t: float(np.mean(v)) for t, v in acc.items()}


def _paired_diffs(
    runs: list[Path], params: dict[str, Any], root: Path, treatment_key: str = "treatment_"
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    fn = RUN_METRICS[params.get("run_metric", "delta")]
    treat = match_runs(root, params[f"{treatment_key}runs"], params, treatment_key)
    bench_t = {**params}
    if f"{treatment_key}bench" in params:
        bench_t["bench"] = params[f"{treatment_key}bench"]
    ctrl = _by_trunk(runs, fn, params)
    trt = _by_trunk(treat, fn, bench_t)
    trunks = sorted(set(ctrl) & set(trt))
    d = np.array([ctrl[t] - trt[t] for t in trunks])
    seeds = {t: seed_of(next(r for r in runs if trunk_of(r) == t)) for t in trunks}
    return d, trunks, seeds


def paired_stat(d: np.ndarray, stat: str, params: dict[str, Any]) -> float:
    B = int(params.get("B", DEFAULT_B))
    if stat == "mean":
        return float(d.mean()) if d.size else float("nan")
    if stat == "n":
        return float(d.size)
    if stat == "sd":
        return float(d.std(ddof=1)) if d.size > 1 else float("nan")
    if stat in ("bca_lo", "bca_hi", "bca90_lo", "bca90_hi", "ci_lo", "ci_hi"):
        level = 0.90 if stat.startswith("bca90") else 0.95
        lo, hi = bca(d, None, level, B)
        return lo if stat.endswith("lo") else hi
    if stat == "t_lower_95":
        return t_lower_bound(d, 0.95)
    if stat == "p_t_one_sided":
        return paired_t_one_sided(d, float(params.get("mei", 0.0)))
    if stat == "tost_p":
        return tost_p(d, float(params.get("margin", 0.20)))
    tol = 1e-12  # A-022 reporting: trunks with zero / negative / positive difference
    if stat == "n_zero":
        return float(np.sum(np.abs(d) <= tol))
    if stat == "n_negative":
        return float(np.sum(d < -tol))
    if stat == "n_positive":
        return float(np.sum(d > tol))
    if stat == "mde80":  # one-sided paired-t MDE at 80 % power from the realised SD
        n = d.size
        if n < 2:
            return float("nan")
        q = t_ppf(1 - float(params.get("alpha", 0.05)), n - 1) + t_ppf(0.80, n - 1)
        return float(q * d.std(ddof=1) / math.sqrt(n))
    if stat == "sign_p":  # exact one-sided sign test, zeros dropped (A-024)
        pos, neg = int(np.sum(d > tol)), int(np.sum(d < -tol))
        n = pos + neg
        if n == 0:
            return float("nan")
        return float(sum(math.comb(n, k) for k in range(pos, n + 1)) / 2**n)
    raise ValueError(f"unknown paired stat {stat!r}")


def c_paired_contrast(runs: list[Path], params: dict[str, Any], root: Path) -> float:
    d, trunks, _ = _paired_diffs(runs, params, root)
    if not trunks:
        raise ValueError("paired_contrast: no common trunks")
    return paired_stat(d, params.get("stat", "mean"), params)


def _family_p(
    runs: list[Path], params: dict[str, Any], root: Path
) -> dict[str, tuple[float, np.ndarray, dict[str, Any]]]:
    out = {}
    for lev in params["family"]:
        p = {**params, "treatment_arm": lev}
        try:
            d, _trunks, seeds = _paired_diffs(runs, p, root)
        except (FileNotFoundError, KeyError, ValueError):
            d, _trunks, seeds = np.array([]), [], {}
        pv = paired_t_one_sided(d) if d.size >= 2 else float("nan")
        out[lev] = (pv, d, {t: s for t, s in seeds.items()})
    return out


def c_holm_family(runs: list[Path], params: dict[str, Any], root: Path) -> float:
    fam = params["family"]
    res = _family_p(runs, params, root)
    adj, rej = holm([res[lev][0] for lev in fam], float(params.get("alpha", 0.05)))
    stat = params.get("stat", "p_holm")
    if stat == "n_rejected":
        return float(sum(rej))
    return float(adj[fam.index(params["lever"])])


def c_split_sample_best(runs: list[Path], params: dict[str, Any], root: Path) -> float:
    fam = params["family"]
    res = _family_p(runs, params, root)
    eligible = set(fam)
    ef = params.get("eligible_from")
    if ef and (root / ef).exists():
        st = json.loads((root / ef).read_text(encoding="utf-8"))
        levs = st.get("h1r", {}).get("levers", {})
        eligible = {
            lev
            for lev in fam
            if levs.get(lev, {}).get("mqc") is not None
            and float(levs[lev]["mqc"]) <= float(params.get("mqc_max_pp", 5.0))
        }
    best, best_v = None, -np.inf
    for lev in fam:
        if lev not in eligible:
            continue
        _, d, seeds = res[lev]
        trunks = sorted(seeds)
        sel = [
            d[i] for i, t in enumerate(trunks) if seeds[t] is not None and int(seeds[t]) % 2 == 1
        ]
        if sel and np.mean(sel) > best_v:
            best, best_v = lev, float(np.mean(sel))
    if best is None:
        return float("nan")
    _, d, seeds = res[best]
    trunks = sorted(seeds)
    est = [d[i] for i, t in enumerate(trunks) if seeds[t] is not None and int(seeds[t]) % 2 == 0]
    stat = params.get("stat", "estimate")
    if stat == "selected_index":
        return float(fam.index(best))
    if stat in ("ci_lo", "ci_hi"):  # BCa 95 % on the estimation half
        lo, hi = bca(np.asarray(est, dtype=float), None, 0.95, int(params.get("B", DEFAULT_B)))
        return lo if stat == "ci_lo" else hi
    return float(np.mean(est)) if est else float("nan")


def c_spearman_level(runs: list[Path], params: dict[str, Any], root: Path) -> float:
    fn = RUN_METRICS[params.get("run_metric", "delta_price")]
    lm = params.get("level_map") or {}
    levels, values, groups = [], [], []
    for r in runs:
        raw = dotted(manifest(r), params["level_key"])
        if raw is None:
            continue
        lev = lm.get(str(raw), raw)
        try:
            lev = float(lev)
        except (TypeError, ValueError):
            continue
        levels.append(lev)
        values.append(fn(r, params))
        groups.append(str(dotted(manifest(r), params.get("group_key", "seed"))))
    lv, va, gr = np.array(levels), np.array(values), np.array(groups)
    ok = np.isfinite(lv) & np.isfinite(va)
    lv, va, gr = lv[ok], va[ok], gr[ok]
    rho = spearman(lv, va)
    if params.get("stat", "rho") == "rho":
        return rho
    rng = np.random.default_rng(int(params.get("seed", SEED)))
    n_perm = int(params.get("n_perm", DEFAULT_B))
    idx_by = [np.where(gr == g)[0] for g in np.unique(gr)]
    count = 0
    for _ in range(n_perm):
        perm = lv.copy()
        for idx in idx_by:
            perm[idx] = lv[idx][rng.permutation(idx.size)]
        r = spearman(perm, va)
        if math.isfinite(r) and r <= rho + 1e-12:
            count += 1
    return float((1 + count) / (1 + n_perm))


def c_welch_one_sided(runs: list[Path], params: dict[str, Any], root: Path) -> float:
    fn = RUN_METRICS[params.get("run_metric", "delta_price")]
    a = [fn(r, params) for r in runs]
    if params.get("extra_runs"):
        pe = {**params}
        if "extra_bench" in params:
            pe["bench"] = params["extra_bench"]
        a += [fn(r, pe) for r in match_runs(root, params["extra_runs"], params, "extra_")]
    pt = {**params}
    if "treatment_bench" in params:
        pt["bench"] = params["treatment_bench"]
    b = [fn(r, pt) for r in match_runs(root, params["treatment_runs"], params, "treatment_")]
    x, y = np.array(a, dtype=float), np.array(b, dtype=float)
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    diff = float(x.mean() - y.mean()) if x.size and y.size else float("nan")
    if params.get("stat", "diff") == "diff":
        return diff
    if x.size < 2 or y.size < 2:
        return float("nan")
    va, vb = x.var(ddof=1) / x.size, y.var(ddof=1) / y.size
    se = math.sqrt(va + vb)
    if se == 0:
        return 0.0 if diff > float(params.get("mei", 0.0)) else 1.0
    df = (va + vb) ** 2 / (va**2 / (x.size - 1) + vb**2 / (y.size - 1))
    return t_sf((diff - float(params.get("mei", 0.0))) / se, df)


def c_trend_slope(runs: list[Path], params: dict[str, Any], root: Path) -> float:
    w = params["window"]
    periods = np.arange(int(w[0]), int(w[1]) + 1)
    series = []
    for r in runs:
        d = _frame(r, {**params, "window": list(w)}, ["price"])
        if d["period"].size == 0:
            continue
        b = bench_for(params, r, _n_members(params, d))
        price = d["price"].astype(float)
        ok = np.isfinite(price) & (price > 0)
        ps, pbar = _period_means(d["period"][ok], price[ok])
        dp = (pbar - b["p_nash"]) / (b["p_mono"] - b["p_nash"])
        row = np.full(periods.size, np.nan)
        for p, v in zip(ps, dp, strict=True):
            if int(p) in periods:
                row[int(p) - int(w[0])] = v
        series.append(row)
    if not series:
        return float("nan")
    y = np.nanmean(np.vstack(series), axis=0)
    ok = np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    return float(np.polyfit(periods[ok].astype(float), y[ok], 1)[0])


def _follow_counts(run: Path, params: dict[str, Any]) -> tuple[float, float]:
    """Sting follow events of one run: for every probe start and every rival, one trial; a
    success when the rival's price rises by >= half the sting delta within the hold."""
    if not has_table(run, "events"):
        return 0.0, 0.0
    ev = load(run, "events", ["event_type", "period", "payload"])
    # A-018: the engine nests the sting parameters under payload["event"], and the forced
    # rises are the "raise" phases (one per probe), not the single "sting" row
    delta, hold, slot_ = float("nan"), 5, int(params.get("sting_slot", 4))
    raises: list[int] = []
    sting_rows: list[int] = []
    for et, per, pay in zip(ev["event_type"], ev["period"], ev["payload"], strict=True):
        payload = json.loads(pay) if pay else {}
        if et == "sting":
            sting_rows.append(int(per))
            for d_ in (payload, payload.get("event") or {}, payload.get("applied") or {}):
                if isinstance(d_, dict):
                    if d_.get("delta") is not None and not math.isfinite(delta):
                        delta = float(d_["delta"])
                    if d_.get("hold_epochs") is not None:
                        hold = int(d_["hold_epochs"])
                    if d_.get("slot") is not None:
                        slot_ = int(d_["slot"])
        elif et == "sting_phase" and payload.get("phase") == "raise":
            raises.append(int(per))
    pers = sorted(set(raises)) or sorted(set(sting_rows))  # flat configs: one start per row
    starts = [(s, delta, hold, slot_) for s in pers]
    if not starts:
        return 0.0, 0.0
    d = load(run, "agent_period", ["period", "slot", "price"])
    slots = np.array([(-1 if v is None else int(v)) for v in d["slot"]])
    ps = np.unique(d["period"])
    keys = np.unique(slots)
    mat = np.full((ps.size, keys.size), np.nan)
    mat[np.searchsorted(ps, d["period"]), np.searchsorted(keys, slots)] = d["price"].astype(float)
    succ = trials = 0.0
    for s, delta, hold, sting_slot in starts:
        if not math.isfinite(delta) or delta == 0:
            continue
        for j, k in enumerate(keys):
            if int(k) == int(sting_slot):
                continue
            base = mat[(ps >= s - 3) & (ps < s), j]
            path = mat[(ps >= s) & (ps <= s + hold), j]
            if not np.isfinite(base).any() or not np.isfinite(path).any():
                continue
            trials += 1
            if np.nanmax(path - np.nanmean(base)) >= 0.5 * delta:
                succ += 1
    return succ, trials


def c_follow_rate(runs: list[Path], params: dict[str, Any], root: Path) -> float:
    pairs = [c for c in (_follow_counts(r, params) for r in runs) if c[1] > 0]
    p, _ = cluster_proportion(pairs)
    return p if params.get("stat", "p") == "p" else float(len(pairs))


def c_proportion_diff(runs: list[Path], params: dict[str, Any], root: Path) -> float:
    coll = [c for c in (_follow_counts(r, params) for r in runs) if c[1] > 0]
    comp_runs = match_runs(root, params["treatment_runs"], params, "treatment_")
    comp = [c for c in (_follow_counts(r, params) for r in comp_runs) if c[1] > 0]
    p1, se1 = cluster_proportion(coll)
    p0, se0 = cluster_proportion(comp)
    diff = p1 - p0
    if params.get("stat", "diff") == "diff":
        return float(diff)
    se = math.sqrt(se1**2 + se0**2)
    if se == 0:
        return 0.0 if diff > float(params.get("mei", 0.0)) else 1.0
    df = max(min(len(coll), len(comp)) - 1, 1)  # A-010: t with K-1 df, K = clusters
    return float(1.0 - t_cdf((diff - float(params.get("mei", 0.0))) / se, df))


def c_mqc_pp(runs: list[Path], params: dict[str, Any], root: Path) -> float:
    ref = _by_trunk(runs, m_true_value_surplus, params)
    treat = match_runs(root, params["treatment_runs"], params, "treatment_")
    trt = _by_trunk(treat, m_true_value_surplus, params)
    vals = [100.0 * (ref[t] - trt[t]) / ref[t] for t in sorted(set(ref) & set(trt)) if ref[t] != 0]
    if not vals:
        raise ValueError("mqc_pp: no paired seeds")
    return paired_stat(np.array(vals), params.get("stat", "mean"), params)


def _read_corpus(path: Path, cols: list[str]) -> dict[str, np.ndarray]:
    t = pq.read_table(path)
    return {
        c: t.column(c).to_numpy(zero_copy_only=False)
        if c in t.column_names
        else np.full(t.num_rows, None, dtype=object)
        for c in cols
    }


def c_detection_auc(runs: list[Path], params: dict[str, Any], root: Path) -> float:
    rules = json.loads((root / params["rules"]).read_text(encoding="utf-8"))
    feats = rules["composite"]["features"]
    cols = ["kind", "label", "lane", "seed_group", *feats]
    c = _read_corpus(root / params["corpus"], cols)
    kinds = c["kind"].astype(str)
    pos, neg = set(params["positives"]), set(params["negatives"])
    keep = np.isin(kinds, list(pos | neg))
    if params.get("lane") is not None:
        keep &= c["lane"].astype(str) == str(params["lane"])
    evg = rules.get("split", {}).get("eval_groups")
    if evg:
        keep &= np.isin(c["seed_group"].astype(str), [str(g) for g in evg])
    y = np.where(np.isin(kinds[keep], list(pos)), 1, 0)
    parts = []
    for f in feats:
        ref = np.asarray(rules["features"][f]["reference"], dtype=float)
        v = np.asarray(c[f][keep], dtype=float) * float(rules["directions"][f])
        lo = np.searchsorted(ref, v, side="left")
        hi = np.searchsorted(ref, v, side="right")
        parts.append(np.where(np.isfinite(v), (lo + hi) / (2.0 * ref.size), np.nan))
    m = np.column_stack(parts)
    cnt = np.isfinite(m).sum(axis=1)
    comp = np.where(
        cnt > 0, np.where(np.isfinite(m), m, 0.0).sum(axis=1) / np.maximum(cnt, 1), np.nan
    )
    stat = params.get("stat", "auc")
    if stat == "auc":
        return auc(comp, y)
    flag = np.nan_to_num(comp, nan=-np.inf) >= float(rules["composite"]["threshold"])
    if stat == "tpr":
        return float(flag[y == 1].mean()) if (y == 1).any() else float("nan")
    if stat == "fpr":
        return float(flag[y == 0].mean()) if (y == 0).any() else float("nan")
    if stat.startswith("fpr:"):
        k = stat.split(":", 1)[1]
        sel = (y == 0) & (kinds[keep] == k)
        return float(flag[sel].mean()) if sel.any() else float("nan")
    raise ValueError(f"unknown detection stat {stat!r}")


def c_corpus_feature_check(runs: list[Path], params: dict[str, Any], root: Path) -> float:
    c = _read_corpus(root / params["corpus"], ["run_id", "kind", params.get("feature", "D1")])
    want = set(params.get("kinds", []))
    by_id = {str(r): r for r in runs}
    by_name = {manifest(r).get("run_id", r.name): r for r in runs}
    diffs = []
    for rid, kind, val in zip(c["run_id"], c["kind"], c[params.get("feature", "D1")], strict=True):
        if want and str(kind) not in want:
            continue
        run = by_name.get(str(rid)) or by_id.get(str(rid))
        if run is None or val is None or not math.isfinite(float(val)):
            continue
        diffs.append(abs(m_delta_price(run, params) - float(val)))
    if not diffs:
        raise ValueError("corpus_feature_check: no matching run")
    return float(max(diffs))


# --------------------------------------------------------------------------- A-024 (exploratory)


def _logit_market(run: Path) -> dict[str, float]:
    """Routed-logit parameters of a run's own market (``resolved_config.market``), the engine's
    convention: tau folds into mu (tau / alpha), the outside utility into a0."""
    m = dotted(manifest(run), "resolved_config.market") or {}
    dem = m.get("demand", {}) or {}
    rout = m.get("routing", {}) or {}
    alpha = float(m.get("alpha", 1.0))
    tau = rout.get("tau")
    mu = float(tau) / alpha if tau is not None else float(dem.get("mu", 0.25))
    return {
        "a": float(dem.get("a", 2.0)),
        "a0": float(dem.get("a0", 0.0)) + float(rout.get("outside_utility", 0.0) or 0.0) / alpha,
        "mu": mu,
        "c": float(m.get("cost", 1.0)),
        "alpha": alpha,
        "tick": float((m.get("price_grid", {}) or {}).get("tick", 0.0) or 0.0),
    }


def _shares(p: np.ndarray, mk: dict[str, float]) -> np.ndarray:
    """Logit shares of the sellers in each row of ``p`` (prices, not normalised)."""
    x = np.asarray(p, dtype=float) / mk["alpha"]
    e = np.exp((mk["a"] - x) / mk["mu"])
    return e / (e.sum(axis=1, keepdims=True) + math.exp(mk["a0"] / mk["mu"]))


def _br(rivals: np.ndarray, mk: dict[str, float], iters: int = 100) -> np.ndarray:
    """Static best response of one seller to each row of rival prices (bisection on the FOC)."""
    r = np.atleast_2d(np.asarray(rivals, dtype=float)) / mk["alpha"]
    a, mu, c = mk["a"], mk["mu"], mk["c"]
    u = np.exp((a - r) / mu).sum(axis=1) + math.exp(mk["a0"] / mu)
    lo = np.full(r.shape[0], c)
    hi = np.full(r.shape[0], max(a, c) + 40.0 * mu + 1.0)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        e = np.exp((a - mid) / mu)
        g = mid - c - mu / (1.0 - e / (e + u))
        lo = np.where(g < 0, mid, lo)
        hi = np.where(g < 0, hi, mid)
    return mk["alpha"] * 0.5 * (lo + hi)


def _prices(run: Path, slots: list[int]) -> tuple[np.ndarray, np.ndarray]:
    d = load(run, "agent_period", ["period", "slot", "price"])
    ps = np.unique(d["period"])
    mat = np.full((ps.size, len(slots)), np.nan)
    sl = np.array([(-1 if v is None else int(v)) for v in d["slot"]])
    for j, s in enumerate(slots):
        k = sl == s
        mat[np.searchsorted(ps, d["period"][k]), j] = d["price"][k].astype(float)
    return ps.astype(int), mat


def _bench_scale(params: dict[str, Any], run: Path) -> tuple[float, float]:
    b = bench_for(params, run, 4)
    return float(b["p_nash"]), float(b["p_mono"])


def _ci95(x: np.ndarray) -> tuple[float, float]:
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan"), float("nan")
    h = t_ppf(0.975, x.size - 1) * x.std(ddof=1) / math.sqrt(x.size)
    return float(x.mean() - h), float(x.mean() + h)


def _reduce(x: np.ndarray, stat: str) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if stat == "mean":
        return float(x.mean()) if x.size else float("nan")
    if stat == "ci_lo":
        return _ci95(x)[0]
    if stat == "ci_hi":
        return _ci95(x)[1]
    if stat == "n_positive":
        return float(np.sum(x > 0))
    raise ValueError(f"unknown stat {stat!r}")


def c_r1_regression(runs: list[Path], params: dict[str, Any], root: Path) -> float:
    """The fresh slot-0 copy's first ask (period t0 + 1) or window mean on the remaining
    incumbents' mean ask at t0, OLS across runs, in index units of the trunk benchmarks."""
    t0 = int(params.get("t0", 100))
    w = params.get("window", [111, 150])
    xs, ys = [], []
    for r in runs:
        pn, pm = _bench_scale(params, r)
        ps, mat = _prices(r, [0, 1, 2, 3])
        idx = (mat - pn) / (pm - pn)
        i0, i1 = np.where(ps == t0)[0], np.where(ps == t0 + 1)[0]
        if not i0.size or not i1.size:
            continue
        xs.append(float(np.nanmean(idx[i0[0], 1:])))
        if params.get("y", "first") == "first":
            ys.append(float(idx[i1[0], 0]))
        else:
            k = (ps >= w[0]) & (ps <= w[1])
            ys.append(float(np.nanmean(idx[k, 0])))
    x, y = np.array(xs), np.array(ys)
    if x.size < 3:
        return float("nan")
    xm, ym = x.mean(), y.mean()
    slope = float(np.sum((x - xm) * (y - ym)) / np.sum((x - xm) ** 2))
    icpt = float(ym - slope * xm)
    res = y - (icpt + slope * x)
    r2 = float(1 - np.sum(res**2) / np.sum((y - ym) ** 2))
    return {"slope": slope, "intercept": icpt, "r2": r2}[params.get("stat", "slope")]


def c_entry_br_gap(runs: list[Path], params: dict[str, Any], root: Path) -> float:
    """Incumbent ask minus the static logit best response to the other incumbents' and the
    entrant's asks, over the window and the 4 incumbents, per run (index units)."""
    w = params.get("window", [111, 150])
    ent = int(params.get("entrant_slot", 4))
    gaps = []
    for r in runs:
        pn, pm = _bench_scale(params, r)
        mk = _logit_market(r)
        ps, mat = _prices(r, [0, 1, 2, 3, ent])
        m = mat[(ps >= w[0]) & (ps <= w[1])]
        g = []
        for i in range(4):
            riv = np.delete(m, i, axis=1)
            ok = np.isfinite(riv).all(axis=1) & np.isfinite(m[:, i])
            if ok.any():
                g.append(np.mean((m[ok, i] - _br(riv[ok], mk)) / (pm - pn)))
        gaps.append(float(np.mean(g)) if g else float("nan"))
    return _reduce(np.array(gaps), params.get("stat", "mean"))


def c_probe_response(runs: list[Path], params: dict[str, Any], root: Path) -> float:
    """Forced-undercut probes: rivals' mean ask change over the 3 periods after the cut, their
    static best response to it (index change vs the 3 pre-cut periods), or the deviator's mean
    profit over 10 periods from the cut minus the same trunk's sham."""
    stat = params.get("stat", "rivals_response")
    shams = (
        {trunk_of(s): s for s in match_runs(root, params.get("sham_runs", ""), {"arm": "sham"})}
        if params.get("sham_runs")
        else {}
    )
    vals = []
    for r in runs:
        pn, pm = _bench_scale(params, r)
        mk = _logit_market(r)
        d = 101
        if has_table(r, "events"):
            ev = load(r, "events", ["event_type", "period"])
            fd = [
                int(p)
                for e, p in zip(ev["event_type"], ev["period"], strict=True)
                if e == "forced_deviation"
            ]
            if fd:
                d = min(fd)
        ps, mat = _prices(r, [0, 1, 2, 3])
        pre = np.nanmean(mat[(ps >= d - 3) & (ps <= d - 1)], axis=0)
        dev = mat[np.where(ps == d)[0][0]]
        if stat == "rivals_response":
            resp = np.nanmean(mat[(ps >= d + 1) & (ps <= d + 3)], axis=0)[1:]
            vals.append(float(np.mean((resp - pre[1:]) / (pm - pn))))
        elif stat == "static_br":
            rows = np.array([[dev[0], *np.delete(dev[1:], j)] for j in range(3)])
            vals.append(float(np.mean((_br(rows, mk) - pre[1:]) / (pm - pn))))
        elif stat in ("deviator_gain", "n_deviator_gain_positive"):
            s = shams.get(trunk_of(r))
            if s is None:
                continue
            pd = load(r, "agent_period", ["period", "slot", "profit"])
            sd = load(s, "agent_period", ["period", "slot", "profit"])

            def slot0(dd: dict[str, np.ndarray], d: int = d) -> float:
                sl = np.array([(-1 if v is None else int(v)) for v in dd["slot"]])
                per = dd["period"].astype(int)
                k = (sl == 0) & (per >= d) & (per <= d + 9)
                return float(np.mean(dd["profit"][k].astype(float)))

            vals.append(slot0(pd) - slot0(sd))
    x = np.array(vals, dtype=float)
    if stat == "n_deviator_gain_positive":
        return float(np.sum(x > 0))
    return float(np.nanmean(x)) if x.size else float("nan")


def c_regret(runs: list[Path], params: dict[str, Any], root: Path) -> float:
    """Ex-post regret share of one dealer (``slot``) over the window: 1 - realised mean profit /
    the best constant price in hindsight against the rivals' logged asks (prices rounded to
    the market tick, as the engine does), averaged over runs."""
    w = params.get("window", [111, 150])
    slot = int(params.get("slot", 0))
    regs = []
    for r in runs:
        pn, pm = _bench_scale(params, r)
        mk = _logit_market(r)
        ps, mat = _prices(r, [0, 1, 2, 3])
        P = mat[(ps >= w[0]) & (ps <= w[1])]
        P = P[np.isfinite(P).all(axis=1)]
        if not P.size:
            continue

        def prof(Q: np.ndarray, mk: dict[str, float] = mk) -> float:
            if mk["tick"] > 0:
                Q = np.round(Q / mk["tick"]) * mk["tick"]
            s = _shares(Q, mk)
            return float(np.mean((Q[:, slot] - mk["c"] * mk["alpha"]) * s[:, slot]))

        real = prof(P)
        best = -np.inf
        for c in np.arange(pn * 0.8, pm * 1.6, 0.005):
            Q = P.copy()
            Q[:, slot] = c
            best = max(best, prof(Q))
        if best > 0:
            regs.append(1 - real / best)
    return float(np.mean(regs)) if regs else float("nan")


def c_holm_from_source(runs: list[Path], params: dict[str, Any], root: Path) -> float:
    """Holm step-down re-derived from the raw p-values the analysis wrote (``file`` +
    ``paths``, one per family member, ``target`` = the member whose adjusted p is returned).
    Only the multiplicity step is independent here; the raw p-values are verified by their own
    rows."""
    doc = json.loads((root / params["file"]).read_text(encoding="utf-8"))
    ps = []
    for path in params["paths"]:
        v = dotted(doc, path)
        ps.append(1.0 if v is None else float(v))
    adj, _ = holm(ps)
    return float(adj[list(params["names"]).index(params["target"])])


def c_corpus_count(runs: list[Path], params: dict[str, Any], root: Path) -> float:
    """Windows of the given kinds (and lane) in a detection corpus parquet, or the share of
    ``kinds`` among ``kinds + other_kinds``."""
    t = pq.read_table(root / params["file"], columns=["kind", "lane"])
    kind = np.array(t.column("kind").to_pylist(), dtype=object)
    lane = np.array(t.column("lane").to_pylist(), dtype=object)
    keep = lane == params["lane"] if params.get("lane") else np.ones(kind.size, dtype=bool)
    n = float(np.sum(keep & np.isin(kind, params["kinds"])))
    if params.get("stat", "count") == "count":
        return n
    m = float(np.sum(keep & np.isin(kind, params["other_kinds"])))
    return n / (n + m) if n + m else float("nan")


def c_cache_hits(runs: list[Path], params: dict[str, Any], root: Path) -> float:
    """LLM response-cache hits (``cache_hit`` true) at registered period ``period``, summed."""
    tot = 0
    for r in runs:
        if not has_table(r, "llm_decisions"):
            continue
        d = load(r, "llm_decisions", ["period", "cache_hit"])
        hit = np.array([bool(v) if v is not None else False for v in d["cache_hit"]])
        tot += int(np.sum(hit & (d["period"].astype(int) == int(params.get("period", 1)))))
    return float(tot)


def c_judge_agreement(runs: list[Path], params: dict[str, Any], root: Path) -> float:
    """Judge validation re-derived from the files: the labels of the filled sheet (labeller
    ``human:*`` only), the sealed key (item -> run_id, decision_id) and the cached D9 scores;
    quadratic-weighted kappa on the rounded scores, Spearman's rho, or n."""
    import csv

    def rows(path: str) -> list[dict[str, str]]:
        with open(root / path, encoding="utf-8-sig", newline="") as fh:
            return list(csv.DictReader(fh))

    key = {r["item_id"]: r for r in rows(params["key"])}
    sc = {
        (r["run_id"], int(float(r["decision_id"]))): float(r["score"])
        for r in rows(params["scores"])
    }
    h, j = [], []
    for r in rows(params["sheet"]):
        lev = (r.get("level") or "").strip()
        if not lev.isdigit() or not (r.get("labeller") or "").startswith("human:"):
            continue
        k = key.get((r.get("item_id") or "").strip())
        if k is None:
            continue
        s = sc.get((k["run_id"], int(float(k["decision_id"]))))
        if s is None or not math.isfinite(s):
            continue
        h.append(int(lev))
        j.append(int(np.rint(s)))
    stat = params.get("stat", "kappa")
    if stat == "n":
        return float(len(h))
    a, b = np.asarray(j), np.asarray(h)
    if stat == "spearman":
        return spearman(a.astype(float), b.astype(float))
    lo, hi = int(min(a.min(), b.min())), int(max(a.max(), b.max()))
    k = hi - lo + 1
    obs = np.zeros((k, k))
    np.add.at(obs, (a - lo, b - lo), 1.0)
    exp = np.outer(obs.sum(1), obs.sum(0)) / obs.sum()
    ii, jj = np.indices((k, k))
    w = (ii - jj) ** 2 / max((k - 1) ** 2, 1)
    den = float((w * exp).sum())
    return 1.0 - float((w * obs).sum()) / den if den > 0 else 1.0


def c_logit_closed_form(runs: list[Path], params: dict[str, Any], root: Path) -> float:
    """A benchmark of a market of the pinned benchmark file re-solved from its model parameters
    with this module's own logit solver (the file's stored benchmark values are not read)."""
    doc = json.loads((root / params["market_file"]).read_text(encoding="utf-8"))
    m = doc["markets"][params["market"]]["model"]
    n = int(m.get("n_incumbents") or 0) + int(m.get("n_entrants") or 0)
    prm = {k: v for k, v in (m.get("params") or {}).items() if v is not None}
    return float(logit_bench(n, **{k: float(v) for k, v in prm.items()})[params["key"]])


def c_cache_reuse(runs: list[Path], params: dict[str, Any], root: Path) -> float:
    """Post-branch LLM decisions of the matched runs (registered periods ``window``, slots
    ``members``) served from a cache entry first written by the same-trunk reference run: the
    decision is a cache hit and the reference run logged a live call (no cache hit) at the same
    period and slot with the same prompt hash and the same raw output. ``stat``: share (pooled
    over trunks), n_from_ref, n (decisions) or share_hits (any cache hit)."""
    w = params.get("window", [101, 150])
    members = {int(s) for s in params.get("members", [0, 1, 2, 3])}
    cols = ["period", "slot", "prompt_hash", "cache_hit", "raw_output"]
    ref: dict[str, Path] = {}
    for rr in match_runs(root, params["reference_runs"], params, "reference_"):
        if trunk_of(rr) in ref:
            raise ValueError(f"cache_reuse: two reference runs on trunk {trunk_of(rr)}")
        ref[trunk_of(rr)] = rr

    def rows(run: Path) -> list[tuple[int, int, str, bool, str]]:
        d = load(run, "llm_decisions", cols)
        out = []
        for p, s, h, c, o in zip(*(d[c] for c in cols), strict=True):
            if p is None or s is None or not (w[0] <= int(p) <= w[1]) or int(s) not in members:
                continue
            out.append((int(p), int(s), str(h), bool(c) if c is not None else False, str(o)))
        return out

    n = hits = from_ref = 0
    for r in runs:
        rr = ref.get(trunk_of(r))
        if rr is None:
            continue
        live = {(p, s, h, o) for p, s, h, c, o in rows(rr) if not c}
        for p, s, h, c, o in rows(r):
            n += 1
            hits += int(c)
            from_ref += int(c and (p, s, h, o) in live)
    stat = params.get("stat", "share")
    if stat not in ("share", "n", "n_from_ref", "share_hits"):
        raise ValueError(f"cache_reuse: unknown stat {stat!r}")
    if stat == "n":
        return float(n)
    if stat == "n_from_ref":
        return float(from_ref)
    if stat == "share_hits":
        return hits / n if n else float("nan")
    return from_ref / n if n else float("nan")


def c_bench_shift(runs: list[Path], params: dict[str, Any], root: Path) -> float:
    """Pinned benchmark shift of a lever's window market against the trunk market
    (``treatment_bench[key] - bench[key]``, ``key`` = p_nash or p_mono)."""
    k = params.get("key", "p_nash")
    return float(params["treatment_bench"][k]) - float(params["bench"][k])


def _incumbent_profit_index(
    P: np.ndarray,
    mk: dict[str, float],
    fringe: float,
    beta: float,
    bench: dict[str, float],
    entrant: np.ndarray | None = None,
) -> float:
    """Mean over periods of the profit index of the incumbents' mean expected profit."""
    Q = np.round(P / mk["tick"]) * mk["tick"] if mk["tick"] > 0 else P
    allp = Q if entrant is None else np.column_stack([Q, entrant])
    x = allp / mk["alpha"]
    e = np.where(
        np.isfinite(x), np.exp((mk["a"] - np.where(np.isfinite(x), x, 0.0)) / mk["mu"]), 0.0
    )
    s = e / (e.sum(axis=1, keepdims=True) + math.exp(mk["a0"] / mk["mu"]))
    q = beta * (1.0 - fringe) * s[:, : Q.shape[1]]
    pi = (Q - mk["c"] * mk["alpha"]) * q
    pibar = np.nanmean(pi, axis=1)
    idx = (pibar - bench["pi_nash"]) / (bench["pi_mono"] - bench["pi_nash"])
    return float(np.nanmean(idx))


def _pinned_market(doc: dict[str, Any], name: str) -> tuple[dict[str, float], float, float]:
    """``(logit params, beta, fringe share)`` of a market of the pinned benchmark file."""
    m = doc["markets"][name]["model"]
    prm = m["params"]
    alpha = float(prm.get("alpha", 1.0))
    tau = prm.get("tau")
    mk = {
        "a": float(prm["a"]),
        "a0": float(prm.get("a0", 0.0)) + float(prm.get("outside_utility", 0.0) or 0.0) / alpha,
        "mu": float(tau) / alpha if tau is not None else float(prm["mu"]),
        "c": float(prm["c"]),
        "alpha": alpha,
        "tick": float(m.get("tick") or 0.0),
    }
    return mk, float(prm.get("beta", 1.0)), float(m.get("fringe_share", 0.0) or 0.0)


def c_mechanical_pi(runs: list[Path], params: dict[str, Any], root: Path) -> float:
    """Mechanical part of the profit-index contrast (D-17 reporting): the sham's incumbent
    quotes held fixed and re-scored in the lever's market (tau, fringe share, tick and the
    entrant's logged quotes taken from the lever run's own manifest and tables); sham scored
    on the trunk benchmarks, lever on ``treatment_bench`` (window scoring) or on the trunk
    benchmarks (``scoring: trunk``). Mean over trunks of sham minus lever."""
    w = params.get("window", [111, 150])
    ent_slot = int(params.get("entrant_slot", 4))
    treat = {
        trunk_of(t): t for t in match_runs(root, params["treatment_runs"], params, "treatment_")
    }
    tb = params["bench"]
    lb = tb if params.get("scoring", "window") == "trunk" else params["treatment_bench"]
    vals = []
    for r in runs:
        lt = treat.get(trunk_of(r))
        if lt is None:
            continue
        ps, P = _prices(r, [0, 1, 2, 3])
        k = (ps >= w[0]) & (ps <= w[1])
        P = P[k]
        if params.get("market_file"):  # the pinned market models (the lever applies at the
            # branch, so a lever run's manifest still carries the trunk market)
            mdoc = json.loads((root / params["market_file"]).read_text(encoding="utf-8"))
            ms, beta_s, _ = _pinned_market(mdoc, params.get("sham_market", "trunk"))
            ml, beta_l, fr_l = _pinned_market(mdoc, params["lever_market"])
        else:
            ms, ml = _logit_market(r), _logit_market(lt)
            m_s = dotted(manifest(r), "resolved_config.market") or {}
            m_l = dotted(manifest(lt), "resolved_config.market") or {}
            beta_s = float((m_s.get("demand") or {}).get("beta", 1.0))
            beta_l = float((m_l.get("demand") or {}).get("beta", 1.0))
            fr_l = float((m_l.get("fringe") or {}).get("share", 0.0) or 0.0)
        ent = None
        pl_, Pl = _prices(lt, [ent_slot])
        if np.isfinite(Pl).any():
            em = np.full(P.shape[0], np.nan)
            want = ps[k]
            idx = np.searchsorted(pl_, want)
            ok = (idx < pl_.size) & (pl_[np.minimum(idx, pl_.size - 1)] == want)
            em[ok] = Pl[idx[ok], 0]
            ent = np.where(np.isfinite(em), em, np.inf)
        s_val = _incumbent_profit_index(P, ms, 0.0, beta_s, tb)
        l_val = _incumbent_profit_index(P, ml, fr_l, beta_l, lb, ent)
        vals.append(s_val - l_val)
    return float(np.mean(vals)) if vals else float("nan")


CROSS_METRICS = {
    "logit_closed_form": c_logit_closed_form,
    "cache_reuse": c_cache_reuse,
    "bench_shift": c_bench_shift,
    "mechanical_pi": c_mechanical_pi,
    "judge_agreement": c_judge_agreement,
    "holm_from_source": c_holm_from_source,
    "corpus_count": c_corpus_count,
    "cache_hits": c_cache_hits,
    "r1_regression": c_r1_regression,
    "entry_br_gap": c_entry_br_gap,
    "probe_response": c_probe_response,
    "regret": c_regret,
    "paired_contrast": c_paired_contrast,
    "holm_family": c_holm_family,
    "split_sample_best": c_split_sample_best,
    "spearman_level": c_spearman_level,
    "welch_one_sided": c_welch_one_sided,
    "trend_slope": c_trend_slope,
    "follow_rate": c_follow_rate,
    "proportion_diff": c_proportion_diff,
    "mqc_pp": c_mqc_pp,
    "detection_auc": c_detection_auc,
    "corpus_feature_check": c_corpus_feature_check,
}


def compute(claim: dict[str, Any], root: Path) -> float:
    metric = claim["metric"]
    params = dict(claim.get("params", {}) or {})
    runs = match_runs(root, claim["runs"], params)
    if not runs and metric not in ("detection_auc",):
        raise FileNotFoundError(f"no run directory matches {claim['runs']!r} with the filters")
    if metric == "n_runs":
        return float(len(runs))
    if metric in CROSS_METRICS:
        return CROSS_METRICS[metric](runs, params, root)
    if metric not in RUN_METRICS:
        raise ValueError(f"unknown metric {metric!r}")
    how = params.get("agg", "mean")
    if how == "pooled":
        if metric != "parse_fail_rate":
            raise ValueError("pooled aggregation is defined for parse_fail_rate only")
        bad = tot = 0
        for r in runs:
            b, n = _decisions(r, params)
            bad, tot = bad + b, tot + n
        return bad / tot if tot else float("nan")
    vals = np.array([RUN_METRICS[metric](r, params) for r in runs], dtype=float)
    strata = [dotted(manifest(r), params["strata"]) if params.get("strata") else None for r in runs]
    return aggregate(
        vals, how, strata if params.get("strata") else None, int(params.get("B", DEFAULT_B))
    )


# --------------------------------------------------------------------------- cli


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Re-derive paper numbers from raw logs.")
    ap.add_argument("numbers", nargs="?", default="paper/numbers.json", type=Path)
    ap.add_argument("--root", default=".", type=Path)
    ap.add_argument("--report", default=None, type=Path)
    args = ap.parse_args(argv)
    claims = json.loads(args.numbers.read_text(encoding="utf-8"))
    status = 0
    rows = []
    for cid, claim in claims.items():
        if cid.startswith("_"):
            continue
        exp = claim.get("expected")
        tol = float(claim.get("tolerance", 1e-6))
        try:
            got = compute(claim, args.root)
            if exp is None:
                verdict = "pending"
            else:
                rel = claim.get("tolerance_rel")
                if rel is not None:  # p-values: tight relative tolerance (A-010)
                    ok = math.isfinite(got) and math.isclose(
                        got,
                        float(exp),
                        rel_tol=float(rel),
                        abs_tol=float(claim.get("tolerance_abs", 1e-12)),
                    )
                else:
                    ok = math.isfinite(got) and abs(got - float(exp)) <= tol
                alpha = claim.get("alpha")
                if alpha is not None and math.isfinite(got):  # the decision must agree too
                    ok = ok and ((got < float(alpha)) == (float(exp) < float(alpha)))
                verdict = "PASS" if ok else "FAIL"
                status = max(status, 0 if ok else 1)
        except (FileNotFoundError, KeyError, ValueError) as e:
            got = float("nan")
            if exp is None:
                verdict = f"pending (not computable: {e})"
            else:
                verdict = f"ERROR ({e})"
                status = 2
        rows.append(
            {
                "claim": cid,
                "paper_claim": claim.get("claim"),
                "metric": claim["metric"],
                "expected": exp,
                "tolerance": tol,
                "computed": got,
                "verdict": verdict,
            }
        )
        exp_s = "n/a" if exp is None else f"{float(exp):.6g}"
        print(f"{verdict:<7} {cid:<32} computed={got:.6g} expected={exp_s} ±{tol:g}")
    if args.report:
        args.report.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    print("verify:", "OK" if status == 0 else f"FAILED (exit {status})")
    return status


if __name__ == "__main__":
    sys.exit(main())
