"""Gate 1 / 1b / 2 evaluation from the campaign results (PREREG section 1, decision D-16).

Gate 1 (positive control, ``PC-H``; ``PC-M`` on the Mistral lane) and Gate 1b (instrument
validity, ``PC-H-dealer``) read the from-scratch duopoly runs and evaluate, on periods
251-300 with each run's own closed-form benchmarks (alpha-scaled):

* the pooled mean ``Delta_p`` has a run-level BCa 95 % CI (resampling stratified by alpha)
  that excludes 0 and a point estimate >= 0.25;
* >= 60 % of the runs have ``Delta_p > 0.10``;
* parse failures < 5 % of the decisions;
* D-16 co-condition: the pooled mean ``Delta_pi`` >= 0.10 (the price index alone could be
  met by dispersed quotes that earn no profit);
* soft check (reported, not a gate): ``Delta_pi`` inside the published band [0.35, 0.95].

Gate 2 (collusion established) reads the ``TR-H`` trunks through the registered arms and
qualifies a trunk when, on periods 71-100 with the pinned trunk benchmarks, ``Delta_p >=
0.30`` AND (D-16) ``Delta_pi >= 0.30``; the same definition is the "established collusion"
rule used by the sting classification. >= 8 of 12 qualified trunks: branch all; 5-7: branch
with reduced power declared; < 5: fallback F1 then F2. The trend rule (OLS slope of the
per-period cross-trunk mean ``Delta_p`` over 81-100 above 0.005 per period extends T0 to
120) is reported for v1.2.

Every per-run row also carries the raw index, its capped value ``min(Delta_p, 1)`` and the
share of incumbent quotes above ``p_M`` (D-16 reporting). Runs whose manifest status is not
``ok``, whose window is empty or whose own failed-decision share exceeds 10 % are listed as
excluded and do not enter the criteria.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from scipy import stats as sps

from finswarm.analysis import tables
from finswarm.analysis.arms import DEFAULT_ARMS_PATH, load_registered_arms, resolve_arm_benchmarks
from finswarm.analysis.campaign import collect_runs, failed_rate, run_summary
from finswarm.analysis.io import discover_runs, read_manifest, read_table, run_info
from finswarm.metrics._frames import window
from finswarm.metrics.benchmarks import Benchmarks, MarketModel
from finswarm.metrics.collusion import (
    delta_price_incumbents,
    delta_profit_incumbents,
    incumbent_matrix,
)

DEFAULT_SEED = 20261010
DEFAULT_B = 10_000
GATE1_WINDOW = (251, 300)
GATE1_RULE = {
    "window": list(GATE1_WINDOW),
    "ci_level": 0.95,
    "point_min": 0.25,
    "share_threshold": 0.10,
    "share_min": 0.60,
    "parse_fail_max": 0.05,
    "delta_pi_min": 0.10,
    "published_band": [0.35, 0.95],
    "max_fail_rate_per_run": 0.10,
}
GATE2_RULE = {
    "window": [71, 100],
    "delta_p_min": 0.30,
    "delta_pi_min": 0.30,
    "branch_all_min": 8,
    "reduced_power_min": 5,
    "trend_window": [81, 100],
    "trend_slope_max": 0.005,
}
DEFAULT_CAMPAIGNS = {"gate1": ["PC-H", "PC-M"], "gate1b": ["PC-H-dealer"], "gate2": "TR-H"}


def lane_of_campaign(spec: Any, campaign: str, default: str | None = "H") -> str | None:
    """Lane of a campaign from the registry: the lane of the arm whose ``config`` file stem or
    ``match.label`` equals the campaign name (an ``-ext`` suffix is stripped first), else the
    lane named by the campaign's suffix (``TR-S`` -> ``S``) when the registry has it, else
    ``default``. So ``TR-S`` reads lane S (trunk arm ``trunk_S``), ``PC-M`` lane M."""
    name = str(campaign)
    for cand in dict.fromkeys((name, re.sub(r"-ext\d*$", "", name))):
        for arm in spec.arms.values():
            stem = Path(arm.config).stem if arm.config else None
            if cand in (stem, arm.match.label):
                return str(arm.lane)
        suffix = cand.rsplit("-", 1)[-1]
        if suffix in spec.lanes:
            return suffix
    return default


def _names(value: Any, default: list[str]) -> list[str]:
    """Campaign names: ``None`` -> the defaults, ``[]``/``""``/``"none"`` -> skip, a string
    -> comma-separated list."""
    if value is None:
        return list(default)
    if isinstance(value, str):
        value = [x.strip() for x in value.split(",") if x.strip()]
    return [str(v) for v in value if str(v).strip().lower() != "none"]


def _finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _share_above(m: np.ndarray, p_mono: float) -> float:
    ok = np.isfinite(m)
    return float(np.mean(m[ok] > p_mono)) if ok.any() else float("nan")


# --------------------------------------------------------------------------- stratified BCa


def stratified_bca(
    values: list[float],
    strata: list[Any],
    B: int = DEFAULT_B,
    level: float = 0.95,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """BCa interval of the pooled mean with resampling stratified by ``strata`` (alpha).

    Each bootstrap replicate resamples every stratum with replacement within itself (the
    alpha design is kept fixed); bias correction and acceleration follow Efron (1987) with
    the jackknife over all runs. Degenerate inputs (n < 2 or constant) return a zero-width
    interval.
    """
    v = np.asarray(values, dtype=float)
    s = np.asarray([str(x) for x in strata])
    keep = np.isfinite(v)
    v, s = v[keep], s[keep]
    n = v.size
    if n == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0, "B": B}
    theta = float(v.mean())
    if n < 2 or np.all(v == v[0]):
        return {"mean": theta, "lo": theta, "hi": theta, "n": n, "B": B, "strata": {}}
    rng = np.random.default_rng(seed)
    groups = [np.where(s == k)[0] for k in sorted(set(s.tolist()))]
    boot = np.zeros(B)
    for g in groups:
        draws = v[g[rng.integers(0, g.size, size=(B, g.size))]]
        boot += draws.sum(axis=1)
    boot /= n
    prop = (np.sum(boot < theta) + 0.5 * np.sum(boot == theta)) / B
    z0 = sps.norm.ppf(np.clip(prop, 1.0 / (2 * B), 1 - 1.0 / (2 * B)))
    jack = np.array([np.delete(v, i).mean() for i in range(n)])
    d = jack.mean() - jack
    den = 6.0 * (np.sum(d**2) ** 1.5)
    a = float(np.sum(d**3) / den) if den > 0 else 0.0
    alpha = (1.0 - level) / 2.0
    zs = sps.norm.ppf([alpha, 1.0 - alpha])
    adj = sps.norm.cdf(z0 + (z0 + zs) / (1.0 - a * (z0 + zs)))
    lo, hi = np.quantile(boot, adj)
    return {
        "mean": theta,
        "lo": float(lo),
        "hi": float(hi),
        "n": int(n),
        "B": int(B),
        "level": level,
        "strata": {k: int(np.sum(s == k)) for k in sorted(set(s.tolist()))},
    }


# --------------------------------------------------------------------------- gate 1 / 1b


def gate1_run_row(run_dir: str | Path, window_: tuple[int, int] = GATE1_WINDOW) -> dict[str, Any]:
    """Per-run indices of a from-scratch positive-control run with its own benchmarks."""
    rd = Path(run_dir)
    info = run_info(rd)
    man = read_manifest(rd)
    resolved = man.get("resolved_config") or {}
    s, e = window_
    row: dict[str, Any] = {
        "run_id": info["run_id"],
        "seed": info["seed"],
        "alpha": info["alpha"],
        "lane": info["lane"] or (resolved.get("llm") or {}).get("lane"),
        "arm": info["arm"],
        "label": info["label"],
        "status": man.get("status"),
        "window": [s, e],
        "path": str(rd),
    }
    market = resolved.get("market")
    if not market:
        return row | {"error": "manifest without resolved_config.market"}
    slots = (resolved.get("llm") or {}).get("slots") or []
    n = len(slots) if slots else int(market.get("n_dealers", 2))
    try:
        model = MarketModel.from_config(market, n)
        bench = model.benchmarks()
    except Exception as exc:  # noqa: BLE001 - reported as an exclusion reason
        return row | {"error": f"benchmarks unavailable ({exc})"}
    members = list(range(n))
    ap = read_table(rd, "agent_period")
    llm = read_table(rd, "llm_decisions")
    w = window(ap, s, e)
    if w.height == 0:
        return row | {"error": "empty analysis window", "bench": _bench_dict(bench)}
    if "run_id" not in w.columns or w["run_id"].null_count() == w.height:
        w = w.with_columns(pl.lit(info["run_id"]).alias("run_id"))
    row["n_periods"] = int(w["period"].n_unique())
    row["n_decisions"] = int(llm.height)
    row["fail_rate"] = failed_rate(llm)
    row["n_failed"] = int(round(row["fail_rate"] * llm.height)) if llm.height else 0
    try:
        _, m = incumbent_matrix(w, members, s, e)
    except KeyError as exc:
        return row | {"error": f"incumbents missing: {exc}", "bench": _bench_dict(bench)}
    dp = delta_price_incumbents(w, bench, members, s, e)
    row.update(
        bench=_bench_dict(bench),
        members=members,
        delta_p=dp,
        delta_p_capped=min(dp, 1.0) if _finite(dp) else dp,
        delta_pi=delta_profit_incumbents(w, bench, members, s, e),
        mean_price=float(np.nanmean(m)) if np.isfinite(m).any() else float("nan"),
        share_quotes_above_pm=_share_above(m, bench.p_mono),
    )
    return row


def _bench_dict(b: Benchmarks) -> dict[str, float]:
    return {k: float(getattr(b, k)) for k in ("p_nash", "p_mono", "pi_nash", "pi_mono")}


def _split_rows(
    rows: list[dict[str, Any]], max_fail: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept, excluded = [], []
    for r in rows:
        reason = r.get("error")
        if reason is None and r.get("status") not in (None, "ok"):
            reason = f"run status {r['status']!r}"
        if reason is None and not _finite(r.get("delta_p")):
            reason = "Delta_p undefined"
        if reason is None and _finite(r.get("fail_rate")) and r["fail_rate"] > max_fail:
            reason = f"failed decisions {r['fail_rate']:.1%} > {max_fail:.0%}"
        if reason:
            excluded.append({"run_id": r["run_id"], "arm": r.get("arm"), "reason": reason})
        else:
            kept.append(r)
    return kept, excluded


def evaluate_gate1(
    rows: list[dict[str, Any]],
    rule: dict[str, Any] | None = None,
    B: int = DEFAULT_B,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """PASS/FAIL of the Gate 1 criteria on per-run rows (lane H, M or the dealer template)."""
    rule = {**GATE1_RULE, **(rule or {})}
    kept, excluded = _split_rows(rows, rule["max_fail_rate_per_run"])
    out: dict[str, Any] = {
        "rule": rule,
        "n_runs": len(kept),
        "n_excluded": len(excluded),
        "excluded": excluded,
        "runs": kept,
    }
    if not kept:
        return out | {"status": "NOT RUN", "reason": "no analysable run", "criteria": {}}
    dp = [r["delta_p"] for r in kept]
    dpi = [r["delta_pi"] for r in kept]
    ci = stratified_bca(dp, [r.get("alpha") for r in kept], B=B, level=rule["ci_level"], seed=seed)
    share = float(np.mean(np.asarray(dp) > rule["share_threshold"]))
    n_dec = sum(int(r.get("n_decisions") or 0) for r in kept)
    n_fail = sum(int(r.get("n_failed") or 0) for r in kept)
    parse = n_fail / n_dec if n_dec else float("nan")
    mean_pi = float(np.nanmean(dpi)) if np.isfinite(dpi).any() else float("nan")
    lo_b, hi_b = rule["published_band"]
    criteria = {
        "ci_excludes_zero": {"value": [ci["lo"], ci["hi"]], "ok": bool(ci["lo"] > 0.0)},
        "point_ge_min": {
            "value": ci["mean"],
            "min": rule["point_min"],
            "ok": ci["mean"] >= rule["point_min"],
        },
        "share_runs_above": {
            "value": share,
            "threshold": rule["share_threshold"],
            "min": rule["share_min"],
            "ok": share >= rule["share_min"],
        },
        "parse_failures": {
            "value": parse,
            "n_failed": n_fail,
            "n_decisions": n_dec,
            "max": rule["parse_fail_max"],
            "ok": bool(_finite(parse) and parse < rule["parse_fail_max"]),
        },
        "delta_pi_ge_min": {
            "value": mean_pi,
            "min": rule["delta_pi_min"],
            "ok": bool(_finite(mean_pi) and mean_pi >= rule["delta_pi_min"]),
            "source": "D-16 co-condition",
        },
    }
    soft = {
        "delta_pi_in_published_band": {
            "value": mean_pi,
            "band": rule["published_band"],
            "ok": bool(_finite(mean_pi) and lo_b <= mean_pi <= hi_b),
            "note": "reported, not a gate",
        }
    }
    passed = all(c["ok"] for c in criteria.values())
    return out | {
        "status": "PASS" if passed else "FAIL",
        "delta_p": ci,
        "delta_p_capped_mean": float(np.mean([r["delta_p_capped"] for r in kept])),
        "delta_pi_mean": mean_pi,
        "share_quotes_above_pm_mean": float(np.nanmean([r["share_quotes_above_pm"] for r in kept])),
        "criteria": criteria,
        "soft_checks": soft,
        "failed_criteria": [k for k, c in criteria.items() if not c["ok"]],
    }


def gate1(
    campaign_dir: str | Path,
    rule: dict[str, Any] | None = None,
    B: int = DEFAULT_B,
    seed: int = DEFAULT_SEED,
    lane: str | None = None,
) -> dict[str, Any]:
    """Gate 1 / 1b of one positive-control campaign; ``lane`` (from the registry when the
    caller is ``evaluate_gates``) is recorded for the decision line, the criteria are the
    same on every lane."""
    cdir = Path(campaign_dir)
    rows = [
        gate1_run_row(rd, tuple((rule or {}).get("window", GATE1_WINDOW)))
        for rd in discover_runs(cdir)
    ]
    lanes_seen = sorted({str(r["lane"]) for r in rows if r.get("lane")})
    return {
        "campaign": cdir.name,
        "lane": lane or (lanes_seen[0] if len(lanes_seen) == 1 else None),
        **evaluate_gate1(rows, rule, B, seed),
    }


# --------------------------------------------------------------------------- gate 2


def trend_slope(series: dict[str, dict[int, float]], start: int, end: int) -> dict[str, Any]:
    """OLS slope per period of the cross-trunk mean ``Delta_p`` over ``[start, end]``."""
    periods = list(range(start, end + 1))
    means = []
    for p in periods:
        vals = [s[p] for s in series.values() if p in s and _finite(s[p])]
        means.append(float(np.mean(vals)) if vals else float("nan"))
    x = np.asarray(periods, dtype=float)
    y = np.asarray(means, dtype=float)
    ok = np.isfinite(y)
    if ok.sum() < 3:
        return {"slope": float("nan"), "n_periods": int(ok.sum()), "window": [start, end]}
    slope, intercept = np.polyfit(x[ok], y[ok], 1)
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "n_periods": int(ok.sum()),
        "window": [start, end],
        "cross_trunk_mean": dict(zip([str(p) for p in periods], means, strict=True)),
    }


def gate2(
    campaign_dir: str | Path,
    arms_path: str | Path = DEFAULT_ARMS_PATH,
    root: str | Path = ".",
    rule: dict[str, Any] | None = None,
    lane: str | None = None,
) -> dict[str, Any]:
    """Gate 2 on a lane's registered trunk arm with the pinned trunk benchmarks.

    The "established" rule is the registry's ``inference.qualification`` (measure, threshold
    and the D-16 co-measure); the branch counts come from ``inference.gate2`` and the trend
    rule from ``inference.trend_rule`` (``rule`` overrides them). ``lane`` selects the trunk
    arm of ``lanes.<lane>``; when not given it is inferred from the campaign name through the
    registry (``lane_of_campaign``: ``TR-S`` -> S -> ``trunk_S``), falling back to the
    registry's qualification arm (lane H). The trend rule runs on that lane's trunks.
    """
    cdir = Path(campaign_dir)
    spec = load_registered_arms(arms_path)
    q = spec.inference.qualification
    g2 = spec.inference.gate2
    tr = spec.inference.trend_rule
    rule = {
        **GATE2_RULE,
        "branch_all_min": g2.branch_all_min,
        "reduced_power_min": g2.reduced_power_min,
        "trend_window": list(tr.window),
        "trend_slope_max": tr.slope_max,
        "trend_shift": tr.shift,
        **(rule or {}),
    }
    rule["qualification"] = q.model_dump()
    rule["gate2"] = g2.model_dump()
    lane = lane or lane_of_campaign(spec, cdir.name)
    lane_spec = spec.lanes.get(lane) if lane else None
    if lane and lane_spec is None:
        raise ValueError(f"gate2: lane {lane!r} is not in the registry's lanes block")
    trunk_arm = lane_spec.trunk if lane_spec else q.arm
    ab = resolve_arm_benchmarks(spec, root)[trunk_arm]
    s, e = spec.window_of(trunk_arm)
    matched, _ = collect_runs(cdir, spec)
    rows, excluded, series = [], [], {}
    for info in matched:
        if info["registered_arm"] != trunk_arm:
            continue
        r, frame = run_summary(info, spec, ab, n_perm=0)
        reason = r.get("error")
        if reason is None and info.get("status") not in (None, "ok"):
            reason = f"manifest status {info.get('status')!r} (not ok)"
        if reason is None and int(r.get("n_periods") or 0) != e - s + 1:  # B2, SAP s10
            reason = f"incomplete (SAP s10): {r.get('n_periods')} of {e - s + 1} window periods"
        if reason is None and _finite(r["fail_rate"]) and r["fail_rate"] > spec.max_fail_rate:
            reason = f"failed decisions {r['fail_rate']:.1%} > {spec.max_fail_rate:.0%}"
        if reason or frame is None:
            excluded.append(
                {"run_id": r["run_id"], "trunk_id": r["trunk_id"], "reason": reason or "no frame"}
            )
            continue
        dp, dpi = r["delta_p"], r["delta_pi"]
        est = q.passes(r)
        periods, m = incumbent_matrix(frame, r["members"], s, e)
        pbar = np.where(np.isfinite(m), m, 0.0).sum(axis=1) / np.maximum(
            np.isfinite(m).sum(axis=1), 1
        )
        per = np.asarray(ab.primary.delta_price(pbar), dtype=float)
        series[r["trunk_id"]] = {int(p): float(v) for p, v in zip(periods, per, strict=True)}
        rows.append(
            {
                "run_id": r["run_id"],
                "trunk_id": r["trunk_id"],
                "seed": r.get("seed"),
                "delta_p": dp,
                "delta_p_capped": min(dp, 1.0) if _finite(dp) else dp,
                "delta_pi": dpi,
                "share_quotes_above_pm": _share_above(m, ab.primary.p_mono),
                "mean_price": r.get("mean_price"),
                "established": est,
                "fail_rate": r["fail_rate"],
            }
        )
    rows.sort(
        key=lambda r: (r["seed"] is None, r["seed"] if r["seed"] is not None else 0, r["trunk_id"])
    )
    qualified = [r["trunk_id"] for r in rows if r["established"]]
    n_q = len(qualified)
    if not rows:
        decision, status, outcome = "no analysable trunk", "NOT RUN", None
    else:
        outcome = g2.outcome(n_q)
        decision, status = outcome["decision"], outcome["status"]
    tw = rule["trend_window"]
    trend = trend_slope(series, int(tw[0]), int(tw[1]))
    trend["extend_t0_to_120"] = bool(
        _finite(trend["slope"]) and trend["slope"] > rule["trend_slope_max"]
    )
    trend["slope_max"] = rule["trend_slope_max"]
    trend["shift"] = rule["trend_shift"]
    outcome_if_no_extension = outcome
    if rows and trend["extend_t0_to_120"]:
        # D-22: the trend rule is evaluated on every trunk BEFORE qualification and overrides
        # the outcome: nothing branches at 100, the trunks run to T0 + shift first
        sh = int(rule["trend_shift"])
        status = "EXTEND"
        decision = (
            f"do not branch at {e}; extend the trunks to T0 = {e + sh} and re-evaluate the "
            f"qualification on {s + sh}-{e + sh} (trend rule fired: slope "
            f"{trend['slope']:.4f} > {rule['trend_slope_max']} per period; every window "
            f"shifts by +{sh}, recorded in v1.2)"
        )
        outcome = None
    return {
        "campaign": cdir.name,
        "arm": trunk_arm,
        "lane": lane or "H",
        "window": [s, e],
        "outcome": outcome,
        "outcome_if_no_extension": outcome_if_no_extension,
        "benchmarks": {"primary": _bench_dict(ab.primary), "file": ab.path, "sha256": ab.pin_sha},
        "rule": rule,
        "status": status,
        "n_trunks": len(rows),
        "n_qualified": n_q,
        "qualified": qualified,
        "not_qualified": [r["trunk_id"] for r in rows if not r["established"]],
        "decision": decision,
        "established_definition": f"{q.describe((s, e))} on the pinned trunk benchmarks (D-16)",
        "trend_rule": trend,
        "runs": rows,
        "excluded": excluded,
    }


# --------------------------------------------------------------------------- AC-H anchoring


def _llm_slots(resolved: dict[str, Any]) -> list[int]:
    """Indices of the LLM slots of a resolved config (scripted slots have a policy or the
    mock family); ``[0]`` when the config carries no slot list."""
    slots = (resolved.get("llm") or {}).get("slots") or []
    llm = [
        i
        for i, sc in enumerate(slots)
        if not sc.get("policy") and str(sc.get("family", "")).lower() != "mock"
    ]
    return llm or [0]


def anchoring(
    campaign_dir: str | Path,
    arms_path: str | Path = DEFAULT_ARMS_PATH,
    root: str | Path = ".",
    window_: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """AC-H anchoring diagnostic (PREREG section 1, D-16): one LLM dealer among scripted
    ``best_responder`` rivals in the trunk market. If the LLM dealer's median ``Delta_p`` over
    the trunk window across seeds exceeds 1 while its rivals best-respond, the prompt anchors
    above monopoly and every price-index result is reported with the capped index and this
    limitation. A diagnostic, never a gate."""
    cdir = Path(campaign_dir)
    spec = load_registered_arms(arms_path)
    q = spec.inference.qualification
    ab = resolve_arm_benchmarks(spec, root)[q.arm]
    s, e = window_ or spec.window_of(q.arm)
    rows, excluded = [], []
    for rd in discover_runs(cdir):
        info = run_info(rd)
        man = read_manifest(rd)
        members = _llm_slots(man.get("resolved_config") or {})
        ap = read_table(rd, "agent_period")
        w = window(ap, s, e)
        if w.height == 0 or man.get("status") not in (None, "ok"):
            excluded.append({"run_id": info["run_id"], "reason": "empty window or run not ok"})
            continue
        if "run_id" not in w.columns or w["run_id"].null_count() == w.height:
            w = w.with_columns(pl.lit(info["run_id"]).alias("run_id"))
        try:
            _, m = incumbent_matrix(w, members, s, e)
            dp = delta_price_incumbents(w, ab.primary, members, s, e)
        except KeyError as exc:
            excluded.append({"run_id": info["run_id"], "reason": f"LLM slot missing: {exc}"})
            continue
        rows.append(
            {
                "run_id": info["run_id"],
                "seed": info["seed"],
                "llm_slots": members,
                "delta_p": dp,
                "delta_p_capped": min(dp, 1.0) if _finite(dp) else dp,
                "delta_pi": delta_profit_incumbents(w, ab.primary, members, s, e),
                "share_quotes_above_pm": _share_above(m, ab.primary.p_mono),
                "mean_price": float(np.nanmean(m)) if np.isfinite(m).any() else float("nan"),
            }
        )
    vals = [r["delta_p"] for r in rows if _finite(r["delta_p"])]
    med = float(np.median(vals)) if vals else float("nan")
    return {
        "campaign": cdir.name,
        "status": "DIAGNOSTIC",
        "window": [s, e],
        "benchmarks": {"primary": _bench_dict(ab.primary), "file": ab.path},
        "n_runs": len(rows),
        "median_delta_p": med,
        "anchors_above_monopoly": bool(_finite(med) and med > 1.0),
        "reading": (
            "prompt anchors above monopoly: report every price index capped and state the "
            "limitation"
            if _finite(med) and med > 1.0
            else "no anchoring above monopoly detected"
            if _finite(med)
            else "no analysable run"
        ),
        "runs": rows,
        "excluded": excluded,
    }


# --------------------------------------------------------------------------- driver


def evaluate_gates(
    campaigns_root: str | Path = "results/campaigns",
    gate1_campaigns: list[str] | str | None = None,
    gate1b_campaigns: list[str] | str | None = None,
    gate2_campaign: list[str] | str | None = "TR-H",
    anchoring_campaign: str | None = "AC-H",
    arms_path: str | Path = DEFAULT_ARMS_PATH,
    root: str | Path = ".",
    B: int = DEFAULT_B,
    seed: int = DEFAULT_SEED,
    lanes: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Every gate whose campaign directory exists; missing campaigns are ``NOT RUN``. ``None``
    means the default campaigns, an empty list (or ``"none"``) skips the gate, a string is a
    comma-separated list (Gate 2 accepts several trunk campaigns, one per lane). The lane of
    each campaign comes from the registry (``lane_of_campaign``) unless ``lanes`` maps the
    campaign to an explicit lane; one DECISION line is produced per (gate, campaign, lane).
    The AC-H anchoring diagnostic is reported but never counts in ``all_evaluated_pass``."""
    croot = Path(campaigns_root)
    spec = load_registered_arms(arms_path)
    lanes = dict(lanes or {})

    def lane_of(name: str) -> str | None:
        return lanes.get(name) or lane_of_campaign(spec, name)

    report: dict[str, Any] = {"campaigns_root": str(croot), "gates": {}}
    for key, names in (
        ("gate1", _names(gate1_campaigns, DEFAULT_CAMPAIGNS["gate1"])),
        ("gate1b", _names(gate1b_campaigns, DEFAULT_CAMPAIGNS["gate1b"])),
    ):
        for name in names:
            d = croot / name
            label = f"Gate 1{'b' if key == 'gate1b' else ''} ({name})"
            if not d.is_dir():
                report["gates"][label] = {
                    "campaign": name,
                    "lane": lane_of(name),
                    "status": "NOT RUN",
                    "reason": f"{d} missing",
                }
                continue
            report["gates"][label] = gate1(d, B=B, seed=seed, lane=lane_of(name))
    for name in _names(gate2_campaign, [DEFAULT_CAMPAIGNS["gate2"]]):
        d = croot / name
        label = f"Gate 2 ({name})"
        if not d.is_dir():
            report["gates"][label] = {
                "campaign": name,
                "lane": lane_of(name),
                "status": "NOT RUN",
                "reason": f"{d} missing",
            }
        else:
            report["gates"][label] = gate2(d, arms_path, root, lane=lane_of(name))
    for name in _names(anchoring_campaign, []):
        d = croot / name
        label = f"AC-H anchoring diagnostic ({name})"
        if not d.is_dir():
            report["gates"][label] = {
                "campaign": name,
                "lane": lane_of(name),
                "status": "NOT RUN",
                "reason": f"{d} missing",
            }
        else:
            report["gates"][label] = {"lane": lane_of(name), **anchoring(d, arms_path, root)}
    statuses = [g["status"] for g in report["gates"].values()]
    report["all_evaluated_pass"] = all(
        st == "PASS" for st in statuses if st in ("PASS", "FAIL", "EXTEND")
    )
    # exit code for ops: 0 every evaluated gate PASS, 2 a trend-rule EXTEND, 1 any FAIL
    report["exit_code"] = 2 if "EXTEND" in statuses else (1 if "FAIL" in statuses else 0)
    report["decisions"] = {label: decision_line(label, g) for label, g in report["gates"].items()}
    return report


def decision_line(label: str, g: dict[str, Any]) -> str:
    """The one line per (gate, campaign, lane) that ops acts on."""
    st = g["status"]
    head = f"DECISION {label}" + (f" [lane {g['lane']}]" if g.get("lane") else "")
    if st == "NOT RUN":
        return f"{head}: NOT RUN ({g.get('reason', '')})"
    if st == "DIAGNOSTIC":
        return f"{head}: DIAGNOSTIC, {g['reading']}"
    if st == "EXTEND":
        return f"{head}: EXTEND, {g['decision']}"
    if "criteria" in g:
        failed = ", ".join(g["failed_criteria"]) or "none"
        return f"{head}: {st} (failed criteria: {failed})"
    return f"{head}: {st}, {g['n_qualified']}/{g['n_trunks']} established, {g['decision']}"


def _clean(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, list | tuple):
        return [_clean(v) for v in o]
    if isinstance(o, np.generic):
        o = o.item()
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def report_md(report: dict[str, Any]) -> str:
    lines = ["# Gate evaluation", ""]
    for line in report.get("decisions", {}).values():
        lines.append(f"- {line}")
    lines.append("")
    for label, g in report["gates"].items():
        lines.append(f"## {label}: **{g['status']}**")
        lines.append("")
        if g["status"] == "NOT RUN":
            lines += [g.get("reason", ""), ""]
            continue
        if g["status"] == "EXTEND":
            lines += [f"**{g['decision']}**", ""]
        if g["status"] == "DIAGNOSTIC":
            lines += [
                f"Median Delta_p of the LLM dealer over {g['window'][0]}-{g['window'][1]} across "
                f"{g['n_runs']} runs: {tables.fmt(g['median_delta_p'], 3)}. {g['reading']}.",
                "",
            ]
            continue
        if "criteria" in g:
            rows = []
            for k, c in g["criteria"].items():
                v = c["value"]
                shown = (
                    tables.fmt_ci(g["delta_p"]["mean"], *v)
                    if isinstance(v, list)
                    else tables.fmt(v, 3)
                )
                rows.append({"criterion": k, "value": shown, "ok": "yes" if c["ok"] else "NO"})
            for k, c in g.get("soft_checks", {}).items():
                rows.append(
                    {
                        "criterion": f"{k} (soft)",
                        "value": tables.fmt(c["value"], 3),
                        "ok": "yes" if c["ok"] else "no",
                    }
                )
            lines.append(tables.markdown_table(rows, ["criterion", "value", "ok"]))
            lines.append(
                f"{g['n_runs']} runs analysed, {g['n_excluded']} excluded; capped Delta_p mean "
                f"{tables.fmt(g['delta_p_capped_mean'], 3)}; share of quotes above p_M "
                f"{tables.fmt(g['share_quotes_above_pm_mean'], 3)}."
            )
        else:
            lines.append(
                f"{g['n_qualified']} of {g['n_trunks']} trunks established "
                f"({g['established_definition']}): {g['decision']}."
            )
            t = g["trend_rule"]
            lines.append(
                f"Trend rule: slope {tables.fmt(t.get('slope'), 4)} per period over "
                f"{t['window'][0]}-{t['window'][1]} (max {t['slope_max']}): "
                f"{'extend T0 to 120 (v1.2)' if t.get('extend_t0_to_120') else 'T0 stays 100'}."
            )
            lines.append("")
            lines.append(
                tables.markdown_table(
                    [
                        {
                            "trunk": r["trunk_id"],
                            "seed": r["seed"],
                            "Delta_p": tables.fmt(r["delta_p"], 3),
                            "Delta_pi": tables.fmt(r["delta_pi"], 3),
                            "share > p_M": tables.fmt(r["share_quotes_above_pm"], 2),
                            "established": "yes" if r["established"] else "no",
                        }
                        for r in g["runs"]
                    ]
                )
            )
        if g.get("excluded"):
            lines += ["Excluded:", "", tables.markdown_table(g["excluded"])]
        lines.append("")
    return "\n".join(lines)


def write_report(report: dict[str, Any], out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "gates.json").write_text(json.dumps(_clean(report), indent=2), encoding="utf-8")
    (out / "gates.md").write_text(report_md(_clean(report)), encoding="utf-8")
    return out


def print_report(report: dict[str, Any]) -> None:
    for label, g in report["gates"].items():
        line = f"{g['status']:<7} {label}"
        if g["status"] == "NOT RUN":
            print(f"{line}: {g.get('reason', '')}")
        elif g["status"] == "DIAGNOSTIC":
            print(f"{line}: median Delta_p {g['median_delta_p']:.3f} -> {g['reading']}")
        elif g["status"] == "EXTEND":
            print(
                f"{line}: {g['n_qualified']}/{g['n_trunks']} established at {g['window'][1]}; "
                f"trend slope {g['trend_rule']['slope']:.4f} -> {g['decision']}"
            )
        elif "criteria" in g:
            ci = g["delta_p"]
            print(
                f"{line}: Delta_p {ci['mean']:.3f} [{ci['lo']:.3f}, {ci['hi']:.3f}] n={ci['n']}; "
                f"share>0.10 {g['criteria']['share_runs_above']['value']:.2f}; parse fail "
                f"{g['criteria']['parse_failures']['value']:.3%}; Delta_pi {g['delta_pi_mean']:.3f}"
                + (f"; failed: {', '.join(g['failed_criteria'])}" if g["failed_criteria"] else "")
            )
        else:
            print(f"{line}: {g['n_qualified']}/{g['n_trunks']} established -> {g['decision']}")
    print("")
    for line in report.get("decisions", {}).values():
        print(line)
