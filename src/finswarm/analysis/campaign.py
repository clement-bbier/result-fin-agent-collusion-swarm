"""Pre-registered analysis of one campaign directory (used by ``scripts/analyze.py``).

Revised SAP, driven by ``configs/analysis/registered_arms.yaml``:

* only runs matching a REGISTERED arm are read (top-level run directories; ghost sub-runs
  only for the ``ghost`` arm, never pooled with their parent); everything else is listed in
  ``ignored_runs``;
* estimands are computed over the arm's incumbent slot whitelist only. Primary: price-based
  ``Delta_p`` with the pre-lever (trunk) benchmarks; secondary: ``Delta_pi`` with the arm's
  per-window benchmarks (plus ``Delta_pi`` on trunk benchmarks, ``Delta_p`` on window
  benchmarks and the quantity-weighted ``Delta_p``); benchmarks come from the pinned
  ``benchmarks.json`` (SHA-256 verified, stale markets refused);
* paired contrasts ``Delta C = Delta(sham) - Delta(arm)`` on the qualified trunks (Gate 2),
  one-sided (reduction), BCa 95 % CI, Wilcoxon, TOST; each lever's Delta C is split into its
  frozen-quote mechanical part and the behavioural remainder;
* H1 (D-3'): the single pre-specified ``primary_lever``, one-sided paired t-test at
  alpha = 0.05 with no multiplicity (Wilcoxon as sensitivity); supported iff p < alpha and
  MQC <= 5 pp, practically significant iff also the one-sided 95 % lower bound > 0.10;
  TOST +/- 0.20 when not supported;
* H1-R: the 8 levers, Holm over m = 8 on the full qualified sample (missing levers at
  p = 1), BCa CIs, and a DESCRIPTIVE split-sample estimate of the empirical best lever
  (odd-seed trunks select, even-seed trunks estimate); ``sigma_d`` is the pooled SD of the
  realised lever-minus-sham differences.

Exclusions (SAP): failed decisions above ``max_fail_rate`` among the run's own decisions,
an empty window, or a whitelisted incumbent missing from the window.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from finswarm.analysis import stats, tables
from finswarm.analysis.arms import (
    DEFAULT_ARMS_PATH,
    ArmBenchmarks,
    PinError,
    RegisteredArms,
    file_sha256,
    load_pinned,
    load_registered_arms,
    resolve_arm_benchmarks,
)
from finswarm.analysis.io import discover_ghosts, discover_runs, read_manifest, read_table, run_info
from finswarm.analysis.mqc import analyze_mqc, load_mqc, mqc_point_estimates
from finswarm.metrics._frames import pivot_matrix, window
from finswarm.metrics.collusion import (
    MECHANICAL_MEASURES,
    capped,
    d7_price,
    delta_price_incumbents,
    delta_profit_incumbents,
    incumbent_matrix,
    mechanical_delta_c,
    undercut_rate,
)

PRIMARY = "delta_p"  # capped min(Delta_p, 1), trunk scoring (D-16)
OWN = "delta_p_own"  # D-17 own scoring: sham on the trunk benchmarks, lever on its own market's
MEASURES: tuple[str, ...] = MECHANICAL_MEASURES  # delta_p first (primary)
SIDEDNESS = (
    "one-sided (alternative='greater'): H1 is Delta(sham) - Delta(arm) > 0, a reduction of "
    "collusion; BCa intervals are two-sided 95 %; TOST is two one-sided tests by construction"
)
MEI = 0.25
MQC_UPPER_MAX_PP = 7.0  # best-lever eligibility: MQC 95 % upper bound <= 7 pp (PREREG s2)
OWN_INFORMATIVE_MIN = 0.005  # A-003: own and trunk scorings closer than this are one yardstick
L2_ASK_FALL_FACTOR = 2.0  # A-003: the LLM ask fall must exceed twice the SC1 ask fall


def failed_rate(llm_decisions: pl.DataFrame) -> float:
    """Share of LLM decisions that failed or were flagged (PREREG section 2): parse failure,
    provider error or refusal (default action flagged). NaN without decisions."""
    if llm_decisions.height == 0:
        return float("nan")
    bad = (
        (~pl.col("parse_ok").fill_null(False))
        | (pl.col("error").is_not_null() & (pl.col("error") != ""))
        | pl.col("refusal").fill_null(False)
    )
    return float(llm_decisions.select(bad.mean()).item())


def post_branch_cache_hits(llm_decisions: pl.DataFrame, branch_period: int | None) -> int:
    """Number of post-branch LLM calls served from the response cache (fresh-draw rule)."""
    if llm_decisions.height == 0 or "cache_hit" not in llm_decisions.columns:
        return 0
    d = llm_decisions
    if branch_period is not None:
        d = d.filter(pl.col("period") > int(branch_period))
    return int(d.select(pl.col("cache_hit").fill_null(False).sum()).item())


# --------------------------------------------------------------------------- matching


def _dotted(d: dict[str, Any], path: str) -> Any:
    cur: Any = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def _equal(a: Any, b: Any) -> bool:
    if isinstance(a, int | float) and isinstance(b, int | float):
        return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-12)
    return a == b


def match_arm(spec: RegisteredArms, info: dict[str, Any], manifest: dict[str, Any]) -> str | None:
    """Registered arm of a run, ``None`` when unregistered; ambiguity is an error."""
    is_ghost = bool(info.get("is_ghost") or info.get("ghost_of"))
    resolved = manifest.get("resolved_config") or {}
    hits = []
    for name, a in spec.arms.items():
        m = a.match
        if m.ghost != is_ghost or m.arm != info["arm"]:
            continue
        if m.label is not None and m.label != info.get("label"):
            continue
        if any(not _equal(_dotted(resolved, p), v) for p, v in m.config.items()):
            continue
        hits.append(name)
    if len(hits) > 1:
        raise ValueError(f"run {info['run_id']} matches several registered arms: {hits}")
    return hits[0] if hits else None


def collect_runs(
    campaign_dir: str | Path, spec: RegisteredArms
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """``(matched, ignored)`` run infos; ghosts only from ``<run>/ghost/<k>/``."""
    matched, ignored = [], []
    want_ghosts = any(a.match.ghost for a in spec.arms.values())
    for rd in discover_runs(campaign_dir):
        cands = [(rd, False)]
        if want_ghosts:
            cands += [(g, True) for g in discover_ghosts(rd)]
        for d, ghost in cands:
            info = run_info(d)
            info["is_ghost"] = ghost or bool(info.get("ghost_of"))
            man = read_manifest(d)
            info["resolved_market"] = (man.get("resolved_config") or {}).get("market")
            info["status"] = man.get("status")
            arm = match_arm(spec, info, man)
            if arm is None:
                ignored.append(
                    {
                        "run_id": info["run_id"],
                        "arm": info["arm"],
                        "label": info["label"],
                        "ghost": info["is_ghost"],
                        "reason": "not a registered arm",
                    }
                )
            else:
                matched.append({**info, "registered_arm": arm})
    return matched, ignored


def market_mismatch(resolved_market: dict[str, Any] | None, ab: ArmBenchmarks) -> list[str]:
    """Differences between a run's resolved market (manifest) and its pinned markets.

    Demand parameters (``a, a0, mu, beta, alpha, cost``) and the number of dealers must equal
    the pre-lever (primary) market; the lever-dependent knobs (routing ``tau`` and outside
    utility, fringe share, tick) must equal either the pre-lever or the arm's window market
    (a branch manifest may record the configuration before or after its own lever). Returns
    an empty list when the manifest carries no resolved config (nothing to check).
    """
    from finswarm.metrics.benchmarks import MarketModel

    if not resolved_market:
        return []
    try:
        m = MarketModel.from_config(resolved_market)
    except Exception as exc:  # noqa: BLE001 - reported as an exclusion reason
        return [f"unreadable resolved market ({exc})"]
    p, w = ab.primary_model, ab.window_model
    out = []
    for k in ("a", "a0", "mu", "beta", "alpha", "c"):
        if not _equal(getattr(m.params, k), getattr(p.params, k)):
            out.append(f"{k}={getattr(m.params, k)} (registered {getattr(p.params, k)})")
    if m.n_incumbents != p.n_incumbents:
        out.append(f"n_dealers={m.n_incumbents} (registered {p.n_incumbents})")
    knobs = {
        "tau": lambda x: x.params.tau_eff,
        "outside_utility": lambda x: x.params.outside_utility,
        "fringe_share": lambda x: x.fringe_share,
        "tick": lambda x: x.tick,
    }
    for k, get in knobs.items():
        v = get(m)
        if not (_equal(v, get(p)) or _equal(v, get(w))):
            out.append(f"{k}={v} (registered {get(p)} or {get(w)})")
    return out


def members_of(spec: RegisteredArms, arm: str, info: dict[str, Any]) -> list[int]:
    mem = spec.arms[arm].members
    if mem == "ghost_slot":
        if info.get("ghost_slot") is None:
            raise KeyError(f"ghost run {info['run_id']} has no ghost_slot tag")
        return [int(info["ghost_slot"])]
    return [int(m) for m in mem]


# --------------------------------------------------------------------------- per run


def run_summary(
    info: dict[str, Any],
    spec: RegisteredArms,
    ab: ArmBenchmarks,
    n_perm: int = 200,
) -> tuple[dict[str, Any], pl.DataFrame | None]:
    """Per-run estimands over the whitelist; returns ``(summary, window frame or None)``."""
    arm = info["registered_arm"]
    s, e = spec.window_of(arm)
    members = members_of(spec, arm, info)
    ap = read_table(info["path"], "agent_period")
    llm = read_table(info["path"], "llm_decisions")
    bp = info.get("branch_period")
    own = llm.filter(pl.col("period") > int(bp)) if bp is not None and llm.height else llm
    out: dict[str, Any] = {
        **{k: v for k, v in info.items() if k != "path"},
        "path": info["path"],
        "role": spec.arms[arm].role,
        "lane": spec.arms[arm].lane,
        "members": members,
        "window": [s, e],
        "fail_rate": failed_rate(own),
        "post_branch_cache_hits": post_branch_cache_hits(llm, bp),
    }
    w = window(ap, s, e)
    if "run_id" not in w.columns or w["run_id"].null_count() == w.height:
        w = w.with_columns(pl.lit(info["run_id"]).alias("run_id"))
    slots = sorted(int(x) for x in w["slot"].drop_nulls().unique().to_list()) if w.height else []
    out["n_periods"] = int(w["period"].n_unique()) if w.height else 0
    out["slots_seen"] = slots
    out["slots_excluded"] = [x for x in slots if x not in members]
    out["config_checked"] = bool(info.get("resolved_market"))
    out.pop("resolved_market", None)
    mism = market_mismatch(info.get("resolved_market"), ab)
    if mism:
        return out | {
            "error": "config mismatch with the registered market: " + "; ".join(mism)
        }, None
    if w.height == 0:
        return out | {"error": "empty analysis window"}, None
    try:
        raw = delta_price_incumbents(w, ab.primary, members, s, e)
    except KeyError as exc:
        return out | {"error": f"whitelisted incumbents missing: {exc}"}, None
    raw_own = delta_price_incumbents(w, ab.window, members, s, e)
    raw_vw = delta_price_incumbents(w, ab.primary, members, s, e, weight="volume")
    # D-16: confirmatory contrasts use the capped index; the raw index is reported alongside
    out["delta_p_raw"], out[PRIMARY] = raw, capped(raw)
    out["delta_p_window_raw"], out["delta_p_window"] = raw_own, capped(raw_own)
    out["delta_p_vw_raw"], out["delta_p_vw"] = raw_vw, capped(raw_vw)
    out["delta_pi"] = delta_profit_incumbents(w, ab.primary, members, s, e)
    out["delta_pi_window"] = delta_profit_incumbents(w, ab.window, members, s, e)
    _, m = incumbent_matrix(w, members, s, e)
    _, q = incumbent_matrix(w, members, s, e, value="quantity", strict=False)
    out["mean_price"] = float(np.nanmean(m)) if np.isfinite(m).any() else float("nan")
    out["delta_p_sens"] = _sensitivity_indices(ap, ab, members, spec)
    out["trade_weighted_price"] = _trade_weighted(m, q)
    out["share_quotes_above_pm"] = _share_above(m, ab.primary.p_mono)
    out["share_quotes_above_pm_own"] = _share_above(m, ab.window.p_mono)
    out["undercut_rate"] = undercut_rate(_members_frame(w, members), key="slot")
    d7 = d7_price(
        w,
        ab.window_model.best_response(),
        ab.window,
        members,
        s,
        e,
        d7_cartel=_d7c(ab),
        n_perm=n_perm,
    )
    out["d7"] = d7.to_dict()
    return out, w


def _sensitivity_indices(
    ap: pl.DataFrame, ab: ArmBenchmarks, members: list[int], spec: RegisteredArms
) -> dict[str, float | None]:
    """Capped Delta_p (trunk scoring) over the registered sensitivity windows of the branch
    (``sens_late`` 121-150 and ``sens_full`` 101-150, shifted with the trend rule; m12), on
    the periods the run holds; None when the run has no period in the window."""
    out: dict[str, float | None] = {}
    for key in ("sens_late", "sens_full"):
        win = spec.windows.get(key)
        if win is None:
            continue
        a, b = int(win[0]), int(win[1])
        sub = window(ap, a, b)
        if sub.height == 0:
            out[f"{a}-{b}"] = None
            continue
        try:
            out[f"{a}-{b}"] = capped(delta_price_incumbents(sub, ab.primary, members, a, b))
        except KeyError:
            out[f"{a}-{b}"] = None
    return out


def shift_windows(
    spec: RegisteredArms, gates_json: str | Path | None = None, window_shift: int = 0
) -> tuple[RegisteredArms, dict[str, Any]]:
    """Registry windows shifted by the trend rule (m12). ``gates_json`` is the Gate report of
    ``--gates``: when its Gate 2 says EXTEND, ``window_shift`` must equal the recorded shift
    (else ValueError: the analysis must never read unshifted windows after an extension).
    The trunk and branch windows move by the shift; the registered sensitivity windows
    121-150 (``sens_late``) and 101-150 (``sens_full``) are added, shifted alike."""
    required = 0
    status = None
    if gates_json is not None and Path(gates_json).exists():
        rep = json.loads(Path(gates_json).read_text(encoding="utf-8"))
        g2 = (
            next((g for k, g in (rep.get("gates") or {}).items() if k.startswith("Gate 2")), None)
            or (rep.get("gates") or {}).get("gate2")
            or {}
        )
        status = g2.get("status")
        if status == "EXTEND":
            required = int((g2.get("trend_rule") or {}).get("shift") or 0)
    if required and int(window_shift) != required:
        raise ValueError(
            f"{gates_json}: Gate 2 says EXTEND (+{required}) but the analysis windows are "
            f"shifted by {window_shift}; pass window_shift={required} (m12)"
        )
    sh = int(window_shift)
    wins: dict[str, Any] = {}
    for k, v in spec.windows.items():
        a, b = int(v[0]), int(v[1])
        wins[k] = (a + sh, b + sh) if k in ("trunk", "branch") else (a, b)
    wins["sens_late"] = (121 + sh, 150 + sh)
    wins["sens_full"] = (101 + sh, 150 + sh)
    info = {
        "shift": sh,
        "gates_json": str(gates_json) if gates_json is not None else None,
        "gate2_status": status,
        "required_shift": required,
        "windows": {k: list(v) for k, v in wins.items()},
    }
    return spec.model_copy(update={"windows": wins}), info


def _sign_counts(d: list[float], tol: float = 1e-12) -> dict[str, int]:
    """Trunks whose paired difference is zero (|d| <= tol), negative or positive (A-022)."""
    x = np.asarray(d, dtype=float)
    x = x[np.isfinite(x)]
    return {
        "zero": int(np.sum(np.abs(x) <= tol)),
        "negative": int(np.sum(x < -tol)),
        "positive": int(np.sum(x > tol)),
    }


def _d7c(ab: ArmBenchmarks) -> float:
    from finswarm.metrics.benchmarks import d7_cartel

    return float(d7_cartel(ab.window_model))


def _members_frame(w: pl.DataFrame, members: list[int]) -> pl.DataFrame:
    return w.filter(pl.col("slot").is_in(members))


def _share_above(m: np.ndarray, p_mono: float) -> float:
    """Share of the incumbents' posted quotes strictly above ``p_mono`` (D-16 reporting)."""
    ok = np.isfinite(m)
    return float(np.mean(m[ok] > p_mono)) if ok.any() else float("nan")


def _trade_weighted(m: np.ndarray, q: np.ndarray) -> float:
    """Quantity-weighted mean incumbent price over the window (D-17 reporting)."""
    if q.shape != m.shape:
        return float("nan")
    ok = np.isfinite(m) & np.isfinite(q) & (q > 0)
    den = float(q[ok].sum())
    return float((m[ok] * q[ok]).sum() / den) if den > 0 else float("nan")


def _entrant_matrix(frame: pl.DataFrame, members: list[int], periods: np.ndarray) -> np.ndarray:
    """Prices of the non-member sellers (entrants), aligned on ``periods``."""
    other = frame.filter(~pl.col("slot").is_in(members))
    if other.height == 0:
        return np.empty((len(periods), 0))
    idx, _, m = pivot_matrix(other, "price", key="slot")
    out = np.full((len(periods), m.shape[1]), np.nan)
    pos = {int(p): i for i, p in enumerate(idx)}
    for i, p in enumerate(periods):
        if int(p) in pos:
            out[i] = m[pos[int(p)]]
    return np.where(np.isfinite(out) & (out > 0), out, np.nan)


# --------------------------------------------------------------------------- contrasts


def _arm_summary(rows: list[dict[str, Any]], B: int, seed: int) -> dict[str, Any]:
    out: dict[str, Any] = {"n": len(rows)}
    for k in MEASURES:
        ci = stats.bca_ci([r.get(k, float("nan")) for r in rows], B=B, seed=seed)
        out[k] = {"mean": ci.estimate, "lo": ci.lo, "hi": ci.hi}
    out["mean"], out["lo"], out["hi"] = (out[PRIMARY][x] for x in ("mean", "lo", "hi"))
    return out


def _by_trunk(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    acc: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        acc.setdefault(r["trunk_id"], []).append(r)
    if any(len(v) > 1 for v in acc.values()):
        dup = sorted(t for t, v in acc.items() if len(v) > 1)
        raise ValueError(f"several runs of the same arm on trunks {dup}; resolve before analysis")
    return {t: v[0] for t, v in acc.items()}


def _mechanical(
    ref_run: dict[str, Any],
    ref_frame: pl.DataFrame,
    arm_frame: pl.DataFrame,
    ab: ArmBenchmarks,
) -> dict[str, float] | str:
    """Mechanical Delta C of one trunk (or the reason it cannot be computed)."""
    s, e = ref_run["window"]
    members = ref_run["members"]
    periods, sham_p = incumbent_matrix(ref_frame, members, s, e)
    model = ab.window_model
    ent = None
    if model.n_entrants:
        em = _entrant_matrix(arm_frame, members, periods)
        if em.shape[1] != model.n_entrants or not np.isfinite(em).any():
            return f"entrant quotes unavailable ({em.shape[1]} columns, model {model.n_entrants})"
        ent = np.where(np.isfinite(em), em, np.inf)  # an absent entrant takes no flow
    dec = mechanical_delta_c(
        sham_p,
        model,
        ab.primary,
        sham_market_params=ab.primary_model,
        lever_benchmarks=ab.window,
        entrant_quotes=ent,
    )
    return dec.mechanical


def qualification(spec: RegisteredArms, runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Gate 2 per trunk and per lane from each lane's trunk arm (D-16 rule, D-25 outcome).

    ``qualified`` is the union over the lanes present in the campaign (trunk ids are unique
    across lanes); ``None`` when no lane has trunk runs (every paired trunk is then used, and
    the summary says so). ``by_lane[lane]["outcome"]`` is the Gate 2 outcome table row.
    """
    q = spec.inference.qualification
    g2 = spec.inference.gate2
    by_lane: dict[str, dict[str, Any]] = {}
    for lane, ls in spec.lanes.items():
        trunk_runs = {r["trunk_id"]: r for r in runs if r["registered_arm"] == ls.trunk}
        if not trunk_runs:
            by_lane[lane] = {"available": False, "trunk_arm": ls.trunk}
            continue
        ok = {t: q.passes(r) for t, r in trunk_runs.items()}
        qualified = sorted(t for t, v in ok.items() if v)
        by_lane[lane] = {
            "available": True,
            "trunk_arm": ls.trunk,
            "qualified": qualified,
            "not_qualified": sorted(t for t, v in ok.items() if not v),
            "values": {t: r.get(q.measure) for t, r in trunk_runs.items()},
            "co_values": {t: r.get(q.co_measure) for t, r in trunk_runs.items()}
            if q.co_measure
            else {},
            "n_qualified": len(qualified),
            "outcome": g2.outcome(len(qualified)),
        }
    avail = [v for v in by_lane.values() if v["available"]]
    base = {
        "rule": q.model_dump(),
        "definition": q.describe(spec.window_of(q.arm)),
        "gate2": g2.model_dump(),
        "by_lane": by_lane,
    }
    if not avail:
        return base | {
            "available": False,
            "qualified": None,
            "note": "no trunk runs of any lane in this campaign: every paired trunk is used",
        }
    merged: dict[str, Any] = {"qualified": [], "not_qualified": [], "values": {}, "co_values": {}}
    for v in avail:
        merged["qualified"] += v["qualified"]
        merged["not_qualified"] += v["not_qualified"]
        merged["values"].update(v["values"])
        merged["co_values"].update(v["co_values"])
    merged["qualified"].sort()
    merged["not_qualified"].sort()
    return base | {"available": True, **merged}


def contrasts_for(
    arm: str,
    spec: RegisteredArms,
    runs: list[dict[str, Any]],
    frames: dict[str, pl.DataFrame],
    abench: dict[str, ArmBenchmarks],
    B: int,
    seed: int,
    qualified: list[str] | None = None,
) -> dict[str, Any] | None:
    ref_name = spec.arms[arm].paired_with
    if ref_name is None:
        return None
    ref = _by_trunk([r for r in runs if r["registered_arm"] == ref_name])
    trt = _by_trunk([r for r in runs if r["registered_arm"] == arm])
    common = sorted(set(ref) & set(trt))
    trunks = common if qualified is None else [t for t in common if t in set(qualified)]
    out: dict[str, Any] = {
        "reference": ref_name,
        "trunks": trunks,
        "n_pairs": len(trunks),
        "unqualified_trunks_dropped": sorted(set(common) - set(trunks)),
    }
    margin = spec.inference.tost_margin
    for k in MEASURES:
        d = [ref[t][k] - trt[t][k] for t in trunks]
        out[k] = stats.one_sample(d, spec.inference.sidedness, B, seed, margin)
        out[k]["diffs"] = dict(zip(trunks, d, strict=True))
        out[k]["signs"] = _sign_counts(d)  # A-022 reporting: zero / negative / positive
    # D-17 own scoring = the delta_p_window contrast: the sham on its own (trunk) market's
    # benchmarks, the lever on the lever market's own pinned benchmarks. A lever that only
    # lowers the competitive price leaves the dealers as far above THEIR new Nash as the
    # sham sits above the trunk's, so d_own stays near 0 ("competitive adjustment"); one
    # yardstick for both arms would cancel p_N and reproduce the trunk contrast.
    out[OWN] = dict(out["delta_p_window"])
    out[OWN]["scoring"] = (
        "d_own = Delta_p,sham(trunk benchmarks) - Delta_p,lever(lever market's own pinned "
        "benchmarks), capped index (D-17 ruling)"
    )
    out["seeds"] = {t: ref[t].get("seed") for t in trunks}
    # A-003 reporting: how far the own scoring departs from the trunk scoring per trunk
    # (a lever whose own benchmarks coincide with the trunk's has an uninformative co-condition)
    gaps = [
        abs(out[OWN]["diffs"][t] - out[PRIMARY]["diffs"][t])
        for t in trunks
        if np.isfinite(out[OWN]["diffs"][t]) and np.isfinite(out[PRIMARY]["diffs"][t])
    ]
    out["own_vs_trunk_max_abs_diff"] = float(max(gaps)) if gaps else None
    out["own_scoring_informative"] = bool(gaps) and float(max(gaps)) >= OWN_INFORMATIVE_MIN
    # the LLM incumbents' mean ask fall (reference minus arm, posted prices) for the A-003 rule
    falls = {
        t: ref[t]["mean_price"] - trt[t]["mean_price"]
        for t in trunks
        if np.isfinite(ref[t].get("mean_price", np.nan))
        and np.isfinite(trt[t].get("mean_price", np.nan))
    }
    out["ask_fall"] = {
        "definition": "mean posted price of the whitelisted incumbents, reference minus arm, "
        "over the analysis window",
        "mean": float(np.mean(list(falls.values()))) if falls else None,
        "per_trunk": falls,
    }
    for key in ("sens_late", "sens_full"):  # m12: registered sensitivity windows
        win = spec.windows.get(key)
        if win is None:
            continue
        name = f"{int(win[0])}-{int(win[1])}"
        d_ = [
            ref[t]["delta_p_sens"][name] - trt[t]["delta_p_sens"][name]
            for t in trunks
            if ref[t].get("delta_p_sens", {}).get(name) is not None
            and trt[t].get("delta_p_sens", {}).get(name) is not None
        ]
        if len(d_) >= 2:
            ci = stats.bca_ci(d_, B=B, seed=seed)
            out[f"delta_p_sens_{key}"] = {
                "window": name,
                "mean": ci.estimate,
                "bca_lo": ci.lo,
                "bca_hi": ci.hi,
                "n": ci.n,
            }
        else:
            out[f"delta_p_sens_{key}"] = {"window": name, "n": len(d_), "mean": None}
    # D-16 / D-17 reporting next to every Delta C: raw-index contrast, share of quotes
    # above p_M, benchmark shift (trunk -> own market), trade-weighted market price
    if trunks and "delta_p_raw" in ref[trunks[0]]:
        raw = [ref[t]["delta_p_raw"] - trt[t]["delta_p_raw"] for t in trunks]
        ci = stats.bca_ci(raw, B=B, seed=seed)
        margin = spec.inference.tost_margin
        tr = stats.tost(raw, -margin, margin, spec.inference.alpha)
        out["delta_p_raw"] = {
            "mean": ci.estimate,
            "bca_lo": ci.lo,
            "bca_hi": ci.hi,
            "n": ci.n,
            "sd": float(np.std(raw, ddof=1)) if len(raw) > 1 else float("nan"),
            # A-022 reporting (no inference change): raw-index TOST and the sign counts
            "tost": {"margin": margin, **tr},
            "diffs": dict(zip(trunks, raw, strict=True)),
        }
        out["delta_p_raw"]["signs"] = _sign_counts(raw)
        out["share_quotes_above_pm"] = {
            "reference": float(np.nanmean([ref[t]["share_quotes_above_pm"] for t in trunks])),
            "arm": float(np.nanmean([trt[t]["share_quotes_above_pm"] for t in trunks])),
        }
        out["trade_weighted_price"] = {
            "reference": float(np.nanmean([ref[t]["trade_weighted_price"] for t in trunks])),
            "arm": float(np.nanmean([trt[t]["trade_weighted_price"] for t in trunks])),
        }
    ab_arm = abench[arm]
    out["benchmark_shift"] = {
        "trunk": {"p_nash": ab_arm.primary.p_nash, "p_mono": ab_arm.primary.p_mono},
        "own": {"p_nash": ab_arm.window.p_nash, "p_mono": ab_arm.window.p_mono},
        "d_p_nash": ab_arm.window.p_nash - ab_arm.primary.p_nash,
        "d_p_mono": ab_arm.window.p_mono - ab_arm.primary.p_mono,
    }
    out["dual_benchmark"] = {
        "p_trunk": out[PRIMARY]["t"]["p"],
        "p_own": out[OWN]["t"]["p"],
        "label": stats.dual_benchmark_label(
            out[PRIMARY]["t"]["p"], out[OWN]["t"]["p"], spec.inference.alpha
        ),
    }
    # frozen-quote mechanical decomposition (levers and any arm with its own window market)
    if spec.arms[arm].role == "primary_lever" and trunks:
        mech_by: dict[str, dict[str, float]] = {}
        notes = {}
        for t in trunks:
            fr, fa = frames.get(ref[t]["run_id"]), frames.get(trt[t]["run_id"])
            if fr is None or fa is None:
                notes[t] = "frame missing"
                continue
            res = _mechanical(ref[t], fr, fa, abench[arm])
            if isinstance(res, str):
                notes[t] = res
            else:
                mech_by[t] = res
        mech, beh = {}, {}
        for k in MEASURES:
            ts = [t for t in trunks if t in mech_by]
            mv = [mech_by[t][k] for t in ts]
            bv = [out[k]["diffs"][t] - mech_by[t][k] for t in ts]
            ci_m = stats.bca_ci(mv, B=B, seed=seed)
            mech[k] = {"mean": ci_m.estimate, "lo": ci_m.lo, "hi": ci_m.hi, "n": ci_m.n}
            beh[k] = stats.one_sample(bv, spec.inference.sidedness, B, seed, margin)
        out["mechanical"] = mech
        out["behavioural"] = beh
        out["mechanical_notes"] = notes
        out["window_market_method"] = abench[arm].window.method
    return out


# --------------------------------------------------------------------------- driver


def analyze_campaign(
    campaign_dir: str | Path,
    arms_path: str | Path = DEFAULT_ARMS_PATH,
    root: str | Path = ".",
    out_root: str | Path = "results/analysis",
    mqc: dict[str, Any] | str | Path | None = None,
    mqc_campaigns: list[str | Path] | None = None,
    B: int | None = None,
    seed: int | None = None,
    allow_stale_benchmarks: bool = False,
    n_perm: int = 200,
    gates_json: str | Path | None = None,
    window_shift: int = 0,
) -> dict[str, Any]:
    """Run the registered analysis; writes ``summary.md``, ``stats.json``,
    ``exclusions.json`` under ``out_root/<campaign>``. ``gates_json`` / ``window_shift``
    apply the trend-rule shift (m12); non-registered overrides (arms file, seed, stale pins)
    are recorded as WARNING lines in ``stats.json`` and printed."""
    cdir = Path(campaign_dir)
    spec = load_registered_arms(arms_path)
    spec, shift_info = shift_windows(spec, gates_json, window_shift)
    warnings: list[str] = []
    if Path(arms_path).resolve() != (Path(root) / DEFAULT_ARMS_PATH).resolve():
        warnings.append(
            f"WARNING: non-registered arms file {arms_path} (registered: {DEFAULT_ARMS_PATH})"
        )
    if seed is not None and int(seed) != int(spec.inference.seed):
        warnings.append(
            f"WARNING: bootstrap seed override {seed} (registered {spec.inference.seed})"
        )
    if allow_stale_benchmarks:
        warnings.append("WARNING: allow_stale_benchmarks set: stale pins are not refused")
    for w_ in warnings:
        print(w_)
    B = int(B if B is not None else spec.inference.B)
    seed = int(seed if seed is not None else spec.inference.seed)
    abench = resolve_arm_benchmarks(spec, root)
    stale = _stale(spec, root)
    if stale and not allow_stale_benchmarks:
        raise PinError(
            f"pinned benchmarks are stale for {stale}: the configs changed after pinning; "
            "run scripts/analyze.py --make-benchmarks and re-pin"
        )
    # fringe arms solved by the closed-form flow diversion (sweep not pinned yet)
    provisional = sorted(
        a
        for a, ab in abench.items()
        if "flow_diversion" in ab.window.method or "flow_diversion" in ab.primary.method
    )
    out = Path(out_root) / cdir.name
    out.mkdir(parents=True, exist_ok=True)

    matched, ignored = collect_runs(cdir, spec)
    runs, exclusions, frames = [], [], {}
    for info in matched:
        arm_name = info["registered_arm"]
        ab = abench.get(arm_name)
        if ab is None:  # instrument arm: closed form of the run's own market (alpha-scaled)
            ab = _run_closed_form_benchmarks(info, spec)
            if ab is None:
                exclusions.append(
                    {
                        "run_id": info["run_id"],
                        "arm": arm_name,
                        "reason": "instrument run without a resolved market in its manifest",
                    }
                )
                continue
        r, frame = run_summary(info, spec, ab, n_perm)
        reason = r.get("error")
        # B2 (SAP s10): the same run filter as the gates: manifest status ok and a complete
        # analysis window; a dropped trunk or sham drops its pairs (no contrast without both)
        s_, e_ = spec.window_of(arm_name)
        if reason is None and info.get("status") not in (None, "ok"):
            reason = f"manifest status {info.get('status')!r} (not ok)"
        if reason is None and int(r.get("n_periods") or 0) != e_ - s_ + 1:
            reason = (
                f"incomplete (SAP s10): {r.get('n_periods')} of {e_ - s_ + 1} periods in the "
                f"window {s_}-{e_}"
            )
        if reason is None and np.isfinite(r["fail_rate"]) and r["fail_rate"] > spec.max_fail_rate:
            reason = f"failed decisions {r['fail_rate']:.1%} > {spec.max_fail_rate:.0%}"
        if (
            reason is None
            and spec.arms[arm_name].fresh_draw_required
            and r.get("post_branch_cache_hits", 0) > 0
        ):
            reason = (
                f"{r['post_branch_cache_hits']} post-branch cache hit(s): fresh-draw rule "
                "(PREREG section 4), exclude and relaunch"
            )
        if reason:
            exclusions.append({"run_id": r["run_id"], "arm": r["registered_arm"], "reason": reason})
            continue
        runs.append(r)
        frames[r["run_id"]] = frame

    arms = {
        a: _arm_summary([r for r in runs if r["registered_arm"] == a], B, seed)
        for a in spec.arms
        if any(r["registered_arm"] == a for r in runs)
    }
    qual = qualification(spec, runs)
    contrasts = {}
    for a in spec.arms:
        c = contrasts_for(a, spec, runs, frames, abench, B, seed, qual["qualified"])
        if c is not None and c["n_pairs"]:
            contrasts[a] = c

    if mqc is None and mqc_campaigns:
        mqc = analyze_mqc(mqc_campaigns, spec, B, seed)
    elif isinstance(mqc, str | Path):
        mqc = load_mqc(mqc)
    mqc_pts = mqc_point_estimates(mqc)

    fam = spec.lever_family
    alpha = spec.inference.alpha
    inf_spec = spec.inference
    # H1 (D-3', D-16, D-17, D-19): one pre-specified lever, capped index, both scorings
    plev = spec.primary_lever
    pc = contrasts.get(plev, {})
    if pc and mqc_pts.get(plev) is None:
        raise ValueError(
            f"MQC of the primary lever {plev!r} is absent from the MQC document: the "
            "registered H1 reading needs it (m9); pass --mqc or --mqc-campaign"
        )
    h1 = stats.primary_h1_test(
        list(pc.get(PRIMARY, {}).get("diffs", {}).values()),
        mqc_pts.get(plev),
        alpha,
        inf_spec.h1.mqc_max_pp,
        inf_spec.h1.practical_lower_bound_gt,
        inf_spec.tost_margin,
        B,
        seed,
        d_own=list(pc[OWN]["diffs"].values()) if pc else None,
    )
    h1 = {
        "lever": plev,
        "lane": "H",
        "hypothesis": spec.lanes["H"].hypothesis if "H" in spec.lanes else "H1",
        "estimand": PRIMARY,
        "own_estimand": OWN,
        "trunks": pc.get("trunks", []),
        "benchmark_shift": pc.get("benchmark_shift"),
        "sc1_price_response": _sc1_response(mqc).get(plev),
        **h1,
    }
    _apply_l2_rule(h1, plev, pc, _sc1_response(mqc), alpha)
    _apply_gate2_outcome(h1, qual, "H")
    # H1-M and H-S (D-23): the lane's pre-specified lever on its own qualified trunks
    h1_by_lane: dict[str, dict[str, Any]] = {}
    for lane, ls in spec.lanes.items():
        if lane == "H" or ls.primary_lever is None or ls.primary_lever not in contrasts:
            continue
        lc = contrasts[ls.primary_lever]
        hl = stats.primary_h1_test(
            list(lc[PRIMARY]["diffs"].values()),
            mqc_pts.get(spec.primary_lever),  # the MQC of the lever is lane-independent (CPU)
            alpha,
            inf_spec.h1.mqc_max_pp,
            inf_spec.h1.practical_lower_bound_gt,
            inf_spec.tost_margin,
            B,
            seed,
            d_own=list(lc[OWN]["diffs"].values()),
        )
        hl = {
            "lever": ls.primary_lever,
            "lane": lane,
            "hypothesis": ls.hypothesis,
            "model": ls.model,
            "trunks": lc.get("trunks", []),
            "benchmark_shift": lc.get("benchmark_shift"),
            "note": ls.description,
            **hl,
        }
        _apply_l2_rule(hl, ls.primary_lever, lc, _sc1_response(mqc), alpha)
        _apply_gate2_outcome(hl, qual, lane)
        h1_by_lane[lane] = hl
    # H1-R: the fixed family of 8 levers, Holm m = 8 on the full qualified sample, within
    # each scoring; the dual-benchmark label per lever (D-17)
    full_p = {lev: contrasts.get(lev, {}).get(PRIMARY, {}).get("t", {}).get("p") for lev in fam}
    own_p = {lev: contrasts.get(lev, {}).get(OWN, {}).get("t", {}).get("p") for lev in fam}
    holm_full = stats.holm_fixed(full_p, fam, alpha)
    holm_own = stats.holm_fixed(own_p, fam, alpha)
    diffs = {lev: contrasts[lev][PRIMARY]["diffs"] for lev in fam if lev in contrasts}
    seeds = {}
    for lev in fam:
        for t, sd in contrasts.get(lev, {}).get("seeds", {}).items():
            if sd is not None:
                seeds[t] = int(sd)
    sc1 = _sc1_response(mqc)
    h1r = {
        "family": fam,
        "m": len(fam),
        "levers": {
            lev: {
                "n": contrasts.get(lev, {}).get(PRIMARY, {}).get("n", 0),
                "mean": contrasts.get(lev, {}).get(PRIMARY, {}).get("mean"),
                "bca_lo": contrasts.get(lev, {}).get(PRIMARY, {}).get("bca_lo"),
                "bca_hi": contrasts.get(lev, {}).get(PRIMARY, {}).get("bca_hi"),
                "mqc": mqc_pts.get(lev),
                **holm_full[lev],
                "own_scoring": {
                    "mean": contrasts.get(lev, {}).get(OWN, {}).get("mean"),
                    "bca_lo": contrasts.get(lev, {}).get(OWN, {}).get("bca_lo"),
                    "bca_hi": contrasts.get(lev, {}).get(OWN, {}).get("bca_hi"),
                    **holm_own[lev],
                },
                "dual_benchmark_label": _family_label(lev, holm_full, holm_own, contrasts),
                "label_basis": "Holm decisions (H1-R, m = 8) under both scorings (B1)",
                "own_vs_trunk_max_abs_diff": contrasts.get(lev, {}).get(
                    "own_vs_trunk_max_abs_diff"
                ),
                "own_scoring_informative": contrasts.get(lev, {}).get("own_scoring_informative"),
                "benchmark_shift": contrasts.get(lev, {}).get("benchmark_shift"),
                "sc1_price_response": sc1.get(lev),
            }
            for lev in fam
        },
        "best_lever_split_sample": stats.split_sample_best(
            diffs,
            seeds,
            fam,
            B,
            seed,
            min_per_half=inf_spec.gate2.split_estimate_min_per_half,
            mqc=_mqc_eligible(mqc, inf_spec.h1.mqc_max_pp, MQC_UPPER_MAX_PP),
            mqc_max_pp=inf_spec.h1.mqc_max_pp,
            min_total=inf_spec.gate2.split_estimate_min_qualified
            if qual["by_lane"].get("H", {}).get("available")
            else None,
            n_qualified=qual["by_lane"].get("H", {}).get("n_qualified"),
        ),
    }
    h1r["n_rejected_trunk"] = int(sum(1 for lev in fam if holm_full[lev]["reject"]))
    h1r["n_rejected_own"] = int(sum(1 for lev in fam if holm_own[lev]["reject"]))
    for lev, row in h1r["levers"].items():  # A-003: the L2 rule inside the family
        if lev in contrasts:
            row.update(_l2_rule_fields(lev, contrasts[lev], sc1, bool(row["reject"])))
            row["supported"] = bool(row["reject"]) and row["dual_benchmark_label"] == (
                "collusion reduction"
            )
        else:
            row["supported"] = False
    h1r["n_supported"] = int(sum(1 for r_ in h1r["levers"].values() if r_["supported"]))
    # the contrast-level label of a family lever is the family (Holm) reading, so figures and
    # tables never show a raw-p reading for a member of the family (B1)
    for lev, row in h1r["levers"].items():
        if lev in contrasts and "dual_benchmark" in contrasts[lev]:
            contrasts[lev]["dual_benchmark"]["label"] = row["dual_benchmark_label"]
            contrasts[lev]["dual_benchmark"]["basis"] = row["label_basis"]
    for a, c in contrasts.items():
        if a not in fam and "dual_benchmark" in c:
            c["dual_benchmark"]["basis"] = "raw p (single contrast)"
    _apply_fallback_labels(h1r, contrasts, qual, spec)
    sig = stats.pooled_sd({lev: list(d.values()) for lev, d in diffs.items()})
    sham_pairs = contrasts.get("sham2", {}).get(PRIMARY, {}).get("sd")
    n_pairs = {lev: contrasts.get(lev, {}).get("n_pairs", 0) for lev in fam}
    power = {
        "mei": MEI,
        "h1": stats.power_one_sided(n_pairs[plev], sig["sigma_d"], MEI, alpha),
        "h1r_holm_worst_case": {
            lev: stats.power_one_sided(n_pairs[lev], sig["sigma_d"], MEI, alpha / len(fam))
            for lev in fam
        },
    }

    result = {
        "campaign": cdir.name,
        "registry": {
            "path": str(arms_path),
            "sha256": file_sha256(arms_path),
            "prereg_id": spec.prereg_id,
        },
        "windows": {k: list(v) for k, v in spec.windows.items()},
        "window_shift": shift_info,
        "warnings": warnings,
        "estimands": {
            "primary": "delta_p: incumbents' CAPPED price index min(Delta_p, 1) with the "
            "pre-lever (trunk) benchmarks (D-16; raw index reported as delta_p_raw)",
            "own_scoring": "delta_p_own: capped Delta_p of the sham on the trunk benchmarks minus "
            "capped Delta_p of the lever on its own market's pinned benchmarks (= the "
            "delta_p_window contrast), the D-17 co-condition for 'collusion reduction'",
            "secondary": "delta_pi_window: incumbents' profit-based Delta, per-window benchmarks",
            "robustness": ["delta_pi", "delta_p_vw", "delta_p_raw", "delta_p_window_raw"],
            "reported": ["share_quotes_above_pm", "trade_weighted_price", "benchmark_shift"],
        },
        "benchmarks": {a: ab.to_dict() for a, ab in abench.items()},
        "stale_benchmarks": stale,
        "provisional_benchmarks": provisional,
        "inference": {
            "sidedness": SIDEDNESS,
            "alternative": spec.inference.sidedness,
            "alpha": alpha,
            "h1_rule": "D-3'/D-16/D-17/D-19: one pre-specified lever, capped index, one-sided "
            "paired t-tests on all qualified trunks under the trunk and the own scoring, no "
            "multiplicity; 'collusion reduction' iff p < alpha under both and the MQC point "
            f"estimate <= {inf_spec.h1.mqc_max_pp:g} pp (trunk-only: 'price reduction "
            "consistent with competitive adjustment'); practically significant iff the "
            f"one-sided 95 % paired-t lower bound > {inf_spec.h1.practical_lower_bound_gt:g} "
            "(inconclusive when the 90 % interval straddles 0.10-0.20; BCa 90 % alongside)",
            "primary_lever": plev,
            "h1r_family": fam,
            "m": len(fam),
            "missing_levers_at_p1": [lev for lev in fam if holm_full[lev]["placeholder"]],
            "split_sample": "descriptive only: best eligible lever (MQC point <= 5 pp, upper "
            "bound <= 7 pp) selected on odd-seed trunks, estimated on even-seed trunks; off "
            "with fewer than 3 trunks per half (D-22)",
            "qualification": qual,
            "sigma_d": {
                "estimate": sig["sigma_d"],
                "df": sig["df"],
                "source": "pooled SD of realised lever-minus-sham differences across trunks",
                "label": "pooled lever-minus-sham SD over the whole family, effective levers included "
                "(power input only; NOT a noise scale, A-022)",
                "by_lever": sig["by"],
                "sham_vs_sham_sd_descriptive": sham_pairs,
                "sham_vs_sham_sd_raw": contrasts.get("sham2", {}).get("delta_p_raw", {}).get("sd"),
            },
            "power": power,
            "B": B,
            "seed": seed,
        },
        "n_runs": len(runs),
        "n_excluded": len(exclusions),
        "n_ignored": len(ignored),
        "runs": [{k: v for k, v in r.items()} for r in runs],
        "ignored_runs": ignored,
        "arms": arms,
        "contrasts": contrasts,
        "h1": h1,
        "h1_by_lane": h1_by_lane,
        "h1r": h1r,
        "mqc": mqc,
    }
    (out / "stats.json").write_text(json.dumps(_clean(result), indent=2), encoding="utf-8")
    (out / "exclusions.json").write_text(json.dumps(exclusions, indent=2), encoding="utf-8")
    (out / "summary.md").write_text(_summary_md(result, exclusions), encoding="utf-8")
    return result


def _run_closed_form_benchmarks(info: dict[str, Any], spec: RegisteredArms) -> ArmBenchmarks | None:
    """Benchmarks of an instrument run (Gate 1 / 1b / AC-H) from its own resolved market."""
    from finswarm.metrics.benchmarks import MarketModel

    market = info.get("resolved_market")
    if not market:
        return None
    members = spec.arms[info["registered_arm"]].members
    n = len(members) if isinstance(members, list) else None
    try:
        model = MarketModel.from_config(market, n)
    except Exception:  # noqa: BLE001 - reported as an exclusion by the caller
        return None
    return ArmBenchmarks.from_model(model)


def _apply_gate2_outcome(h1: dict[str, Any], qual: dict[str, Any], lane: str) -> None:
    """Gate 2 outcome table (D-25) on a lane's H1: fewer qualified trunks than the fallback
    threshold -> H1 'not tested' (estimates kept for transparency, never a decision); a
    reduced-power sample is flagged; no trunk runs -> status 'not evaluable'."""
    bl = qual.get("by_lane", {}).get(lane, {})
    if not bl.get("available"):
        h1["gate2"] = None
        h1["status"] = (
            "not evaluable: no trunk runs of this lane in the campaign (qualification unknown)"
            if h1.get("n", 0)
            else "not tested: no paired contrast"
        )
        return
    out = bl["outcome"]
    h1["gate2"] = out
    if not out["h1_tested"]:
        h1["status"] = (
            f"not tested: {out['n_qualified']} qualified trunks, below the fallback threshold "
            f"({out['decision']}); estimates shown for transparency only"
        )
        h1["h1_supported"] = False
        h1["h1_practically_significant"] = False
        h1["dual_benchmark_label"] = "not tested (Gate 2 fallback)"
    elif out["reduced_power"]:
        h1["status"] = f"tested with reduced power ({out['n_qualified']} qualified trunks)"
    else:
        h1["status"] = f"tested ({out['n_qualified']} qualified trunks)"


def _apply_fallback_labels(
    h1r: dict[str, Any],
    contrasts: dict[str, Any],
    qual: dict[str, Any],
    spec: RegisteredArms,
) -> None:
    """Under a Gate 2 fallback (fewer qualified trunks than the threshold) no lever of that
    lane may read 'collusion reduction': H1-R rows and the lane's contrasts are labelled
    'not tested (fallback ...)' and their Holm rejections withdrawn (D-25)."""
    by_lane = qual.get("by_lane", {})
    for lane, bl in by_lane.items():
        if not bl.get("available") or bl["outcome"]["h1_tested"]:
            continue
        label = f"not tested (fallback: {bl['outcome']['decision']})"
        for arm, c in contrasts.items():
            if spec.arms[arm].lane == lane and "dual_benchmark" in c:
                c["dual_benchmark"]["label"] = label
        if lane == "H":
            h1r["status"] = label
            for lev in h1r["levers"].values():
                lev["dual_benchmark_label"] = label
                lev["reject"] = False
                if isinstance(lev.get("own_scoring"), dict):
                    lev["own_scoring"]["reject"] = False


def _family_label(
    lev: str,
    holm_full: dict[str, dict[str, Any]],
    holm_own: dict[str, dict[str, Any]],
    contrasts: dict[str, Any],
) -> str:
    """D-17 label of a family lever from the HOLM decisions under both scorings (B1)."""
    if holm_full[lev].get("placeholder"):
        return "not run (enters Holm at p = 1)"
    own_avail = lev in contrasts and OWN in contrasts[lev]
    return stats.label_from_decisions(
        bool(holm_full[lev]["reject"]), bool(holm_own[lev]["reject"]) if own_avail else None
    )


def _is_l2(lever: str) -> bool:
    return lever == "L2" or lever.startswith("L2_")


def _l2_rule_fields(
    lever: str, contrast: dict[str, Any], sc1: dict[str, float | None], trunk_rejected: bool
) -> dict[str, Any]:
    """A-003 (the L2 rule): the pinned L2 own benchmarks coincide with the trunk's within
    0.005 over p in [1.5, 2.0], so the D-17 own-scoring co-condition is uninformative for L2.
    For L2 the label "collusion reduction" requires (a) the trunk-scored test significant
    (Holm inside H1-R, raw p for H1 / H1-M / H-S) AND (b) the LLM incumbents' mean ask fall
    under L2 (sham minus lever, posted prices, 111-150) exceeding TWICE the SC1
    best-responders' mean ask fall under the same lever (MQC-L2 vs SC-1 sham, CPU, computed
    before any L2 LLM data are read; ``sc1_price_response`` = treated minus reference, so the
    fall is its negative); otherwise "price reduction consistent with competitive adjustment".
    Returns the fields to merge into the lever's row (label only overridden for L2)."""
    if not _is_l2(lever):
        return {}
    base = lever.split("_")[0]
    resp = sc1.get(base)
    llm_fall = (contrast.get("ask_fall") or {}).get("mean")
    rule: dict[str, Any] = {
        "applies": True,
        "trunk_test_significant": bool(trunk_rejected),
        "llm_ask_fall": llm_fall,
        "sc1_ask_fall": None if resp is None else -float(resp),
        "factor": L2_ASK_FALL_FACTOR,
        "own_vs_trunk_max_abs_diff": contrast.get("own_vs_trunk_max_abs_diff"),
    }
    if resp is None or llm_fall is None or not np.isfinite(float(llm_fall)):
        rule["status"] = "not evaluable: SC1 ask fall (MQC of the lever) or LLM ask fall missing"
        rule["passes"] = None
        label = (
            "not evaluable (A-003: SC1 ask fall missing)"
            if trunk_rejected
            else "no significant reduction"
        )
    else:
        rule["threshold"] = L2_ASK_FALL_FACTOR * rule["sc1_ask_fall"]
        rule["passes"] = bool(float(llm_fall) > rule["threshold"])
        rule["status"] = "ok"
        if not trunk_rejected:
            label = "no significant reduction"
        elif rule["passes"]:
            label = "collusion reduction"
        else:
            label = "price reduction consistent with competitive adjustment"
    return {"a003_l2_rule": rule, "dual_benchmark_label": label}


def _apply_l2_rule(
    h: dict[str, Any],
    lever: str,
    contrast: dict[str, Any],
    sc1: dict[str, float | None],
    alpha: float,
) -> None:
    """A-003 on a single-contrast reading (H1, H1-M, H-S): the label and ``h1_supported``
    follow the L2 rule instead of the own-scoring co-condition."""
    if not _is_l2(lever) or not contrast:
        return
    sig = bool(h.get("checks", {}).get("p_lt_alpha"))
    fields = _l2_rule_fields(lever, contrast, sc1, sig)
    h.update(fields)
    mqc_ok = bool(h.get("checks", {}).get("mqc_ok"))
    passes = fields["a003_l2_rule"].get("passes")
    h["h1_supported"] = bool(sig and passes and mqc_ok)
    h["h1_practically_significant"] = bool(
        h["h1_supported"] and h.get("practical_verdict") == "practically significant"
    )
    h["checks"]["a003_l2_passes"] = passes
    h["rule"]["a003"] = (
        "A-003 (the fringe lever): collusion reduction = trunk-scored significance AND LLM "
        "ask fall > 2 x SC1 ask fall; the own-scoring co-condition is uninformative because "
        "the lever's own benchmarks coincide with the trunk's (no lever name here: blinding)"
    )


def _sc1_response(mqc: dict[str, Any] | None) -> dict[str, float | None]:
    """``{lever: SC1 best-responder price response}`` from an MQC document (D-17 reporting)."""
    if not isinstance(mqc, dict):
        return {}
    levers = mqc.get("levers", {})
    return {
        lev: v.get("sc1_price_response")
        for lev, v in levers.items()
        if isinstance(v, dict) and v.get("status", "ok") == "ok"
    }


def _mqc_eligible(
    mqc: dict[str, Any] | None, max_pp: float, max_upper_pp: float
) -> dict[str, float | None]:
    """Best-lever eligibility (PREREG s2): ``{lever: MQC point}`` with ``None`` for levers
    whose point estimate exceeds ``max_pp`` or whose 95 % upper bound exceeds
    ``max_upper_pp`` (or that have no MQC)."""
    if not isinstance(mqc, dict):
        return {}
    levers = mqc.get("levers", mqc)
    out: dict[str, float | None] = {}
    for lev, v in levers.items():
        if isinstance(v, dict):
            if v.get("status", "ok") != "ok" or v.get("mean") is None:
                out[lev] = None
                continue
            m, hi = float(v["mean"]), v.get("bca_hi")
            # m11: a missing upper bound makes the lever ineligible for the split estimate
            ok_hi = hi is not None and np.isfinite(float(hi)) and float(hi) <= max_upper_pp
            out[lev] = m if m <= max_pp and ok_hi else None
        elif v is not None:
            out[lev] = None  # a bare point estimate has no upper bound: ineligible (m11)
    return out


def _stale(spec: RegisteredArms, root: str | Path) -> list[str]:
    from finswarm.analysis.benchmarks_file import stale_markets

    out = []
    for pin in spec.benchmark_files:
        doc = load_pinned(spec, pin, root)
        out += [f"{pin}:{m}" for m in stale_markets(spec, pin, doc, root)]
    return out


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


def _summary_md(result: dict[str, Any], exclusions: list[dict[str, Any]]) -> str:
    w = result["windows"]
    inf = result["inference"]
    lines = [
        f"# Analysis — {result['campaign']}",
        "",
        f"Registry `{result['registry']['path']}` (sha256 {result['registry']['sha256'][:12]}); "
        f"windows: branch {w['branch'][0]}–{w['branch'][1]}, trunk {w['trunk'][0]}–"
        f"{w['trunk'][1]}; runs analysed: {result['n_runs']}; excluded: "
        f"{result['n_excluded']}; ignored (unregistered): {result['n_ignored']}; bootstrap "
        f"B = {inf['B']}, seed {inf['seed']}.",
        "",
        f"Primary estimand: {result['estimands']['primary']}. Tests: {inf['sidedness']}.",
        "",
        "Provisional (closed-form fringe) benchmarks: "
        f"{', '.join(result['provisional_benchmarks']) or 'none'}; re-pin with "
        "scripts/analyze.py --make-benchmarks before the registered analysis.",
        "",
        "## Arms",
        "",
        tables.arm_table(result["arms"]),
    ]
    h1, h1r = result["h1"], result["h1r"]
    yn = {True: "yes", False: "no"}
    own = h1.get("own_scoring") or {}
    prac = h1.get("practical_significance") or {}
    lines += [
        f"## H1 — primary lever {h1['lever']} (no multiplicity; capped index, both scorings)",
        "",
        f"Gate 2 outcome: {h1.get('status', '–')}.",
        "",
        f"n = {h1['n']} qualified trunks; ΔC_p (trunk scoring) = "
        f"{tables.fmt_ci(h1['mean'], h1['bca_lo'], h1['bca_hi'])} (BCa 95 %); one-sided "
        f"p = {tables.fmt_p(h1['p_one_sided'])} (Wilcoxon sensitivity "
        f"{tables.fmt_p(h1['wilcoxon_sensitivity'].get('p'))}); own scoring ΔC_p = "
        f"{tables.fmt_ci(own.get('mean'), own.get('bca_lo'), own.get('bca_hi'))}, one-sided "
        f"p = {tables.fmt_p(own.get('p_one_sided'))}; MQC {tables.fmt(h1['mqc'])} pp. "
        f"Label (D-17): {h1.get('dual_benchmark_label', '–')}. Supported: "
        f"{yn[h1['h1_supported']]}. Practical significance (D-19): {prac.get('verdict', '–')} "
        f"(one-sided 95 % lower bound {tables.fmt(h1['lower_bound_one_sided_95'])}; 90 % t "
        f"interval {tables.fmt_ci(h1['mean'], *prac.get('t_interval_90', [None, None]))}; "
        f"BCa 90 % {tables.fmt_ci(h1['mean'], *prac.get('bca_90', [None, None]))})"
        + (
            f"; TOST ±0.20: {'equivalent' if h1['tost'].get('equivalent') else 'not shown'}."
            if h1.get("tost")
            else "."
        ),
        "",
    ]
    for lane, hl in result.get("h1_by_lane", {}).items():
        o = hl.get("own_scoring") or {}
        lines += [
            f"## {hl.get('hypothesis', lane)} — lever {hl['lever']} on lane {lane} "
            f"({hl.get('model', '')}; {hl.get('status', '–')})",
            "",
            f"n = {hl['n']} qualified trunks; ΔC_p (trunk scoring) = "
            f"{tables.fmt_ci(hl['mean'], hl['bca_lo'], hl['bca_hi'])}, one-sided p = "
            f"{tables.fmt_p(hl['p_one_sided'])}; own scoring p = "
            f"{tables.fmt_p(o.get('p_one_sided'))}; label: {hl.get('dual_benchmark_label', '–')}; "
            f"supported: {yn[bool(hl['h1_supported'])]}. {hl.get('note', '')}",
            "",
        ]
    if result["contrasts"]:
        lines += [
            f"## H1-R — paired contrasts (primary Delta_p; Holm over m = {inf['m']})",
            "",
            tables.contrast_table(result["contrasts"], h1r["levers"]),
            f"Missing levers counted at p = 1: {', '.join(inf['missing_levers_at_p1']) or 'none'}.",
            "",
            "## Mechanical vs behavioural Delta C (levers)",
            "",
            tables.decomposition_table(result["contrasts"]),
        ]
    best = h1r["best_lever_split_sample"]
    est = best.get("estimate") or {}
    lines += [
        "## Empirical best lever (descriptive split sample)",
        "",
        f"Status: {best.get('status', 'ok')}. Selected on odd-seed trunks "
        f"({', '.join(best['selection_trunks']) or '–'}): {best.get('best_lever') or 'none'}; "
        f"estimated on even-seed trunks ({', '.join(best['estimation_trunks']) or '–'}): "
        f"{tables.fmt_ci(est.get('mean'), est.get('bca_lo'), est.get('bca_hi'))}. "
        "Descriptive only, no decision. sigma_d (realised, pooled lever-minus-sham) = "
        f"{tables.fmt(inf['sigma_d']['estimate'], 3)}.",
        "",
    ]
    if exclusions:
        lines += ["## Exclusions", "", tables.markdown_table(exclusions)]
    return "\n".join(lines)
