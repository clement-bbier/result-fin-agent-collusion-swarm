"""Market-quality cost per lever from the scripted MQC campaigns (0 LLM calls).

Each ``MQC-<lever>`` campaign runs the scripted competitive market SC1 to period 100 and
branches it into a sham and the lever (20 seeds). Per seed the MQC is
``100 * (TS_sham - TS_lever) / TS_sham`` with ``TS`` the true-value surplus per period over
the registered window (:func:`finswarm.metrics.quality.true_value_surplus`: buyers' value
minus price for executed trades plus seller profit, the fringe through its trades; one unit
for every lever, no taste shock, no inclusive-value term). Every arm is scored from its OWN
trades, so no market parameters are shared between the sham and the lever. Per lever: mean
over seeds with a BCa 95 % CI (unit = seed), plus the ratio-of-means variant.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from finswarm.analysis import stats
from finswarm.analysis.arms import RegisteredArms
from finswarm.analysis.io import discover_runs, read_table, run_info
from finswarm.metrics.quality import mqc_pp, true_value_surplus


def analyze_mqc(
    campaign_dirs: list[str | Path] | str | Path,
    spec: RegisteredArms,
    B: int = stats.DEFAULT_B,
    seed: int = stats.DEFAULT_SEED,
) -> dict[str, Any]:
    """MQC of every lever of the fixed family (levers without data are listed as such)."""
    dirs = [campaign_dirs] if isinstance(campaign_dirs, str | Path) else list(campaign_dirs)
    infos = [run_info(r) for d in dirs for r in discover_runs(d)]
    s, e = spec.windows[spec.mqc.window]
    out: dict[str, Any] = {
        "window": [s, e],
        "unit": "pp of true-value surplus per period (sham - lever) / sham",
        "reference_arm": spec.mqc.reference_arm,
        "max_pp": spec.mqc.max_pp,
        "levers": {},
    }
    for lev in spec.lever_family:
        label = spec.arms[lev].mqc_label
        if not label:
            out["levers"][lev] = {"status": "no MQC campaign registered", "n": 0}
            continue
        ref = {
            i["trunk_id"]: i
            for i in infos
            if i["label"] == label and i["arm"] == spec.mqc.reference_arm
        }
        trt = {i["trunk_id"]: i for i in infos if i["label"] == label and i["arm"] == lev}
        per_seed, ts_r, ts_t, pr_r, pr_t = {}, [], [], [], []
        for t in sorted(set(ref) & set(trt)):
            a = true_value_surplus(read_table(ref[t]["path"], "trades"), s, e)
            b = true_value_surplus(read_table(trt[t]["path"], "trades"), s, e)
            per_seed[str(ref[t].get("seed", t))] = mqc_pp(a, b)
            ts_r.append(a)
            ts_t.append(b)
            pr_r.append(mean_posted_price(read_table(ref[t]["path"], "agent_period"), s, e))
            pr_t.append(mean_posted_price(read_table(trt[t]["path"], "agent_period"), s, e))
        vals = np.array([v for v in per_seed.values() if np.isfinite(v)])
        if vals.size < 2:
            out["levers"][lev] = {
                "status": "too few paired seeds",
                "n": int(vals.size),
                "label": label,
            }
            continue
        ci = stats.bca_ci(vals, B=B, seed=seed)
        out["levers"][lev] = {
            "status": "ok",
            "label": label,
            "n": int(vals.size),
            "seeds_registered": spec.mqc.n_seeds,
            "mean": ci.estimate,
            "bca_lo": ci.lo,
            "bca_hi": ci.hi,
            "ratio_of_means": mqc_pp(float(np.mean(ts_r)), float(np.mean(ts_t))),
            "ts_reference_mean": float(np.mean(ts_r)),
            "ts_treated_mean": float(np.mean(ts_t)),
            "per_seed": per_seed,
            # D-17 reporting: the SC1 best responders' price response to the lever
            "price_reference_mean": float(np.nanmean(pr_r)),
            "price_treated_mean": float(np.nanmean(pr_t)),
            "sc1_price_response": float(np.nanmean(np.asarray(pr_t) - np.asarray(pr_r))),
        }
    return out


def mean_posted_price(agent_period: pl.DataFrame, start: int, end: int) -> float:
    """Mean positive posted price of every seller over the window (NaN without rows)."""
    if agent_period.height == 0 or "price" not in agent_period.columns:
        return float("nan")
    w = agent_period.filter(
        (pl.col("period") >= start) & (pl.col("period") <= end) & (pl.col("price") > 0)
    )
    return float(w["price"].mean()) if w.height else float("nan")


def mqc_point_estimates(mqc: dict[str, Any] | None) -> dict[str, float | None]:
    """``{lever: mean MQC or None}`` from an :func:`analyze_mqc` document (or a plain
    ``{lever: value}`` mapping)."""
    if not mqc:
        return {}
    levers = mqc.get("levers", mqc)
    out: dict[str, float | None] = {}
    for k, v in levers.items():
        if isinstance(v, dict):
            m = v.get("mean")
            out[k] = float(m) if m is not None and v.get("status", "ok") == "ok" else None
        elif v is not None:
            out[k] = float(v)
    return out


def load_mqc(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
