"""A-024 EXPLORATORY descriptives (no inference change, labelled exploratory everywhere).

(a) replacement regression (R1): the fresh copy's first post-replacement ask and its window mean
    on the remaining incumbents' ask at T0 (period 100), OLS across the qualified trunks;
(b) post-entry incumbent asks vs the static logit best response to the entrant's ask (gap with
    95 % t CI), per entrant arm L6 / L6b / L6c;
(c) the forced-undercut probes (S2): rivals' response vs the static best response to the cut,
    profit vs the trunk's sham, reversion to the pre-probe ask;
(d) per lever: exact sign test, per-trunk difference table, uncapped index with 90 % CI,
    screened (all trunks) vs qualified trunks.

Prices are in index units (p - p_N) / (p_M - p_N) of the pinned trunk benchmarks unless said
otherwise; periods are registered (1-based, A-008). Reads the registered outputs and the raw
tables (``finswarm.analysis.io.read_table``); no LLM call, no new run::

    uv run python scripts/a024_exploratory.py --out-root results/analysis
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from scipy import stats as sps

from finswarm.analysis.arms import load_registered_arms, resolve_arm_benchmarks
from finswarm.analysis.io import read_table

LEVERS = ("L1", "L2", "L3", "L4a", "L5", "L6", "L7", "L6b", "L6c")
ENTRANTS = ("L6", "L6b", "L6c")
T0 = 100  # last trunk period; the branch (replacement, entry, probe) acts from period 101
WINDOW = (111, 150)


def t_ci(x: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan"), float("nan")
    h = sps.t.ppf(0.5 + level / 2, x.size - 1) * x.std(ddof=1) / math.sqrt(x.size)
    return float(x.mean() - h), float(x.mean() + h)


def ols(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = x.size
    if n < 3 or np.ptp(x) == 0:
        return {"n": int(n), "slope": None, "intercept": None}
    r = sps.linregress(x, y)
    tq = sps.t.ppf(0.975, n - 2)
    return {
        "n": int(n),
        "slope": float(r.slope),
        "slope_ci95": [float(r.slope - tq * r.stderr), float(r.slope + tq * r.stderr)],
        "intercept": float(r.intercept),
        "intercept_ci95": [
            float(r.intercept - tq * r.intercept_stderr),
            float(r.intercept + tq * r.intercept_stderr),
        ],
        "r2": float(r.rvalue**2),
    }


def price_matrix(path: str, slots: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """``(periods, prices[period, slot])`` of the run (registered periods)."""
    ap = read_table(path, "agent_period").filter(pl.col("slot").is_in(slots))
    if ap.height == 0:
        return np.array([]), np.empty((0, len(slots)))
    wide = ap.pivot(values="price", index="period", on="slot", aggregate_function="mean").sort(
        "period"
    )
    cols = [str(s) for s in slots]
    per = wide["period"].to_numpy()
    mat = np.column_stack(
        [wide[c].to_numpy() if c in wide.columns else np.full(len(per), np.nan) for c in cols]
    )
    return per, mat.astype(float)


def profit_matrix(path: str, slots: list[int]) -> tuple[np.ndarray, np.ndarray]:
    ap = read_table(path, "agent_period").filter(pl.col("slot").is_in(slots))
    wide = ap.pivot(values="profit", index="period", on="slot", aggregate_function="mean").sort(
        "period"
    )
    per = wide["period"].to_numpy()
    mat = np.column_stack([wide[str(s)].to_numpy() for s in slots]).astype(float)
    return per, mat


def at(per: np.ndarray, mat: np.ndarray, p: int) -> np.ndarray:
    i = np.where(per == p)[0]
    return mat[i[0]] if i.size else np.full(mat.shape[1], np.nan)


def window_mean(per: np.ndarray, mat: np.ndarray, lo: int, hi: int) -> np.ndarray:
    k = (per >= lo) & (per <= hi)
    return np.nanmean(mat[k], axis=0) if k.any() else np.full(mat.shape[1], np.nan)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-root", type=Path, default=Path("results/analysis"))
    ap.add_argument("--arms", type=Path, default=Path("configs/analysis/registered_arms.yaml"))
    ap.add_argument("--root", type=Path, default=Path("."))
    args = ap.parse_args(argv)
    st = json.loads((args.out_root / "TR-H" / "stats.json").read_text(encoding="utf-8"))
    spec = load_registered_arms(args.arms)
    ab = resolve_arm_benchmarks(spec, args.root)
    tb = ab["sham"].primary
    pn, pm = float(tb.p_nash), float(tb.p_mono)

    def idx(p: Any) -> Any:
        return (np.asarray(p, dtype=float) - pn) / (pm - pn)

    qualified = set(st["inference"]["qualification"]["by_lane"]["H"]["qualified"])
    runs: dict[str, dict[str, dict[str, Any]]] = {}
    for r in st["runs"]:
        runs.setdefault(r["registered_arm"], {})[str(r["trunk_id"])] = r
    out: dict[str, Any] = {
        "status": "exploratory (A-024), no inference change",
        "benchmarks": {"p_nash": pn, "p_mono": pm},
        "n_qualified": len(qualified),
    }

    # (a) replacement regression (R1)
    xs, y1, y2, rows_a = [], [], [], []
    for t, r in sorted(runs.get("R1", {}).items()):
        if t not in qualified:
            continue
        per, mat = price_matrix(r["path"], [0, 1, 2, 3])
        x = float(np.nanmean(idx(at(per, mat, T0)[1:])))  # remaining incumbents at T0
        first = float(idx(at(per, mat, T0 + 1)[0]))  # the fresh copy's first ask
        wmean = float(np.nanmean(idx(window_mean(per, mat, *WINDOW)[0])))
        xs.append(x)
        y1.append(first)
        y2.append(wmean)
        rows_a.append({"trunk": t, "incumbents_T0": x, "first_ask": first, "window_mean": wmean})
    out["a_replacement"] = {
        "definition": "x = mean ask (index) of slots 1-3 at period 100; y = the fresh slot-0 "
        "copy's ask at period 101 (first) and its mean over 111-150 (window)",
        "first_ask": ols(np.array(xs), np.array(y1)),
        "window_mean": ols(np.array(xs), np.array(y2)),
        "rows": rows_a,
    }

    # (b) post-entry incumbent asks vs the static best response to the entrant's ask
    res_b: dict[str, Any] = {}
    for arm in ENTRANTS:
        br = ab[arm].window_model.best_response()
        gaps, rows_b = [], []
        for t, r in sorted(runs.get(arm, {}).items()):
            if t not in qualified:
                continue
            per, mat = price_matrix(r["path"], [0, 1, 2, 3, 4])
            k = (per >= WINDOW[0]) & (per <= WINDOW[1])
            m = mat[k]
            if not m.size:
                continue
            g = []
            for i in range(4):
                rivals = np.delete(m, i, axis=1)  # the 3 other incumbents and the entrant
                ok = np.isfinite(rivals).all(axis=1) & np.isfinite(m[:, i])
                if ok.any():
                    b = np.asarray(br(rivals[ok]), dtype=float)
                    g.append(np.mean((m[ok, i] - b) / (pm - pn)))
            gap = float(np.mean(g)) if g else float("nan")
            gaps.append(gap)
            rows_b.append(
                {
                    "trunk": t,
                    "gap_index": gap,
                    "entrant_ask_index": float(np.nanmean(idx(m[:, 4]))),
                    "incumbent_ask_index": float(np.nanmean(idx(m[:, :4]))),
                }
            )
        x = np.asarray(gaps, dtype=float)
        res_b[arm] = {
            "n": int(np.isfinite(x).sum()),
            "mean_gap_index": float(np.nanmean(x)) if x.size else None,
            "ci95": list(t_ci(x)),
            "rows": rows_b,
        }
    out["b_entry_best_response_gap"] = {
        "definition": "gap = incumbent ask - static logit best response to the other incumbents' "
        "and the entrant's asks (window market), averaged over 111-150 and the 4 incumbents, in "
        "index units; > 0 = incumbents price above their static best response",
        "arms": res_b,
    }

    # (c) forced-undercut probes (S2)
    rows_c = []
    br_t = ab["sham"].window_model.best_response()
    for t, r in sorted(runs.get("S2", {}).items()):
        if t not in qualified:
            continue
        ev = read_table(r["path"], "events")
        d = T0 + 1
        if ev.height and "event_type" in ev.columns:
            fd = ev.filter(pl.col("event_type") == "forced_deviation")
            if fd.height:
                d = int(fd["period"].min())
        per, mat = price_matrix(r["path"], [0, 1, 2, 3])
        pre = window_mean(per, mat, d - 3, d - 1)
        dev = at(per, mat, d)
        b = np.asarray(br_t(np.array([[dev[0], *np.delete(dev[1:], j)] for j in range(3)])))
        resp = window_mean(per, mat, d + 1, d + 3)[1:]
        sham = runs.get("sham", {}).get(t)
        prof = None
        if sham is not None:
            pp, pr = profit_matrix(r["path"], [0, 1, 2, 3])
            sp, sr = profit_matrix(sham["path"], [0, 1, 2, 3])
            a_ = window_mean(pp, pr, d, d + 9)
            s_ = window_mean(sp, sr, d, d + 9)
            prof = {
                "deviator_vs_sham": float(a_[0] - s_[0]),
                "rivals_vs_sham": float(np.mean(a_[1:] - s_[1:])),
            }
        tol = 0.10 * (pm - pn)
        rev = None
        for p in per[per > d]:
            row = at(per, mat, int(p))
            if np.all(np.abs(row - pre) <= tol):
                rev = int(p) - d
                break
        rows_c.append(
            {
                "trunk": t,
                "deviation_period": d,
                "cut_index": float(idx(pre[0]) - idx(dev[0])),
                "rivals_response_index": float(np.mean(idx(resp) - idx(pre[1:]))),
                "rivals_static_br_index": float(np.mean(idx(b) - idx(pre[1:]))),
                "profit_10p": prof,
                "reversion_periods": rev,
            }
        )

    def _m(key: str, sub: str | None = None) -> float | None:
        v = [
            (r[key] or {}).get(sub) if sub else r[key]
            for r in rows_c
            if (r[key] if not sub else (r[key] or {}).get(sub)) is not None
        ]
        return float(np.mean(v)) if v else None

    summary_c = {
        "n": len(rows_c),
        "rivals_response_mean": _m("rivals_response_index"),
        "static_br_mean": _m("rivals_static_br_index"),
        "deviator_gain_mean": _m("profit_10p", "deviator_vs_sham"),
        "n_deviator_gain_positive": int(
            sum(1 for r in rows_c if (r["profit_10p"] or {}).get("deviator_vs_sham", 0) > 0)
        ),
        "n_reverted": int(sum(1 for r in rows_c if r["reversion_periods"] is not None)),
    }
    out["c_forced_undercut"] = {
        "summary": summary_c,
        "definition": "pre = mean ask over the 3 periods before the cut; response = rivals' mean "
        "ask change over the 3 periods after it; static BR = rivals' best response to the cut "
        "price (index change vs pre); profit = mean per-period profit over 10 periods from the "
        "cut minus the same trunk's sham; reversion = periods until every slot is back within "
        "0.10 index units of its pre-probe ask (None = not within the run)",
        "rows": rows_c,
    }

    # (d) per lever: sign test, per-trunk table, uncapped 90 % CI, screened vs qualified
    res_d: dict[str, Any] = {}
    all_trunks = sorted(runs.get("sham", {}))
    for lev in LEVERS:
        if lev not in runs:
            continue
        entry: dict[str, Any] = {}
        for label, trunks in (("qualified", sorted(qualified)), ("screened", all_trunks)):
            pair = [t for t in trunks if t in runs["sham"] and t in runs[lev]]
            dc = np.array([runs["sham"][t]["delta_p"] - runs[lev][t]["delta_p"] for t in pair])
            dr = np.array(
                [runs["sham"][t]["delta_p_raw"] - runs[lev][t]["delta_p_raw"] for t in pair]
            )
            blk: dict[str, Any] = {"n": len(pair)}
            for name, d in (("capped", dc), ("raw", dr)):
                pos, neg = int(np.sum(d > 1e-12)), int(np.sum(d < -1e-12))
                p_sign = (
                    float(sps.binomtest(pos, pos + neg, 0.5, alternative="greater").pvalue)
                    if pos + neg
                    else None
                )
                blk[name] = {
                    "mean": float(d.mean()) if d.size else None,
                    "ci90_t": list(t_ci(d, 0.90)),
                    "signs": {"positive": pos, "negative": neg, "zero": int(d.size - pos - neg)},
                    "sign_test_p_one_sided": p_sign,
                }
            if label == "qualified":
                blk["per_trunk"] = [
                    {"trunk": t, "capped": float(c), "raw": float(w)}
                    for t, c, w in zip(pair, dc, dr, strict=True)
                ]
            entry[label] = blk
        res_d[lev] = entry
    out["d_levers"] = res_d

    # (e) ex-post regret per dealer in the sham windows (best fixed price / best response)
    model = ab["sham"].window_model
    br_s = model.best_response()
    grid = np.arange(pn * 0.8, pm * 1.6, 0.005)
    reg: dict[int, list[dict[str, float]]] = {i: [] for i in range(4)}
    for t, r in sorted(runs.get("sham", {}).items()):
        if t not in qualified:
            continue
        per, mat = price_matrix(r["path"], [0, 1, 2, 3])
        k = (per >= WINDOW[0]) & (per <= WINDOW[1])
        P = mat[k]
        P = P[np.isfinite(P).all(axis=1)]
        if not P.size:
            continue
        real = np.nanmean(model.profits(P), axis=0)
        for i in range(4):
            best_fixed = -np.inf
            for c in grid:
                Q = P.copy()
                Q[:, i] = c
                best_fixed = max(best_fixed, float(np.nanmean(model.profits(Q)[:, i])))
            b = np.asarray(br_s(np.delete(P, i, axis=1)), dtype=float)
            Q = P.copy()
            Q[:, i] = b
            best_br = float(np.nanmean(model.profits(Q)[:, i]))
            reg[i].append(
                {
                    "trunk": t,
                    "realised": float(real[i]),
                    "best_fixed": best_fixed,
                    "best_response": best_br,
                    "regret_fixed_share": 1 - float(real[i]) / best_fixed
                    if best_fixed > 0
                    else None,
                    "regret_br_share": 1 - float(real[i]) / best_br if best_br > 0 else None,
                }
            )
    summ = {}
    for i, rows in reg.items():
        rf = np.array(
            [x["regret_fixed_share"] for x in rows if x["regret_fixed_share"] is not None]
        )
        rb = np.array([x["regret_br_share"] for x in rows if x["regret_br_share"] is not None])
        summ[f"slot{i}"] = {
            "n": len(rows),
            "regret_fixed_share_mean": float(rf.mean()) if rf.size else None,
            "regret_fixed_share_ci95": list(t_ci(rf)),
            "regret_br_share_mean": float(rb.mean()) if rb.size else None,
            "regret_br_share_ci95": list(t_ci(rb)),
            "rows": rows,
        }
    out["e_regret"] = {
        "definition": "sham windows 111-150 of the qualified trunks, trunk market's expected "
        "profit: realised = the dealer's mean profit at the logged asks; best fixed = the best "
        "constant price in hindsight against the rivals' logged asks (grid step 0.005); best "
        "response = the per-period static best response to them; regret = 1 - realised / best "
        "(share of the attainable profit left on the table)",
        "dealers": summ,
    }

    # (f) LLM response-cache hits at the first period of every trunk (registered period 1)
    hits, n_tr = 0, 0
    for _t, r in sorted(runs.get("trunk", {}).items()):
        dec = read_table(r["path"], "llm_decisions")
        n_tr += 1
        if dec.height and "cache_hit" in dec.columns:
            hits += int(
                dec.filter((pl.col("period") == 1) & pl.col("cache_hit").fill_null(False)).height
            )
    out["f_cache_hits_period1_trunks"] = {
        "definition": "llm_decisions rows with cache_hit = true at registered period 1 (store "
        "period 0), summed over the TR-H trunk runs",
        "n_trunks": n_tr,
        "hits": hits,
    }

    path = args.out_root / "a024_exploratory.json"
    path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    md = render(out)
    (args.out_root / "a024_exploratory.md").write_text(md, encoding="utf-8")
    print(f"-> {path}")
    return 0


def _f(v: Any, d: int = 3) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "-"
    return f"{v:.{d}f}" if isinstance(v, float) else str(v)


def render(o: dict[str, Any]) -> str:
    L = ["# A-024 exploratory descriptives (no inference change)", ""]
    a = o["a_replacement"]
    L += [
        "## (a) Replacement regression (R1)",
        "",
        a["definition"],
        "",
        "| y | n | slope [95 % CI] | intercept [95 % CI] | R2 |",
        "|---|---|---|---|---|",
    ]
    for k in ("first_ask", "window_mean"):
        r = a[k]
        if r.get("slope") is None:
            L.append(f"| {k} | {r['n']} | - | - | - |")
            continue
        L.append(
            f"| {k} | {r['n']} | {_f(r['slope'])} [{_f(r['slope_ci95'][0])}, {_f(r['slope_ci95'][1])}] | "
            f"{_f(r['intercept'])} [{_f(r['intercept_ci95'][0])}, {_f(r['intercept_ci95'][1])}] | {_f(r['r2'])} |"
        )
    b = o["b_entry_best_response_gap"]
    L += [
        "",
        "## (b) Post-entry asks vs the static best response",
        "",
        b["definition"],
        "",
        "| arm | n | mean gap [95 % CI] |",
        "|---|---|---|",
    ]
    for arm, r in b["arms"].items():
        L.append(
            f"| {arm} | {r['n']} | {_f(r['mean_gap_index'])} [{_f(r['ci95'][0])}, {_f(r['ci95'][1])}] |"
        )
    c = o["c_forced_undercut"]
    L += [
        "",
        "## (c) Forced-undercut probes (S2)",
        "",
        c["definition"],
        "",
        "| trunk | cut | rivals' response | static BR | deviator profit vs sham | rivals' profit vs sham | reversion (periods) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in c["rows"]:
        pf = r["profit_10p"] or {}
        L.append(
            f"| {r['trunk'][-6:]} | {_f(r['cut_index'])} | {_f(r['rivals_response_index'])} | {_f(r['rivals_static_br_index'])} | "
            f"{_f(pf.get('deviator_vs_sham'))} | {_f(pf.get('rivals_vs_sham'))} | {_f(r['reversion_periods'])} |"
        )
    L += [
        "",
        "## (d) Levers: sign test, uncapped index (90 % CI), screened vs qualified",
        "",
        "| lever | set | n | capped mean (+/0/-) sign p | raw mean [90 % CI] (+/0/-) sign p |",
        "|---|---|---|---|---|",
    ]
    for lev, e in o["d_levers"].items():
        for label in ("qualified", "screened"):
            b_ = e[label]
            cp, rw = b_["capped"], b_["raw"]
            L.append(
                f"| {lev} | {label} | {b_['n']} | {_f(cp['mean'])} ({cp['signs']['positive']}/{cp['signs']['zero']}/{cp['signs']['negative']}) {_f(cp['sign_test_p_one_sided'], 4)} | "
                f"{_f(rw['mean'])} [{_f(rw['ci90_t'][0])}, {_f(rw['ci90_t'][1])}] ({rw['signs']['positive']}/{rw['signs']['zero']}/{rw['signs']['negative']}) {_f(rw['sign_test_p_one_sided'], 4)} |"
            )
    e = o.get("e_regret")
    if e:
        L += [
            "",
            "## (e) Ex-post regret per dealer (sham windows)",
            "",
            e["definition"],
            "",
            "| dealer | n | regret vs best fixed [95 % CI] | regret vs best response [95 % CI] |",
            "|---|---|---|---|",
        ]
        for k, r in e["dealers"].items():
            L.append(
                f"| {k} | {r['n']} | {_f(r['regret_fixed_share_mean'])} [{_f(r['regret_fixed_share_ci95'][0])}, "
                f"{_f(r['regret_fixed_share_ci95'][1])}] | {_f(r['regret_br_share_mean'])} "
                f"[{_f(r['regret_br_share_ci95'][0])}, {_f(r['regret_br_share_ci95'][1])}] |"
            )
    L += [
        "",
        "Per-trunk differences: a024_exploratory.json, d_levers.<lever>.qualified.per_trunk.",
        "",
    ]
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
