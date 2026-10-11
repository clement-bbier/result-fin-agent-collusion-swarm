"""Appendix A6 "statistical coherence": per claim the test, margin, N, realised SD, minimum
detectable effect, multiplicity, registration status and data seen, plus two sensitivity tables
(qualification threshold 0.25 / 0.30 / 0.35 and leave-one-trunk-out) and the TOST-margin
justification against the MEI and the ceiling of the capped index.

Reads only analysis outputs (no LLM call, no new run)::

    uv run python scripts/a6_coherence.py --out-root results/analysis \\
        --amendments docs/03_preregistration/AMENDMENTS.md

Writes ``<out-root>/a6_coherence.md`` and ``a6_coherence.json``. Every number is read from the
registered outputs (stats.json, family_s.json, gates.json, detection report) or recomputed from
the per-run indices they hold; nothing here is a test of its own (reporting only, A-022).
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats as sps

ALPHA = 0.05
POWER = 0.80
MEI = 0.25
TOST_MARGIN = 0.20
FAMILY = ("L1", "L2", "L3", "L4a", "L5", "L6", "L7", "L8")


def _load(p: Path) -> dict[str, Any] | None:
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def mde(sd: float | None, n: int, alpha: float = ALPHA, power: float = POWER) -> float | None:
    """One-sided paired-t minimum detectable effect at ``power``:
    (t_{1-alpha, n-1} + t_{power, n-1}) * sd / sqrt(n)."""
    if sd is None or not math.isfinite(sd) or n < 2:
        return None
    q = sps.t.ppf(1 - alpha, n - 1) + sps.t.ppf(power, n - 1)
    return float(q * sd / math.sqrt(n))


def amendments(path: Path | None) -> dict[str, dict[str, str]]:
    """``{A-id: {"data_seen": ..., "what": ...}}`` from AMENDMENTS.md (read only)."""
    if path is None or not path.exists():
        return {}
    out: dict[str, dict[str, str]] = {}
    cur = None
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^### (A-\d{3})", line)
        if m:
            cur = m.group(1)
            out.setdefault(cur, {})
            continue
        if cur and line.startswith("- Data seen before the change?:"):
            out[cur]["data_seen"] = line.split(":", 1)[1].strip()
        if cur and line.startswith("- Affected analysis becomes exploratory?:"):
            out[cur]["exploratory"] = line.split(":", 1)[1].strip()
    return out


def _paired(st: dict[str, Any], lever: str) -> dict[str, Any]:
    c = st.get("contrasts", {}).get(lever) or {}
    d = c.get("delta_p") or {}
    return {
        "n": int(c.get("n_pairs") or d.get("n") or 0),
        "sd": d.get("sd"),
        "mean": d.get("mean"),
        "p": (d.get("t") or {}).get("p"),
        "tost_p": (d.get("tost") or {}).get("p"),
        "raw_tost_p": ((c.get("delta_p_raw") or {}).get("tost") or {}).get("p"),
        "raw_mean": (c.get("delta_p_raw") or {}).get("mean"),
        "share_ref": (c.get("share_quotes_above_pm") or {}).get("reference"),
        "share_arm": (c.get("share_quotes_above_pm") or {}).get("arm"),
        "signs": d.get("signs"),
    }


def claim_rows(
    st: dict[str, Any],
    fam_s: dict[str, Any] | None,
    st_m: dict[str, Any] | None,
    st_s: dict[str, Any] | None,
    gates: dict[str, Any] | None,
    det: dict[str, Any] | None,
    am: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def seen(ids: list[str], base: str) -> str:
        notes = [f"{i}: {am[i]['data_seen']}" for i in ids if i in am and am[i].get("data_seen")]
        return base + ("; " + " | ".join(notes) if notes else "")

    # C1 / C1b gates
    for label, cid, gname in (
        ("C1", "Gate 1 (PC-H)", "Gate 1 (PC-H)"),
        ("C1b", "Gate 1b", "Gate 1b (PC-H-dealer)"),
    ):
        g = ((gates or {}).get("gates") or {}).get(gname) or {}
        rows.append(
            {
                "claim": label,
                "hypothesis": cid,
                "test": "alpha-stratified run BCa (point >= 0.25, share of runs > 0.10)",
                "margin": "0.25 point / 0.10 share",
                "unit": "run",
                "n": g.get("n_runs"),
                "sd": None,
                "mde": None,
                "multiplicity": "none (gate)",
                "registration": "registered (PREREG s1, D-16); A-008 period convention",
                "data_seen": seen(["A-008"], "rule frozen before the gate read (A-000-1 sha)"),
            }
        )
    # H1 (L2)
    h1 = _paired(st, "L2")
    rows.append(
        {
            "claim": "C3a",
            "hypothesis": "H1 (L2)",
            "test": "one-sided paired t, capped index, trunk and own scoring; BCa 95 %; D-19 bound",
            "margin": f"MEI {MEI}; practical bound 0.10",
            "unit": "qualified trunk",
            "n": h1["n"],
            "sd": h1["sd"],
            "mde": mde(h1["sd"], h1["n"]),
            "multiplicity": "none (single pre-specified lever)",
            "registration": "registered (D-3', D-16, D-17, D-19); A-003 L2 rule",
            "data_seen": seen(["A-003"], "stage-2 hash (A-001) before any lever contrast"),
        }
    )
    # H1-R family
    for lev in FAMILY:
        if lev == "L2":
            continue
        r = _paired(st, lev)
        rows.append(
            {
                "claim": "C3b",
                "hypothesis": f"H1-R ({lev})",
                "test": "one-sided paired t, capped index, Holm over 8 levers",
                "margin": f"MEI {MEI}",
                "unit": "qualified trunk",
                "n": r["n"],
                "sd": r["sd"],
                "mde": mde(r["sd"], r["n"], ALPHA / len(FAMILY)),
                "multiplicity": "Holm m = 8 (MDE at alpha / 8, worst case)",
                "registration": "registered (SAP s3)"
                + (" - never built, p = 1 placeholder" if r["n"] == 0 else ""),
                "data_seen": "stage-2 hash (A-001) before any lever contrast",
            }
        )
    # C4 nulls (TOST)
    for lev in FAMILY:
        r = _paired(st, lev)
        if not r["n"]:
            continue
        half = None
        if r["sd"] is not None and r["n"] > 1:
            half = float(sps.t.ppf(1 - ALPHA, r["n"] - 1) * r["sd"] / math.sqrt(r["n"]))
        rows.append(
            {
                "claim": "C4",
                "hypothesis": f"null equivalence ({lev})",
                "test": "TOST on the capped paired difference (90 % CI inside the margins); DEGENERATE VARIANCE: the capped differences are exactly 0 in most trunks, so this is the registered output, not evidence of closeness",
                "margin": f"+/-{TOST_MARGIN}",
                "unit": "qualified trunk",
                "n": r["n"],
                "sd": r["sd"],
                "mde": half,
                "multiplicity": "none (descriptive equivalence; Holm family is C3b)",
                "registration": "registered (SAP s3); raw-index TOST added as reporting (A-022)",
                "data_seen": seen(["A-022"], "margin registered before data"),
            }
        )
    # replication lanes
    for cid, hyp, doc, lev, reg in (
        ("C3m", "H1-M", st_m, "L2_M", "registered (D-23); model switched to Llama-3.3-70B (A-005)"),
        ("C-S", "H-S", st_s, "L2_S", "registered (D-23)"),
    ):
        if doc is None:
            continue
        r = _paired(doc, lev)
        rows.append(
            {
                "claim": cid,
                "hypothesis": hyp,
                "test": "one-sided paired t, capped index",
                "margin": f"MEI {MEI}",
                "unit": "qualified trunk",
                "n": r["n"],
                "sd": r["sd"],
                "mde": mde(r["sd"], r["n"]),
                "multiplicity": "none (secondary, outside the H1-R family)",
                "registration": reg,
                "data_seen": seen(
                    ["A-005"] if cid == "C3m" else [], "registered before the lane's lever data"
                ),
            }
        )
    # family S
    names = {
        "H2": ("C2", "Spearman within seed groups (permutation) + Welch s1 - s4"),
        "H4": ("C3b", "one-sided t vs MEI (L3)"),
        "H5": ("C5", "two one-sided |d| test"),
        "H6-carrier": ("C5", "one-sided t on D5-net fall; ghost-null guard"),
        "H7-same": ("C6", "max(TOST, sign test on recovery <= 20 periods)"),
        "H7-other": ("C6", "one-sided t vs 0.15 (R3)"),
        "H8": ("C7", "one-sided follow-rate difference, cluster-robust SE, t with K-1 df"),
        "H10": ("C9", "TOST +/-0.15"),
    }
    for h, v in ((fam_s or {}).get("hypotheses") or {}).items():
        cid, test = names.get(h, ("?", "?"))
        t = v.get("test") or {}
        n = t.get("n") or v.get("n_gradient_runs") or t.get("df")
        rows.append(
            {
                "claim": cid,
                "hypothesis": h,
                "test": test,
                "margin": f"MEI {v.get('mei')}" if v.get("mei") is not None else "",
                "unit": "trunk / run (see test)",
                "n": n,
                "sd": t.get("sd"),
                "mde": None,
                "multiplicity": "Holm m = 8 (family S)",
                "registration": "registered (family S); " + v.get("status", ""),
                "data_seen": seen(
                    ["A-010", "A-018"] if h in ("H7-same", "H8") else [],
                    "family S rules frozen with stage 2 (A-001)",
                ),
            }
        )
    # detection
    if det:
        rows.append(
            {
                "claim": "C8",
                "hypothesis": "H9",
                "test": "LLM-only AUC, FPR at the frozen threshold, seed-group clusters",
                "margin": "AUC >= 0.85; FPR <= 5 % (ghost)",
                "unit": "window (seed-group clusters)",
                "n": det.get("n_eval"),
                "sd": None,
                "mde": None,
                "multiplicity": "none (single frozen rule set, one evaluation)",
                "registration": "registered (DETECTION_PROTOCOL, D-21); A-013/A-014 pipeline",
                "data_seen": seen(
                    ["A-013", "A-014"], "rules frozen on the dev half before the eval half"
                ),
            }
        )
    # A-019 entrant type, exploratory
    for lev in ("L6", "L6b", "L6c"):
        r = _paired(st, lev)
        if not r["n"]:
            continue
        rows.append(
            {
                "claim": "C11",
                "hypothesis": f"entrant type ({lev})",
                "test": "descriptive paired contrast with BCa 95 % (no test)",
                "margin": "",
                "unit": "qualified trunk",
                "n": r["n"],
                "sd": r["sd"],
                "mde": mde(r["sd"], r["n"]),
                "multiplicity": "none (exploratory)",
                "registration": "exploratory (A-019)" if lev != "L6" else "registered (H1-R lever)",
                "data_seen": seen(
                    ["A-019"], "decided after the registered reads" if lev != "L6" else "stage 2"
                ),
            }
        )
    return rows


def _by_arm(st: dict[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for r in st.get("runs", []):
        out.setdefault(r["registered_arm"], {})[str(r["trunk_id"])] = r
    return out


def threshold_sensitivity(st: dict[str, Any], levers: tuple[str, ...] = ("L2", "L6")) -> list[dict]:
    """Qualification at 0.25 / 0.30 / 0.35 (Delta_p AND Delta_pi over 71-100 on the pinned trunk
    benchmarks, D-16) and the paired capped contrasts of ``levers`` on each qualified set."""
    arms = _by_arm(st)
    trunks = arms.get("trunk", {})
    out = []
    for thr in (0.25, 0.30, 0.35):
        q = sorted(
            t
            for t, r in trunks.items()
            if r.get("delta_p") is not None
            and r.get("delta_pi") is not None
            and r["delta_p"] >= thr
            and r["delta_pi"] >= thr
        )
        row: dict[str, Any] = {"threshold": thr, "n_qualified": len(q)}
        for lev in levers:
            d = [
                arms["sham"][t]["delta_p"] - arms[lev][t]["delta_p"]
                for t in q
                if t in arms.get("sham", {}) and t in arms.get(lev, {})
            ]
            if len(d) >= 2:
                x = np.asarray(d, dtype=float)
                tt = sps.ttest_1samp(x, 0.0, alternative="greater")
                row[lev] = {"n": len(d), "mean": float(x.mean()), "p_one_sided": float(tt.pvalue)}
            else:
                row[lev] = {"n": len(d)}
        out.append(row)
    return out


def raw_mde(st: dict[str, Any]) -> dict[str, Any]:
    """MDE on the RAW index at the realised raw sham-sham2 SD (the noise floor of a null lever)
    and the number of qualified trunks, one-sided alpha 0.05, 80 % power."""
    c = st.get("contrasts", {}).get("sham2", {}).get("delta_p_raw") or {}
    sd, n = c.get("sd"), int(c.get("n") or 0)
    return {"sd_sham_sham2_raw": sd, "n": n, "mde80": mde(sd, n), "alpha": ALPHA, "power": POWER}


def jackknife(st: dict[str, Any], levers: tuple[str, ...] = ("L2", "L6")) -> dict[str, Any]:
    """Leave-one-trunk-out range of the paired capped contrast (trunk clustering check)."""
    out = {}
    for lev in levers:
        d = (st.get("contrasts", {}).get(lev, {}).get("delta_p") or {}).get("diffs") or {}
        vals = np.asarray(list(d.values()), dtype=float)
        if vals.size < 3:
            continue
        means, ps = [], []
        for i in range(vals.size):
            x = np.delete(vals, i)
            means.append(float(x.mean()))
            ps.append(float(sps.ttest_1samp(x, 0.0, alternative="greater").pvalue))
        out[lev] = {
            "n": int(vals.size),
            "mean_full": float(vals.mean()),
            "mean_min": min(means),
            "mean_max": max(means),
            "p_max": max(ps),
            "p_min": min(ps),
        }
    return out


def _fmt(v: Any, d: int = 3) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        if not math.isfinite(v):
            return "-"
        return f"{v:.2e}" if (v != 0 and abs(v) < 10 ** (-d)) else f"{v:.{d}f}"
    return str(v)


def render(doc: dict[str, Any]) -> str:
    cols = [
        "claim",
        "hypothesis",
        "test",
        "margin",
        "unit",
        "n",
        "sd",
        "mde",
        "multiplicity",
        "registration",
        "data_seen",
    ]
    lines = ["# A6 Statistical coherence", "", "## Claims", ""]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "---|" * len(cols))
    for r in doc["claims"]:
        lines.append("| " + " | ".join(_fmt(r.get(c)) for c in cols) + " |")
    lines += [
        "",
        "MDE: one-sided paired-t minimum detectable effect at 80 % power and the realised SD "
        "(alpha / 8 for the Holm family); for C4 the column holds the half-width of the 90 % "
        "equivalence interval.",
        "",
    ]
    lines += [
        "## Sensitivity: qualification threshold",
        "",
        "| threshold | n qualified | L2 mean (p) | L6 mean (p) |",
        "|---|---|---|---|",
    ]
    for r in doc["threshold_sensitivity"]:
        cells = [_fmt(r["threshold"], 2), str(r["n_qualified"])]
        for lev in ("L2", "L6"):
            x = r.get(lev, {})
            cells.append(
                f"{_fmt(x.get('mean'))} ({_fmt(x.get('p_one_sided'))})" if "mean" in x else "-"
            )
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Sensitivity: trunk clustering (leave one trunk out)",
        "",
        "| lever | n | mean | min-max over drops | max p |",
        "|---|---|---|---|---|",
    ]
    for lev, r in doc["jackknife"].items():
        lines.append(
            f"| {lev} | {r['n']} | {_fmt(r['mean_full'])} | {_fmt(r['mean_min'])} - {_fmt(r['mean_max'])} | {_fmt(r['p_max'])} |"
        )
    lines += [
        "",
        "Every paired test uses one difference per trunk (the trunk is the unit); the levers "
        "share each trunk's sham, so the H1-R p-values are positively dependent, which Holm "
        "tolerates (valid under arbitrary dependence).",
        "",
    ]
    lines += [
        "## TOST margin against the MEI and the ceiling (degenerate variance)",
        "",
        "| lever | capped TOST p | raw TOST p | raw mean | quotes > p_M sham -> lever |",
        "|---|---|---|---|---|",
    ]
    for r in doc["tost"]:
        lines.append(
            f"| {r['lever']} | {_fmt(r['tost_p'])} | {_fmt(r['raw_tost_p'])} | {_fmt(r['raw_mean'])} | "
            f"{_fmt(r['share_ref'], 2)} -> {_fmt(r['share_arm'], 2)} |"
        )
    lines += ["", doc["tost_note"], ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-root", type=Path, default=Path("results/analysis"))
    ap.add_argument(
        "--amendments", type=Path, default=Path("docs/03_preregistration/AMENDMENTS.md")
    )
    args = ap.parse_args(argv)
    o = args.out_root
    st = _load(o / "TR-H" / "stats.json")
    if st is None:
        print(f"error: {o / 'TR-H' / 'stats.json'} not found")
        return 2
    doc: dict[str, Any] = {
        "claims": claim_rows(
            st,
            _load(o / "family_s" / "family_s.json"),
            _load(o / "TR-M" / "stats.json"),
            _load(o / "TR-S" / "stats.json"),
            _load(o / "gates" / "gates.json"),
            _load(o / "detection" / "report.json"),
            amendments(args.amendments),
        ),
        "raw_mde": raw_mde(st),
        "threshold_sensitivity": threshold_sensitivity(st),
        "jackknife": jackknife(st),
        "tost": [{"lever": lev, **_paired(st, lev)} for lev in FAMILY if _paired(st, lev)["n"]],
        "tost_note": (
            f"The equivalence margin +/-{TOST_MARGIN} is tighter than the minimum effect of interest "
            f"({MEI}): an equivalent lever is ruled out from moving the capped index by an effect "
            "as large as the smallest one that matters. The capped index saturates at 1 when quotes "
            "sit above the monopoly price (most sham quotes do), so capped equivalence means no "
            "movement of the capped index, not necessarily no price change above p_M; the raw-index "
            "TOST and the share of quotes above p_M (A-022) show which levers also leave the uncapped "
            "prices unchanged."
        ),
        "inputs": {
            k: str(o / p)
            for k, p in (
                ("trunk_stats", "TR-H/stats.json"),
                ("family_s", "family_s/family_s.json"),
                ("gates", "gates/gates.json"),
                ("detection", "detection/report.json"),
            )
        },
    }
    (o / "a6_coherence.json").write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
    (o / "a6_coherence.md").write_text(render(doc), encoding="utf-8")
    print(f"-> {o / 'a6_coherence.md'}, {o / 'a6_coherence.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
