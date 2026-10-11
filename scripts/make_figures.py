"""Produce the paper figures (PNG 200 dpi + PDF + JSON sidecar with run ids).

Usage::

    uv run python scripts/make_figures.py results/campaigns/<campaign> [--out DIR]
        [--stats results/analysis/<campaign>/stats.json] [--mqc mqc.json]
        [--detection detection_report.json] [--gradient-arms trunk,GR-s2,GR-s3,GR-s4]
        [--ablation-arms sham,CA --t0 100] [--pc-arms PC-H,PC-H-dealer,PC-M,PC-S]
    uv run python scripts/make_figures.py --demo OUT_DIR     # synthetic preview of all figures

Fig. 1 defaults to the instrument arms (Gate 1 / 1b positive controls) present in the
campaign and shows Delta_p (Gate 1 lines) and Delta_pi (published band) on separate panels;
the AC-H anchoring median is a marker when the campaign holds AC-H runs. Fig. 2 draws the
Fish et al. (P1) reference as a Delta_p mark (their Delta_pi belongs in the caption; both
values sit in the sidecar) and distinguishes the pooled TR-H s1 (trunk role, square) from
the CRN-paired GR-H levels (circles). Fig. 3 plots the total capped Delta C_p under the
trunk scoring with the mechanical part and the own-scoring (D-17) marker; the sham band is
the SD over trunks of Delta_p(sham) - Delta_p(sham2). Fig. 4 series are the per-period
capped price index against the trunk benchmarks.

``mqc.json`` maps an arm name to its market-quality cost in pp (from the scripted SC1
market). ``detection_report.json`` holds ``{"curves": {label: {"fpr": [...], "tpr": [...],
"auc": x}}}``. Figures whose inputs are missing are skipped with a message.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

INSTRUMENT_ARMS = ("PC-H", "PC-H-dealer", "PC-M", "PC-S")
ANCHORING_ARM = "AC-H"
FISH_P1_DELTA_P = 0.70  # Fish et al. (2024), P1 prompt: price index (the Fig. 2 mark)
FISH_P1_DELTA_PI = 0.43  # the same runs' profit index (caption only, never on the Delta_p axis)
# The Fish et al. values are UNVERIFIED (lead ruling Q7): no reference is drawn unless
# --fish-reference / --fish-band is passed, and then it is labelled unverified, informational.
GRADIENT_REFERENCES: dict[str, tuple[float, float]] = {}
FISH_REFERENCES: dict[str, tuple[float, float]] = {
    "Fish et al. (P1), unverified": (FISH_P1_DELTA_P, FISH_P1_DELTA_P),
}
FISH_BAND = (0.35, 0.95)
GRADIENT_NOTES = {
    "markers": "square = pooled TR-H s1 (trunk role); circle = CRN-paired GR-H levels",
}

BenchOf = Callable[[dict[str, Any]], "tuple[Any, float] | None"]


def _arm_of(r: dict[str, Any]) -> str:
    return str(r.get("registered_arm", r.get("arm")))


def _nanmean_rows(m: np.ndarray) -> np.ndarray:
    cnt = np.isfinite(m).sum(axis=1)
    tot = np.where(np.isfinite(m), m, 0.0).sum(axis=1)
    return np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)


def _bench(st: dict[str, Any], arm: str) -> Any:
    """Pre-lever (trunk) benchmarks of ``arm`` from ``stats.json`` (pinned values)."""
    from finswarm.analysis.arms import benchmarks_from_entry

    entry = st.get("benchmarks", {}).get(arm)
    if not entry:
        return None
    return benchmarks_from_entry({"benchmarks": entry["primary"], "model": {}})


def _run_market(r: dict[str, Any]) -> Any:
    """``MarketModel`` of a run's own resolved market (from its manifest), or None."""
    from finswarm.analysis.io import read_manifest
    from finswarm.metrics.benchmarks import MarketModel

    market = (read_manifest(Path(r["path"])).get("resolved_config") or {}).get("market")
    if not market:
        return None
    members = r.get("members")
    try:
        return MarketModel.from_config(market, len(members) if isinstance(members, list) else None)
    except Exception:  # noqa: BLE001 - the run is skipped with a message by the caller
        return None


def _bench_of(st: dict[str, Any], r: dict[str, Any]) -> tuple[Any, float] | None:
    """``(benchmarks, alpha)`` of a run: the pinned trunk benchmarks of its arm when
    ``stats.json`` holds them, else the closed form of the run's own market (instrument arms,
    ``benchmarks_mode: run_closed_form``); ``alpha`` normalises the price paths."""
    model = _run_market(r)
    alpha = float(model.params.alpha) if model is not None else 1.0
    b = _bench(st, _arm_of(r))
    if b is not None:
        return b, alpha
    if model is None:
        return None
    return model.benchmarks(), alpha


def _paths(
    runs: list[dict[str, Any]], arm: str, bench_of: BenchOf, delta: bool = False
) -> tuple[np.ndarray, list[str], tuple[float, float] | None, np.ndarray | None]:
    """Seeds x periods matrix for every run of ``arm`` (truncated to the shortest run): the
    mean posted price of the whitelisted incumbents normalised by alpha, or with ``delta``
    the per-period CAPPED price index ``min(Delta_p,t, 1)`` against the run's trunk
    benchmarks. Also returns the run ids, ``(p_nash, p_mono) / alpha`` of the first run (the
    reference lines of Fig. 1) and the first run's period index."""
    import polars as pl

    from finswarm.analysis.io import read_table
    from finswarm.metrics.collusion import incumbent_matrix

    paths, ids, lines, periods = [], [], None, None
    for r in runs:
        if _arm_of(r) != arm:
            continue
        bb = bench_of(r)
        if bb is None:
            print(f"skip {r['run_id']}: no benchmarks (neither pinned nor a resolved market)")
            continue
        bench, alpha = bb
        ap = read_table(r["path"], "agent_period").with_columns(pl.lit(r["run_id"]).alias("run_id"))
        members = r.get("members")
        if isinstance(members, list) and members and "slot" in ap.columns:
            idx, m = incumbent_matrix(ap, members, key="slot", strict=False)
            pbar = _nanmean_rows(m)
        else:
            per = ap.group_by("period").agg(pl.col("price").mean().alias("p")).sort("period")
            idx, pbar = per["period"].to_numpy(), per["p"].to_numpy()
        series = np.minimum(np.asarray(bench.delta_price(pbar)), 1.0) if delta else pbar / alpha
        paths.append(np.asarray(series, dtype=float))
        ids.append(r["run_id"])
        if lines is None:
            lines = (float(bench.p_nash) / alpha, float(bench.p_mono) / alpha)
            periods = np.asarray(idx)
    if not paths:
        return np.empty((0, 0)), ids, None, None
    t = min(len(p) for p in paths)
    return np.stack([p[:t] for p in paths]), ids, lines, periods


def _trunk_arms(arms_path: Path | None) -> set[str]:
    """The lane trunk arms of the registry (the pooled s1 level of Fig. 2); empty when the
    registry is not readable, so every level falls back to the circle marker."""
    if arms_path is None or not Path(arms_path).exists():
        return set()
    from finswarm.analysis.arms import load_registered_arms

    try:
        spec = load_registered_arms(arms_path)
    except Exception:  # noqa: BLE001 - a figure never fails on the registry
        return set()
    return {lane.trunk for lane in spec.lanes.values()}


def load_stats(paths: list[Path]) -> dict[str, Any]:
    """Several ``stats.json`` merged: runs (by run id), arms and benchmarks are unions, every
    other block (contrasts, h1, h1r, mqc, ...) comes from the first file that has it."""
    docs = [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths]
    st: dict[str, Any] = {}
    for d in docs:
        for k, v in d.items():
            if k not in st:
                st[k] = v
    seen: set[str] = set()
    runs: list[dict[str, Any]] = []
    for d in docs:
        for r in d.get("runs", []):
            if str(r.get("run_id")) not in seen:
                seen.add(str(r.get("run_id")))
                runs.append(r)
    st["runs"] = runs
    for key in ("arms", "benchmarks"):
        merged: dict[str, Any] = {}
        for d in docs:
            for a, v in (d.get(key) or {}).items():
                merged.setdefault(a, v)
        st[key] = merged
    return st


def _pool(runs: list[dict[str, Any]], token: str) -> list[dict[str, Any]]:
    """Runs of a Fig. 2 level: ``arm`` or ``arm1+arm2`` (pooled, e.g. trunk+GR-s1)."""
    arms = set(token.split("+"))
    return [r for r in runs if _arm_of(r) in arms]


def from_campaign(args: argparse.Namespace) -> list[Path]:
    """Figures from one or several ``stats.json`` (registered-arm layout of
    ``analysis.campaign``)."""
    import hashlib

    from finswarm.analysis import figures

    stats_paths = list(
        args.stats or [Path("results/analysis") / args.campaign_dir.name / "stats.json"]
    )
    missing = [p for p in stats_paths if not Path(p).exists()]
    if missing:
        print(f"error: {missing[0]} not found; run scripts/analyze.py first", file=sys.stderr)
        return []
    st = load_stats(stats_paths)
    runs = st["runs"]
    figures.set_provenance(
        role=args.role,
        inputs=[
            {"path": str(p), "sha256": hashlib.sha256(Path(p).read_bytes()).hexdigest()}
            for p in stats_paths
        ],
    )
    gate_ci: dict[str, Any] = {}
    if args.gates and Path(args.gates).exists():
        rep = json.loads(Path(args.gates).read_text(encoding="utf-8"))
        for label, g in (rep.get("gates") or {}).items():
            if isinstance(g, dict) and isinstance(g.get("delta_p"), dict):
                gate_ci[label] = g["delta_p"]
    out = args.out or Path("results/figures") / args.campaign_dir.name
    written: list[Path] = []

    def bench_of(r: dict[str, Any]) -> tuple[Any, float] | None:
        return _bench_of(st, r)

    # Fig. 1: instrument arms (Gate 1 / 1b), Delta_p and Delta_pi on separate panels
    arms_seen = sorted(st.get("arms") or {_arm_of(r) for r in runs})
    default_pc = [a for a in INSTRUMENT_ARMS if a in arms_seen] or arms_seen[:4]
    pc_arms = args.pc_arms.split(",") if args.pc_arms else default_pc
    paths: dict[str, np.ndarray] = {}
    deltas: dict[str, list[float]] = {}
    deltas_pi: dict[str, list[float]] = {}
    ids: list[str] = []
    lines, periods = None, None
    for arm in pc_arms:
        m, rid, ln, per = _paths(runs, arm, bench_of)
        if m.size:
            rows = [r for r in runs if _arm_of(r) == arm]
            paths[arm] = m
            deltas[arm] = [float(r["delta_p"]) for r in rows]
            deltas_pi[arm] = [float(r["delta_pi"]) for r in rows if r.get("delta_pi") is not None]
            ids += rid
            if lines is None:
                lines, periods = ln, per
    ac = [float(r["delta_p"]) for r in runs if _arm_of(r) == ANCHORING_ARM]
    anchoring = float(np.median(ac)) if ac else None
    if paths and lines is not None:
        written += figures.plot_positive_control(
            paths,
            deltas,
            lines[0],
            lines[1],
            out / "fig1_positive_control",
            ids,
            periods=periods,
            delta_pi_by_seed=deltas_pi if any(deltas_pi.values()) else None,
            anchoring=anchoring,
            published_band=FISH_BAND if args.fish_band else None,
            gate_ci=gate_ci,
        )
    else:
        print("skip fig1: no runs (or no benchmarks) for the positive-control arms")

    # Fig. 2: structural gradient in Delta_p; the trunk role (pooled s1) gets a square
    if args.gradient_arms:
        levels = args.gradient_arms.split(",")
        vals = [[float(r["delta_p"]) for r in _pool(runs, lv)] for lv in levels]
        gids = [r["run_id"] for lv in levels for r in _pool(runs, lv)]
        pooled = _trunk_arms(args.arms)
        markers = ["s" if (set(lv.split("+")) & pooled) else "o" for lv in levels]
        written += figures.plot_structural_gradient(
            levels,
            vals,
            out / "fig2_gradient",
            gids,
            references=FISH_REFERENCES if args.fish_reference else GRADIENT_REFERENCES,
            markers=markers,
            notes=GRADIENT_NOTES,
        )

    # Fig. 3: frontier in total capped Delta C_p (trunk scoring) with the D-17 marker
    mqc = json.loads(Path(args.mqc).read_text(encoding="utf-8")) if args.mqc else None
    pts, band, fids, not_run = figures.frontier_from_stats(st, mqc)
    if pts:
        written += figures.plot_frontier(
            pts,
            out / "fig3_frontier",
            fids,
            sham_band=band,
            mqc_max=float((st.get("mqc") or {}).get("max_pp") or 5.0),
            not_run=not_run,
        )
    else:
        print("skip fig3: needs paired lever contrasts and MQC estimates (stats.json or --mqc)")

    # Fig. 4: per-period capped Delta_p around the branch period
    if args.ablation_arms and args.t0 is not None:
        from finswarm.analysis.figures import mean_ci

        series = {}
        for arm in args.ablation_arms.split(","):
            m, _, _, per = _paths(runs, arm, bench_of, delta=True)
            if not m.size or per is None:
                continue
            per = np.asarray(per)[: m.shape[1]]
            keep = per >= args.t0 - 20
            if keep.any():
                mm, lo, hi = mean_ci(m[:, keep])
                series[arm] = (per[keep], mm, lo, hi)
        if series:
            written += figures.plot_action_carrier_ablation(
                series, args.t0, out / "fig4_action_carrier", [r["run_id"] for r in runs]
            )
        else:
            print("skip fig4: no runs with benchmarks for the ablation arms")

    if args.detection:
        rep = json.loads(Path(args.detection).read_text(encoding="utf-8"))
        curves = {
            k: (np.asarray(v["fpr"]), np.asarray(v["tpr"]), float(v["auc"]))
            for k, v in rep.get("curves", {}).items()
        }
        if curves:
            written += figures.plot_detector_roc(
                curves, out / "fig5_detector_roc", rep.get("run_ids", [])
            )
    return written


def demo(out: Path) -> list[Path]:
    """Synthetic inputs for every figure (layout preview and smoke test)."""
    from finswarm.analysis import figures
    from finswarm.metrics.benchmarks import logit_closed_form
    from finswarm.metrics.detection import auc, roc_curve

    rng = np.random.default_rng(0)
    b = logit_closed_form()
    t = np.arange(1, 301)
    target = {"PC-H": 1.80, "PC-M": 1.70}
    paths = {
        arm: b.p_nash
        + (p - b.p_nash) * (1 - np.exp(-t / 60))[None, :]
        + rng.normal(0, 0.04, (n, t.size))
        for (arm, p), n in zip(target.items(), (5, 8), strict=True)
    }
    deltas = {
        arm: rng.normal(0.6 if arm == "PC-H" else 0.45, 0.15, n)
        for arm, n in (("PC-H", 5), ("PC-M", 8))
    }
    deltas_pi = {arm: v * 0.8 + rng.normal(0, 0.05, v.size) for arm, v in deltas.items()}
    ids = [f"demo-{i}" for i in range(13)]
    w = figures.plot_positive_control(
        paths,
        deltas,
        b.p_nash,
        b.p_mono,
        out / "fig1_positive_control",
        ids,
        periods=t,
        delta_pi_by_seed=deltas_pi,
        anchoring=0.92,
    )
    lv = ["trunk (s1)", "GR-s2", "GR-s3", "GR-s4"]
    w += figures.plot_structural_gradient(
        lv,
        [rng.normal(m, 0.12, 6) for m in (0.55, 0.40, 0.22, 0.08)],
        out / "fig2_gradient",
        ids,
        references=GRADIENT_REFERENCES,
        markers=["s", "o", "o", "o"],
        notes=GRADIENT_NOTES,
    )
    ca = "price reduction consistent with competitive adjustment"
    pts = [
        {
            "label": "L1 routing",
            "mqc": 1.2,
            "dc": 0.32,
            "lo": 0.18,
            "hi": 0.45,
            "mechanical": 0.05,
            "behavioural": 0.27,
            "own": 0.30,
            "own_lo": 0.15,
            "own_hi": 0.44,
            "dual_label": "collusion reduction",
        },
        {
            "label": "L2 fringe",
            "mqc": 2.5,
            "dc": 0.41,
            "lo": 0.27,
            "hi": 0.55,
            "own": 0.38,
            "own_lo": 0.22,
            "own_hi": 0.52,
            "dual_label": "collusion reduction",
            "primary": True,
        },
        {
            "label": "L3 tick",
            "mqc": 0.4,
            "dc": 0.12,
            "lo": -0.02,
            "hi": 0.26,
            "mechanical": 0.10,
            "behavioural": 0.02,
            "own": -0.01,
            "own_lo": -0.12,
            "own_hi": 0.10,
            "dual_label": ca,
        },
        {"label": "L6 sting", "mqc": 6.5, "dc": 0.20, "lo": 0.05, "hi": 0.35},
        {"label": "L2 (M)", "mqc": 2.5, "dc": 0.36, "lo": 0.15, "hi": 0.57, "replication": True},
    ]
    w += figures.plot_frontier(
        pts, out / "fig3_frontier", ids, sham_band=0.08, not_run=["L4a", "L5", "L7", "L8"]
    )
    tt = np.arange(80, 150)
    series = {}
    for k, (lab, drop) in enumerate((("sham", 0.0), ("CA (anonymised)", 0.35))):
        m = 0.55 - drop * (tt >= 100) * (1 - np.exp(-(tt - 100).clip(0) / 8)) + 0.01 * k
        series[lab] = (tt, m, m - 0.1, m + 0.1)
    w += figures.plot_action_carrier_ablation(
        series,
        100,
        out / "fig4_action_carrier",
        ids,
        dependence={"sham": (0.12, 0.06, 0.18), "CA": (0.05, 0.0, 0.10)},
    )
    y = np.r_[np.ones(40), np.zeros(60)]
    curves = {}
    for lab, sep in (("composite", 2.0), ("D1 level", 1.0), ("D7 non-BR", 1.5)):
        s = rng.normal(0, 1, 100) + sep * y
        fpr, tpr = roc_curve(s, y)
        curves[lab] = (fpr, tpr, auc(s, y))
    w += figures.plot_detector_roc(curves, out / "fig5_detector_roc", ids)
    return w


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("campaign_dir", nargs="?", type=Path)
    ap.add_argument("--demo", type=Path, default=None, metavar="OUT_DIR")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument(
        "--stats",
        type=Path,
        nargs="+",
        default=None,
        help="one or several stats.json (runs, arms, benchmarks merged; e.g. TR-H PC-H PC-S AC-H)",
    )
    ap.add_argument("--gates", type=Path, default=None, help="gates.json (gate CIs to the sidecar)")
    ap.add_argument("--role", default="confirmatory", help="role label written to the sidecars")
    ap.add_argument("--fish-band", action="store_true", help="draw the unverified Fish band")
    ap.add_argument(
        "--fish-reference", action="store_true", help="draw the unverified Fish reference mark"
    )
    ap.add_argument("--mqc", type=Path, default=None)
    ap.add_argument("--detection", type=Path, default=None)
    ap.add_argument(
        "--pc-arms",
        default=None,
        help="Fig. 1 arms (default: the instrument arms PC-H, PC-H-dealer, PC-M, PC-S present)",
    )
    ap.add_argument("--gradient-arms", default=None)
    ap.add_argument(
        "--arms",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "configs"
        / "analysis"
        / "registered_arms.yaml",
        help="registered arms (lane trunks = the pooled s1 level of Fig. 2)",
    )
    ap.add_argument("--ablation-arms", default=None)
    ap.add_argument("--t0", type=int, default=None)
    args = ap.parse_args(argv)
    if args.demo:
        written = demo(args.demo)
    elif args.campaign_dir:
        written = from_campaign(args)
    else:
        ap.error("give a campaign directory or --demo OUT_DIR")
    for p in written:
        print(p)
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
