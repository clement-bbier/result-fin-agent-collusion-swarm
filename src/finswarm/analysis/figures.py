"""Paper figures (D.5). Object-oriented matplotlib (no pyplot state), ``finswarm.mplstyle``.

Conventions: colour-blind-safe categorical order (validated), CI bands or error bars on
every estimate, no in-figure titles (captions live in the paper), PNG at 200 dpi plus PDF,
and a JSON sidecar listing the run ids and the plotted numbers.
"""

from __future__ import annotations

import functools
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib import style as mpl_style
from matplotlib.figure import Figure
from scipy import stats

STYLE = Path(__file__).with_name("finswarm.mplstyle")
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK = {
    "primary": "#0b0b0b",
    "secondary": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "band": "#f0efec",
}


def styled(fn: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with mpl_style.context(str(STYLE)):
            return fn(*args, **kwargs)

    return wrapper


def mean_ci(x: np.ndarray, level: float = 0.95, axis: int = 0) -> tuple[np.ndarray, ...]:
    """Mean and t-based CI along ``axis`` (NaN-aware)."""
    a = np.asarray(x, dtype=float)
    n = np.sum(np.isfinite(a), axis=axis)
    m = np.nanmean(a, axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        se = np.nanstd(a, axis=axis, ddof=1) / np.sqrt(n)
        q = stats.t.ppf(0.5 + level / 2, np.maximum(n - 1, 1))
    half = np.where(n > 1, q * se, 0.0)
    return m, m - half, m + half


def _jsonable(o: Any) -> Any:
    if isinstance(o, np.ndarray):
        return [_jsonable(v) for v in o.tolist()]
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, list | tuple):
        return [_jsonable(v) for v in o]
    if isinstance(o, float | np.floating):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, np.integer):
        return int(o)
    return o


#: provenance written into every sidecar (A-011): set once per run by ``make_figures``
PROVENANCE: dict[str, Any] = {}


def set_provenance(**kw: Any) -> None:
    """Role (confirmatory / exploratory), input files with sha256, ... for the sidecars."""
    PROVENANCE.clear()
    PROVENANCE.update(kw)


def _code_sha() -> str | None:
    import subprocess

    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10, check=True
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def save_figure(
    fig: Figure,
    stem: str | Path,
    run_ids: Sequence[str],
    data: Mapping[str, Any] | None = None,
    kind: str = "",
) -> list[Path]:
    """Write ``stem.png`` (200 dpi), ``stem.pdf`` and the ``stem.json`` sidecar."""
    s = Path(stem)
    s.parent.mkdir(parents=True, exist_ok=True)
    png, pdf, side = s.with_suffix(".png"), s.with_suffix(".pdf"), s.with_suffix(".json")
    fig.savefig(png, dpi=200)
    fig.savefig(pdf)
    side.write_text(
        json.dumps(
            {
                "figure": kind or s.name,
                "code_sha": _code_sha(),
                **_jsonable(dict(PROVENANCE)),
                "run_ids": sorted(set(map(str, run_ids))),
                "data": _jsonable(dict(data or {})),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return [png, pdf, side]


def _hline(ax: Any, y: float, label: str, x_text: float = 1.0) -> None:
    ax.axhline(y, color=INK["muted"], lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.annotate(
        label,
        xy=(x_text, y),
        xycoords=("axes fraction", "data"),
        xytext=(-2, 3),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=7.5,
        color=INK["secondary"],
    )


def _seed_panel(
    sx: Any,
    by_seed: Mapping[str, Sequence[float]],
    data: dict[str, Any],
    prefix: str,
    ylabel: str,
) -> None:
    rng = np.random.default_rng(0)
    for k, (lane, d) in enumerate(by_seed.items()):
        v = np.asarray(d, dtype=float)
        col = PALETTE[k % len(PALETTE)]
        sx.scatter(
            k + rng.uniform(-0.15, 0.15, v.size),
            v,
            s=14,
            color=col,
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
        if v.size:
            m, lo, hi = mean_ci(v)
            sx.errorbar(
                k + 0.32,
                m,
                yerr=[[m - lo], [hi - m]],
                fmt="o",
                ms=4,
                color=col,
                elinewidth=1.2,
                capsize=0,
                zorder=4,
            )
        data[f"{prefix}{lane}"] = v
    sx.axhline(0, color=INK["axis"], lw=0.6)
    sx.set_xticks(range(len(by_seed)), list(by_seed), rotation=30, ha="right")
    sx.set_xlim(-0.6, len(by_seed) - 0.2)
    sx.set_ylabel(ylabel)


@styled
def plot_positive_control(
    price_paths: Mapping[str, np.ndarray],
    delta_by_seed: Mapping[str, Sequence[float]],
    p_nash: float,
    p_mono: float,
    stem: str | Path,
    run_ids: Sequence[str],
    periods: np.ndarray | None = None,
    published_band: tuple[float, float] | None = None,
    delta_pi_by_seed: Mapping[str, Sequence[float]] | None = None,
    thresholds: tuple[float, float] = (0.25, 0.10),
    anchoring: float | None = None,
    gate_ci: Mapping[str, Any] | None = None,
    ylim: tuple[float, float] | None = None,
) -> list[Path]:
    """Fig. 1: mean price path per instrument arm (95 % CI band across seeds) with Nash and
    monopoly lines; middle panel: the price index Delta_p per seed with the Gate 1 lines
    (point estimate >= ``thresholds[0]``, share of runs above ``thresholds[1]``) and the AC-H
    anchoring median as a marker when given; right panel: Delta_pi per seed with the
    Fish band shaded ONLY when ``published_band`` is given (default off: the Fish values are
    unverified, the band is informational and labelled so). The price panel is clipped to
    ``ylim`` (default: Nash - 0.2 .. monopoly + 0.3) with the opening noted; ``gate_ci`` (the
    alpha-stratified run bootstrap of the gate rule) goes to the sidecar. The two indices are
    never drawn on one axis."""
    with_pi = delta_pi_by_seed is not None
    fig = Figure(figsize=(7.6 if with_pi else 6.4, 3.2))
    axes = fig.subplots(
        1, 3 if with_pi else 2, gridspec_kw={"width_ratios": [3.0, 1, 1] if with_pi else [3.4, 1]}
    )
    ax, sx = axes[0], axes[1]
    data: dict[str, Any] = {
        "p_nash": p_nash,
        "p_mono": p_mono,
        "band": published_band,
        "thresholds": list(thresholds),
        "anchoring_median_delta_p": anchoring,
        "gate_ci": dict(gate_ci or {}),
        "ylim": list(ylim) if ylim else [p_nash - 0.2, p_mono + 0.3],
        "ylim_note": "price axis clipped; the opening periods can lie outside it",
    }
    for k, (lane, paths) in enumerate(price_paths.items()):
        arr = np.atleast_2d(np.asarray(paths, dtype=float))
        t = np.arange(arr.shape[1]) if periods is None else np.asarray(periods)[: arr.shape[1]]
        m, lo, hi = mean_ci(arr)
        col = PALETTE[k % len(PALETTE)]
        ax.fill_between(t, lo, hi, color=col, alpha=0.18, lw=0)
        ax.plot(t, m, color=col, label=f"{lane} (n={arr.shape[0]})")
        data[lane] = {"mean": m, "lo": lo, "hi": hi}
    _hline(ax, p_nash, "Nash")
    _hline(ax, p_mono, "Monopoly")
    ax.set_xlabel("Period")
    ax.set_ylabel("Price (normalised by α), clipped")
    ax.set_ylim(*(ylim or (p_nash - 0.2, p_mono + 0.3)))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=7, frameon=False)
    _seed_panel(sx, delta_by_seed, data, "delta_p_", "Δ_p per seed (mean, 95 % CI)")
    _hline(sx, thresholds[0], f"point ≥ {thresholds[0]:g}")
    _hline(sx, thresholds[1], f"share > {thresholds[1]:g}")
    if anchoring is not None and np.isfinite(anchoring):
        # a separate point on the 4-dealer trunk benchmarks, not a PC-H-dealer observation
        xa = len(delta_by_seed) - 0.1
        sx.scatter([xa], [anchoring], marker="X", s=46, color=INK["primary"], zorder=5)
        sx.annotate(
            f"AC-H median (trunk market, n = 4): {anchoring:.2f}",
            xy=(xa, anchoring),
            xytext=(-4, 7),
            textcoords="offset points",
            ha="right",
            fontsize=6.5,
            color=INK["primary"],
        )
        sx.set_xlim(-0.6, len(delta_by_seed) + 0.1)
    if with_pi:
        px = axes[2]
        if published_band is not None:
            px.axhspan(*published_band, color=INK["band"], lw=0, zorder=0)
            px.annotate(
                "Fish band (unverified, informational)",
                xy=(0.0, published_band[0]),
                xycoords=("axes fraction", "data"),
                xytext=(2, -9),
                textcoords="offset points",
                fontsize=6,
                color=INK["secondary"],
            )
        _seed_panel(px, delta_pi_by_seed or {}, data, "delta_pi_", "Δ_π per seed (mean, 95 % CI)")
        _hline(px, 0.10, "Δ_π ≥ 0.10")
    return save_figure(fig, stem, run_ids, data, "positive_control")


@styled
def plot_structural_gradient(
    levels: Sequence[str],
    deltas: Sequence[Sequence[float]],
    stem: str | Path,
    run_ids: Sequence[str],
    cis: Sequence[tuple[float, float]] | None = None,
    references: Mapping[str, tuple[float, float]] | None = None,
    markers: Sequence[str] | None = None,
    notes: Mapping[str, Any] | None = None,
) -> list[Path]:
    """Fig. 2: mean Delta_p per routing-sensitivity level with CIs, seeds as dots, and
    published references: a range is a band, a single value (``lo == hi``) a dashed mark
    (the Fish et al. P1 reference is a Delta_p mark; its Delta_pi 0.43 belongs in the
    caption). ``markers`` distinguishes the pooled TR-H s1 from the CRN-paired GR-H levels."""
    fig = Figure(figsize=(4.6, 3.2))
    ax = fig.add_subplot(1, 1, 1)
    rng = np.random.default_rng(0)
    data: dict[str, Any] = {"notes": dict(notes or {})}
    for k, (lvl, d) in enumerate(zip(levels, deltas, strict=True)):
        v = np.asarray(d, dtype=float)
        m, lo, hi = mean_ci(v)
        if cis is not None:
            lo, hi = cis[k]
        mk = markers[k] if markers is not None else "o"
        ax.scatter(
            k + rng.uniform(-0.12, 0.12, v.size),
            v,
            s=10,
            color=PALETTE[0],
            alpha=0.35,
            lw=0,
            zorder=2,
        )
        ax.errorbar(
            k,
            m,
            yerr=[[m - lo], [hi - m]],
            fmt=mk,
            color=PALETTE[0],
            ms=6,
            capsize=0,
            elinewidth=1.5,
            zorder=3,
        )
        data[lvl] = {"mean": m, "lo": lo, "hi": hi, "values": v, "marker": mk}
    for j, (name, (lo, hi)) in enumerate((references or {}).items()):
        if hi > lo:
            ax.axhspan(lo, hi, color=INK["band"], lw=0, zorder=0)
        else:
            ax.axhline(lo, color=INK["muted"], lw=0.8, ls=(0, (4, 3)), zorder=1)
        ax.annotate(
            name,
            xy=(1.0, hi),
            xycoords=("axes fraction", "data"),
            xytext=(-2, 2 + 8 * j),
            textcoords="offset points",
            ha="right",
            fontsize=7.5,
            color=INK["secondary"],
        )
        data[f"ref_{name}"] = [lo, hi]
    ax.set_xticks(range(len(levels)), list(levels))
    ax.set_xlabel("Routing sensitivity")
    ax.set_ylabel("Mean Δ_p (95 % CI)")
    return save_figure(fig, stem, run_ids, data, "structural_gradient")


@styled
def plot_frontier(
    points: Sequence[Mapping[str, Any]],
    stem: str | Path,
    run_ids: Sequence[str],
    sham_band: float = 0.0,
    mei: float = 0.25,
    mqc_max: float = 5.0,
    not_run: Sequence[str] = (),
) -> list[Path]:
    """Fig. 3: cost-effectiveness frontier of the levers.

    Each point is ``{label, mqc, dc, lo, hi, mechanical?, behavioural?, own?, own_lo?,
    own_hi?, dual_label?, replication?, primary?}``: x = market-quality cost (pp, true-value
    surplus), y = the TOTAL paired capped Delta C_p under the trunk scoring with its BCa 95 %
    CI. The mechanical (frozen-quote) part is a hollow circle at the same x joined by a dotted
    segment, so the behavioural part is the remainder; the own-scoring contrast (D-17) is a
    second, smaller triangle marker. A lever labelled "price reduction consistent with
    competitive adjustment" is drawn with a white face and an orange edge; replication lanes
    (M, S) use hollow diamonds; the pre-specified H1 lever gets a bold label; levers not run
    are listed in the corner. The sham sits at the origin with the band +/- ``sham_band`` =
    the SD over trunks of Delta_p(sham) - Delta_p(sham2) (SAP section 8).

    Ceiling panel (right, when the points carry ``raw`` / ``share_ref`` / ``share_arm``): the
    capped index saturates at 1 when the dealers quote above the monopoly price, so a lever
    can lower prices without moving it. The panel shows, per lever, the RAW-index paired
    contrast (BCa 95 %) and the share of incumbent quotes above p_M under the sham and under
    the lever, so the ceiling caveat is visible in the figure (lead ruling).
    """
    ceiling = [p for p in points if p.get("raw") is not None or p.get("share_arm") is not None]
    fig = Figure(figsize=(8.6 if ceiling else 5.5, 3.6))
    if ceiling:
        ax, cx = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1.6, 1]})
    else:
        ax, cx = fig.add_subplot(1, 1, 1), None
    xs = [float(p["mqc"]) for p in points] + [0.0]
    xmax = max(max(xs) * 1.15, mqc_max * 1.6)
    xmin = min(min(xs) - 0.5, -0.5)
    ax.axvspan(xmin, mqc_max, color=INK["band"], lw=0, zorder=0)
    ax.annotate(
        f"MQC ≤ {mqc_max:g} pp",
        xy=(mqc_max, 1.0),
        xycoords=("data", "axes fraction"),
        xytext=(-3, -10),
        textcoords="offset points",
        ha="right",
        fontsize=7.5,
        color=INK["secondary"],
    )
    if sham_band > 0:
        ax.axhspan(-sham_band, sham_band, color=PALETTE[0], alpha=0.08, lw=0, zorder=0)
    _hline(ax, mei, f"MEI = {mei:g}")
    ax.axhline(0, color=INK["axis"], lw=0.6, zorder=1)
    ax.scatter([0], [0], marker="s", s=28, color=INK["secondary"], zorder=4)
    ax.annotate(
        "sham ± SD(sham − sham2)" if sham_band > 0 else "sham",
        (0, 0),
        xytext=(4, -10),
        textcoords="offset points",
        fontsize=7.5,
        color=INK["secondary"],
    )
    data: dict[str, Any] = {
        "y": "total capped Delta C_p, trunk scoring (BCa 95 % CI); hollow circle = mechanical "
        "part; triangle = own scoring (D-17)",
        "x": "MQC (pp of true-value surplus)",
        "sham_band": sham_band,
        "sham_band_definition": "SD over trunks of Delta_p(sham) - Delta_p(sham2)",
        "mei": mei,
        "mqc_max": mqc_max,
        "not_run": list(not_run),
        "points": [],
    }
    ca_label = "price reduction consistent with competitive adjustment"
    for p in points:
        x, y, lo, hi = float(p["mqc"]), float(p["dc"]), float(p["lo"]), float(p["hi"])
        rep = bool(p.get("replication", False))
        ca = str(p.get("dual_label", "")) == ca_label
        if rep:
            x += 0.015 * (xmax - xmin)  # visual nudge only; the sidecar keeps the true MQC
        col = PALETTE[1] if ca else PALETTE[0]
        ax.errorbar(
            x,
            y,
            yerr=[[max(y - lo, 0.0)], [max(hi - y, 0.0)]],
            fmt="none",
            ecolor=col,
            elinewidth=1.2,
            alpha=0.6 if rep else 1.0,
            zorder=2,
        )
        ax.scatter(
            [x],
            [y],
            s=34,
            zorder=3,
            marker="D" if rep else "o",
            linewidth=1.4,
            facecolor="white" if (rep or ca) else col,
            edgecolor=col,
        )
        mech = p.get("mechanical")
        if mech is not None and np.isfinite(float(mech)):
            ax.plot([x, x], [y, float(mech)], ls=":", lw=0.8, color=col, zorder=2)
            ax.scatter(
                [x],
                [float(mech)],
                s=22,
                zorder=3,
                linewidth=1.0,
                facecolor="white",
                edgecolor=INK["secondary"],
            )
        own = p.get("own")
        if own is not None and np.isfinite(float(own)):
            olo, ohi = p.get("own_lo"), p.get("own_hi")
            xo = x + 0.03 * (xmax - xmin)
            if olo is not None and ohi is not None:
                ax.errorbar(
                    xo,
                    float(own),
                    yerr=[[max(float(own) - float(olo), 0.0)], [max(float(ohi) - float(own), 0.0)]],
                    fmt="none",
                    ecolor=col,
                    elinewidth=0.8,
                    alpha=0.6,
                    zorder=2,
                )
            ax.scatter([xo], [float(own)], s=22, marker="^", color=col, zorder=3)
        ax.annotate(
            str(p["label"]) + (" (CA)" if ca else ""),
            (x, y),
            xytext=(5, 3),
            textcoords="offset points",
            fontsize=7.5,
            fontweight="bold" if p.get("primary") else "normal",
            color=INK["primary"],
        )
        data["points"].append({k: p[k] for k in p})
    if not_run:
        ax.annotate(
            "not run: " + ", ".join(not_run),
            xy=(1.0, 0.0),
            xycoords="axes fraction",
            xytext=(-3, 3),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=7,
            color=INK["secondary"],
        )
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel("Market-quality cost, MQC (pp of true-value surplus)")
    ax.set_ylabel("Capped ΔC_p, trunk scoring (95 % CI)")
    ax.set_title("○ mechanical part   ▲ own scoring (D-17)", fontsize=7.5, loc="left")
    if cx is not None:
        labels = [str(p["label"]) for p in ceiling]
        yy = np.arange(len(ceiling))[::-1]
        for y, p in zip(yy, ceiling, strict=True):
            raw = p.get("raw")
            if raw is not None and np.isfinite(float(raw)):
                lo, hi = p.get("raw_lo"), p.get("raw_hi")
                if lo is not None and hi is not None:
                    cx.errorbar(
                        float(raw),
                        y,
                        xerr=[
                            [max(float(raw) - float(lo), 0.0)],
                            [max(float(hi) - float(raw), 0.0)],
                        ],
                        fmt="o",
                        ms=4,
                        mfc="white",
                        color=PALETTE[0],
                        elinewidth=1.0,
                        capsize=0,
                    )
                else:
                    cx.scatter([float(raw)], [y], s=18, facecolor="white", edgecolor=PALETTE[0])
            sr, sa = p.get("share_ref"), p.get("share_arm")
            if sr is not None and sa is not None:
                cx.annotate(
                    f"> p_M: {100 * float(sr):.0f} % → {100 * float(sa):.0f} %",
                    xy=(1.0, y),
                    xycoords=("axes fraction", "data"),
                    xytext=(-2, 4),
                    textcoords="offset points",
                    ha="right",
                    fontsize=6.5,
                    color=INK["secondary"],
                )
        cx.axvline(0, color=INK["axis"], lw=0.6)
        cx.set_yticks(yy, labels)
        cx.set_ylim(-0.7, len(ceiling) - 0.3)
        cx.set_xlabel("Raw ΔC_p (uncapped, 95 % CI)")
        cx.set_title("Ceiling check: raw index, share of quotes > p_M", fontsize=7.5)
        data["ceiling_panel"] = {
            "note": "the capped index saturates at 1 above the monopoly price; raw-index contrast "
            "and share of incumbent quotes above p_M (sham -> lever) per lever",
            "levers": [
                {
                    k: p.get(k)
                    for k in ("label", "raw", "raw_lo", "raw_hi", "share_ref", "share_arm")
                }
                for p in ceiling
            ],
        }
    return save_figure(fig, stem, run_ids, data, "frontier")


def frontier_from_stats(
    st: Mapping[str, Any],
    mqc: Mapping[str, Any] | None = None,
    estimand: str = "delta_p",
) -> tuple[list[dict[str, Any]], float, list[str], list[str]]:
    """Frontier points, sham band and run ids from a ``stats.json`` document.

    One point per lever of the H1-R family with a paired contrast and an MQC estimate
    (``mqc`` overrides ``st["mqc"]``): ``dc/lo/hi`` = the TOTAL capped Delta C_p under the
    trunk scoring (BCa 95 %), ``mechanical`` / ``behavioural`` = the frozen-quote split,
    ``own/own_lo/own_hi`` = the own-scoring contrast (D-17), ``dual_label`` = its D-17
    label. ``band`` = the SD over trunks of Delta_p(sham) - Delta_p(sham2) (the sham2
    contrast's ``sd``; 0 with a note when sham2 is absent). Levers of the family without a
    contrast are returned as ``not_run`` (fourth element).
    """
    from finswarm.analysis.mqc import mqc_point_estimates

    mq = mqc_point_estimates(dict(mqc) if mqc is not None else st.get("mqc"))
    contrasts = st.get("contrasts", {})
    family = st.get("h1r", {}).get("family") or sorted(contrasts)
    primary = st.get("h1", {}).get("lever")
    pts: list[dict[str, Any]] = []
    not_run: list[str] = []
    for lev in family:
        c = contrasts.get(lev)
        if not c or estimand not in c:
            not_run.append(lev)
            continue
        if mq.get(lev) is None:
            continue
        tot = c[estimand]
        own = c.get("delta_p_own") or {}
        pts.append(
            {
                "label": lev,
                "mqc": float(mq[lev]),  # type: ignore[arg-type]
                "dc": tot.get("mean"),
                "lo": tot.get("bca_lo"),
                "hi": tot.get("bca_hi"),
                "mechanical": c.get("mechanical", {}).get(estimand, {}).get("mean"),
                "behavioural": c.get("behavioural", {}).get(estimand, {}).get("mean"),
                "own": own.get("mean"),
                "own_lo": own.get("bca_lo"),
                "own_hi": own.get("bca_hi"),
                "dual_label": (c.get("dual_benchmark") or {}).get("label"),
                # ceiling caveat (D-16 reporting): raw-index contrast and share above p_M
                "raw": (c.get("delta_p_raw") or {}).get("mean"),
                "raw_lo": (c.get("delta_p_raw") or {}).get("bca_lo"),
                "raw_hi": (c.get("delta_p_raw") or {}).get("bca_hi"),
                "share_ref": (c.get("share_quotes_above_pm") or {}).get("reference"),
                "share_arm": (c.get("share_quotes_above_pm") or {}).get("arm"),
                "n": c.get("n_pairs", 0),
                "primary": lev == primary,
            }
        )
    pts = [p for p in pts if all(p[k] is not None for k in ("dc", "lo", "hi"))]
    sham2 = contrasts.get("sham2", {}).get(estimand, {})
    sd = sham2.get("sd")
    band = float(sd) if sd is not None and np.isfinite(float(sd)) else 0.0
    run_ids = [str(r.get("run_id")) for r in st.get("runs", [])]
    return pts, band, run_ids, not_run


@styled
def plot_action_carrier_ablation(
    series: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    t0: int,
    stem: str | Path,
    run_ids: Sequence[str],
    dependence: Mapping[str, tuple[float, float, float]] | None = None,
) -> list[Path]:
    """Fig. 4: the capped price index Delta_p around the branch period for the sham vs the
    view arms (CI bands across trunks); right panel: net dependence per arm with CIs."""
    ncol = 2 if dependence else 1
    fig = Figure(figsize=(6.4 if dependence else 4.6, 3.2))
    axes = fig.subplots(1, ncol, gridspec_kw={"width_ratios": [2.2, 1]} if dependence else None)
    ax = axes[0] if dependence else axes
    data: dict[str, Any] = {"t0": t0}
    for k, (lab, (t, m, lo, hi)) in enumerate(series.items()):
        col = PALETTE[k % len(PALETTE)]
        ax.fill_between(t, lo, hi, color=col, alpha=0.18, lw=0)
        ax.plot(t, m, color=col, label=lab)
        data[lab] = {"t": t, "mean": m, "lo": lo, "hi": hi}
    ax.axvline(t0, color=INK["muted"], lw=0.8, ls=(0, (4, 3)))
    ax.set_xlabel("Period")
    ax.set_ylabel("Capped Δ_p (95 % CI)")
    ax.legend(loc="lower left")
    if dependence:
        bx = axes[1]
        labs = list(dependence)
        for k, lab in enumerate(labs):
            v, lo, hi = dependence[lab]
            bx.bar(k, v, width=0.45, color=PALETTE[k % len(PALETTE)])
            bx.errorbar(
                k, v, yerr=[[v - lo], [hi - v]], fmt="none", ecolor=INK["secondary"], elinewidth=1.0
            )
        bx.set_xticks(range(len(labs)), labs, rotation=0)
        bx.set_ylabel("Net dependence (vs ghost)")
        data["dependence"] = dict(dependence)
    return save_figure(fig, stem, run_ids, data, "action_carrier_ablation")


@styled
def plot_detector_roc(
    curves: Mapping[str, tuple[np.ndarray, np.ndarray, float]],
    stem: str | Path,
    run_ids: Sequence[str],
    max_fpr: float = 0.05,
) -> list[Path]:
    """Detector ROC curves on the held-out half; legend carries the AUCs."""
    fig = Figure(figsize=(3.8, 3.6))
    ax = fig.add_subplot(1, 1, 1)
    ax.grid(True, axis="both")
    ax.plot([0, 1], [0, 1], color=INK["axis"], lw=0.8)
    ax.axvline(max_fpr, color=INK["muted"], lw=0.8, ls=(0, (4, 3)))
    data: dict[str, Any] = {"max_fpr": max_fpr}
    for k, (lab, (fpr, tpr, a)) in enumerate(curves.items()):
        ax.step(
            fpr, tpr, where="post", color=PALETTE[k % len(PALETTE)], label=f"{lab} (AUC {a:.2f})"
        )
        data[lab] = {"fpr": fpr, "tpr": tpr, "auc": a}
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("True-positive rate")
    ax.legend(loc="lower right")
    return save_figure(fig, stem, run_ids, data, "detector_roc")
