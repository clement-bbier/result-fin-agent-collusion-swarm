"""Markdown tables with confidence intervals for ``summary.md`` and the paper."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


def fmt(x: Any, digits: int = 2) -> str:
    if x is None:
        return "–"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, int):
        return str(x)
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not math.isfinite(v):
        return "–"
    return f"{v:.{digits}f}"


def fmt_ci(est: Any, lo: Any, hi: Any, digits: int = 2) -> str:
    """``0.43 [0.31, 0.55]``."""
    return f"{fmt(est, digits)} [{fmt(lo, digits)}, {fmt(hi, digits)}]"


def fmt_p(p: Any) -> str:
    try:
        v = float(p)
    except (TypeError, ValueError):
        return "–"
    if not math.isfinite(v):
        return "–"
    return "<0.001" if v < 0.001 else f"{v:.3f}"


def markdown_table(
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[str] | None = None,
    align: Mapping[str, str] | None = None,
) -> str:
    """Render rows (already-formatted values or numbers) as a GitHub markdown table."""
    if not rows:
        return "_(no rows)_\n"
    cols = list(columns or rows[0].keys())
    al = align or {}
    sep = {"l": ":---", "r": "---:", "c": ":---:"}
    head = "| " + " | ".join(cols) + " |"
    rule = "| " + " | ".join(sep.get(al.get(c, "l"), "---") for c in cols) + " |"
    body = [
        "| "
        + " | ".join(v if isinstance(v, str) else fmt(v) for v in (r.get(c) for c in cols))
        + " |"
        for r in rows
    ]
    return "\n".join([head, rule, *body]) + "\n"


def arm_table(arms: Mapping[str, Mapping[str, Any]], digits: int = 2) -> str:
    """One row per arm: n runs and mean Delta with its BCa CI."""
    rows = [
        {
            "arm": a,
            "n": int(v.get("n", 0)),
            "Delta [95% CI]": fmt_ci(v.get("mean"), v.get("lo"), v.get("hi"), digits),
        }
        for a, v in arms.items()
    ]
    return markdown_table(rows, ["arm", "n", "Delta [95% CI]"], {"n": "r"})


def contrast_table(
    contrasts: Mapping[str, Mapping[str, Any]],
    holm: Mapping[str, Mapping[str, Any]] | None = None,
    measure: str = "delta_p",
    digits: int = 2,
) -> str:
    """One row per paired contrast (reference minus arm) on ``measure``: Delta C with BCa
    CI, one-sided t and Wilcoxon p, Holm (fixed family, full sample) and TOST.

    Accepts the campaign layout (``contrasts[arm][measure]`` bundles with ``reference``) or
    flat bundles (``contrasts[name]`` is itself the bundle)."""
    rows = []
    for name, c0 in contrasts.items():
        c = c0.get(measure, c0)
        ref = c0.get("reference")
        h = (holm or {}).get(name, {})
        rows.append(
            {
                "contrast": f"{ref} − {name}" if ref else name,
                "n pairs": int(c.get("n", 0)),
                "ΔC [BCa 95% CI]": fmt_ci(c.get("mean"), c.get("bca_lo"), c.get("bca_hi"), digits),
                "p (t, one-sided)": fmt_p(c.get("t", {}).get("p")),
                "p (Wilcoxon)": fmt_p(c.get("wilcoxon", {}).get("p")),
                "p Holm (m = 8)": fmt_p(h.get("p_holm", c.get("p_holm"))),
                "TOST ±0.20": "equivalent" if c.get("tost", {}).get("equivalent") else "not shown",
                "dual-benchmark (D-17)": h.get("dual_benchmark_label")
                or c0.get("dual_benchmark", {}).get("label")
                or "–",
            }
        )
    return markdown_table(rows, None, {"n pairs": "r"})


def decomposition_table(
    contrasts: Mapping[str, Mapping[str, Any]],
    measures: Sequence[str] = ("delta_p", "delta_pi_window"),
    digits: int = 2,
) -> str:
    """Total, mechanical (frozen sham quotes re-scored under the lever's market) and
    behavioural (total - mechanical) Delta C per lever."""
    rows = []
    for name, c in contrasts.items():
        if "mechanical" not in c:
            continue
        for m in measures:
            t, mech, beh = c[m], c["mechanical"][m], c["behavioural"][m]
            rows.append(
                {
                    "lever": name,
                    "measure": m,
                    "total ΔC": fmt_ci(t.get("mean"), t.get("bca_lo"), t.get("bca_hi"), digits),
                    "mechanical": fmt_ci(mech.get("mean"), mech.get("lo"), mech.get("hi"), digits),
                    "behavioural": fmt_ci(
                        beh.get("mean"), beh.get("bca_lo"), beh.get("bca_hi"), digits
                    ),
                }
            )
    return markdown_table(rows, None) if rows else "_(no lever contrasts)_\n"


def detector_table(report: Mapping[str, Any], digits: int = 2) -> str:
    """Held-out detection summary: composite and scorecard rates, per-feature AUC."""
    comp, sc = report["composite"], report["scorecard"]
    rows = [
        {
            "detector": "composite (D1, D7, D5net)",
            "AUC [95% CI]": fmt_ci(comp["auc"], *comp["auc_ci"], digits=digits),
            "TPR": fmt(comp["tpr"], digits),
            "FPR": fmt(comp["fpr"], digits),
        },
        {
            "detector": "scorecard D1∧D7∧D5net",
            "AUC [95% CI]": "–",
            "TPR": fmt(sc["tpr"], digits),
            "FPR": fmt(sc["fpr"], digits),
        },
    ]
    for f, a in report.get("feature_auc", {}).items():
        rows.append({"detector": f, "AUC [95% CI]": fmt(a, digits), "TPR": "–", "FPR": "–"})
    out = markdown_table(rows, ["detector", "AUC [95% CI]", "TPR", "FPR"])
    types = comp.get("fpr_by_type", {})
    if types:
        out += "\n" + markdown_table(
            [
                {
                    "negative type": k,
                    "FPR composite": fmt(v, digits),
                    "FPR scorecard": fmt(sc["fpr_by_type"].get(k), digits),
                }
                for k, v in types.items()
            ]
        )
    return out
