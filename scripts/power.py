"""Monte Carlo power for the pre-registered H1 rule (decision D-3').

Model. One paired difference per qualified trunk, ``d_i = Delta_bar(sham) - Delta_bar(lever)`` on
periods 111-150, with mean ``effect`` and standard deviation ``sigma_d``. In the 8-lever family
every contrast shares the trunk's sham, ``d_{l,i} = effect_l + u_{sham,i} - u_{l,i}`` with
``u ~ N(0, sigma_d^2 / 2)``, so ``sd(d_l) = sigma_d`` and contrasts correlate at 0.5.

Tables (written to ``docs/03_preregistration/power_table.md``):

1. PRIMARY, the pre-specified lever: one-sided paired t at alpha = 0.05 (no multiplicity); power
   of "p < 0.05" and of "one-sided 95 % t lower bound > 0.10".
2. SECONDARY, the Holm family: 8 levers, one with the true effect and seven nulls, one-sided
   Holm (m = 8, alpha = 0.05) on the full sample; power to flag the true lever and family-wise
   false-positive rate (at least one null lever flagged).
3. DESCRIPTIVE, winner's curse: the naive "best of 8 levers" estimate (select and estimate on all
   trunks) against the split-sample estimate (select on odd seeds, estimate on even seeds),
   under all-null and one-true-effect scenarios. Bias = estimate - true effect of the lever
   actually selected.

Usage: ``uv run python scripts/power.py [--reps 20000] [--seed 20261010] [--no-write]``.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "03_preregistration" / "power_table.md"
SEED = 20261010
REPS = 20_000
NS = (5, 6, 7, 8, 10, 12)  # D-19: small-N rows for fewer qualified trunks
SIGMAS = (0.15, 0.20, 0.30)
EFFECTS = (0.15, 0.25, 0.35)
ALPHA = 0.05
LOWER_BOUND = 0.10
M_LEVERS = 8


# ------------------------------------------------------------------------------- statistics
def one_sided_t(d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One-sided paired t over the last axis: (p-value for H1 mean > 0, 95 % lower bound)."""
    n = d.shape[-1]
    mean = d.mean(axis=-1)
    se = d.std(axis=-1, ddof=1) / np.sqrt(n)
    p = stats.t.sf(mean / se, df=n - 1)
    lower = mean - stats.t.ppf(1.0 - ALPHA, df=n - 1) * se
    return p, lower


def holm_reject(p: np.ndarray, alpha: float = ALPHA) -> np.ndarray:
    """Holm step-down rejections over the last axis (same shape as ``p``)."""
    m = p.shape[-1]
    order = np.argsort(p, axis=-1)
    ps = np.take_along_axis(p, order, axis=-1)
    ok = ps <= alpha / (m - np.arange(m))
    accepted_sorted = np.cumprod(ok, axis=-1).astype(bool)  # reject until the first failure
    out = np.zeros_like(accepted_sorted)
    np.put_along_axis(out, order, accepted_sorted, axis=-1)
    return out


def family_draws(
    rng: np.random.Generator, reps: int, n: int, sigma: float, effects: np.ndarray
) -> np.ndarray:
    """Paired differences ``(reps, m, n)`` of m contrasts sharing the trunk's sham."""
    m = len(effects)
    half = sigma / np.sqrt(2.0)
    sham = rng.standard_normal((reps, 1, n)) * half
    lever = rng.standard_normal((reps, m, n)) * half
    return effects[None, :, None] + sham - lever


# ---------------------------------------------------------------------------------- tables
@dataclass(frozen=True)
class PrimaryRow:
    n: int
    sigma: float
    effect: float
    power_p: float
    power_lb: float


def primary(rng: np.random.Generator, reps: int) -> tuple[list[PrimaryRow], dict[int, float]]:
    rows = []
    for n in NS:
        for sigma in SIGMAS:
            for effect in EFFECTS:
                d = effect + sigma * rng.standard_normal((reps, n))
                p, lower = one_sided_t(d)
                rows.append(
                    PrimaryRow(
                        n,
                        sigma,
                        effect,
                        float((p < ALPHA).mean()),
                        float((lower > LOWER_BOUND).mean()),
                    )
                )
    size = {}
    for n in NS:  # type I error of "p < 0.05" at effect 0 (scale-free)
        p, _ = one_sided_t(rng.standard_normal((reps, n)))
        size[n] = float((p < ALPHA).mean())
    return rows, size


@dataclass(frozen=True)
class HolmRow:
    n: int
    sigma: float
    effect: float
    power_true: float
    fwer: float


def secondary(rng: np.random.Generator, reps: int) -> tuple[list[HolmRow], dict[int, float]]:
    rows = []
    for n in NS:
        for sigma in SIGMAS:
            for effect in EFFECTS:
                eff = np.zeros(M_LEVERS)
                eff[0] = effect
                p, _ = one_sided_t(family_draws(rng, reps, n, sigma, eff))
                rej = holm_reject(p)
                rows.append(
                    HolmRow(
                        n,
                        sigma,
                        effect,
                        float(rej[:, 0].mean()),
                        float(rej[:, 1:].any(axis=1).mean()),
                    )
                )
    null_fwer = {}
    for n in NS:  # all eight levers null (scale-free)
        p, _ = one_sided_t(family_draws(rng, reps, n, 1.0, np.zeros(M_LEVERS)))
        null_fwer[n] = float(holm_reject(p).any(axis=1).mean())
    return rows, null_fwer


@dataclass(frozen=True)
class BiasRow:
    n: int
    sigma: float
    scenario: str
    naive_est: float
    naive_bias: float
    naive_hit: float
    split_est: float
    split_bias: float
    split_hit: float


def selection_bias(
    d: np.ndarray, effects: np.ndarray
) -> tuple[float, float, float, float, float, float]:
    """Naive (all trunks) and split-sample (select on odd seeds, estimate on even seeds)."""
    reps = d.shape[0]
    idx = np.arange(reps)
    full = d.mean(axis=-1)
    w = full.argmax(axis=1)
    naive = full[idx, w]
    odd, even = d[..., 1::2].mean(axis=-1), d[..., 0::2].mean(axis=-1)
    ws = odd.argmax(axis=1)
    split = even[idx, ws]
    true_lever = effects.argmax() if effects.max() > 0 else -1
    return (
        float(naive.mean()),
        float((naive - effects[w]).mean()),
        float((w == true_lever).mean()) if true_lever >= 0 else float("nan"),
        float(split.mean()),
        float((split - effects[ws]).mean()),
        float((ws == true_lever).mean()) if true_lever >= 0 else float("nan"),
    )


def descriptive(rng: np.random.Generator, reps: int) -> list[BiasRow]:
    rows = []
    for n in NS:
        for sigma in SIGMAS:
            for effect in (0.0, *EFFECTS):
                eff = np.zeros(M_LEVERS)
                eff[0] = effect
                d = family_draws(rng, reps, n, sigma, eff)
                label = "all null" if effect == 0 else f"one true {effect:.2f}"
                rows.append(BiasRow(n, sigma, label, *selection_bias(d, eff)))
    return rows


# --------------------------------------------------------------------------------- output
def fmt(x: float, nd: int = 3) -> str:
    return "n/a" if x != x else f"{x:.{nd}f}"


def render(
    primary_rows: list[PrimaryRow],
    size: dict[int, float],
    holm_rows: list[HolmRow],
    null_fwer: dict[int, float],
    bias_rows: list[BiasRow],
    *,
    seed: int,
    reps: int,
) -> str:
    se = 0.5 / np.sqrt(reps)
    lines = [
        "# Power table for H1 (decision rule D-3')",
        "",
        f"Generated by `uv run python scripts/power.py --reps {reps} --seed {seed}` "
        f"(numpy PCG64, simulation seed **{seed}**, {reps:,} replications per cell; Monte Carlo "
        f"standard error at most {se:.4f}). Do not edit by hand: rerun the script.",
        "",
        "Model: one paired difference per qualified trunk, d = Delta_bar(sham) - Delta_bar(lever) "
        "on periods 111-150, normal with mean = true effect and SD = sigma_d. In the 8-lever family "
        "every contrast shares the trunk's sham (sham and lever noise each sigma_d^2 / 2), so "
        "contrasts correlate at 0.5. sigma_d = 0.20 is the planning value (PREREG A-4); the double "
        "sham measures it.",
        "",
        "## 1. Primary pre-specified lever (one-sided paired t, alpha = 0.05, no multiplicity)",
        "",
        "Power of the two parts of the primary rule: p < 0.05, and the one-sided 95 % t lower "
        f"bound of the mean difference above {LOWER_BOUND:.2f}.",
        "",
        "| N trunks | sigma_d | true effect | P(p < 0.05) | P(lower 95 % bound > 0.10) |",
        "| --- | --- | --- | --- | --- |",
    ]
    lines += [
        f"| {r.n} | {r.sigma:.2f} | {r.effect:.2f} | {fmt(r.power_p)} | {fmt(r.power_lb)} |"
        for r in primary_rows
    ]
    lines += [
        "",
        "Size check (true effect 0): P(p < 0.05) = "
        + ", ".join(f"{fmt(v)} at N = {n}" for n, v in size.items())
        + ".",
        "",
        "## 2. Secondary Holm family (8 levers: one true effect, seven nulls; one-sided Holm, "
        "m = 8, alpha = 0.05, full sample)",
        "",
        "| N trunks | sigma_d | true effect | P(true lever flagged) | FWER (>= 1 null lever flagged) |",
        "| --- | --- | --- | --- | --- |",
    ]
    lines += [
        f"| {r.n} | {r.sigma:.2f} | {r.effect:.2f} | {fmt(r.power_true)} | {fmt(r.fwer)} |"
        for r in holm_rows
    ]
    lines += [
        "",
        "All eight levers null: FWER = "
        + ", ".join(f"{fmt(v)} at N = {n}" for n, v in null_fwer.items())
        + " (Holm controls it at 0.05 under any dependence).",
        "",
        "## 3. Descriptive: winner's curse of the best-of-8 estimate",
        "",
        "Naive: pick the lever with the largest mean difference over all trunks and report that "
        "mean. Split sample: pick on odd seeds, estimate on even seeds. Bias = estimate - true "
        "effect of the lever actually picked; P(true) = probability that the picked lever is the "
        "one with the true effect.",
        "",
        "| N trunks | sigma_d | scenario | naive estimate | naive bias | naive P(true) | "
        "split estimate | split bias | split P(true) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    lines += [
        f"| {r.n} | {r.sigma:.2f} | {r.scenario} | {fmt(r.naive_est)} | {fmt(r.naive_bias)} | "
        f"{fmt(r.naive_hit)} | {fmt(r.split_est)} | {fmt(r.split_bias)} | {fmt(r.split_hit)} |"
        for r in bias_rows
    ]
    lines += [
        "",
        "Reading: the naive best-of-8 estimate is biased upward (by about the expected maximum "
        "of eight correlated noise means when all levers are null), which is why the primary "
        "lever is pre-specified and the descriptive frontier reports split-sample estimates.",
        "",
    ]
    return "\n".join(lines)


def run(seed: int = SEED, reps: int = REPS) -> str:
    rng = np.random.Generator(np.random.PCG64(seed))
    p_rows, size = primary(rng, reps)
    h_rows, null_fwer = secondary(rng, reps)
    b_rows = descriptive(rng, reps)
    return render(p_rows, size, h_rows, null_fwer, b_rows, seed=seed, reps=reps)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--no-write", action="store_true", help="print only")
    args = ap.parse_args(argv)
    text = run(args.seed, args.reps)
    print(text)
    if not args.no_write:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8", newline="\n")
        print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
