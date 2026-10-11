"""Monte Carlo power script for the H1 decision rule (D-3', ``scripts/power.py``): statistics,
Holm step-down, family draws, analytic cross-checks against the noncentral t and the rendered
table. The planning values quoted in PREREG.md come from this script, so its primitives are
checked against closed forms, not only against themselves."""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
POWER_TABLE = ROOT / "docs" / "03_preregistration" / "power_table.md"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        f"_script_{name}", ROOT / "scripts" / f"{name}.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses resolve string annotations through sys.modules
    spec.loader.exec_module(mod)
    return mod


power = _load("power")


def _holm_reference(p: np.ndarray, alpha: float) -> np.ndarray:
    """Textbook scalar Holm step-down for one vector of p-values."""
    m = len(p)
    order = np.argsort(p)
    out = np.zeros(m, dtype=bool)
    for k, idx in enumerate(order):
        if p[idx] <= alpha / (m - k):
            out[idx] = True
        else:
            break
    return out


# ------------------------------------------------------------------------------ primitives
def test_one_sided_t_matches_scipy() -> None:
    rng = np.random.default_rng(1)
    d = 0.2 + 0.3 * rng.standard_normal((5, 8))
    p, lower = power.one_sided_t(d)
    ref = stats.ttest_1samp(d, 0.0, axis=-1, alternative="greater")
    np.testing.assert_allclose(p, ref.pvalue, rtol=1e-10)
    se = d.std(axis=-1, ddof=1) / np.sqrt(8)
    expected_lower = d.mean(axis=-1) - stats.t.ppf(0.95, df=7) * se
    np.testing.assert_allclose(lower, expected_lower, rtol=1e-12)
    assert np.all(lower < d.mean(axis=-1))


def test_holm_reject_known_cases() -> None:
    # 0.001 <= 0.05/8 rejected; 0.02 > 0.05/7 stops the step-down, 0.03 is not reached
    p = np.array([0.001, 0.02, 0.03, 0.5, 0.6, 0.7, 0.8, 0.9])
    assert power.holm_reject(p[None, :]).tolist() == [[True] + [False] * 7]
    # three small p-values in a row pass their step-down thresholds 0.05/8, 0.05/7, 0.05/6
    p = np.array([0.004, 0.006, 0.007, 0.9, 0.9, 0.9, 0.9, 0.9])
    assert power.holm_reject(p[None, :]).tolist() == [[True, True, True] + [False] * 5]
    # eight p-values below alpha but above alpha / 8: Holm rejects nothing
    assert not power.holm_reject(np.full((1, 8), 0.01)).any()
    # rejections are scattered back to the original positions, not left in sorted order
    p = np.array([0.5, 0.002, 0.9, 0.01, 0.9, 0.9, 0.9, 0.9])
    assert power.holm_reject(p[None, :]).tolist() == [
        [False, True, False, False, False, False, False, False]
    ]


def test_holm_reject_matches_scalar_reference() -> None:
    rng = np.random.default_rng(2)
    p = rng.uniform(0.0, 0.2, size=(200, 8)) ** 2  # many small values so rejections happen
    got = power.holm_reject(p)
    ref = np.stack([_holm_reference(row, power.ALPHA) for row in p])
    assert got.dtype == bool
    np.testing.assert_array_equal(got, ref)
    assert got.any(), "the reference case must exercise some rejections"


def test_family_draws_mean_sd_and_shared_sham_correlation() -> None:
    rng = np.random.default_rng(3)
    effects = np.array([0.10, 0.0, 0.0])
    d = power.family_draws(rng, 40_000, 4, 0.20, effects)
    assert d.shape == (40_000, 3, 4)
    np.testing.assert_allclose(d[:, 0].mean(), 0.10, atol=0.005)
    np.testing.assert_allclose(d[:, 1].mean(), 0.0, atol=0.005)
    np.testing.assert_allclose(d.std(axis=0).mean(), 0.20, atol=0.005)
    # two contrasts on the same trunk share the sham noise: correlation 0.5
    rho = np.corrcoef(d[:, 0, 0], d[:, 1, 0])[0, 1]
    assert abs(rho - 0.5) < 0.02
    # contrasts on different trunks are independent
    rho_cross = np.corrcoef(d[:, 0, 0], d[:, 1, 1])[0, 1]
    assert abs(rho_cross) < 0.02


# ------------------------------------------------------------------- analytic cross-checks
def _nct_power(n: int, sigma: float, shift: float) -> float:
    """Exact power of the one-sided t at level ALPHA for a mean shift of ``shift``."""
    df = n - 1
    t_crit = stats.t.ppf(1.0 - power.ALPHA, df)
    return float(stats.nct.sf(t_crit, df, shift / sigma * np.sqrt(n)))


def test_primary_table_matches_noncentral_t() -> None:
    reps = 4_000
    rng = np.random.Generator(np.random.PCG64(4))
    rows, size = power.primary(rng, reps)
    assert len(rows) == len(power.NS) * len(power.SIGMAS) * len(power.EFFECTS)
    tol = 4.5 * 0.5 / np.sqrt(reps)  # 4.5 Monte Carlo standard errors
    for r in rows:
        assert abs(r.power_p - _nct_power(r.n, r.sigma, r.effect)) < tol, r
        # "lower bound > 0.10" is the same t test against the shifted null 0.10
        lb = _nct_power(r.n, r.sigma, r.effect - power.LOWER_BOUND)
        assert abs(r.power_lb - lb) < tol, r
        assert r.power_lb <= r.power_p + 1e-12
    for n, s in size.items():
        assert abs(s - power.ALPHA) < tol, (n, s)


def test_planning_cell_is_above_point_eight() -> None:
    """PREREG section 2 planning value: N = 12, sigma_d 0.20, effect 0.25 (analytic)."""
    assert _nct_power(12, 0.20, 0.25) > 0.98
    assert 0.75 < _nct_power(12, 0.20, 0.25 - power.LOWER_BOUND) < 0.82


def test_secondary_holm_power_bounded_and_fwer_controlled() -> None:
    reps = 4_000
    rng = np.random.Generator(np.random.PCG64(5))
    rows, null_fwer = power.secondary(rng, reps)
    assert len(rows) == len(power.NS) * len(power.SIGMAS) * len(power.EFFECTS)
    slack = 4.5 * 0.5 / np.sqrt(reps)
    for r in rows:
        df = r.n - 1
        nc = r.effect / r.sigma * np.sqrt(r.n)
        bonferroni = float(stats.nct.sf(stats.t.ppf(1 - power.ALPHA / power.M_LEVERS, df), df, nc))
        unadjusted = _nct_power(r.n, r.sigma, r.effect)
        # Holm is at least as powerful as Bonferroni and never beats the unadjusted test
        assert r.power_true >= bonferroni - slack, r
        assert r.power_true <= unadjusted + slack, r
        assert r.fwer <= power.ALPHA + slack, r
    for n, f in null_fwer.items():
        assert 0.02 < f <= power.ALPHA + slack, (n, f)
    # power increases with the effect at fixed (n, sigma)
    by_cell: dict[tuple[int, float], list[float]] = {}
    for r in rows:
        by_cell.setdefault((r.n, r.sigma), []).append(r.power_true)
    for cell, seq in by_cell.items():
        assert seq == sorted(seq), cell


def test_selection_bias_winner_curse_and_split_sample() -> None:
    rng = np.random.default_rng(6)
    null = np.zeros(power.M_LEVERS)
    d = power.family_draws(rng, 20_000, 12, 0.20, null)
    naive_est, naive_bias, naive_hit, split_est, split_bias, split_hit = power.selection_bias(
        d, null
    )
    assert naive_est == naive_bias  # all levers null: the estimate is pure bias
    assert naive_bias > 0.04  # expected max of 8 correlated means at sd 0.20 / sqrt(12)
    assert abs(split_bias) < 0.01  # selecting on odd seeds removes the curse on even seeds
    assert np.isnan(naive_hit) and np.isnan(split_hit)
    one = null.copy()
    one[0] = 0.35
    d = power.family_draws(rng, 20_000, 12, 0.20, one)
    _, naive_bias, naive_hit, _, split_bias, split_hit = power.selection_bias(d, one)
    assert naive_hit > 0.99 and split_hit > 0.95
    assert abs(naive_bias) < 0.01 and abs(split_bias) < 0.01


# ------------------------------------------------------------------------------- rendering
def test_run_is_deterministic_and_renders_every_cell() -> None:
    text = power.run(seed=7, reps=200)
    assert text == power.run(seed=7, reps=200)
    assert text != power.run(seed=8, reps=200)
    assert "simulation seed **7**" in text and "200 replications" in text
    n_pat = "|".join(str(n) for n in power.NS)
    data_rows = [ln for ln in text.splitlines() if re.match(rf"^\| ({n_pat}) \| ", ln)]
    n_cells = len(power.NS) * len(power.SIGMAS)
    assert len(data_rows) == n_cells * len(power.EFFECTS) * 2 + n_cells * (len(power.EFFECTS) + 1)
    assert text.count("| all null |") == n_cells
    assert text.count("n/a") == 2 * n_cells  # P(true) undefined when every lever is null
    for header in ("## 1. Primary", "## 2. Secondary Holm", "## 3. Descriptive"):
        assert header in text


def test_fmt_handles_nan() -> None:
    assert power.fmt(float("nan")) == "n/a"
    assert power.fmt(0.12345) == "0.123"
    assert power.fmt(0.5, 1) == "0.5"


def test_main_writes_table_or_prints_only(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    out = tmp_path / "power_table.md"
    assert power.main(["--reps", "100", "--seed", "11", "--out", str(out)]) == 0
    assert out.read_text(encoding="utf-8") == power.run(seed=11, reps=100)
    assert "wrote" in capsys.readouterr().err
    out2 = tmp_path / "unwritten.md"
    assert power.main(["--reps", "100", "--seed", "11", "--out", str(out2), "--no-write"]) == 0
    assert not out2.exists()
    assert "Power table for H1" in capsys.readouterr().out


@pytest.mark.slow
def test_committed_power_table_is_reproducible() -> None:
    """The committed table must be the script's output at the registered seed and reps."""
    if not POWER_TABLE.exists():
        pytest.skip("docs/03_preregistration/power_table.md not generated yet")
    assert POWER_TABLE.read_text(encoding="utf-8") == power.run()
