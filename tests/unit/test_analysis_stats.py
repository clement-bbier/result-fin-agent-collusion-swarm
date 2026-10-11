import numpy as np
import pytest
from scipy import stats as sps

from finswarm.analysis import stats as S


def test_bca_matches_scipy_and_is_deterministic() -> None:
    x = np.random.default_rng(7).normal(0.3, 0.2, 12)
    ci = S.bca_ci(x)  # B = 10,000, fixed seed
    ref = sps.bootstrap(
        (x,), np.mean, n_resamples=10_000, method="BCa", rng=np.random.default_rng(1)
    ).confidence_interval
    assert ci.estimate == pytest.approx(x.mean())
    assert ci.lo == pytest.approx(ref.low, abs=0.01)
    assert ci.hi == pytest.approx(ref.high, abs=0.01)
    assert S.bca_ci(x) == ci
    assert ci.B == 10_000 and ci.method == "bca"
    # skewed data: BCa shifts the interval relative to the percentile one
    y = np.random.default_rng(3).exponential(1.0, 15)
    med = S.bca_ci(y, stat=np.median, B=4000)
    assert med.lo < np.median(y) < med.hi


def test_bca_degenerate_inputs() -> None:
    c = S.bca_ci([0.4] * 6)
    assert c.lo == c.estimate == c.hi == pytest.approx(0.4)
    assert np.isnan(S.bca_ci([]).estimate)
    p = S.paired_bca([1.0, 2.0, 3.0, 4.0], [0.5, 1.5, 2.0, 3.6], B=2000)
    assert p.estimate == pytest.approx(np.mean([0.5, 0.5, 1.0, 0.4]))


def test_paired_t_and_wilcoxon_match_scipy() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(0.5, 0.2, 10)
    b = a - 0.2 + rng.normal(0, 0.1, 10)
    t = S.paired_t(a, b)
    ref = sps.ttest_rel(a, b)
    assert t["t"] == pytest.approx(ref.statistic) and t["p"] == pytest.approx(ref.pvalue)
    assert t["lo"] < t["mean"] < t["hi"]
    w = S.wilcoxon(a, b)
    assert w["p"] == pytest.approx(sps.wilcoxon(a - b).pvalue)
    assert S.wilcoxon([1, 2], [1, 2])["p"] == 1.0


def test_holm_known_values() -> None:
    adj, rej = S.holm([0.01, 0.04, 0.03, 0.005])
    assert adj.tolist() == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert rej.tolist() == [True, False, False, True]
    adj2, rej2 = S.holm([0.2, float("nan")])
    assert adj2[0] == pytest.approx(0.2) and np.isnan(adj2[1]) and not rej2[1]


def test_tost_known_inputs() -> None:
    d = np.array([0.1, -0.1] * 10)
    r = S.tost(d, -0.2, 0.2)
    se = d.std(ddof=1) / np.sqrt(d.size)
    assert r["p_lower"] == pytest.approx(sps.t.sf(0.2 / se, 19))
    assert r["p_upper"] == pytest.approx(sps.t.cdf(-0.2 / se, 19))
    assert r["equivalent"] and r["p"] < 0.001
    far = S.tost(d + 0.5)
    assert not far["equivalent"] and far["p"] > 0.5
    # a small, noisy sample cannot establish equivalence
    noisy = S.tost(np.array([0.3, -0.3, 0.25, -0.2]))
    assert not noisy["equivalent"]


def test_cluster_proportion_ci() -> None:
    r = S.cluster_proportion_ci([1, 1, 0, 1], [2, 2, 2, 2])
    assert r["p"] == pytest.approx(0.375)
    assert r["se"] == pytest.approx(0.125)
    assert r["lo"] == pytest.approx(0.375 - 1.959964 * 0.125, abs=1e-5)
    assert S.cluster_proportion_ci([0, 0], [3, 3])["hi"] == 0.0


def test_paired_contrast_bundle() -> None:
    sham = np.array([0.6, 0.55, 0.7, 0.62, 0.58, 0.66, 0.61, 0.59])
    lever = sham - np.array([0.3, 0.25, 0.35, 0.28, 0.31, 0.27, 0.33, 0.29])
    c = S.paired_contrast(sham, lever, B=3000)
    assert c["n"] == 8 and c["mean"] == pytest.approx(0.2975)
    assert c["bca_lo"] > 0.25 and c["t"]["p"] < 1e-6 and not c["tost"]["equivalent"]
