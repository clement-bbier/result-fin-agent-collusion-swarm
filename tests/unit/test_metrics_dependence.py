import numpy as np
import pytest
from _metrics_synth import agent_period, colluding_prices, competitive_prices

from finswarm.metrics import dependence as D


def _leader_follower(t: int = 300, seed: int = 0, noise: float = 0.3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    lead = rng.normal(size=t)
    follow = np.r_[0.0, lead[:-1]] + noise * rng.normal(size=t)
    return np.column_stack([lead, follow])


def test_xcorr_known_value_and_direction() -> None:
    x = _leader_follower(noise=0.0)
    r = D.lagged_xcorr(x, max_lag=2)
    assert r.matrix[0, 1] == pytest.approx(1.0, abs=1e-9)  # leader -> follower, lag 1
    assert r.matrix[1, 0] < 0.2
    assert np.isnan(r.matrix[0, 0])


def test_dependence_separates_colluding_from_competitive() -> None:
    col = agent_period(colluding_prices(200, seed=3))
    comp = agent_period(competitive_prices(200, seed=3))
    xc_c = D.lagged_xcorr(col, diff=True).summary
    xc_n = D.lagged_xcorr(comp, diff=True).summary
    assert xc_c > 0.4 > 0.2 > xc_n
    g_c = D.granger_pairwise(col, diff=True)
    g_n = D.granger_pairwise(comp, diff=True)
    assert g_c.p[0, 1] < 1e-6 and g_c.share_significant >= 0.5
    assert g_n.share_significant == 0.0
    te_c = D.transfer_entropy(col, diff=True, n_surrogates=60)
    te_n = D.transfer_entropy(comp, diff=True, method="quantile", n_surrogates=60)
    assert te_c.te_net[0, 1] > 0.3 and te_c.p[0, 1] < 0.05
    assert abs(te_n.summary) < 0.03


def test_granger_handles_constant_series() -> None:
    x = np.column_stack([np.zeros(50), np.random.default_rng(0).normal(size=50)])
    g = D.granger_pairwise(x)
    assert g.p[0, 1] == 1.0 and g.p[1, 0] == 1.0


def test_transfer_entropy_plugin_value() -> None:
    # y copies x with one lag, x iid uniform over 3 symbols: TE_{x->y} -> log 3
    rng = np.random.default_rng(1)
    x = rng.integers(0, 3, 4000) - 1.0
    y = np.r_[0.0, x[:-1]]
    r = D.transfer_entropy(np.column_stack([x, y]), n_surrogates=20)
    assert r.te[0, 1] == pytest.approx(np.log(3), abs=0.01)
    assert r.te[1, 0] == pytest.approx(0.0, abs=0.01)


def test_conditional_mutual_information() -> None:
    rng = np.random.default_rng(2)
    s = rng.integers(0, 2, 5000)
    x = (s + (rng.random(5000) < 0.5)) % 2
    y = x.copy()  # perfectly dependent: I(X;Y|S) = H(X|S) = log 2
    assert D.cond_mutual_info(x, y, s, 2, 2, 2) == pytest.approx(np.log(2), abs=0.01)
    z = rng.integers(0, 2, 5000)
    assert D.cond_mutual_info(x, z, s, 2, 2, 2) == pytest.approx(0.0, abs=0.01)


def test_net_influence_rps_always_colluding_pair_detected() -> None:
    res = D.validate_net_influence_rps(n_games=60, collusion_prob=1.0, n_trials=30, n_boot=200)
    assert res["detection_rate"] >= 0.95
    assert res["top_pair_rate"] >= 0.95
    assert res["false_flag_rate"] <= 0.10


def test_net_influence_conditioning_removes_common_input() -> None:
    a, s = D.simulate_rps(400, collusion_prob=0.0, common_input=0.8, seed=5)
    uncond = D.net_influence(a, None, n_boot=100, seed=1)
    cond = D.net_influence(a, s, n_boot=100, seed=1)
    assert uncond.gamma[0, 1] > 0.1  # shared public input makes raw actions dependent
    assert cond.gamma[0, 1] < 0.03  # ... but not given the market state
    assert cond.flagged == []


def test_net_influence_symmetric_cartel_is_not_a_pair() -> None:
    # all three move together: Gamma subtracts the third party's influence
    rng = np.random.default_rng(3)
    base = rng.integers(0, 3, 200)
    a = np.column_stack([base, base, base])
    r = D.net_influence(a, None, n_boot=50)
    assert r.statistic == pytest.approx(0.0, abs=1e-9)
    assert r.flagged == []


def _common_input_run(seed: int, coupled: bool) -> np.ndarray:
    rng = np.random.default_rng(seed)
    common = np.zeros(200)
    for t in range(1, 200):  # autocorrelated common shock seen by both dealers
        common[t] = 0.8 * common[t - 1] + rng.normal()
    a = common + 0.5 * rng.normal(size=200)
    b = common + 0.5 * rng.normal(size=200)
    if coupled:
        b[1:] += 0.8 * a[:-1]
    return np.column_stack([a, b])


def _xc(x: np.ndarray) -> float:
    return D.lagged_xcorr(x).summary


def test_net_of_ghost_removes_common_input_correlation() -> None:
    # the raw statistic is large under a common input alone ...
    assert _xc(_common_input_run(0, False)) > 0.4
    # ... but net of ghost runs (same common input, no interaction) it is null and the
    # permutation test is calibrated: rejection rate ~ 5 % over replicated campaigns
    pvals, stats_ = [], []
    for rep in range(20):
        ghost = [_common_input_run(1000 * rep + k, False) for k in range(8)]
        treat = [_common_input_run(1000 * rep + 500 + k, False) for k in range(8)]
        r = D.net_of_ghost(_xc, treat, ghost, n_perm=500, seed=rep)
        pvals.append(r.p_value)
        stats_.append(r.stat)
    assert np.mean(np.array(pvals) < 0.05) <= 0.15
    assert abs(np.mean(stats_)) < 0.03
    # genuine interaction survives the ghost baseline (unpaired and paired tests)
    ghost = [_common_input_run(100 + k, False) for k in range(8)]
    coupled = [_common_input_run(300 + k, True) for k in range(8)]
    alt = D.net_of_ghost(_xc, coupled, ghost, n_perm=2000)
    assert alt.stat > 0.1 and alt.p_value < 0.01
    assert alt.baseline.shape == (8,) and alt.treatment.shape == (8,)
    paired = D.net_of_ghost(_xc, coupled, ghost, n_perm=2000, paired=True)
    assert paired.paired and paired.p_value < 0.01


@pytest.mark.slow
def test_net_influence_rps_power_report() -> None:
    always = D.validate_net_influence_rps(n_games=60, collusion_prob=1.0, n_trials=100)
    cp40 = D.validate_net_influence_rps(n_games=300, collusion_prob=0.4, n_trials=100)
    weak = D.validate_net_influence_rps(n_games=60, collusion_prob=0.2, n_trials=100)
    print("RPS always@60", always, "cp0.4@300", cp40, "cp0.2@60", weak)
    assert always["detection_rate"] >= 0.95
    assert cp40["detection_rate"] >= 0.95
    assert weak["detection_rate"] < always["detection_rate"]
