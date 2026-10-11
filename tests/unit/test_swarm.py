import numpy as np
import pytest
from scipy import stats

from finswarm.config.schema import SwarmConfig
from finswarm.core.types import SwarmType
from finswarm.swarm import kernels
from finswarm.swarm.population import alias_table, build_population
from finswarm.swarm.shocks import apply_info_shock, geometric_delays, remove_liquidity
from finswarm.swarm.state import SwarmState


def test_alias_sampling_frequencies_proportional_to_lam() -> None:
    rng = np.random.default_rng(11)
    lam = rng.lognormal(0.0, 0.8, size=40)
    active = np.ones(40, dtype=bool)
    active[[3, 17]] = False
    lam[25] = 0.0
    prob, alias, total = alias_table(lam * active)
    assert total == pytest.approx((lam * active).sum())
    draws = kernels.alias_sample_many(prob, alias, rng.random(400_000))
    counts = np.bincount(draws, minlength=40)
    assert counts[3] == counts[17] == counts[25] == 0
    w = lam * active
    keep = w > 0
    expected = w[keep] / w[keep].sum() * counts.sum()
    chi2, p = stats.chisquare(counts[keep], expected)
    assert p > 1e-3, (chi2, p)


def test_poisson_inversion_moments() -> None:
    u = np.random.default_rng(1).random(50_000)
    for lam in (0.5, 7.0, 29.0):
        x = np.array([kernels.poisson_inv(v, lam) for v in u[:20_000]])
        assert x.mean() == pytest.approx(lam, rel=0.03)
        assert x.var() == pytest.approx(lam, rel=0.06)
    assert kernels.n_poisson_pieces(100.0) == 4


def _small_state(n: int, rng: np.random.Generator, attention: float = 0.3) -> SwarmState:
    st = SwarmState.empty(n)
    st.sentiment[:] = rng.standard_normal(n)
    st.attention[:] = attention
    st.belief_F[:] = 2.0
    return st


def test_lazy_refresh_equals_eager_refresh_deterministic_part() -> None:
    rho, sigma, n, horizon = 0.9, 0.0, 40, 60
    rng = np.random.default_rng(5)
    eager = _small_state(n, rng)
    lazy = eager.copy()
    shocks = {7: 0.5, 21: -0.2}
    g_e, g_l = np.random.default_rng(9), np.random.default_rng(9)
    f_e = f_l = 2.0
    zeros = np.zeros(n)
    for t in range(1, horizon + 1):
        if t in shocks:
            f_e = apply_info_shock(
                eager,
                tick=t,
                fundamental=f_e,
                d_fundamental=shocks[t],
                diffusion="attention",
                rng=g_e,
            )
            f_l = apply_info_shock(
                lazy,
                tick=t,
                fundamental=f_l,
                d_fundamental=shocks[t],
                diffusion="attention",
                rng=g_l,
            )
        kernels.refresh_eager_step(
            t,
            zeros,
            eager.sentiment,
            eager.last_update,
            eager.belief_F,
            eager.shock_delay,
            rho,
            sigma,
            f_e,
        )
        ids = np.flatnonzero(rng.random(n) < 0.1)  # a few lazy arrivals
        kernels.refresh_lazy_many(
            ids,
            t,
            zeros[: ids.size],
            lazy.sentiment,
            lazy.last_update,
            lazy.belief_F,
            lazy.shock_delay,
            rho,
            sigma,
            f_l,
        )
    ids = np.arange(n)
    kernels.refresh_lazy_many(
        ids,
        horizon,
        zeros,
        lazy.sentiment,
        lazy.last_update,
        lazy.belief_F,
        lazy.shock_delay,
        rho,
        sigma,
        f_l,
    )
    assert np.allclose(lazy.sentiment, eager.sentiment, rtol=1e-12, atol=1e-15)
    assert np.array_equal(lazy.belief_F, eager.belief_F)
    assert np.array_equal(lazy.shock_delay, eager.shock_delay)
    assert np.array_equal(lazy.last_update, eager.last_update)


def test_lazy_refresh_matches_eager_in_distribution() -> None:
    rho, sigma, n, k = 0.95, 0.05, 20_000, 40
    rng = np.random.default_rng(2)
    eager = SwarmState.empty(n)
    lazy = SwarmState.empty(n)
    for t in range(1, k + 1):
        kernels.refresh_eager_step(
            t,
            rng.standard_normal(n),
            eager.sentiment,
            eager.last_update,
            eager.belief_F,
            eager.shock_delay,
            rho,
            sigma,
            2.0,
        )
    kernels.refresh_lazy_many(
        np.arange(n),
        k,
        rng.standard_normal(n),
        lazy.sentiment,
        lazy.last_update,
        lazy.belief_F,
        lazy.shock_delay,
        rho,
        sigma,
        2.0,
    )
    sd = kernels.ar1_gap_sd(rho, sigma, k)
    assert lazy.sentiment.std() == pytest.approx(sd, rel=0.03)
    assert stats.ks_2samp(lazy.sentiment, eager.sentiment).pvalue > 1e-3


def test_shock_delays_are_geometric() -> None:
    p, n = 0.2, 100_000
    d = geometric_delays(np.full(n, p), np.random.default_rng(4))
    assert d.min() == 1
    assert d.mean() == pytest.approx(1 / p, rel=0.02)
    ks = np.arange(1, 16)
    observed = np.array([(d == k).sum() for k in ks] + [(d > 15).sum()])
    pmf = np.append(p * (1 - p) ** (ks - 1), (1 - p) ** 15)
    assert stats.chisquare(observed, pmf * n).pvalue > 1e-3
    st = SwarmState.empty(5)
    st.attention[:] = 1.0  # fully attentive agents learn at the shock tick
    f = apply_info_shock(
        st,
        tick=10,
        fundamental=2.0,
        d_fundamental=0.3,
        diffusion="attention",
        rng=np.random.default_rng(0),
    )
    assert f == pytest.approx(2.3) and np.all(st.shock_delay == 10)


def test_population_sampling() -> None:
    cfg = SwarmConfig(n_agents=50_000)
    a = build_population(cfg, np.random.default_rng(0))
    b = build_population(cfg, np.random.default_rng(0))
    for key, arr in a.state().items():
        assert np.array_equal(arr, b.state()[key])
    freq = np.bincount(a.stype, minlength=6) / a.n
    for name, w in cfg.type_weights.items():
        assert freq[SwarmType[name]] == pytest.approx(w, abs=0.01)
    assert a.lam.mean() == pytest.approx(cfg.arrivals.lam_mean, rel=0.03)
    informed = a.stype == SwarmType.INFORMED
    assert np.all(a.attention[informed] == 1.0) and np.all(a.noise_sd[informed] == 0.0)
    assert a.attention[~informed].mean() == pytest.approx(cfg.attention_mean, rel=0.05)
    assert np.all(a.trend_k[a.stype != SwarmType.TREND] == 0.0)
    with pytest.raises(ValueError, match="unknown swarm type"):
        build_population(SwarmConfig(type_weights={"WHALE": 1.0}), np.random.default_rng(0))


def test_remove_liquidity_updates_alias() -> None:
    cfg = SwarmConfig(n_agents=1000, type_weights={"ZI_BUYER": 0.5, "MM": 0.5})
    st = build_population(cfg, np.random.default_rng(1))
    before = st.total_intensity()
    removed = remove_liquidity(st, 0.5, ["MM"], np.random.default_rng(2))
    assert removed.size == round(0.5 * (st.stype == SwarmType.MM).sum())
    assert not st.active[removed].any()
    assert st.total_intensity() < before
    assert st.total_intensity() == pytest.approx((st.lam * st.active).sum())
