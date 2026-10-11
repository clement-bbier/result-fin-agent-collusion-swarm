"""Sample swarm heterogeneity from ``SwarmConfig`` (``population`` stream only).

Every field is drawn for every agent in a fixed order, independently of the type mix, so
changing ``type_weights`` changes types but not the other draws (common random numbers).
"""

from __future__ import annotations

import math

import numpy as np

from finswarm.config.schema import RunConfig, SwarmConfig
from finswarm.core.rng import RNGRegistry
from finswarm.core.types import SwarmType
from finswarm.swarm.kernels import build_alias
from finswarm.swarm.state import SwarmState

#: noise traders' per-arrival signal noise, as a multiple of ``signal_noise_sd``
NOISE_SD_MULT = 5.0
#: concentration of the Beta distribution of attention (mean = ``attention_mean``)
ATTENTION_KAPPA = 4.0
#: inventory cap of heuristic market makers (stub in the dealer mechanism)
MM_MAX_INV = 10.0


def swarm_size(cfg: RunConfig) -> int:
    """Number of swarm agents present in the market (ids ``0..N-1``); 0 unless ``agents``."""
    return cfg.swarm.n_agents if cfg.swarm.mode == "agents" else 0


def parse_type_weights(weights: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(type_values int8, probabilities)``; names must be ``SwarmType`` names."""
    if not weights:
        raise ValueError("swarm.type_weights is empty")
    values, probs = [], []
    for name, w in weights.items():
        try:
            values.append(int(SwarmType[name]))
        except KeyError as exc:
            raise ValueError(
                f"unknown swarm type {name!r}; expected one of {[t.name for t in SwarmType]}"
            ) from exc
        if not (w >= 0.0 and math.isfinite(w)):
            raise ValueError(f"type weight for {name} must be finite and >= 0, got {w}")
        probs.append(float(w))
    p = np.asarray(probs, dtype=np.float64)
    if p.sum() <= 0:
        raise ValueError("swarm.type_weights must have a positive sum")
    return np.asarray(values, dtype=np.int8), p / p.sum()


def alias_table(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Vose alias table over non-negative weights (e.g. ``lam * active``)."""
    w = np.ascontiguousarray(weights, dtype=np.float64)
    if np.any(w < 0) or not np.all(np.isfinite(w)):
        raise ValueError("alias weights must be finite and non-negative")
    prob, alias, total = build_alias(w)
    return prob, alias, float(total)


def build_population(
    cfg: SwarmConfig,
    rng: np.random.Generator,
    *,
    fundamental: float = 2.0,
    horizon_ticks: int = 50,
) -> SwarmState:
    """Draw a swarm of ``cfg.n_agents`` agents.

    * ``lam`` ~ lognormal with mean ``arrivals.lam_mean`` and log-sd ``arrivals.lam_sd_log``;
    * ``value_base`` ~ N(0, value_private_sd) (private value deviation, alpha units);
    * ``noise_sd`` = signal_noise_sd (ZI, trend, sellers, MM), 0 (informed),
      ``NOISE_SD_MULT * signal_noise_sd`` (noise traders);
    * sentiment deviation from its stationary law N(0, sigma^2 / (1 - rho^2));
    * attention ~ Beta with mean ``attention_mean`` (informed agents: 1);
    * trend loading U(0.5, 1.5) for trend followers; horizon U(0.5, 1.5) * horizon_ticks.
    """
    n = int(cfg.n_agents)
    if n < 0:
        raise ValueError("swarm.n_agents must be >= 0")
    if not 0.0 < cfg.attention_mean <= 1.0:
        raise ValueError("swarm.attention_mean must lie in (0, 1]")
    values, probs = parse_type_weights(cfg.type_weights)

    # fixed draw order (each n-sized), independent of the type mix
    type_idx = rng.choice(len(values), size=n, p=probs)
    z_lam = rng.standard_normal(n)
    z_val = rng.standard_normal(n)
    z_sent = rng.standard_normal(n)
    m = cfg.attention_mean
    if m >= 1.0:
        att = np.ones(n)
        rng.random(n)  # keep the draw count fixed
    else:
        att = rng.beta(ATTENTION_KAPPA * m, ATTENTION_KAPPA * (1.0 - m), size=n)
    u_trend = rng.random(n)
    u_horizon = rng.random(n)
    z_aversion = rng.standard_normal(n)

    st = SwarmState.empty(n)
    stype = values[type_idx]
    st.stype[:] = stype
    sd = cfg.arrivals.lam_sd_log
    st.lam[:] = cfg.arrivals.lam_mean * np.exp(sd * z_lam - 0.5 * sd * sd)
    st.value_base[:] = cfg.value_private_sd * z_val

    noise = np.full(n, cfg.signal_noise_sd)
    noise[stype == SwarmType.INFORMED] = 0.0
    noise[stype == SwarmType.NOISE] = NOISE_SD_MULT * cfg.signal_noise_sd
    st.noise_sd[:] = noise

    rho, sigma = cfg.sentiment_rho, cfg.sentiment_sigma
    stat_sd = sigma / math.sqrt(1.0 - rho * rho) if rho < 1.0 else 0.0
    st.sentiment[:] = stat_sd * z_sent

    att = np.clip(att, 1e-6, 1.0)
    att[stype == SwarmType.INFORMED] = 1.0
    st.attention[:] = att

    st.trend_k[:] = np.where(stype == SwarmType.TREND, 0.5 + u_trend, 0.0)
    st.horizon[:] = np.maximum(1, np.rint(horizon_ticks * (0.5 + u_horizon))).astype(np.int64)
    st.aversion[:] = np.where(stype == SwarmType.MM, np.exp(0.5 * z_aversion), 0.0)
    st.max_inv[:] = np.where(stype == SwarmType.MM, MM_MAX_INV, np.inf)
    st.belief_F[:] = fundamental
    st.invalidate_alias()
    return st


def build_population_for_run(cfg: RunConfig, rng: RNGRegistry) -> SwarmState:
    return build_population(
        cfg.swarm,
        rng["population"],
        fundamental=cfg.market.demand.a,
        horizon_ticks=cfg.market.fringe.horizon_ticks,
    )
