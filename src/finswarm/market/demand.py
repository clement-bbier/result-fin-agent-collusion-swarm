"""Calvano et al. (2020) logit demand, Bertrand-Nash and monopoly benchmarks.

``q_i = beta * exp((a - p_i/alpha)/mu) / (sum_j exp((a - p_j/alpha)/mu) + exp(a0/mu))``,
marginal cost ``c * alpha``. Prices scale exactly with ``alpha`` and profits with
``alpha * beta``. Benchmarks are solved by bisection on the first-order conditions (the
fixed-point iteration ``p <- c + mu/(1 - n s)`` diverges for the monopoly FOC).

Golden values (a=2, a0=0, mu=0.25, c=1, alpha=1, n=2): p_N = 1.4729, pi_N = 0.2229 beta,
p_M = 1.9250, pi_M = 0.3375 beta.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np

from finswarm.config.schema import MarketConfig


@dataclass(frozen=True, slots=True)
class DemandParams:
    a: float = 2.0
    a0: float = 0.0
    mu: float = 0.25
    beta: float = 100.0
    alpha: float = 1.0
    cost: float = 1.0  # c, in alpha units (marginal cost is c * alpha)

    @classmethod
    def from_market(cls, cfg: MarketConfig) -> DemandParams:
        d = cfg.demand
        return cls(a=d.a, a0=d.a0, mu=d.mu, beta=d.beta, alpha=cfg.alpha, cost=cfg.cost)

    @property
    def marginal_cost(self) -> float:
        return self.cost * self.alpha


@dataclass(frozen=True, slots=True)
class Benchmarks:
    """Symmetric per-firm benchmarks."""

    p_nash: float
    pi_nash: float
    p_mono: float
    pi_mono: float
    q_nash: float = float("nan")
    q_mono: float = float("nan")
    n: int = 2


def routed_params(cfg: MarketConfig) -> DemandParams:
    """Calvano parameters with the routing knobs folded in, as the dealer market uses them.

    ``routing.tau`` (price units) replaces ``alpha*mu`` (so ``mu = tau/alpha``) and
    ``routing.outside_utility`` shifts the outside option (``a0 + outside_utility/alpha``).
    With the defaults this is exactly ``DemandParams.from_market(cfg)``.
    """
    prm = DemandParams.from_market(cfg)
    tau, u0 = cfg.routing.tau, cfg.routing.outside_utility
    mu = prm.mu if tau is None else float(tau) / prm.alpha
    a0 = prm.a0 if u0 == 0.0 else prm.a0 + float(u0) / prm.alpha
    return prm if (mu == prm.mu and a0 == prm.a0) else replace(prm, mu=mu, a0=a0)


def _params(cfg: MarketConfig | DemandParams) -> DemandParams:
    return cfg if isinstance(cfg, DemandParams) else DemandParams.from_market(cfg)


def shares(prices: np.ndarray | list[float], cfg: MarketConfig | DemandParams) -> np.ndarray:
    """Logit market shares along the last axis; ``inf``/``nan`` prices mean "not quoting"."""
    prm = _params(cfg)
    p = np.asarray(prices, dtype=np.float64)
    x = (prm.a - p / prm.alpha) / prm.mu
    x = np.where(np.isfinite(x), x, -np.inf)
    x0 = prm.a0 / prm.mu
    m = np.maximum(np.max(x, axis=-1, keepdims=True), x0)
    e = np.exp(x - m)
    e0 = np.exp(x0 - m)
    return e / (np.sum(e, axis=-1, keepdims=True) + e0)


def quantities(prices: np.ndarray | list[float], cfg: MarketConfig | DemandParams) -> np.ndarray:
    return _params(cfg).beta * shares(prices, cfg)


def profits(prices: np.ndarray | list[float], cfg: MarketConfig | DemandParams) -> np.ndarray:
    """Per-firm profits ``(p_i - c*alpha) * q_i`` (0 for firms that do not quote)."""
    prm = _params(cfg)
    p = np.asarray(prices, dtype=np.float64)
    q = quantities(p, prm)
    with np.errstate(invalid="ignore"):
        return np.where(q > 0, (p - prm.marginal_cost) * q, 0.0)


def _bisect(f: Callable[[float], float], lo: float, hi: float, tol: float = 1e-13) -> float:
    flo, fhi = f(lo), f(hi)
    if not (flo <= 0.0 <= fhi):
        raise ValueError(
            f"bisection bracket does not straddle the root: f({lo})={flo}, f({hi})={fhi}"
        )
    for _ in range(2000):
        mid = 0.5 * (lo + hi)
        if f(mid) < 0.0:
            lo = mid
        else:
            hi = mid
        if hi - lo <= tol * max(1.0, abs(mid)):
            break
    return 0.5 * (lo + hi)


def _sym_terms(p: float, prm: DemandParams, n: int) -> tuple[float, float]:
    """(own share s, outside share s0) when all ``n`` firms charge ``p``."""
    r = math.exp(min((prm.a - p / prm.alpha - prm.a0) / prm.mu, 700.0))  # e / e0
    s0 = 1.0 / (n * r + 1.0)
    return r * s0, s0


def nash_price(cfg: MarketConfig | DemandParams, n: int | None = None) -> float:
    """Symmetric Bertrand-Nash price: root of ``p - c*alpha - alpha*mu / (1 - s(p))``."""
    prm = _params(cfg)
    n = n if n is not None else _n(cfg)
    c = prm.marginal_cost
    am = prm.alpha * prm.mu

    def g(p: float) -> float:
        s, _ = _sym_terms(p, prm, n)
        return p - c - am / (1.0 - s)

    hi = c + am / (1.0 - _sym_terms(c, prm, n)[0]) + 1e-9
    return _bisect(g, c, hi)


def monopoly_price(cfg: MarketConfig | DemandParams, n: int | None = None) -> float:
    """Symmetric joint-profit-maximising price: root of ``p - c*alpha - alpha*mu / s0(p)``."""
    prm = _params(cfg)
    n = n if n is not None else _n(cfg)
    c = prm.marginal_cost
    am = prm.alpha * prm.mu

    def g(p: float) -> float:
        _, s0 = _sym_terms(p, prm, n)
        return p - c - am / s0

    hi = c + am / _sym_terms(c, prm, n)[1] + 1e-9
    return _bisect(g, c, hi)


def best_response_price(
    rival_prices: np.ndarray | list[float], cfg: MarketConfig | DemandParams
) -> float:
    """Static best response to fixed rival prices (own FOC ``p = c*alpha + alpha*mu/(1-s_i)``)."""
    prm = _params(cfg)
    rivals = np.asarray(rival_prices, dtype=np.float64)
    c = prm.marginal_cost
    am = prm.alpha * prm.mu

    def own_share(p: float) -> float:
        return float(shares(np.concatenate([[p], rivals]), prm)[0])

    def g(p: float) -> float:
        # ``1 - s_i`` can underflow to 0 at small tau (near price-priority routing, e.g. s4 =
        # 0.02) when the own price undercuts every rival: floor it instead of dividing by zero.
        return p - c - am / max(1.0 - own_share(p), 1e-12)

    # g(c) < 0 and g is increasing; expand the bracket geometrically until g(hi) > 0 instead of
    # using c + am / (1 - s_i(c)), which is infinite when s_i(c) rounds to 1.
    hi = c + 2.0 * am
    while g(hi) <= 0.0 and hi - c < am * 1e12:
        hi = c + 2.0 * (hi - c)
    return _bisect(g, c, hi)


def _n(cfg: MarketConfig | DemandParams) -> int:
    if isinstance(cfg, DemandParams):
        raise ValueError("pass n explicitly when using DemandParams")
    return int(cfg.n_dealers)


def benchmarks(cfg: MarketConfig | DemandParams, n: int | None = None) -> Benchmarks:
    prm = _params(cfg)
    n = n if n is not None else _n(cfg)
    pn = nash_price(prm, n)
    pm = monopoly_price(prm, n)
    qn = float(quantities(np.full(n, pn), prm)[0])
    qm = float(quantities(np.full(n, pm), prm)[0])
    return Benchmarks(
        p_nash=pn,
        pi_nash=(pn - prm.marginal_cost) * qn,
        p_mono=pm,
        pi_mono=(pm - prm.marginal_cost) * qm,
        q_nash=qn,
        q_mono=qm,
        n=n,
    )


def delta(pi: float | np.ndarray, bench: Benchmarks) -> float | np.ndarray:
    """Profit gain index ``(pi - pi_N) / (pi_M - pi_N)`` (per-firm profit)."""
    return (np.asarray(pi, dtype=np.float64) - bench.pi_nash) / (bench.pi_mono - bench.pi_nash)


def delta_price(p: float | np.ndarray, bench: Benchmarks) -> float | np.ndarray:
    """Price-based index ``(p - p_N) / (p_M - p_N)``."""
    return (np.asarray(p, dtype=np.float64) - bench.p_nash) / (bench.p_mono - bench.p_nash)
