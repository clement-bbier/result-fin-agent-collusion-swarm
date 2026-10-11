"""Competitive (Nash) and collusive (joint-monopoly) benchmarks.

Three routes, as in the analysis plan (SAP, D.3):

* :func:`logit_closed_form` -- the Calvano et al. (2020) / Fish et al. (2024) logit
  demand, solved by bisection on the first-order conditions. It wraps the sim-core solver
  ``finswarm.market.demand.benchmarks`` (single source of truth) when that module is
  importable and falls back to an equivalent local bisection otherwise;
* :func:`estimate_by_sweeps` -- any market with a (Monte Carlo) expected-profit function:
  Nash by best-response iteration on a price grid from several starts, monopoly by joint
  argmax;
* :func:`scripted_competitive` -- the empirical competitive benchmark measured on runs
  where every dealer is a scripted competitor (robustness ``Delta_emp``).

Logit demand (normalised prices ``p~ = p / alpha``)::

    q_i = beta * exp((a - p~_i) / mu) / (sum_j exp((a - p~_j) / mu) + exp(a0 / mu))
    pi_i = (p_i - alpha * c) * q_i
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from finswarm.metrics._frames import require, window

ProfitFn = Callable[[np.ndarray], np.ndarray]
BestResponseFn = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class Benchmarks:
    """Per-firm benchmark prices and per-firm per-period profits."""

    p_nash: float
    p_mono: float
    pi_nash: float
    pi_mono: float
    method: str = "closed_form"
    params: dict[str, Any] = field(default_factory=dict, hash=False, compare=False)

    def delta(self, profit: Any) -> Any:
        """Calvano profit gain ``(pi - pi_N) / (pi_M - pi_N)``."""
        return (np.asarray(profit, dtype=float) - self.pi_nash) / (self.pi_mono - self.pi_nash)

    def delta_price(self, price: Any) -> Any:
        """Price analogue ``(p - p_N) / (p_M - p_N)``."""
        return (np.asarray(price, dtype=float) - self.p_nash) / (self.p_mono - self.p_nash)

    def with_competitive(self, comp: CompetitiveBenchmark) -> Benchmarks:
        """Replace the Nash point by an empirical scripted-competitor benchmark."""
        return replace(
            self,
            p_nash=comp.p_comp,
            pi_nash=comp.pi_comp,
            method=f"{self.method}+scripted_competitive",
            params={**self.params, "competitive": asdict(comp)},
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Benchmarks:
        return cls(
            **{k: d[k] for k in ("p_nash", "p_mono", "pi_nash", "pi_mono")},
            method=d.get("method", "closed_form"),
            params=d.get("params", {}),
        )

    def sha256(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, default=float).encode()
        return hashlib.sha256(blob).hexdigest()

    def save(self, path: str | Path) -> str:
        """Write JSON (with its own hash) and return the hash."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        d = self.to_dict()
        d["sha256"] = self.sha256()
        p.write_text(json.dumps(d, indent=2, sort_keys=True, default=float), encoding="utf-8")
        return d["sha256"]

    @classmethod
    def load(cls, path: str | Path) -> Benchmarks:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True)
class CompetitiveBenchmark:
    p_comp: float
    pi_comp: float
    n_runs: int
    n_obs: int
    sd_pi_runs: float


# --------------------------------------------------------------------------- logit demand


def logit_shares(
    prices: Any, a: float = 2.0, a0: float = 0.0, mu: float = 0.25, alpha: float = 1.0
) -> np.ndarray:
    """Market shares (without ``beta``) for prices of shape ``(..., n)``."""
    p = np.asarray(prices, dtype=float) / alpha
    u = (a - p) / mu
    u0 = a0 / mu
    m = np.maximum(u.max(axis=-1, keepdims=True), u0)
    e = np.exp(u - m)
    return e / (e.sum(axis=-1, keepdims=True) + np.exp(u0 - m))


def logit_profits(
    prices: Any,
    a: float = 2.0,
    a0: float = 0.0,
    mu: float = 0.25,
    c: float = 1.0,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> np.ndarray:
    """Per-firm profits for prices of shape ``(..., n)``."""
    p = np.asarray(prices, dtype=float)
    return (p - alpha * c) * beta * logit_shares(p, a, a0, mu, alpha)


@dataclass(frozen=True)
class LogitParams:
    """Logit market parameters in the units of ``config.schema.MarketConfig``.

    Routing: a buyer of value ``v`` picks dealer ``i`` with probability proportional to
    ``exp((v - p_i) / tau)`` against ``exp(u0 / tau)``; ``tau = alpha * mu`` and ``u0 =
    alpha * a0 (+ outside_utility)`` reproduce Calvano's demand with ``v = alpha * a``.
    """

    a: float = 2.0
    a0: float = 0.0
    mu: float = 0.25
    c: float = 1.0
    alpha: float = 1.0
    beta: float = 1.0
    tau: float | None = None
    outside_utility: float = 0.0

    @property
    def tau_eff(self) -> float:
        return self.alpha * self.mu if self.tau is None else float(self.tau)

    @property
    def u0(self) -> float:
        return self.alpha * self.a0 + self.outside_utility

    @property
    def marginal_cost(self) -> float:
        return self.c * self.alpha

    @classmethod
    def coerce(cls, obj: Any = None) -> LogitParams:
        """Accept ``LogitParams``, a dict, a ``MarketConfig`` or a sim-core ``DemandParams``."""
        if obj is None:
            return cls()
        if isinstance(obj, cls):
            return obj
        if isinstance(obj, dict):
            return cls(**obj)
        if hasattr(obj, "demand") and hasattr(obj, "alpha"):  # config.schema.MarketConfig
            d, r = obj.demand, getattr(obj, "routing", None)
            return cls(
                a=d.a,
                a0=d.a0,
                mu=d.mu,
                beta=d.beta,
                c=obj.cost,
                alpha=obj.alpha,
                tau=getattr(r, "tau", None),
                outside_utility=getattr(r, "outside_utility", 0.0),
            )
        if hasattr(obj, "cost") and hasattr(obj, "mu"):  # finswarm.market.demand.DemandParams
            return cls(a=obj.a, a0=obj.a0, mu=obj.mu, beta=obj.beta, c=obj.cost, alpha=obj.alpha)
        raise TypeError(f"cannot read logit parameters from {type(obj).__name__}")


def _bisect(f: Callable[[float], float], lo: float, hi: float, tol: float = 1e-13) -> float:
    flo = f(lo)
    for _ in range(200):
        if np.sign(f(hi)) != np.sign(flo):
            break
        hi = lo + 2.0 * (hi - lo)
    else:  # pragma: no cover - defensive
        raise ValueError("bisection: no sign change")
    for _ in range(400):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if np.sign(fm) == np.sign(flo):
            lo, flo = mid, fm
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def _sim_core_demand() -> Any:
    try:
        from finswarm.market import demand  # sim-core solver, single source of truth
    except ImportError:
        return None
    return demand if hasattr(demand, "benchmarks") and hasattr(demand, "DemandParams") else None


def logit_closed_form(
    a: float = 2.0,
    a0: float = 0.0,
    mu: float = 0.25,
    c: float = 1.0,
    alpha: float = 1.0,
    n: int = 2,
    beta: float = 1.0,
    solver: str = "auto",
) -> Benchmarks:
    """Symmetric Nash and joint-monopoly benchmarks of the logit market, by bisection.

    Nash FOC: ``p~ - c = mu / (1 - s)``; monopoly FOC: ``p~ - c = mu / (1 - n s)``.
    Golden values (a=2, a0=0, mu=0.25, c=1, alpha=1, n=2): p_N=1.472927, p_M=1.924981,
    pi_N=0.222927*beta, pi_M=0.337490*beta. ``solver``: ``"auto"`` uses
    ``finswarm.market.demand.benchmarks`` when importable, ``"local"`` forces the local
    bisection (kept for branches without sim-core and for cross-checks).
    """
    params = {"a": a, "a0": a0, "mu": mu, "c": c, "alpha": alpha, "n": n, "beta": beta}
    md = _sim_core_demand() if solver == "auto" else None
    if md is not None:
        b = md.benchmarks(md.DemandParams(a=a, a0=a0, mu=mu, beta=beta, alpha=alpha, cost=c), n)
        return Benchmarks(
            p_nash=float(b.p_nash),
            p_mono=float(b.p_mono),
            pi_nash=float(b.pi_nash),
            pi_mono=float(b.pi_mono),
            method="closed_form:market.demand",
            params=params,
        )

    def share(pt: float) -> float:
        return float(logit_shares(np.full(n, pt), a, a0, mu, 1.0)[0])

    p_n = _bisect(lambda pt: pt - c - mu / (1.0 - share(pt)), c, c + mu + 1.0)
    p_m = _bisect(lambda pt: pt - c - mu / (1.0 - n * share(pt)), c, c + mu + 1.0)
    return Benchmarks(
        p_nash=alpha * p_n,
        p_mono=alpha * p_m,
        pi_nash=alpha * beta * (p_n - c) * share(p_n),
        pi_mono=alpha * beta * (p_m - c) * share(p_m),
        method="closed_form:local",
        params=params,
    )


def benchmarks_from_config(market: Any, n: int | None = None) -> Benchmarks:
    """Closed-form benchmarks for a ``MarketConfig`` (or ``LogitParams`` plus ``n``)."""
    p = LogitParams.coerce(market)
    nn = n if n is not None else int(getattr(market, "n_dealers", 2))
    return logit_closed_form(p.a, p.a0, p.mu, p.c, p.alpha, nn, p.beta)


def logit_best_response(
    a: float = 2.0,
    a0: float = 0.0,
    mu: float = 0.25,
    c: float = 1.0,
    alpha: float = 1.0,
    iters: int = 100,
) -> BestResponseFn:
    """Vectorised best response of one firm to its rivals' prices in the logit market.

    The returned function maps rival prices of shape ``(m, n-1)`` (or ``(n-1,)``) to the
    profit-maximising own price, shape ``(m,)``. Solved by bisection on the FOC.
    """

    def br(rivals: np.ndarray) -> np.ndarray:
        r = np.atleast_2d(np.asarray(rivals, dtype=float)) / alpha
        u_r = np.exp((a - r) / mu).sum(axis=1) + np.exp(a0 / mu)
        lo = np.full(r.shape[0], c)
        hi = np.full(r.shape[0], max(a, c) + 40.0 * mu + 1.0)
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            e = np.exp((a - mid) / mu)
            s = e / (e + u_r)
            g = mid - c - mu / (1.0 - s)
            neg = g < 0
            lo = np.where(neg, mid, lo)
            hi = np.where(neg, hi, mid)
        return alpha * 0.5 * (lo + hi)

    return br


# --------------------------------------------------------------------------- sweeps


def _batched(profit_fn: ProfitFn, n: int) -> ProfitFn:
    """Wrap ``profit_fn`` so it accepts a ``(m, n)`` batch even if it only takes ``(n,)``."""
    try:
        if np.asarray(profit_fn(np.ones((2, n)))).shape == (2, n):
            return profit_fn
    except Exception:  # a strictly one-dimensional profit function: evaluate row by row
        pass

    def fn(batch: np.ndarray) -> np.ndarray:
        return np.stack([np.asarray(profit_fn(row), dtype=float) for row in batch])

    return fn


def estimate_by_sweeps(
    profit_fn: ProfitFn,
    grid: Sequence[float] | np.ndarray,
    n_agents: int = 2,
    n_starts: int = 10,
    max_iter: int = 200,
    seed: int = 0,
    symmetric_monopoly: bool = True,
    max_joint_cells: int = 2_000_000,
) -> Benchmarks:
    """Benchmarks for markets without a closed form.

    ``profit_fn`` maps a ``(m, n_agents)`` price batch to ``(m, n_agents)`` expected
    profits (use common random numbers inside for Monte Carlo markets). Nash: sequential
    best-response iteration on ``grid`` from ``n_starts`` random starts; the reported
    point averages the converged fixed points. Monopoly: joint argmax of total profit over
    a common price (``symmetric_monopoly``) or the full grid product when small enough.
    """
    g = np.sort(np.asarray(grid, dtype=float))
    fn = _batched(profit_fn, n_agents)
    rng = np.random.default_rng(seed)
    points: list[np.ndarray] = []
    profits: list[np.ndarray] = []
    n_conv = 0
    for s in range(n_starts):
        idx = np.full(n_agents, len(g) // 2) if s == 0 else rng.integers(0, len(g), size=n_agents)
        converged = False
        for _ in range(max_iter):
            changed = False
            for i in range(n_agents):
                cand = np.tile(g[idx], (len(g), 1))
                cand[:, i] = g
                best = int(np.argmax(np.asarray(fn(cand))[:, i]))
                if best != idx[i]:
                    idx[i] = best
                    changed = True
            if not changed:
                converged = True
                break
        if converged:
            n_conv += 1
            points.append(g[idx].copy())
            profits.append(np.asarray(fn(g[idx][None, :]))[0])
    if not points:
        raise RuntimeError("estimate_by_sweeps: best-response iteration never converged")
    pts = np.array(points)
    prf = np.array(profits)

    if symmetric_monopoly:
        cand = np.repeat(g[:, None], n_agents, axis=1)
        joint = np.asarray(fn(cand)).sum(axis=1)
        k = int(np.argmax(joint))
        p_m, pi_m = float(g[k]), float(joint[k] / n_agents)
    else:
        if len(g) ** n_agents > max_joint_cells:
            raise ValueError("grid too large for a full joint argmax; use symmetric_monopoly")
        mesh = np.stack(np.meshgrid(*([g] * n_agents), indexing="ij"), axis=-1)
        cand = mesh.reshape(-1, n_agents)
        joint = np.asarray(fn(cand)).sum(axis=1)
        k = int(np.argmax(joint))
        p_m, pi_m = float(cand[k].mean()), float(joint[k] / n_agents)

    return Benchmarks(
        p_nash=float(pts.mean()),
        p_mono=p_m,
        pi_nash=float(prf.mean()),
        pi_mono=pi_m,
        method="sweep",
        params={
            "grid_min": float(g[0]),
            "grid_max": float(g[-1]),
            "grid_n": len(g),
            "n_agents": n_agents,
            "n_starts": n_starts,
            "n_converged": n_conv,
            "nash_points": pts.tolist(),
            "nash_spread": float(pts.max() - pts.min()),
            "seed": seed,
        },
    )


# --------------------------------------------------------------------------- empirical


def scripted_competitive(
    df_agent_period: pl.DataFrame, start: int | None = None, end: int | None = None
) -> CompetitiveBenchmark:
    """Empirical competitive benchmark from scripted-competitor runs (``agent_period``).

    Per run: mean price and mean per-agent per-period profit over ``[start, end]``; the
    benchmark averages runs with equal weight.
    """
    require(df_agent_period, ["run_id", "period", "price", "profit"], "scripted_competitive")
    d = window(df_agent_period, start, end)
    per_run = d.group_by("run_id").agg(
        pl.col("price").mean().alias("p"), pl.col("profit").mean().alias("pi"), pl.len()
    )
    if per_run.height == 0:
        raise ValueError("scripted_competitive: empty window")
    pis = per_run["pi"].to_numpy()
    return CompetitiveBenchmark(
        p_comp=float(per_run["p"].mean()),
        pi_comp=float(pis.mean()),
        n_runs=per_run.height,
        n_obs=int(per_run["len"].sum()),
        sd_pi_runs=float(pis.std(ddof=1)) if len(pis) > 1 else 0.0,
    )


# --------------------------------------------------------------------------- market model


@dataclass(frozen=True)
class MarketModel:
    """Closed-form description of the market the incumbents face in one window.

    Used to (i) solve the per-window benchmarks of an arm, (ii) re-score frozen quote paths
    under another arm's market (mechanical decomposition, 0 LLM calls) and (iii) compute the
    theoretical D7 of a symmetric cartel. ``params`` are the logit parameters with the
    routing knobs (``tau``, ``outside_utility``) as configured; :meth:`routed` folds them in
    exactly as ``finswarm.market.demand.routed_params`` does.

    * ``n_incumbents`` -- dealers present before the lever (the analysed slots);
    * ``n_entrants`` -- extra sellers added by the lever (``add_scripted_agent``, L6); they
      take routed flow like any dealer and count in the per-window symmetric benchmark;
    * ``fringe_share`` -- share of the buyer flow diverted to the public limit-order fringe
      (L2). Until the engine's fringe lands the documented approximation is a pure flow
      diversion: every dealer's expected quantity is scaled by ``1 - fringe_share``, so the
      benchmark prices are unchanged and the benchmark profits scale by the same factor;
    * ``tick`` -- price grid; frozen quotes are rounded to it when re-scored (L3b).
    """

    params: LogitParams = field(default_factory=LogitParams)
    n_incumbents: int = 2
    n_entrants: int = 0
    fringe_share: float = 0.0
    tick: float | None = None

    def __post_init__(self) -> None:
        if self.n_incumbents < 1 or self.n_entrants < 0:
            raise ValueError("MarketModel: need n_incumbents >= 1 and n_entrants >= 0")
        if not 0.0 <= self.fringe_share < 1.0:
            raise ValueError("MarketModel: fringe_share must lie in [0, 1)")

    @property
    def n_sellers(self) -> int:
        return self.n_incumbents + self.n_entrants

    def routed(self) -> LogitParams:
        """Calvano parameters with ``tau`` and ``outside_utility`` folded into ``mu``/``a0``."""
        p = self.params
        return replace(p, mu=p.tau_eff / p.alpha, a0=p.u0 / p.alpha, tau=None, outside_utility=0.0)

    def benchmarks(self) -> Benchmarks:
        """Symmetric per-firm benchmarks of the window market (all sellers, fringe scaled)."""
        r = self.routed()
        b = logit_closed_form(r.a, r.a0, r.mu, r.c, r.alpha, self.n_sellers, r.beta)
        k = 1.0 - self.fringe_share
        return replace(
            b,
            pi_nash=b.pi_nash * k,
            pi_mono=b.pi_mono * k,
            method=f"{b.method}+market_model",
            params={**b.params, "market": self.to_dict()},
        )

    def best_response(self, solver: str = "auto") -> BestResponseFn:
        """Static best response of one seller to its rivals' prices (fringe-invariant).

        ``solver="auto"`` solves the unique rival rows with the local vectorised bisection
        and cross-checks three of them (first, middle, last) against the sim-core solver
        ``finswarm.market.demand.best_response_price`` (single source of truth); on any
        disagreement beyond ``rtol = 1e-7`` every unique row is solved by sim-core (a row
        sim-core cannot solve, e.g. a division by zero at very small ``tau`` in older
        versions, falls back to the bisection). ``"local"`` forces the vectorised bisection;
        ``"sim_core"`` solves every unique row with sim-core.
        """
        r = self.routed()
        local = logit_best_response(r.a, r.a0, r.mu, r.c, r.alpha)
        md = _sim_core_demand() if solver != "local" else None
        if md is None or not hasattr(md, "best_response_price"):
            return local
        prm = md.DemandParams(a=r.a, a0=r.a0, mu=r.mu, beta=r.beta, alpha=r.alpha, cost=r.c)

        def ref(row: np.ndarray) -> float:
            try:
                return float(md.best_response_price(row, prm))
            except (ZeroDivisionError, ValueError, OverflowError, FloatingPointError):
                return float(local(row[None, :])[0])

        def br(rivals: np.ndarray) -> np.ndarray:
            rows = np.atleast_2d(np.asarray(rivals, dtype=float))
            if rows.shape[0] == 0:
                return np.empty(0)
            uniq, inv = np.unique(rows, axis=0, return_inverse=True)
            inv = np.asarray(inv).reshape(-1)
            if solver == "sim_core":
                return np.array([ref(u) for u in uniq])[inv]
            fast = np.asarray(local(uniq), dtype=float).reshape(-1)
            probe = sorted({0, len(uniq) // 2, len(uniq) - 1})
            agree = bool(np.isfinite(fast).all()) and all(
                math.isclose(ref(uniq[i]), float(fast[i]), rel_tol=1e-7, abs_tol=1e-9)
                for i in probe
            )
            if not agree:
                fast = np.array([ref(u) for u in uniq])
            return fast[inv]

        return br

    def round_prices(self, prices: Any) -> np.ndarray:
        p = np.asarray(prices, dtype=float)
        if self.tick is None or self.tick <= 0:
            return p
        return np.round(p / self.tick) * self.tick

    def quantities(self, prices: Any, entrant_prices: Any = None) -> np.ndarray:
        """Expected quantities of the incumbents, ``(T, n_incumbents)``.

        ``prices`` are the incumbents' quotes ``(T, n_incumbents)`` (NaN = not quoting);
        ``entrant_prices`` the entrants' quotes ``(T,)``, ``(T, n_entrants)`` or a scalar
        (required when ``n_entrants > 0``).
        """
        return self.all_quantities(prices, entrant_prices)[0]

    def all_quantities(self, prices: Any, entrant_prices: Any = None) -> tuple[Any, Any]:
        """``(incumbent quantities (T, n_inc), entrant quantities (T, n_ent))``."""
        p = self.round_prices(np.atleast_2d(np.asarray(prices, dtype=float)))
        t = p.shape[0]
        if self.n_entrants:
            if entrant_prices is None:
                raise ValueError("MarketModel: entrant_prices required when n_entrants > 0")
            e = np.asarray(entrant_prices, dtype=float)
            e = np.broadcast_to(e.reshape(t, -1) if e.ndim else e, (t, self.n_entrants))
            allp = np.concatenate([p, self.round_prices(e)], axis=1)
        else:
            allp = p
        r = self.routed()
        x = np.where(np.isfinite(allp), allp, np.inf)
        s = logit_shares(x, r.a, r.a0, r.mu, r.alpha)
        q = r.beta * (1.0 - self.fringe_share) * s
        return q[:, : p.shape[1]], q[:, p.shape[1] :]

    def profits(self, prices: Any, entrant_prices: Any = None) -> np.ndarray:
        """Expected per-period profits ``(p_i - c alpha) q_i`` of the incumbents."""
        p = self.round_prices(np.atleast_2d(np.asarray(prices, dtype=float)))
        q = self.quantities(p, entrant_prices)
        return np.where(np.isfinite(p), (p - self.params.marginal_cost) * q, np.nan)

    def to_dict(self) -> dict[str, Any]:
        return {
            "params": asdict(self.params),
            "n_incumbents": self.n_incumbents,
            "n_entrants": self.n_entrants,
            "fringe_share": self.fringe_share,
            "tick": self.tick,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MarketModel:
        return cls(
            params=LogitParams(**d["params"]),
            n_incumbents=int(d["n_incumbents"]),
            n_entrants=int(d.get("n_entrants", 0)),
            fringe_share=float(d.get("fringe_share", 0.0)),
            tick=None if d.get("tick") is None else float(d["tick"]),
        )

    @classmethod
    def from_config(
        cls, market: Any, n_incumbents: int | None = None, n_entrants: int = 0
    ) -> MarketModel:
        """From a ``config.schema.MarketConfig`` (or its JSON dict)."""
        if isinstance(market, dict):
            from finswarm.config.schema import MarketConfig

            market = MarketConfig.model_validate(market)
        n = int(n_incumbents if n_incumbents is not None else market.n_dealers)
        return cls(
            params=LogitParams.coerce(market),
            n_incumbents=n,
            n_entrants=int(n_entrants),
            fringe_share=float(market.fringe.share),
            tick=float(market.price_grid.tick),
        )


def market_model(obj: Any, n: int | None = None) -> MarketModel:
    """Coerce a ``MarketModel``, a ``MarketConfig``/``RunConfig``/dict or ``LogitParams``
    (``n`` = number of incumbents when it cannot be read from the object)."""
    if isinstance(obj, MarketModel):
        return obj if n is None else replace(obj, n_incumbents=int(n))
    if hasattr(obj, "market") and hasattr(obj, "llm"):  # config.schema.RunConfig
        obj = obj.market
    if isinstance(obj, dict) and "n_incumbents" in obj:
        mm = MarketModel.from_dict(obj)
        return mm if n is None else replace(mm, n_incumbents=int(n))
    if isinstance(obj, LogitParams):
        return MarketModel(params=obj, n_incumbents=int(n if n is not None else 2))
    if isinstance(obj, dict) or hasattr(obj, "routing"):
        return MarketModel.from_config(obj, n)
    raise TypeError(f"cannot build a MarketModel from {type(obj).__name__}")


def d7_cartel(config: Any, n: int | None = None) -> float:
    """Theoretical price-form D7 of a symmetric cartel pricing at ``p_M``.

    ``D7_cartel = (p_M - BR(p_M, ..., p_M)) / (p_M - p_N)`` on the configured (routed)
    demand with all sellers of the window market (default: the configuration's dealers).
    Duopoly baseline (a = 2, a0 = 0, mu = 0.25, c = 1): 0.582 (BR to 1.925 is 1.662);
    four dealers at tau = 0.25: 0.621.
    """
    m = market_model(config, n)
    b = m.benchmarks()
    br = float(m.best_response()(np.full((1, m.n_sellers - 1), b.p_mono))[0])
    return (b.p_mono - br) / (b.p_mono - b.p_nash)
