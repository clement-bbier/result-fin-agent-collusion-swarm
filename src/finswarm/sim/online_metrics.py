"""Cheap per-period metrics for event conditions and the campaign dashboard.

``delta`` is the Calvano / Fish profit index ``(pi_bar - pi_N) / (pi_M - pi_N)`` computed on the
mean per-dealer profit of the tagged agents in one period. It needs benchmarks: an injected
``Benchmarks`` wins, then ``finswarm.market.demand`` (when importable), then the closed-form
logit solver below (exact for ``demand.kind == closed_form_logit``).

Any base metric ``m`` also exists as a trailing moving average ``m_ma<k>`` (e.g. ``delta_ma20``).
"""

from __future__ import annotations

import math
import re
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from finswarm.config.schema import MarketConfig


@dataclass(frozen=True)
class Benchmarks:
    """Per-firm symmetric one-shot Nash and joint-monopoly prices and per-period profits."""

    p_nash: float
    pi_nash: float
    p_mono: float
    pi_mono: float

    def delta(self, profit: float) -> float:
        span = self.pi_mono - self.pi_nash
        if not math.isfinite(profit) or span == 0:
            return float("nan")
        return (profit - self.pi_nash) / span

    def delta_p(self, price: float) -> float:
        span = self.p_mono - self.p_nash
        if not math.isfinite(price) or span == 0:
            return float("nan")
        return (price - self.p_nash) / span

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


# --------------------------------------------------------------------------------------------
# Closed-form logit (Calvano et al. 2020; Fish et al. 2024 alpha scaling)
# --------------------------------------------------------------------------------------------


def logit_shares(
    prices: list[float], *, alpha: float, a: float, a0: float, tau: float
) -> list[float]:
    """Choice probabilities with utilities ``(alpha*a - p_i)/tau`` and outside ``alpha*a0/tau``."""
    utils = [(alpha * a - p) / tau for p in prices]
    u0 = alpha * a0 / tau
    m = max([u0, *utils])
    ex = [math.exp(u - m) for u in utils]
    denom = sum(ex) + math.exp(u0 - m)
    return [e / denom for e in ex]


def _sym_share(p: float, n: int, alpha: float, a: float, a0: float, tau: float) -> float:
    return logit_shares([p] * n, alpha=alpha, a=a, a0=a0, tau=tau)[0]


def closed_form_benchmarks(
    market: MarketConfig, n_firms: int | None = None, *, tol: float = 1e-12
) -> Benchmarks:
    """Symmetric Nash (FOC bisection) and joint monopoly (golden section) for the logit market."""
    n = int(n_firms or market.n_dealers)
    alpha, c = market.alpha, market.cost * market.alpha
    d = market.demand
    tau = market.routing.tau if market.routing.tau is not None else alpha * d.mu
    beta = d.beta

    def foc(p: float) -> float:  # p - c - tau / (1 - s(p)) = 0 at the symmetric Nash price
        s = _sym_share(p, n, alpha, d.a, d.a0, tau)
        return p - c - tau / (1.0 - s)

    lo, hi = c, c + 1.0
    while foc(hi) < 0:
        hi = c + 2.0 * (hi - c)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if foc(mid) < 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    p_n = 0.5 * (lo + hi)
    pi_n = (p_n - c) * beta * _sym_share(p_n, n, alpha, d.a, d.a0, tau)

    def joint(p: float) -> float:
        return (p - c) * beta * _sym_share(p, n, alpha, d.a, d.a0, tau)

    g = (math.sqrt(5.0) - 1.0) / 2.0
    lo, hi = c, c + 20.0 * tau + 10.0 * alpha
    x1, x2 = hi - g * (hi - lo), lo + g * (hi - lo)
    f1, f2 = joint(x1), joint(x2)
    for _ in range(300):
        if f1 < f2:
            lo, x1, f1 = x1, x2, f2
            x2 = lo + g * (hi - lo)
            f2 = joint(x2)
        else:
            hi, x2, f2 = x2, x1, f1
            x1 = hi - g * (hi - lo)
            f1 = joint(x1)
        if hi - lo < tol:
            break
    p_m = 0.5 * (lo + hi)
    return Benchmarks(p_nash=p_n, pi_nash=pi_n, p_mono=p_m, pi_mono=joint(p_m))


#: pinned benchmark files (``scripts/analyze.py --make-benchmarks``)
PINNED_BENCHMARKS_DIR = Path(__file__).resolve().parents[3] / "configs" / "analysis" / "benchmarks"


class UnpinnedBenchmarksWarning(UserWarning):
    """A fringe market has no pinned sweep benchmarks: the online Delta ignores the fringe."""


def pinned_fringe_benchmarks(
    market: MarketConfig, n_firms: int, pin_dir: str | Path | None = None
) -> Benchmarks | None:
    """Replicated-sweep benchmarks pinned for this fringe market, else ``None``.

    Scans ``pin_dir/*.json`` for a non-provisional ``sweep:*`` entry whose market model
    (routed logit parameters, number of sellers, fringe share, tick) equals the live one.
    """
    import json

    from finswarm.metrics.benchmarks import MarketModel

    d = Path(pin_dir) if pin_dir is not None else PINNED_BENCHMARKS_DIR
    if not d.is_dir():
        return None
    want = MarketModel.from_config(market, int(n_firms), 0)
    for f in sorted(d.glob("*.json")):
        try:
            doc = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for entry in (doc.get("markets") or {}).values():
            if not str(entry.get("method", "")).startswith("sweep") or entry.get("provisional"):
                continue
            got = MarketModel.from_dict(entry["model"])
            if (
                got.n_sellers == want.n_sellers
                and got.routed() == want.routed()
                and math.isclose(got.fringe_share, want.fringe_share)
                and got.tick == want.tick
            ):
                b = entry["benchmarks"]
                return Benchmarks(
                    p_nash=float(b["p_nash"]),
                    pi_nash=float(b["pi_nash"]),
                    p_mono=float(b["p_mono"]),
                    pi_mono=float(b["pi_mono"]),
                )
    return None


def resolve_benchmarks(
    market: MarketConfig,
    override: Benchmarks | None = None,
    n_firms: int | None = None,
    *,
    pin_dir: str | Path | None = None,
    strict: bool = False,
) -> Benchmarks | None:
    """Injected benchmarks, else ``finswarm.market.demand.benchmarks`` on the routed demand
    (``routing.tau`` / ``outside_utility`` folded in) for ``n_firms`` selling agents (default
    ``market.n_dealers``).

    With a public limit-order fringe (``fringe.share > 0``) the closed form ignores the
    diverted flow: the pinned replicated-sweep benchmarks are returned when they exist
    (:func:`pinned_fringe_benchmarks`); otherwise ``strict`` raises ``LookupError`` and the
    default emits an :class:`UnpinnedBenchmarksWarning` and keeps the closed form.
    """
    if override is not None:
        return override
    from finswarm.market import demand as _demand

    n = int(n_firms or market.n_dealers)
    fringe = getattr(market, "fringe", None)
    if float(getattr(fringe, "share", 0.0) or 0.0) > 0:
        pinned = pinned_fringe_benchmarks(market, n, pin_dir)
        if pinned is not None:
            return pinned
        msg = (
            f"fringe share {fringe.share:g}: no pinned sweep benchmarks for this market "  # type: ignore[union-attr]
            "(run scripts/analyze.py --make-benchmarks); online Delta uses the closed form"
        )
        if strict:
            raise LookupError(msg)
        warnings.warn(msg, UnpinnedBenchmarksWarning, stacklevel=2)
    b = _demand.benchmarks(_demand.routed_params(market), n)
    return Benchmarks(p_nash=b.p_nash, pi_nash=b.pi_nash, p_mono=b.p_mono, pi_mono=b.pi_mono)


# --------------------------------------------------------------------------------------------
# Online series
# --------------------------------------------------------------------------------------------

BASE_METRICS: tuple[str, ...] = (
    "delta",
    "delta_p",
    "price_mean",
    "profit_mean",
    "volume",
    "n_trades",
    "parse_fail_count",
    "llm_profit_total",
    "mid_mean",
)

_MA = re.compile(r"^(?P<base>[a-z_]+?)_ma(?P<k>\d+)$")


class OnlineMetrics:
    """Append-only per-period series with trailing moving averages (NaN-aware)."""

    def __init__(self) -> None:
        self.periods: list[int] = []
        self.series: dict[str, list[float]] = {k: [] for k in BASE_METRICS}

    def __len__(self) -> int:
        return len(self.periods)

    def update(self, period: int, values: dict[str, float]) -> None:
        self.periods.append(int(period))
        for k in BASE_METRICS:
            self.series[k].append(float(values.get(k, float("nan"))))

    def known(self, name: str) -> bool:
        if name in self.series or name == "period":
            return True
        m = _MA.match(name)
        return bool(m and m.group("base") in self.series)

    def value(self, name: str, offset: int = 0) -> float:
        """Metric ``name`` at ``offset`` periods before the latest one (NaN if unavailable)."""
        n = len(self.periods)
        idx = n - 1 - offset
        if idx < 0:
            return float("nan")
        if name == "period":
            return float(self.periods[idx])
        if name in self.series:
            return self.series[name][idx]
        m = _MA.match(name)
        if not m or m.group("base") not in self.series:
            raise KeyError(f"unknown online metric {name!r}")
        k = int(m.group("k"))
        if idx + 1 < k:
            return float("nan")
        window = [
            v for v in self.series[m.group("base")][idx + 1 - k : idx + 1] if math.isfinite(v)
        ]
        if len(window) < k:
            return float("nan")
        return sum(window) / k

    def latest(
        self, names: tuple[str, ...] = ("delta", "delta_ma20", "price_mean")
    ) -> dict[str, float]:
        return {k: self.value(k) for k in names}

    def state(self) -> dict[str, Any]:
        return {
            "periods": list(self.periods),
            "series": {k: list(v) for k, v in self.series.items()},
        }

    def restore(self, st: dict[str, Any]) -> None:
        self.periods = list(st["periods"])
        self.series = {k: list(st["series"].get(k, [])) for k in BASE_METRICS}
