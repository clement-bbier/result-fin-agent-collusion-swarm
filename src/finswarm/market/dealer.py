"""Mechanism (A): dealers post asks; background buyers are routed by a logit rule.

Two modes (from ``cfg.swarm.mode``):

``closed_form``
    No arrivals. On the last tick of each period (``ticks_per_period`` ticks) every quoting
    dealer sells ``q_i`` from Calvano's logit demand at its current ask. With
    ``ticks_per_period = 1`` this reproduces Fish et al. / Calvano et al. exactly.
    Buyers are one aggregate background account (``buyer_id = -1``). The logit temperature
    and outside option follow the routing knobs exactly as in agents mode (``tau`` in price
    units replaces ``alpha*mu``; ``outside_utility`` shifts ``alpha*a0``); with the defaults
    (``tau = None``, ``outside_utility = 0``) the demand is Calvano's, bit for bit.
``agents``
    Each tick, ``M_t ~ Poisson(sum_j lam_j active_j)`` swarm agents arrive (identity by Vose
    alias sampling). A buyer with reservation value ``v_j`` goes to dealer ``i`` with
    probability ∝ ``exp((v_j - ask_i)/tau)`` against the outside option ``exp(u0/tau)`` and
    buys one lot. With ``tau = alpha*mu``, ``u0 = alpha*a0`` and no value noise the expected
    shares are exactly Calvano's.
``none``
    Quotes only; no background flow.

Conventions: prices are integer ticks inside the market; dealer ``slot`` has agent id
``agent_id_offset + slot`` where the offset is the number of swarm agents present (0 in
closed-form mode). Dealers sell short: each lot moves ``ask`` cash from buyer to dealer and
one unit of inventory from dealer to buyer (cash and inventory are conserved). Profits
subtract the marginal cost ``c*alpha`` per unit (``PeriodSummary.profit``).

Quotes are ``OrderType.QUOTE`` orders with ``side=SELL`` (the ask). ``CANCEL`` orders
withdraw the sender's quote. Orders take effect at ``apply_tick`` (speed bump); orders due at
the same tick apply in submission order. Bids (``market.two_sided``) are not implemented.

Live changes (``set_param``, at period boundaries; see ``settable_params``):

* ``market.price_grid.tick`` (lever L3, every mode): re-grid. ``p_min`` and ``p_max`` stay;
  every standing ask (dealers, pending quotes, fringe depth) is re-quantised UP to the next
  point of the new grid, so a re-grid never makes a quote more aggressive (a price with no
  image on the new grid is withdrawn). Trades not yet drained keep the grid they traded on.
  The details (old/new tick, per-dealer prices) are in ``last_param_report``.
* ``market.fringe.share`` (lever L2, ``agents`` mode only): the public limit-order fringe of
  mechanism B (``finswarm.market.fringe``). ``share = 0`` with no resting fringe depth runs
  exactly the mechanism-A kernel; other modes raise ``UnsupportedIntervention`` for
  ``share > 0``.
"""

from __future__ import annotations

import dataclasses
import heapq
import math
import os
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from finswarm.config.schema import RunConfig
from finswarm.core.rng import RNGRegistry
from finswarm.core.types import MechanismKind, Order, OrderAck, OrderType, Quote, Side, Trade
from finswarm.market.demand import DemandParams, quantities
from finswarm.market.fringe import FringeBook, check_wheel
from finswarm.market.grid import n_levels_for, regrid_index, regrid_indices
from finswarm.sim.integration import UnsupportedIntervention
from finswarm.swarm import kernels
from finswarm.swarm.population import build_population_for_run
from finswarm.swarm.shocks import apply_info_shock
from finswarm.swarm.state import SwarmState

TAPE_BLOCK = 1 << 17
STATE_VERSION = 2
DEBUG = os.environ.get("FINSWARM_DEBUG", "").strip().lower() not in ("", "0", "false")

TICK_PATH = "market.price_grid.tick"
FRINGE_PATH = "market.fringe.share"
FRINGE_ID = kernels.FRINGE_ID

_SETTABLE = (
    "market.routing.tau",
    "market.routing.outside_utility",
    "market.cost",
    "market.demand.beta",
    "market.demand.mu",
    "market.demand.a0",
    TICK_PATH,
)


@dataclass(slots=True)
class PeriodSummary:
    """Per-dealer outcome of one closed period (arrays indexed by dealer slot)."""

    period: int
    tick_start: int
    tick_end: int  # inclusive
    ask: np.ndarray  # last ask price (NaN when not quoting)
    vwap: np.ndarray  # revenue / qty (NaN without sales)
    qty: np.ndarray
    revenue: np.ndarray
    profit: np.ndarray  # revenue - c*alpha*qty
    n_arrivals: int
    n_outside: int  # arrivals that neither bought nor posted a fringe ask
    fundamental: float
    fringe_qty: float = 0.0  # lots bought from the public fringe
    fringe_revenue: float = 0.0
    fringe_posted: float = 0.0  # lots posted to the fringe
    fringe_resting: float = 0.0  # fringe lots resting at the end of the period

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, np.ndarray):
                d[k] = [None if math.isnan(x) else float(x) for x in v.tolist()]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PeriodSummary:
        arr = {
            k: np.array([np.nan if x is None else x for x in d[k]], dtype=np.float64)
            for k in ("ask", "vwap", "qty", "revenue", "profit")
        }
        return cls(
            period=int(d["period"]),
            tick_start=int(d["tick_start"]),
            tick_end=int(d["tick_end"]),
            n_arrivals=int(d["n_arrivals"]),
            n_outside=int(d["n_outside"]),
            fundamental=float(d["fundamental"]),
            fringe_qty=float(d.get("fringe_qty", 0.0)),
            fringe_revenue=float(d.get("fringe_revenue", 0.0)),
            fringe_posted=float(d.get("fringe_posted", 0.0)),
            fringe_resting=float(d.get("fringe_resting", 0.0)),
            **arr,
        )


def market_mode(cfg: RunConfig) -> str:
    return {"closed_form": "closed_form", "agents": "agents", "none": "none"}[cfg.swarm.mode]


def _order_to_dict(o: Order) -> dict[str, Any]:
    return {
        "order_id": o.order_id,
        "agent_id": o.agent_id,
        "side": int(o.side),
        "otype": int(o.otype),
        "price_ticks": o.price_ticks,
        "size": o.size,
        "tif_ticks": o.tif_ticks,
        "submit_tick": o.submit_tick,
        "apply_tick": o.apply_tick,
    }


def _order_from_dict(d: dict[str, Any]) -> Order:
    return Order(
        order_id=int(d["order_id"]),
        agent_id=int(d["agent_id"]),
        side=Side(int(d["side"])),
        otype=OrderType(int(d["otype"])),
        price_ticks=int(d["price_ticks"]),
        size=float(d["size"]),
        tif_ticks=int(d["tif_ticks"]),
        submit_tick=int(d["submit_tick"]),
        apply_tick=int(d["apply_tick"]),
    )


class DealerMarket:
    kind: MechanismKind = MechanismKind.DEALER

    def __init__(
        self,
        cfg: RunConfig,
        rng: RNGRegistry,
        swarm: SwarmState | None = None,
        *,
        mode: str | None = None,
    ) -> None:
        m = cfg.market
        if m.two_sided:
            raise NotImplementedError(
                "dealer market: two-sided quotes (market.two_sided=true) are not implemented; "
                "use one-sided asks"
            )
        if m.n_dealers < 1:
            raise ValueError("market.n_dealers must be >= 1")
        self.cfg = cfg
        self.rng = rng
        self.mode = mode or market_mode(cfg)
        if self.mode not in ("closed_form", "agents", "none"):
            raise ValueError(f"unknown dealer market mode {self.mode!r}")
        if self.mode == "agents" and swarm is None:
            swarm = build_population_for_run(cfg, rng)
        self.swarm: SwarmState | None = swarm
        self.agent_id_offset = swarm.n if (swarm is not None and self.mode == "agents") else 0
        self.n_dealers = int(m.n_dealers)
        self.tick_size = float(m.price_grid.tick)
        self.p_min = float(m.price_grid.min)
        self.p_max = float(m.price_grid.max)
        self.n_levels = n_levels_for(self.p_min, self.p_max, self.tick_size)
        self.ticks_per_period = int(cfg.ticks_per_period)
        if self.ticks_per_period < 1:
            raise ValueError("ticks_per_period must be >= 1")
        self.alpha = float(m.alpha)
        self.cost = float(m.cost)
        self.a0 = float(m.demand.a0)
        self.mu = float(m.demand.mu)
        self.beta = float(m.demand.beta)
        self._tau_cfg: float | None = m.routing.tau
        self.outside_utility = float(m.routing.outside_utility)
        sc = cfg.swarm
        self.rho = float(sc.sentiment_rho)
        self.sigma = float(sc.sentiment_sigma)
        self.w_s = float(sc.sentiment_weight)

        nd = self.n_dealers
        self.tick = 0
        self.asks = np.full(nd, -1, dtype=np.int64)
        self.ask_size = np.zeros(nd, dtype=np.float64)
        self.ask_order = np.full(nd, -1, dtype=np.int64)
        self.ask_expire = np.full(nd, -1, dtype=np.int64)
        self.dealer_cash = np.zeros(nd, dtype=np.float64)
        self.dealer_inv = np.zeros(nd, dtype=np.float64)
        self.bg_cash = 0.0  # aggregate background buyer (closed form)
        self.bg_inv = 0.0
        self.glob = np.array([0.0, 0.0, float(m.demand.a)])  # G, H, F
        self.trend = np.array([np.nan, np.nan])  # mean ask before / after the last change
        self._pending: list[tuple[int, int, Order]] = []
        self._seq = 0
        self._tape = np.empty(0, dtype=np.float64)
        self._tpos = 0
        cap = 1024
        self._tr = {
            "tick": np.zeros(cap, dtype=np.int64),
            "buyer": np.zeros(cap, dtype=np.int64),
            "dealer": np.zeros(cap, dtype=np.int64),
            "price": np.zeros(cap, dtype=np.int64),
            "size": np.zeros(cap, dtype=np.float64),
            "value": np.zeros(cap, dtype=np.float64),
        }
        self._tr_pos = 0
        self._trade_id_base = 0
        self._tr_stash: dict[str, np.ndarray] | None = None  # drained early by a re-grid
        self._flow: list[tuple[np.ndarray, ...]] = []  # (tick, buy, n_arr, n_out, fbuy, fpost)
        self._p_qty = np.zeros(nd)
        self._p_rev = np.zeros(nd)
        self._p_arr = 0
        self._p_out = 0
        self._p_fringe = np.zeros(3)  # fringe qty, revenue, posted in the open period
        self._periods: list[PeriodSummary] = []
        #: details of the last ``set_param`` that needs them (re-grid, fringe switch)
        self.last_param_report: dict[str, Any] | None = None
        # public limit-order fringe (mechanism B): allocated when first switched on
        self.fringe_share = 0.0
        self.fringe_horizon = int(m.fringe.horizon_ticks)
        self._fr: FringeBook | None = None
        if float(m.fringe.share) > 0.0:
            self._set_fringe_share(float(m.fringe.share))

    # ------------------------------------------------------------------ parameters
    @property
    def settable_params(self) -> tuple[str, ...]:
        """Config paths ``set_param`` can change live (anything else must fail loudly)."""
        return (*_SETTABLE, FRINGE_PATH) if self.mode == "agents" else _SETTABLE

    @property
    def tau(self) -> float:
        return self.alpha * self.mu if self._tau_cfg is None else float(self._tau_cfg)

    @property
    def u0(self) -> float:
        """Outside-option utility in price units: Calvano's ``alpha*a0`` plus a shift."""
        return self.alpha * self.a0 + self.outside_utility

    @property
    def fundamental(self) -> float:
        return float(self.glob[2])

    @property
    def marginal_cost(self) -> float:
        return self.cost * self.alpha

    def demand_params(self) -> DemandParams:
        return DemandParams(
            a=self.fundamental,
            a0=self.a0,
            mu=self.mu,
            beta=self.beta,
            alpha=self.alpha,
            cost=self.cost,
        )

    def routed_demand_params(self) -> DemandParams:
        """Closed-form demand with the live routing knobs (Calvano's when they are defaults).

        Logit utilities ``(alpha*a - p)/tau`` against ``(alpha*a0 + outside_utility)/tau``,
        i.e. Calvano's formula with ``mu = tau/alpha`` and ``a0 + outside_utility/alpha``.
        """
        prm = self.demand_params()
        mu = prm.mu if self._tau_cfg is None else float(self._tau_cfg) / self.alpha
        a0 = prm.a0 if self.outside_utility == 0.0 else prm.a0 + self.outside_utility / self.alpha
        if mu == prm.mu and a0 == prm.a0:
            return prm
        return DemandParams(a=prm.a, a0=a0, mu=mu, beta=prm.beta, alpha=prm.alpha, cost=prm.cost)

    def set_param(self, path: str, value: Any) -> Any:
        """Change a live parameter (``SetParam``); returns the previous value.

        A re-grid (``market.price_grid.tick``) and a fringe switch (``market.fringe.share``)
        leave their details in ``last_param_report``; other paths clear it.
        """
        self.last_param_report = None
        if path == TICK_PATH:
            old_tick = self.tick_size
            self.last_param_report = self._regrid(float(value))
            return old_tick
        if path == FRINGE_PATH:
            old_share = self.fringe_share
            self._set_fringe_share(float(value))
            fb = self._fr
            self.last_param_report = {
                "kind": "fringe_share",
                "path": FRINGE_PATH,
                "old_share": old_share,
                "new_share": self.fringe_share,
                "horizon_ticks": self.fringe_horizon,
                "resting_lots": fb.resting if fb is not None else 0,
                "best_ask": (
                    self.p_min + fb.best * self.tick_size
                    if fb is not None and fb.best >= 0
                    else None
                ),
            }
            return old_share
        if path == "market.routing.tau":
            old, self._tau_cfg = self._tau_cfg, None if value is None else float(value)
        elif path == "market.routing.outside_utility":
            old, self.outside_utility = self.outside_utility, float(value)
        elif path == "market.cost":
            old, self.cost = self.cost, float(value)
        elif path == "market.demand.beta":
            old, self.beta = self.beta, float(value)
        elif path == "market.demand.mu":
            old, self.mu = self.mu, float(value)
        elif path == "market.demand.a0":
            old, self.a0 = self.a0, float(value)
        else:
            raise KeyError(
                f"dealer market cannot set {path!r} live; settable: {self.settable_params}"
            )
        return old

    # ------------------------------------------------------------------ fringe (B)
    def _set_fringe_share(self, share: float) -> None:
        if not (math.isfinite(share) and 0.0 <= share <= 1.0):
            raise ValueError(f"market.fringe.share must lie in [0, 1], got {share!r}")
        if share > 0.0 and self.mode != "agents":
            raise UnsupportedIntervention(
                f"market.fringe.share={share:g}: the public limit-order fringe (mechanism B) "
                f"needs swarm.mode='agents' (this dealer market runs in {self.mode!r} mode)"
            )
        if share > 0.0 and self._fr is None:
            self._fr = FringeBook.empty(self.fringe_horizon, self.n_levels)
        self.fringe_share = share

    @property
    def fringe(self) -> FringeBook | None:
        """The fringe book (``None`` until the fringe is first switched on)."""
        return self._fr

    def fringe_levels(self) -> list[tuple[float, int]]:
        """Fringe-only depth as ``(price, lots)`` by price (empty without a fringe)."""
        fb = self._fr
        if fb is None:
            return []
        return [(self.p_min + k * self.tick_size, n) for k, n in fb.levels()]

    @property
    def fringe_cash(self) -> float:
        return float(self._fr.acct[0]) if self._fr is not None else 0.0

    @property
    def fringe_inv(self) -> float:
        return float(self._fr.acct[1]) if self._fr is not None else 0.0

    # ------------------------------------------------------------------ re-grid (L3)
    def _regrid(self, new_tick: float) -> dict[str, Any]:
        """Move every standing price onto the grid ``p_min + k * new_tick`` (asks round UP).

        Dealer asks and pending quotes with no image on the new grid are withdrawn. Trades not
        yet drained are converted with the old grid first (they keep their traded price).
        """
        old_tick, old_n = self.tick_size, self.n_levels
        new_n = n_levels_for(self.p_min, self.p_max, new_tick)
        if self._fr is not None:
            check_wheel(self._fr.horizon, new_n)
        self._stash_trades()
        new_asks = regrid_indices(self.asks, old_tick, new_tick, new_n, up=True)
        dealers, withdrawn = [], []
        for d in np.flatnonzero(self.asks >= 0).tolist():
            old_k, new_k = int(self.asks[d]), int(new_asks[d])
            dealers.append(
                {
                    "slot": d,
                    "agent_id": self.dealer_agent_id(d),
                    "old_ticks": old_k,
                    "new_ticks": new_k,
                    "old_price": self.p_min + old_k * old_tick,
                    "new_price": None if new_k < 0 else self.p_min + new_k * new_tick,
                }
            )
            if new_k < 0:
                self._withdraw(d)
                withdrawn.append(d)
            else:
                self.asks[d] = new_k
        pending, kept = [], []
        for a, s, o in self._pending:
            if o.otype == OrderType.QUOTE:
                new_k = regrid_index(
                    o.price_ticks, old_tick, new_tick, new_n, up=o.side == Side.SELL
                )
                pending.append(
                    {"order_id": o.order_id, "old_ticks": o.price_ticks, "new_ticks": new_k}
                )
                if new_k < 0:
                    continue
                o = dataclasses.replace(o, price_ticks=new_k)
            kept.append((a, s, o))
        heapq.heapify(kept)
        self._pending = kept
        fringe = self._fr.regrid(old_tick, new_tick, new_n) if self._fr is not None else None
        self.tick_size, self.n_levels = float(new_tick), new_n
        return {
            "kind": "regrid",
            "path": TICK_PATH,
            "old_tick": old_tick,
            "new_tick": float(new_tick),
            "p_min": self.p_min,
            "n_levels": [old_n, new_n],
            "dealers": dealers,
            "withdrawn": withdrawn,
            "pending": pending,
            "fringe": fringe,
        }

    def apply_info_shock(self, d_fundamental: float, diffusion: str = "attention") -> float:
        """Move the fundamental (Calvano's ``a``) by ``d_fundamental``; returns the new value.

        Closed form: demand moves at once. Agents: awareness diffuses with per-agent
        geometric delays drawn from the ``interventions`` stream.
        """
        old = self.fundamental
        if self.mode == "agents" and self.swarm is not None:
            new = apply_info_shock(
                self.swarm,
                tick=self.tick,
                fundamental=old,
                d_fundamental=float(d_fundamental),
                diffusion=diffusion,  # type: ignore[arg-type]
                rng=self.rng["interventions"],
            )
        else:
            new = old + float(d_fundamental)
        self.glob[2] = new
        return new

    # ---------------------------------------------------------------------- ids
    def dealer_agent_id(self, slot: int) -> int:
        return self.agent_id_offset + slot

    def dealer_slot(self, agent_id: int) -> int:
        return agent_id - self.agent_id_offset

    # ------------------------------------------------------------------- orders
    def submit(self, order: Order) -> OrderAck:
        slot = self.dealer_slot(order.agent_id)
        if not 0 <= slot < self.n_dealers:
            return OrderAck(False, order.order_id, f"agent {order.agent_id} is not a dealer")
        if order.apply_tick < order.submit_tick:
            return OrderAck(False, order.order_id, "apply_tick < submit_tick")
        if any(o.order_id == order.order_id for _, _, o in self._pending):
            return OrderAck(False, order.order_id, "duplicate pending order_id")
        if order.otype == OrderType.CANCEL:
            self._enqueue(order)
            return OrderAck(True, order.order_id)
        if order.otype != OrderType.QUOTE:
            return OrderAck(False, order.order_id, "dealer market accepts QUOTE and CANCEL only")
        if order.side == Side.BUY:
            return OrderAck(False, order.order_id, "bid quotes require market.two_sided")
        if not 0 <= order.price_ticks < self.n_levels:
            return OrderAck(False, order.order_id, f"price_ticks {order.price_ticks} off grid")
        if not order.size > 0:
            return OrderAck(False, order.order_id, "quote size must be > 0")
        if order.tif_ticks == 0 or order.tif_ticks < -1:
            return OrderAck(False, order.order_id, "tif_ticks must be -1 (GTC) or >= 1")
        self._enqueue(order)
        return OrderAck(True, order.order_id)

    def _enqueue(self, order: Order) -> None:
        heapq.heappush(self._pending, (order.apply_tick, self._seq, order))
        self._seq += 1

    def cancel(self, order_id: int) -> bool:
        for k, (_, _, o) in enumerate(self._pending):
            if o.order_id == order_id:
                self._pending.pop(k)
                heapq.heapify(self._pending)
                return True
        hit = np.flatnonzero((self.ask_order == order_id) & (self.asks >= 0))
        if hit.size:
            self._withdraw(int(hit[0]))
            return True
        return False

    def _withdraw(self, d: int) -> None:
        self.asks[d] = -1
        self.ask_size[d] = 0.0
        self.ask_order[d] = -1
        self.ask_expire[d] = -1

    def _apply_due(self) -> None:
        changed = False
        for d in np.flatnonzero((self.ask_expire >= 0) & (self.ask_expire <= self.tick)):
            self._withdraw(int(d))
            changed = True
        while self._pending and self._pending[0][0] <= self.tick:
            _, _, o = heapq.heappop(self._pending)
            d = self.dealer_slot(o.agent_id)
            if o.otype == OrderType.CANCEL:
                self._withdraw(d)
            else:
                self.asks[d] = o.price_ticks
                self.ask_size[d] = o.size
                self.ask_order[d] = o.order_id
                self.ask_expire[d] = self.tick + o.tif_ticks if o.tif_ticks > 0 else -1
            changed = True
        if changed:
            live = self.asks >= 0
            new = float(np.mean(self._ask_prices()[live])) if live.any() else np.nan
            if not (new == self.trend[1] or (math.isnan(new) and math.isnan(self.trend[1]))):
                self.trend[0], self.trend[1] = self.trend[1], new

    def _next_event_tick(self) -> int | None:
        cands = [t for t in self.ask_expire if t > self.tick]
        if self._pending:
            cands.append(max(self._pending[0][0], self.tick + 1))
        return int(min(cands)) if cands else None

    def _ask_prices(self) -> np.ndarray:
        return np.where(self.asks >= 0, self.p_min + self.asks * self.tick_size, np.nan)

    def _trend_signal(self) -> float:
        s = float(self.trend[1] - self.trend[0])
        return 0.0 if math.isnan(s) else s

    # ------------------------------------------------------------------ advance
    def advance(self, n_ticks: int) -> None:
        if n_ticks < 0:
            raise ValueError("n_ticks must be >= 0")
        end = self.tick + int(n_ticks)
        tpp = self.ticks_per_period
        while self.tick < end:
            self._apply_due()
            stop = min(end, (self.tick // tpp + 1) * tpp)
            nxt = self._next_event_tick()
            if nxt is not None:
                stop = min(stop, nxt)
            if self.mode == "closed_form":
                self._run_closed_form(self.tick, stop - self.tick)
            elif self.mode == "agents":
                self._run_agents(self.tick, stop - self.tick)
            self.tick = stop
            if self.tick % tpp == 0:
                self._close_period(self.tick // tpp - 1)
        # quotes due at the new tick are visible to observers taken before the next advance
        self._apply_due()

    def _run_closed_form(self, t0: int, n: int) -> None:
        tpp = self.ticks_per_period
        t_last = t0 + n - 1
        if (t_last + 1) % tpp != 0:
            return  # the period's last tick is not in this chunk
        live = self.asks >= 0
        if not live.any():
            return
        prices = np.where(live, self.p_min + self.asks * self.tick_size, np.inf)
        q = np.where(live, quantities(prices, self.routed_demand_params()), 0.0)
        flow = np.zeros((1, self.n_dealers))
        for d in np.flatnonzero(q > 0):
            p = float(prices[d])
            self._push_trade(t_last, -1, int(d), int(self.asks[d]), float(q[d]), np.nan)
            self.dealer_cash[d] += p * q[d]
            self.dealer_inv[d] -= q[d]
            self.bg_cash -= p * q[d]
            self.bg_inv += q[d]
            flow[0, d] = q[d]
        z = np.zeros(1, np.int64)
        self._flow.append(
            (np.array([t_last], dtype=np.int64), flow, z, z, np.zeros(1), np.zeros(1))
        )
        self._p_qty += q
        self._p_rev += np.where(live, prices, 0.0) * q

    def _push_trade(
        self, t: int, buyer: int, dealer: int, price: int, size: float, value: float
    ) -> None:
        self._grow_trades(1)
        k = self._tr_pos
        tr = self._tr
        tr["tick"][k], tr["buyer"][k], tr["dealer"][k] = t, buyer, dealer
        tr["price"][k], tr["size"][k], tr["value"][k] = price, size, value
        self._tr_pos += 1

    def _grow_trades(self, extra: int) -> None:
        need = self._tr_pos + extra
        cap = self._tr["tick"].shape[0]
        if need <= cap:
            return
        new_cap = max(need, 2 * cap)
        for k, arr in self._tr.items():
            out = np.zeros(new_cap, dtype=arr.dtype)
            out[: self._tr_pos] = arr[: self._tr_pos]
            self._tr[k] = out

    def _refill_tape(self) -> None:
        rest = self._tape[self._tpos :]
        self._tape = np.concatenate([rest, self.rng["swarm"].random(TAPE_BLOCK)])
        self._tpos = 0

    def _run_agents(self, t0: int, n: int) -> None:
        sw = self.swarm
        assert sw is not None
        prob, alias, lam_total = sw.alias()
        nd = self.n_dealers
        ask_prices = np.where(self.asks >= 0, self.p_min + self.asks * self.tick_size, 0.0)
        params = np.array(
            [self.rho, self.sigma, self.w_s, self.alpha, self.tau, self.u0, self._trend_signal()]
        )
        flow = np.zeros((n, nd))
        n_arr = np.zeros(n, dtype=np.int64)
        n_out = np.zeros(n, dtype=np.int64)
        f_buy = np.zeros(n)
        f_post = np.zeros(n)
        fb = self._fr
        # mechanism B runs while the fringe is on or still has resting lots; otherwise the
        # mechanism-A kernel runs unchanged (share 0 reproduces mechanism A bit for bit)
        use_fringe = fb is not None and (self.fringe_share > 0.0 or fb.resting > 0)
        if use_fringe:
            assert fb is not None
            fparams = np.array([self.fringe_share, self.marginal_cost, self.p_min, self.tick_size])
            cash0 = float(fb.acct[0])
        done_total = 0
        stalls = 0
        while done_total < n:
            remaining = n - done_total
            need = kernels.uniforms_needed_upper(lam_total, min(remaining, 64))
            if self._tape.shape[0] - self._tpos < need:
                self._refill_tape()
            self._grow_trades(int(lam_total * remaining * 1.1) + 64)
            pos0 = self._tr_pos
            tr = self._tr
            if use_fringe:
                assert fb is not None
                done, tpos, tr_pos = kernels.dealer_fringe_ticks(
                    t0 + done_total,
                    remaining,
                    self._tape,
                    self._tpos,
                    prob,
                    alias,
                    lam_total,
                    sw.stype,
                    sw.value_base,
                    sw.noise_sd,
                    sw.trend_k,
                    sw.sentiment,
                    sw.belief_F,
                    sw.last_update,
                    sw.shock_delay,
                    sw.inventory,
                    sw.cash,
                    sw.max_inv,
                    self.glob,
                    params,
                    fparams,
                    self.asks,
                    ask_prices,
                    self.dealer_cash,
                    self.dealer_inv,
                    fb.depth,
                    fb.wheel,
                    fb.meta,
                    fb.st,
                    fb.acct,
                    tr["tick"],
                    tr["buyer"],
                    tr["dealer"],
                    tr["price"],
                    tr["value"],
                    self._tr_pos,
                    flow[done_total:],
                    f_buy[done_total:],
                    f_post[done_total:],
                    n_arr[done_total:],
                    n_out[done_total:],
                )
            else:
                done, tpos, tr_pos = kernels.dealer_ticks(
                    t0 + done_total,
                    remaining,
                    self._tape,
                    self._tpos,
                    prob,
                    alias,
                    lam_total,
                    sw.stype,
                    sw.value_base,
                    sw.noise_sd,
                    sw.trend_k,
                    sw.sentiment,
                    sw.belief_F,
                    sw.last_update,
                    sw.shock_delay,
                    sw.inventory,
                    sw.cash,
                    sw.max_inv,
                    self.glob,
                    params,
                    self.asks,
                    ask_prices,
                    self.dealer_cash,
                    self.dealer_inv,
                    tr["tick"],
                    tr["buyer"],
                    tr["dealer"],
                    tr["price"],
                    tr["value"],
                    self._tr_pos,
                    flow[done_total:],
                    n_arr[done_total:],
                    n_out[done_total:],
                )
            done, tpos, tr_pos = int(done), int(tpos), int(tr_pos)
            tr["size"][pos0:tr_pos] = 1.0
            self._tpos, self._tr_pos = tpos, tr_pos
            done_total += done
            if done == 0:
                stalls += 1
                if stalls > 64:
                    raise RuntimeError("dealer kernel made no progress (tape/buffer sizing bug)")
                self._refill_tape()
                self._grow_trades(int(4 * lam_total) + 1024)
            else:
                stalls = 0
        qty = flow.sum(axis=0)
        self._p_qty += qty
        self._p_rev += qty * ask_prices
        self._p_arr += int(n_arr.sum())
        self._p_out += int(n_out.sum())
        if use_fringe:
            assert fb is not None
            self._p_fringe += (f_buy.sum(), float(fb.acct[0]) - cash0, f_post.sum())
            if DEBUG:
                fb.check()
        self._flow.append(
            (np.arange(t0, t0 + n, dtype=np.int64), flow, n_arr, n_out, f_buy, f_post)
        )

    def _close_period(self, period: int) -> None:
        qty, rev = self._p_qty.copy(), self._p_rev.copy()
        with np.errstate(invalid="ignore", divide="ignore"):
            vwap = np.where(qty > 0, rev / np.where(qty > 0, qty, 1.0), np.nan)
        tpp = self.ticks_per_period
        self._periods.append(
            PeriodSummary(
                period=period,
                tick_start=period * tpp,
                tick_end=(period + 1) * tpp - 1,
                ask=self._ask_prices(),
                vwap=vwap,
                qty=qty,
                revenue=rev,
                profit=rev - self.marginal_cost * qty,
                n_arrivals=self._p_arr,
                n_outside=self._p_out,
                fundamental=self.fundamental,
                fringe_qty=float(self._p_fringe[0]),
                fringe_revenue=float(self._p_fringe[1]),
                fringe_posted=float(self._p_fringe[2]),
                fringe_resting=float(self._fr.resting) if self._fr is not None else 0.0,
            )
        )
        self._p_qty[:] = 0.0
        self._p_rev[:] = 0.0
        self._p_arr = 0
        self._p_out = 0
        self._p_fringe[:] = 0.0

    # ------------------------------------------------------------------ queries
    def quotes(self) -> list[Quote]:
        return [
            Quote(
                agent_id=self.dealer_agent_id(d),
                ask_ticks=int(self.asks[d]),
                ask_size=float(self.ask_size[d]) if self.asks[d] >= 0 else 0.0,
            )
            for d in range(self.n_dealers)
        ]

    def best_bid(self) -> float | None:
        return None

    def best_ask(self) -> float | None:
        """Lowest ask available to buyers: dealers and the public fringe."""
        live = self.asks >= 0
        best = int(self.asks[live].min()) if live.any() else -1
        fb = self._fr
        if fb is not None and fb.best >= 0 and (best < 0 or fb.best < best):
            best = fb.best
        return None if best < 0 else self.p_min + best * self.tick_size

    def mid(self) -> float | None:
        bid, ask = self.best_bid(), self.best_ask()
        return None if bid is None or ask is None else 0.5 * (bid + ask)

    def snapshot(self, depth: int) -> list[tuple[float, float, float]]:
        """Top ``depth`` ask levels ``(price, 0, size)``: dealer quotes plus fringe lots."""
        levels: dict[int, float] = {}
        for d in np.flatnonzero(self.asks >= 0):
            levels[int(self.asks[d])] = levels.get(int(self.asks[d]), 0.0) + float(self.ask_size[d])
        if self._fr is not None:
            for k, n in self._fr.levels():
                levels[k] = levels.get(k, 0.0) + float(n)
        out = [(self.p_min + k * self.tick_size, 0.0, levels[k]) for k in sorted(levels)]
        return out[: max(0, depth)]

    def accounts(self) -> dict[str, np.ndarray]:
        return {"cash": self.dealer_cash.copy(), "inventory": self.dealer_inv.copy()}

    # ------------------------------------------------------------------- drains
    def drain_trades_arrays(self) -> dict[str, np.ndarray]:
        """Columnar trades since the last drain (fast path for the Parquet writer).

        Fringe trades have ``seller_id = -2`` and ``seller_cost`` = the traded level. The
        lot's reservation price ``r`` was rounded UP to that level, so ``r`` lies in
        ``(level - tick, level]``: the fringe seller's surplus ``price - r`` in ``[0, tick)``
        is not recorded (registered convention; total-surplus bias below ``tick`` per
        fringe trade).
        """
        out = self._drain_buffer()
        stash, self._tr_stash = self._tr_stash, None
        if stash is not None:
            out = {key: np.concatenate([stash[key], out[key]]) for key in out}
        return out

    def _drain_buffer(self) -> dict[str, np.ndarray]:
        k = self._tr_pos
        tr = self._tr
        price_ticks = tr["price"][:k].copy()
        buyer = tr["buyer"][:k]
        dealer = tr["dealer"][:k]
        price = self.p_min + price_ticks * self.tick_size
        if self._fr is not None and (dealer < 0).any():
            seller = np.where(dealer >= 0, dealer + self.agent_id_offset, dealer)
            seller_cost = np.where(dealer >= 0, self.marginal_cost, price)
        else:
            seller = dealer + self.agent_id_offset
            seller_cost = np.full(k, self.marginal_cost, dtype=np.float64)
        out = {
            "trade_id": np.arange(self._trade_id_base, self._trade_id_base + k, dtype=np.int64),
            "tick": tr["tick"][:k].copy(),
            "period": (tr["tick"][:k] // self.ticks_per_period).astype(np.int32),
            "price": price,
            "price_ticks": price_ticks,
            "size": tr["size"][:k].copy(),
            "buyer_id": buyer.astype(np.int32),
            "seller_id": seller.astype(np.int32),
            "aggressor": np.full(k, int(Side.BUY), dtype=np.int8),
            "batch_id": np.full(k, -1, dtype=np.int64),
            "buyer_value": tr["value"][:k].copy(),
            "seller_cost": seller_cost,
        }
        self._trade_id_base += k
        self._tr_pos = 0
        return out

    def _stash_trades(self) -> None:
        """Convert undrained trades now (before a re-grid changes what a tick index means)."""
        if self._tr_pos == 0:
            return
        cols = self._drain_buffer()
        if self._tr_stash is not None:
            cols = {key: np.concatenate([self._tr_stash[key], cols[key]]) for key in cols}
        self._tr_stash = cols

    def drain_trades(self) -> list[Trade]:
        cols = self.drain_trades_arrays()
        mech = self.kind.value
        return [
            Trade(
                trade_id=int(cols["trade_id"][i]),
                tick=int(cols["tick"][i]),
                price_ticks=int(cols["price_ticks"][i]),
                size=float(cols["size"][i]),
                buyer_id=int(cols["buyer_id"][i]),
                seller_id=int(cols["seller_id"][i]),
                aggressor=int(cols["aggressor"][i]),
                mechanism=mech,
                buyer_value=float(cols["buyer_value"][i]),
                seller_cost=float(cols["seller_cost"][i]),
            )
            for i in range(cols["trade_id"].shape[0])
        ]

    def drain_flow(self) -> dict[str, np.ndarray]:
        """Long-format per-tick flow (``flow_per_tick`` columns without ``run_id``).

        ``buy_qty`` = lots bought from the dealer by the swarm; ``n_arrivals``/``n_outside``
        are per-tick totals repeated on every dealer row. Once the public fringe has been
        switched on, every tick also has a ``dealer_id = -2`` row: ``buy_qty`` = lots bought
        from the fringe, ``sell_qty`` = lots posted to the fringe, ``n_arrivals = n_outside =
        0`` (the counts stay on the dealer rows only, so per-tick totals are read from any
        dealer row and never inflated by the fringe row).
        """
        nd = self.n_dealers
        f = self._flow_peek()
        self._flow = []
        ticks, flow, arr, out_ = f["tick"], f["buy"], f["n_arr"], f["n_out"]
        k = ticks.shape[0]
        ids = np.arange(nd) + self.agent_id_offset
        if self._fr is None:
            sell = np.zeros(k * nd)
        else:  # the anonymous fringe as one more seller column
            ids = np.append(ids, FRINGE_ID)
            flow = np.column_stack([flow, f["fr_buy"]])
            sell_m = np.zeros((k, nd + 1))
            sell_m[:, nd] = f["fr_post"]
            sell = sell_m.reshape(-1)
        nc = ids.shape[0]
        n_arr = np.repeat(arr, nc).astype(np.int32)
        n_out = np.repeat(out_, nc).astype(np.int32)
        if self._fr is not None:  # per-tick counts on the dealer rows only, 0 on the fringe row
            n_arr.reshape(k, nc)[:, nd] = 0
            n_out.reshape(k, nc)[:, nd] = 0
        return {
            "tick": np.repeat(ticks, nc),
            "period": np.repeat(ticks // self.ticks_per_period, nc).astype(np.int32),
            "dealer_id": np.tile(ids, k).astype(np.int32),
            "buy_qty": flow.reshape(-1).copy(),
            "sell_qty": sell,
            "n_arrivals": n_arr,
            "n_outside": n_out,
        }

    def drain_periods(self) -> list[PeriodSummary]:
        out, self._periods = self._periods, []
        return out

    # ---------------------------------------------------------------- checkpoint
    def state(self) -> dict[str, Any]:
        """Market-only state (the swarm is checkpointed separately via ``SwarmState.state``)."""
        k = self._tr_pos
        flow = self._flow_peek()
        return {
            "version": STATE_VERSION,
            "kind": self.kind.value,
            "mode": self.mode,
            "n_dealers": self.n_dealers,
            "agent_id_offset": self.agent_id_offset,
            "params": {
                "tau_cfg": self._tau_cfg,
                "outside_utility": self.outside_utility,
                "cost": self.cost,
                "beta": self.beta,
                "mu": self.mu,
                "a0": self.a0,
                "fringe_share": self.fringe_share,
            },
            "grid": {"tick_size": self.tick_size, "n_levels": self.n_levels, "p_max": self.p_max},
            "fringe": self._fr.state() if self._fr is not None else None,
            "trades_stash": (
                {key: v.copy() for key, v in self._tr_stash.items()}
                if self._tr_stash is not None
                else None
            ),
            "tick": self.tick,
            "asks": self.asks.copy(),
            "ask_size": self.ask_size.copy(),
            "ask_order": self.ask_order.copy(),
            "ask_expire": self.ask_expire.copy(),
            "dealer_cash": self.dealer_cash.copy(),
            "dealer_inv": self.dealer_inv.copy(),
            "bg": np.array([self.bg_cash, self.bg_inv]),
            "glob": self.glob.copy(),
            "trend": self.trend.copy(),
            "pending": [
                {"apply_tick": a, "seq": s, "order": _order_to_dict(o)}
                for a, s, o in sorted(self._pending, key=lambda x: (x[0], x[1]))
            ],
            "seq": self._seq,
            "tape": self._tape[self._tpos :].copy(),
            "trades": {name: arr[:k].copy() for name, arr in self._tr.items()},
            "trade_id_base": self._trade_id_base,
            "flow": flow,
            "period_acc": {
                "qty": self._p_qty.copy(),
                "rev": self._p_rev.copy(),
                "counts": np.array([self._p_arr, self._p_out], dtype=np.int64),
                "fringe": self._p_fringe.copy(),
            },
            "periods": [p.to_dict() for p in self._periods],
        }

    _FLOW_KEYS = ("tick", "buy", "n_arr", "n_out", "fr_buy", "fr_post")

    def _flow_peek(self) -> dict[str, np.ndarray]:
        nd = self.n_dealers
        if not self._flow:
            return {
                "tick": np.zeros(0, np.int64),
                "buy": np.zeros((0, nd)),
                "n_arr": np.zeros(0, np.int64),
                "n_out": np.zeros(0, np.int64),
                "fr_buy": np.zeros(0),
                "fr_post": np.zeros(0),
            }
        return {
            key: np.concatenate([c[i] for c in self._flow]) for i, key in enumerate(self._FLOW_KEYS)
        }

    def restore(self, state: dict[str, Any]) -> None:
        if state.get("kind") != self.kind.value:
            raise ValueError(f"cannot restore a {state.get('kind')!r} state into a dealer market")
        if int(state["n_dealers"]) != self.n_dealers:
            raise ValueError("n_dealers mismatch on restore")
        if state["mode"] != self.mode:
            raise ValueError(f"mode mismatch on restore: {state['mode']} != {self.mode}")
        p = state["params"]
        self._tau_cfg = None if p["tau_cfg"] is None else float(p["tau_cfg"])
        self.outside_utility = float(p["outside_utility"])
        self.cost = float(p["cost"])
        self.beta = float(p["beta"])
        self.mu = float(p["mu"])
        self.a0 = float(p["a0"])
        self.fringe_share = float(p.get("fringe_share", 0.0))
        grid = state.get("grid")
        if grid is not None:  # a re-gridded market restores onto its own grid
            self.tick_size = float(grid["tick_size"])
            self.n_levels = int(grid["n_levels"])
        self.agent_id_offset = int(state["agent_id_offset"])
        self._restore_dynamic(state)

    def _restore_dynamic(self, state: dict[str, Any]) -> None:
        fr = state.get("fringe")
        self._fr = FringeBook.from_state(fr) if fr is not None else None
        if self._fr is not None:
            self.fringe_horizon = self._fr.horizon
        stash = state.get("trades_stash")
        self._tr_stash = (
            {key: np.array(v) for key, v in stash.items()} if stash is not None else None
        )
        self.tick = int(state["tick"])
        self.asks = np.array(state["asks"], dtype=np.int64)
        self.ask_size = np.array(state["ask_size"], dtype=np.float64)
        self.ask_order = np.array(state["ask_order"], dtype=np.int64)
        self.ask_expire = np.array(state["ask_expire"], dtype=np.int64)
        self.dealer_cash = np.array(state["dealer_cash"], dtype=np.float64)
        self.dealer_inv = np.array(state["dealer_inv"], dtype=np.float64)
        bg = np.asarray(state["bg"], dtype=np.float64)
        self.bg_cash, self.bg_inv = float(bg[0]), float(bg[1])
        self.glob = np.array(state["glob"], dtype=np.float64)
        self.trend = np.array(state["trend"], dtype=np.float64)
        self._pending = [
            (int(e["apply_tick"]), int(e["seq"]), _order_from_dict(e["order"]))
            for e in state["pending"]
        ]
        heapq.heapify(self._pending)
        self._seq = int(state["seq"])
        self._tape = np.array(state["tape"], dtype=np.float64)
        self._tpos = 0
        trades = state["trades"]
        k = int(np.asarray(trades["tick"]).shape[0])
        cap = max(1024, k)
        self._tr = {}
        for name, dtype in (
            ("tick", np.int64),
            ("buyer", np.int64),
            ("dealer", np.int64),
            ("price", np.int64),
            ("size", np.float64),
            ("value", np.float64),
        ):
            arr = np.zeros(cap, dtype=dtype)
            arr[:k] = np.asarray(trades[name], dtype=dtype)
            self._tr[name] = arr
        self._tr_pos = k
        self._trade_id_base = int(state["trade_id_base"])
        f = state["flow"]
        ticks = np.asarray(f["tick"], dtype=np.int64)
        k = len(ticks)
        self._flow = (
            [
                (
                    ticks,
                    np.asarray(f["buy"], dtype=np.float64).reshape(k, self.n_dealers),
                    np.asarray(f["n_arr"], dtype=np.int64),
                    np.asarray(f["n_out"], dtype=np.int64),
                    np.asarray(f.get("fr_buy", np.zeros(k)), dtype=np.float64),
                    np.asarray(f.get("fr_post", np.zeros(k)), dtype=np.float64),
                )
            ]
            if k
            else []
        )
        acc = state["period_acc"]
        self._p_qty = np.array(acc["qty"], dtype=np.float64)
        self._p_rev = np.array(acc["rev"], dtype=np.float64)
        counts = np.asarray(acc["counts"], dtype=np.int64)
        self._p_arr, self._p_out = int(counts[0]), int(counts[1])
        self._p_fringe = np.array(acc.get("fringe", np.zeros(3)), dtype=np.float64)
        self._periods = [PeriodSummary.from_dict(d) for d in state["periods"]]

    def migrate_from(self, other: Any) -> None:
        """Take over the dynamic state of ``other`` (pure: ``other`` is not modified).

        Parameters (tau, cost, demand, fringe share, tick) stay those of this market's config,
        so a switch can change them; standing prices are re-gridded onto this market's grid
        (asks UP). The swarm object is shared, not copied.
        """
        if isinstance(other, DealerMarket):
            st = other.state()
            if other.mode != self.mode:
                st["periods"] = []
            own_tick = self.tick_size
            self.tick_size, self.n_levels = other.tick_size, other.n_levels
            self._restore_dynamic(st)
            if other.swarm is not None:
                self.swarm = other.swarm
            if own_tick != other.tick_size:
                self._regrid(own_tick)
            return
        self.tick = int(other.tick)
        for name in ("dealer_cash", "dealer_inv"):
            if hasattr(other, name):
                setattr(self, name, np.array(getattr(other, name), dtype=np.float64))
        if getattr(other, "swarm", None) is not None:
            self.swarm = other.swarm
        if hasattr(other, "fundamental"):
            self.glob[2] = float(other.fundamental)
