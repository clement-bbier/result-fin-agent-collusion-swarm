"""Mechanism (B): a public limit-order fringe inside the dealer market (lever L2).

The fringe is an overlay of :class:`finswarm.market.dealer.DealerMarket`, switched by
``market.fringe.share`` (live via ``set_param``, from 0 to 0.3 and back, from any checkpoint);
it needs ``swarm.mode = agents``. Semantics (one-sided market: dealers sell, the swarm buys):

* **Who.** A fraction ``share`` of the arriving ZI buyers (``SwarmType.ZI_BUYER``) act as
  fringe sellers instead of buying. Membership reuses the arrival's routing uniform ``ur``
  (``ur < share``), so the swarm's random numbers are identical with and without the fringe.
* **Cost model.** A fringe seller's reservation price is
  ``r_j = c + u_j * max(0, v_j - c)``, with ``c = market.cost * alpha`` the dealers' marginal
  cost, ``v_j`` the reservation value the agent would have had as a buyer (fundamental belief,
  private value, signal noise, sentiment) and ``u_j = ur / share ~ U[0, 1)``: a cost uniformly
  spread between the dealers' production cost and the agent's own valuation (about
  ``U[1, 2] * alpha`` in the Calvano calibration). The seller rests **one lot** at ``r_j``
  rounded **up** to the dealer grid (never below its reservation price; dropped above the
  grid) for ``market.fringe.horizon_ticks`` ticks. ``seller_cost`` of a fringe trade is the
  traded level (the seller's reservation price up to one tick of rounding).
* **Routing.** Every other buyer chooses by logit over ``{dealer asks..., best fringe ask,
  outside}`` with the same temperature ``tau``; the fringe's utility is ``v - best fringe
  price``. A fringe fill consumes the oldest lot at the best level (price-time priority).
* **Records.** Fringe trades have ``seller_id = -2`` (anonymous fringe); ``flow_per_tick`` gets
  one ``dealer_id = -2`` row per tick (``buy_qty`` = lots bought from the fringe, ``sell_qty``
  = lots posted to the fringe) once the fringe has been switched on; fringe depth shows in
  ``snapshot()`` (hence in the agents' ``book_levels``) and in ``best_ask()``.
* **Accounts.** The fringe is one aggregate account (cash, inventory), so cash and inventory
  stay conserved between dealers, fringe and swarm.
* **Benchmarks.** With the fringe on, the dealers face the residual demand left by a
  competitive fringe whose lots are priced at cost draws, so Calvano's closed-form Nash and
  monopoly prices do not apply. The competitive / collusive benchmarks are the static Nash
  equilibrium and the symmetric joint-profit maximum of the dealers' one-shot pricing game in
  the market *with* the fringe (same share, horizon, ``tau``, grid and swarm as the evaluation
  window): ``finswarm.metrics.benchmarks.estimate_by_sweeps(
  finswarm.market.sweep.profit_fn(cfg), grid, n_dealers)``.

:class:`FringeBook` stores the depth aggregated per grid level with an expiry wheel (no
per-order objects); the hot loop lives in ``finswarm.swarm.kernels.dealer_fringe_ticks``.
``MechanismKind.FRINGE`` is not registered separately: a dealer market with ``share > 0`` is
mechanism B.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from finswarm.market.grid import regrid_indices

#: largest expiry wheel (horizon x grid levels) we allocate (int64 cells, 128 MiB)
MAX_WHEEL_CELLS = 1 << 24

# st[] layout (see finswarm.swarm.kernels)
BEST, RESTING, POSTED, FILLED, EXPIRED = range(5)


@dataclass(eq=False)
class FringeBook:
    """Fringe depth on the dealer grid: ``depth[L]``, expiry ``wheel[t % H, L]``, slot
    ``meta[s] = [lots, lowest level, highest level]``, ``st = [best level, resting lots,
    posted, filled, expired]`` and the aggregate account ``acct = [cash, inventory]``."""

    depth: np.ndarray
    wheel: np.ndarray
    meta: np.ndarray
    st: np.ndarray
    acct: np.ndarray

    @classmethod
    def empty(cls, horizon: int, n_levels: int) -> FringeBook:
        horizon, n_levels = int(horizon), int(n_levels)
        if horizon < 1:
            raise ValueError("market.fringe.horizon_ticks must be >= 1")
        check_wheel(horizon, n_levels)
        st = np.zeros(5, dtype=np.int64)
        st[BEST] = -1
        return cls(
            depth=np.zeros(n_levels, dtype=np.int64),
            wheel=np.zeros((horizon, n_levels), dtype=np.int64),
            meta=np.zeros((horizon, 3), dtype=np.int64),
            st=st,
            acct=np.zeros(2, dtype=np.float64),
        )

    # ------------------------------------------------------------------ queries
    @property
    def horizon(self) -> int:
        return int(self.wheel.shape[0])

    @property
    def best(self) -> int:
        return int(self.st[BEST])

    @property
    def resting(self) -> int:
        return int(self.st[RESTING])

    def levels(self) -> list[tuple[int, int]]:
        """``(level, lots)`` for every level with resting depth, by price."""
        nz = np.flatnonzero(self.depth)
        return [(int(k), int(self.depth[k])) for k in nz]

    def stats(self) -> dict[str, float]:
        return {
            "best_level": float(self.st[BEST]),
            "resting": float(self.st[RESTING]),
            "posted": float(self.st[POSTED]),
            "filled": float(self.st[FILLED]),
            "expired": float(self.st[EXPIRED]),
            "cash": float(self.acct[0]),
            "inventory": float(self.acct[1]),
        }

    def check(self) -> None:
        """Invariants (``FINSWARM_DEBUG=1``): non-negative depth, wheel and depth agree, the
        lot identity ``posted == filled + expired + resting`` holds, the best level is right."""
        if (self.depth < 0).any() or (self.wheel < 0).any():
            raise AssertionError("fringe: negative depth")
        if not np.array_equal(self.wheel.sum(axis=0), self.depth):
            raise AssertionError("fringe: wheel and depth disagree")
        if not np.array_equal(self.wheel.sum(axis=1), self.meta[:, 0]):
            raise AssertionError("fringe: wheel slot totals are stale")
        st = self.st
        if st[POSTED] != st[FILLED] + st[EXPIRED] + st[RESTING] or st[RESTING] != self.depth.sum():
            raise AssertionError("fringe: lot identity broken")
        nz = np.flatnonzero(self.depth)
        if (int(nz[0]) if nz.size else -1) != st[BEST]:
            raise AssertionError("fringe: stale best level")

    # ------------------------------------------------------------------ re-grid
    def regrid(self, old_tick: float, new_tick: float, n_levels: int) -> dict[str, Any]:
        """Move the resting lots onto a new grid (asks round UP); lots above the new grid are
        dropped (counted as expired). Expiry slots and counters are kept."""
        check_wheel(self.horizon, n_levels)
        old_n = self.depth.shape[0]
        new_level = regrid_indices(np.arange(old_n), old_tick, new_tick, n_levels, up=True)
        valid = new_level >= 0
        wheel = np.zeros((self.horizon, n_levels), dtype=np.int64)
        np.add.at(wheel.T, new_level[valid], self.wheel.T[valid])
        dropped = int(self.wheel[:, ~valid].sum())
        before = self.levels()
        self.wheel = np.ascontiguousarray(wheel)
        self._rebuild()
        self.st[EXPIRED] += dropped
        return {
            "resting_before": int(sum(n for _, n in before)),
            "resting_after": self.resting,
            "dropped": dropped,
            "levels": [[int(k), int(new_level[k]), int(n)] for k, n in before],
        }

    def _rebuild(self) -> None:
        """Recompute depth, slot metadata and best level from the wheel."""
        self.depth = np.ascontiguousarray(self.wheel.sum(axis=0))
        h = self.horizon
        meta = np.zeros((h, 3), dtype=np.int64)
        meta[:, 0] = self.wheel.sum(axis=1)
        for s in np.flatnonzero(meta[:, 0]):
            nz = np.flatnonzero(self.wheel[s])
            meta[s, 1], meta[s, 2] = nz[0], nz[-1]
        self.meta = meta
        nz = np.flatnonzero(self.depth)
        self.st[BEST] = int(nz[0]) if nz.size else -1
        self.st[RESTING] = int(self.depth.sum())

    # ------------------------------------------------------------------ state
    def state(self) -> dict[str, np.ndarray]:
        return {
            "depth": self.depth.copy(),
            "wheel": self.wheel.copy(),
            "meta": self.meta.copy(),
            "st": self.st.copy(),
            "acct": self.acct.copy(),
        }

    @classmethod
    def from_state(cls, st: dict[str, Any]) -> FringeBook:
        return cls(
            depth=np.ascontiguousarray(np.array(st["depth"], dtype=np.int64)),
            wheel=np.ascontiguousarray(np.array(st["wheel"], dtype=np.int64)),
            meta=np.ascontiguousarray(np.array(st["meta"], dtype=np.int64)),
            st=np.array(st["st"], dtype=np.int64),
            acct=np.array(st["acct"], dtype=np.float64),
        )


def check_wheel(horizon: int, n_levels: int) -> None:
    if int(horizon) * int(n_levels) > MAX_WHEEL_CELLS:
        raise ValueError(
            f"fringe expiry wheel too large ({horizon} ticks x {n_levels} levels > "
            f"{MAX_WHEEL_CELLS}); use a coarser tick or a shorter market.fringe.horizon_ticks"
        )
