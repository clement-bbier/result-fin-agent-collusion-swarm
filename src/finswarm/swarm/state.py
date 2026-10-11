"""Struct-of-arrays state of the background swarm.

One row per swarm agent (ids ``0..N-1``). Every field is a contiguous numpy array with a
fixed dtype so that kernels can take them directly. Field semantics:

==============  =======  ===================================================================
``stype``       int8     ``SwarmType`` value
``value_base``  float64  private value deviation ``eps_j`` (alpha units)
``noise_sd``    float64  sd of the per-arrival private signal noise (alpha units)
``lam``         float64  arrival intensity per tick
``horizon``     int64    order lifetime in ticks (fringe/LOB mechanisms)
``aversion``    float64  risk aversion (market makers)
``trend_k``     float64  trend-chasing loading on the latest mean-quote change (price units)
``sentiment``   float64  idiosyncratic sentiment deviation ``x_j`` at ``last_update``; the
                         full sentiment is ``H(t) + rho**(t - last_update) * x_j + noise``
``attention``   float64  per-tick probability of learning a pending fundamental shock
``inventory``   float64  units held
``cash``        float64  cash balance (price units)
``belief_F``    float64  last fundamental known by the agent (alpha units)
``last_update`` int64    tick of the last lazy refresh
``shock_delay`` int64    absolute tick at which the agent learns the current fundamental
                         (-1 when up to date)
``active``      bool     participates in arrivals
``max_inv``     float64  inventory cap (buyers stop buying at the cap)
==============  =======  ===================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

FIELD_DTYPES: dict[str, Any] = {
    "stype": np.int8,
    "value_base": np.float64,
    "noise_sd": np.float64,
    "lam": np.float64,
    "horizon": np.int64,
    "aversion": np.float64,
    "trend_k": np.float64,
    "sentiment": np.float64,
    "attention": np.float64,
    "inventory": np.float64,
    "cash": np.float64,
    "belief_F": np.float64,
    "last_update": np.int64,
    "shock_delay": np.int64,
    "active": np.bool_,
    "max_inv": np.float64,
}


@dataclass(eq=False)
class SwarmState:
    stype: np.ndarray
    value_base: np.ndarray
    noise_sd: np.ndarray
    lam: np.ndarray
    horizon: np.ndarray
    aversion: np.ndarray
    trend_k: np.ndarray
    sentiment: np.ndarray
    attention: np.ndarray
    inventory: np.ndarray
    cash: np.ndarray
    belief_F: np.ndarray
    last_update: np.ndarray
    shock_delay: np.ndarray
    active: np.ndarray
    max_inv: np.ndarray
    _alias: tuple[np.ndarray, np.ndarray, float] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        n = len(self.stype)
        for name, dtype in FIELD_DTYPES.items():
            arr = np.ascontiguousarray(getattr(self, name), dtype=dtype)
            if arr.shape != (n,):
                raise ValueError(f"swarm field {name!r} has shape {arr.shape}, expected ({n},)")
            setattr(self, name, arr)

    @property
    def n(self) -> int:
        return int(self.stype.shape[0])

    @classmethod
    def empty(cls, n: int) -> SwarmState:
        arrays = {name: np.zeros(n, dtype=dtype) for name, dtype in FIELD_DTYPES.items()}
        arrays["shock_delay"][:] = -1
        arrays["active"][:] = True
        arrays["max_inv"][:] = np.inf
        arrays["attention"][:] = 1.0
        return cls(**arrays)

    # ---- Vose alias table over lam * active; cached until lam/active change ----
    def alias(self) -> tuple[np.ndarray, np.ndarray, float]:
        if self._alias is None:
            from finswarm.swarm.population import alias_table

            self._alias = alias_table(self.lam * self.active)
        return self._alias

    def invalidate_alias(self) -> None:
        self._alias = None

    def set_active(self, mask: np.ndarray, value: bool) -> None:
        self.active[np.asarray(mask, dtype=bool)] = value
        self.invalidate_alias()

    def set_lam(self, lam: np.ndarray) -> None:
        self.lam[:] = np.asarray(lam, dtype=np.float64)
        self.invalidate_alias()

    def total_intensity(self) -> float:
        return float(self.alias()[2])

    # ---- serialisation (dict of numpy arrays, see core.checkpoint) ----
    def state(self) -> dict[str, Any]:
        return {name: getattr(self, name).copy() for name in FIELD_DTYPES}

    def restore(self, state: dict[str, Any]) -> None:
        for name, dtype in FIELD_DTYPES.items():
            arr = np.array(state[name], dtype=dtype)
            if arr.shape != getattr(self, name).shape:
                raise ValueError(f"swarm field {name!r}: shape mismatch on restore")
            setattr(self, name, np.ascontiguousarray(arr))
        self.invalidate_alias()

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> SwarmState:
        return cls(
            **{name: np.array(state[name], dtype=dtype) for name, dtype in FIELD_DTYPES.items()}
        )

    def copy(self) -> SwarmState:
        return SwarmState.from_state(self.state())
