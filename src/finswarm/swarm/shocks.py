"""Information shocks and liquidity removal for the swarm (``interventions`` stream).

``InfoShock``: the fundamental moves ``F <- F + dF``. Each agent learns it after a
per-agent geometric delay with success probability ``attention_j`` per tick (support
1, 2, ...; delay 1 = learns at the shock tick). Awareness is applied lazily by the kernels
(``refresh_agent``): an agent with ``t >= shock_delay`` adopts the *current* fundamental.

A new shock supersedes pending awareness: agents that had already passed their awareness
tick are first brought up to date with the old fundamental; agents still unaware learn the
cumulative fundamental at their new awareness tick.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

import numpy as np

from finswarm.core.types import SwarmType
from finswarm.swarm.state import SwarmState


def geometric_delays(attention: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Per-agent delays ``D_j ~ Geometric(attention_j)`` on ``{1, 2, ...}`` (int64)."""
    p = np.clip(np.asarray(attention, dtype=np.float64), 1e-12, 1.0)
    return rng.geometric(p).astype(np.int64)


def apply_info_shock(
    swarm: SwarmState,
    *,
    tick: int,
    fundamental: float,
    d_fundamental: float,
    diffusion: Literal["instant", "attention"],
    rng: np.random.Generator,
) -> float:
    """Apply a shock at ``tick``; returns the new fundamental (the caller stores it)."""
    pending = (swarm.shock_delay >= 0) & (swarm.shock_delay <= tick)
    swarm.belief_F[pending] = fundamental
    swarm.shock_delay[pending] = -1
    if diffusion == "instant":
        swarm.shock_delay[:] = tick
    elif diffusion == "attention":
        swarm.shock_delay[:] = tick + geometric_delays(swarm.attention, rng) - 1
    else:
        raise ValueError(f"unknown diffusion {diffusion!r}")
    return float(fundamental + d_fundamental)


def remove_liquidity(
    swarm: SwarmState, share: float, swarm_types: Iterable[str], rng: np.random.Generator
) -> np.ndarray:
    """Deactivate ``round(share * k)`` of the ``k`` active agents of the given types."""
    if not 0.0 <= share <= 1.0:
        raise ValueError("share must lie in [0, 1]")
    codes = [int(SwarmType[name]) for name in swarm_types]
    candidates = np.flatnonzero(np.isin(swarm.stype, codes) & swarm.active)
    k = int(round(share * candidates.size))
    if k == 0:
        return np.empty(0, dtype=np.int64)
    chosen = np.sort(rng.choice(candidates, size=k, replace=False))
    mask = np.zeros(swarm.n, dtype=bool)
    mask[chosen] = True
    swarm.set_active(mask, False)
    return chosen.astype(np.int64)
