"""Mechanism registry: ``MechanismKind`` -> market class, and the ``make_market`` factory."""

from __future__ import annotations

from typing import TYPE_CHECKING

from finswarm.config.schema import RunConfig
from finswarm.core.rng import RNGRegistry
from finswarm.core.types import MechanismKind
from finswarm.market.base import Market
from finswarm.market.dealer import DealerMarket

if TYPE_CHECKING:
    from finswarm.swarm.state import SwarmState

REGISTRY: dict[MechanismKind, type] = {
    MechanismKind.DEALER: DealerMarket,
}


def make_market(cfg: RunConfig, rng: RNGRegistry, swarm: SwarmState | None = None) -> Market:
    """Build the market for ``cfg.market.mechanism``.

    ``swarm`` is the background population (``finswarm.swarm.build_population_for_run``);
    when ``cfg.swarm.mode == "agents"`` and it is omitted, the market draws it from the
    ``population`` stream and exposes it as ``market.swarm``.
    """
    kind = MechanismKind(cfg.market.mechanism)
    try:
        cls = REGISTRY[kind]
    except KeyError:
        avail = sorted(k.value for k in REGISTRY)
        raise NotImplementedError(
            f"mechanism {kind.value!r} is not implemented yet; available: {avail}"
        ) from None
    market = cls(cfg, rng, swarm)
    return market  # type: ignore[no-any-return]
