"""finswarm swarm package: struct-of-arrays background agents and their kernels."""

from finswarm.swarm.population import build_population, build_population_for_run, swarm_size
from finswarm.swarm.state import SwarmState

__all__ = ["SwarmState", "build_population", "build_population_for_run", "swarm_size"]
