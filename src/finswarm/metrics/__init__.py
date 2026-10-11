"""finswarm metrics package.

Importing the package registers every metric in :data:`finswarm.metrics.registry.REGISTRY`
(statsmodels and other heavy dependencies are imported lazily inside the functions).
"""

from finswarm.metrics import (  # noqa: F401  (imported for registration side effects)
    benchmarks,
    collusion,
    consistency,
    dependence,
    detection,
    dynamics,
    judge,
    quality,
)
from finswarm.metrics.registry import REGISTRY, get_metric, register_metric

__all__ = ["REGISTRY", "get_metric", "register_metric"]
