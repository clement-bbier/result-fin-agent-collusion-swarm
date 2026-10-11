"""Static best response stays finite at every registered routing level (s1..s4)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from finswarm.market.demand import DemandParams, best_response_price, nash_price


@pytest.mark.parametrize("tau", [0.25, 0.125, 0.0625, 0.02])
def test_best_response_finite_at_registered_routing_levels(tau: float) -> None:
    prm = DemandParams(alpha=1.0, a=2.0, a0=0.0, mu=tau, cost=1.0)
    rivals = np.full(3, 1.9)  # rivals near the monopoly price
    br = best_response_price(rivals, prm)
    assert math.isfinite(br)
    assert 1.0 < br < 1.9 + 1e-9  # undercut the rivals, stay above cost


def test_best_response_to_symmetric_nash_is_nash() -> None:
    for tau in (0.25, 0.0625, 0.02):
        prm = DemandParams(alpha=1.0, a=2.0, a0=0.0, mu=tau, cost=1.0)
        pn = nash_price(prm, 4)
        br = best_response_price(np.full(3, pn), prm)
        assert br == pytest.approx(pn, abs=2e-3)
