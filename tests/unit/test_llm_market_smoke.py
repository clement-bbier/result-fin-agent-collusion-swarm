"""Smoke: mock-lane LLM dealers drive the real closed-form dealer market (sim-core).

Skipped until ``finswarm.market.registry`` (sim-core) is merged into this branch. The
observation building here is a minimal stand-in for the runner (agents-sim).
"""

import asyncio
import math

import numpy as np
import pytest
from llm_fakes import mock_lane

registry = pytest.importorskip("finswarm.market.registry")

from finswarm.agents.llm.llm_agent import LLMAgent  # noqa: E402
from finswarm.config.schema import RunConfig  # noqa: E402
from finswarm.core.rng import RNGRegistry  # noqa: E402
from finswarm.core.types import Observation, Order, OrderType, PeriodRecord, Side  # noqa: E402
from finswarm.llm.cache import ResponseCache  # noqa: E402
from finswarm.llm.scheduler import LaneScheduler  # noqa: E402
from finswarm.llm.timing import VirtualClock  # noqa: E402


def test_mock_llm_dealers_on_closed_form_market() -> None:
    cfg = RunConfig(name="llm_smoke", periods=12)
    market = registry.make_market(cfg, RNGRegistry(0))
    sched = LaneScheduler(
        {"mock": mock_lane()}, clock=VirtualClock(), cache=ResponseCache(mode="mock")
    )
    policies = ("tit_for_tat", "static_nash")
    agents = [
        LLMAgent(
            agent_id=market.dealer_agent_id(slot),
            slot=slot,
            name=f"Firm {slot}",
            model="mock-v1",
            provider="mock",
            lane="mock",
            scheduler=sched,
            prompt="fish_v1",
            mock_policy=policies[slot],
        )
        for slot in range(2)
    ]
    extra = {
        "max_wtp": cfg.market.price_grid.max,
        "tick_size": market.tick_size,
        "p_min": market.p_min,
    }

    async def run() -> list[list[float]]:
        prices: list[list[float]] = []
        oid = 0
        for period in range(cfg.periods):
            obs = [
                Observation(
                    tick=market.tick,
                    period=period,
                    agent_id=a.agent_id,
                    slot=a.slot,
                    role="dealer",
                    inventory=float(market.dealer_inv[a.slot]),
                    cash=float(market.dealer_cash[a.slot]),
                    cost=market.marginal_cost,
                    extra=dict(extra),
                )
                for a in agents
            ]
            decisions = await asyncio.gather(
                *(a.decide(o) for a, o in zip(agents, obs, strict=True))
            )
            assert all(d.parse_ok for d in decisions)
            for a, d in zip(agents, decisions, strict=True):
                ticks = int(round((d.price - market.p_min) / market.tick_size))
                order = Order(
                    oid,
                    a.agent_id,
                    Side.SELL,
                    OrderType.QUOTE,
                    ticks,
                    1.0,
                    -1,
                    market.tick,
                    market.tick,
                )
                oid += 1
                assert market.submit(order).accepted
            market.advance(cfg.ticks_per_period)
            (summary,) = market.drain_periods()
            for a in agents:
                rivals = tuple(float(summary.ask[j]) for j in range(2) if j != a.slot)
                vwap = summary.vwap[~np.isnan(summary.vwap)]
                a.record_period(
                    PeriodRecord(
                        period=period,
                        own_price=float(summary.ask[a.slot]),
                        own_quantity=float(summary.qty[a.slot]),
                        own_profit=float(summary.profit[a.slot]),
                        rival_prices=rivals,
                        market_price=float(vwap.mean()) if vwap.size else math.nan,
                    )
                )
            prices.append([d.price for d in decisions])
        return prices

    prices = asyncio.run(run())
    assert prices[0] == pytest.approx([1.92, 1.47])  # tit-for-tat opens at the monopoly price
    assert prices[-1] == pytest.approx([1.47, 1.47])  # then matches the static-Nash rival
    for a in agents:
        assert len(a.memory.history) == cfg.periods
        assert all(r.own_profit > 0 for r in a.memory.history)
        final = Observation(
            tick=market.tick,
            period=cfg.periods,
            agent_id=a.agent_id,
            slot=a.slot,
            role="dealer",
            inventory=0.0,
            cash=0.0,
            cost=market.marginal_cost,
            extra=dict(extra),
        )
        _, user = a.render(final)
        assert "Round 12:" in user and "Round 1:" in user
        assert len(a.drain_logs()) == cfg.periods
