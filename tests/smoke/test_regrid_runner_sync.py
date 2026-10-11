"""F2: after a tick re-grid the runner's in-force and pending quotes match the market's."""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import pytest
from fakes import base_config

from finswarm.config.schema import RunConfig
from finswarm.sim.integration import _sync_runner_after_regrid
from finswarm.sim.run import Runner, SlotState, run_config

pytestmark = pytest.mark.smoke


def _market_ask(market: Any, slot: int) -> float:
    k = int(market.asks[slot])
    return math.nan if k < 0 else float(market.p_min + k * market.tick_size)


def _runner_ask(runner: Runner, x: SlotState, tick: int) -> float:
    """The runner's in-force ask once its pending changes up to ``tick`` apply (the market has
    applied those due at ``tick`` when an advance returns; the runner does at the next batch)."""
    ask = x.live_ask
    for a, _, slot, side, px, _ in runner.pending_quotes:
        if a <= tick and slot == x.slot and side in (0, -1):  # cancel, Side.SELL
            ask = math.nan if side == 0 else px
    return ask


def test_sync_runner_regrid_moves_live_and_pending_quotes() -> None:
    s = SlotState(slot=0, agent_id=7, spec={})
    s.ask = s.live_ask = 1.37
    s.bid = s.live_bid = 1.03
    s.live_ask_size = s.live_bid_size = 1.0
    s.live_orders = {1: 3, -1: 4}
    runner = SimpleNamespace(
        slots=[s],
        quote_log=[],
        last_trades=[],
        pending_quotes=[
            (12, 5, 0, -1, 1.52, 1.0),  # an ask: the market moves it to level 5 (1.625)
            (12, 6, 0, 1, 1.21, 1.0),  # a bid withdrawn by the market
            (13, 8, 0, 0, math.nan, 0.0),  # a delayed cancel is kept as is
        ],
    )
    report = {
        "old_tick": 0.01,
        "new_tick": 0.125,
        "n_levels": [101, 9],
        "p_min": 1.0,
        "pending": [
            {"order_id": 5, "old_ticks": 52, "new_ticks": 5},
            {"order_id": 6, "old_ticks": 21, "new_ticks": -1},
        ],
    }
    _sync_runner_after_regrid(runner, report)
    assert s.live_ask == pytest.approx(1.375) and s.ask == pytest.approx(1.375)
    assert s.live_bid == pytest.approx(1.0)
    assert [(q[1], q[4]) for q in runner.pending_quotes][:1] == [(5, pytest.approx(1.625))]
    assert [q[1] for q in runner.pending_quotes] == [5, 8]


def test_coarsening_regrid_with_speed_bump_keeps_runner_on_market_grid(
    tmp_path, monkeypatch
) -> None:
    data = base_config(periods=12).model_dump(mode="json")
    data["ticks_per_period"] = 10
    data["clock"] = {"mode": "sync", "epoch_ticks": 3, "speed_bump_max": 7}
    data["market"]["price_grid"] = {**data["market"]["price_grid"], "tick": 0.01}
    data["events"] = [
        {
            "type": "set_param",
            "path": "market.price_grid.tick",
            "value": 0.125,
            "when": {"at_period": 5},
        }
    ]
    cfg = RunConfig.model_validate(data)
    checked = {"n": 0, "pending": 0}
    orig = Runner._advance_to

    def spy(self: Runner, tick: int, p: int, acc: Any, mids: list[float]) -> None:
        orig(self, tick, p, acc, mids)
        m = self.market
        if m.tick_size == pytest.approx(0.125):
            for x in self.slots:
                want, got = _market_ask(m, x.slot), _runner_ask(self, x, int(m.tick))
                assert (math.isnan(want) and math.isnan(got)) or got == pytest.approx(want), (
                    p,
                    x.slot,
                    got,
                    want,
                )
                checked["n"] += 1
            for q in self.pending_quotes:
                if q[3] != 0:
                    k = (q[4] - m.p_min) / m.tick_size
                    assert k == pytest.approx(round(k)), q
                    checked["pending"] += 1

    monkeypatch.setattr(Runner, "_advance_to", spy)
    r = run_config(cfg, out=tmp_path)
    assert r.status in ("completed", "ok")
    assert checked["n"] > 10 and checked["pending"] > 0
