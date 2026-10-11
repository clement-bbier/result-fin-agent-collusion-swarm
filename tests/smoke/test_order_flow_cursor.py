"""F1: under the poisson clock a slot's observed order flow covers its own decision interval."""

from __future__ import annotations

from typing import Any

import polars as pl
import pytest
from fakes import base_config

from finswarm.config.schema import RunConfig
from finswarm.sim.run import Runner, run_config
from finswarm.store.reader import load_run

pytestmark = pytest.mark.smoke


def _cfg(mode: str, swarm: bool) -> RunConfig:
    data = base_config(periods=25).model_dump(mode="json")
    data["ticks_per_period"] = 10
    data["clock"] = {"mode": mode, "epoch_ticks": 4}
    if swarm:
        data["swarm"] = {"mode": "agents", "n_agents": 500, "arrivals": {"lam_mean": 0.3}}
        data["market"]["demand"] = {"kind": "swarm"}
    return RunConfig.model_validate(data)


@pytest.mark.parametrize("swarm", [False, True])
def test_poisson_flow_since_own_last_decision(tmp_path, monkeypatch, swarm) -> None:
    seen: list[tuple[int, int, int, float, float, int]] = []
    orig = Runner._observe

    def spy(self: Runner, s: Any, p: int, e: int, book: list[Any]) -> Any:
        obs = orig(self, s, p, e, book)
        tr = obs.trades_since_epoch
        own = sum(t.size for t in tr if t.seller_id == s.agent_id)
        seen.append((s.slot, s.agent_id, obs.tick, own, sum(t.size for t in tr), len(tr)))
        return obs

    monkeypatch.setattr(Runner, "_observe", spy)
    r = run_config(_cfg("poisson", swarm), out=tmp_path)
    trades = load_run(r.run_dir).frame("trades")
    last: dict[int, int] = {}
    checked = 0
    for slot, aid, tick, own, vol, n in seen:
        if slot in last:
            win = trades.filter((pl.col("tick") >= last[slot]) & (pl.col("tick") < tick))
            mine = win.filter(pl.col("seller_id") == aid)["size"].sum()
            assert n == win.height, (slot, tick)
            assert own == pytest.approx(mine)
            assert vol == pytest.approx(win["size"].sum())
            checked += 1
        last[slot] = tick
    assert checked > 20
    # staggered decisions: the slots rarely observe at the same tick
    assert len({t for _, _, t, *_ in seen}) > len(seen) // 2
