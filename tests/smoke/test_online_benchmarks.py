"""F5: with the fringe on, delta (profit) is NaN and ``benchmark_valid`` is False."""

from __future__ import annotations

import polars as pl
import pytest
from fakes import base_config

from finswarm.config.schema import RunConfig
from finswarm.sim.run import run_config
from finswarm.store.reader import load_run

pytestmark = pytest.mark.smoke


def _cfg(fringe: float) -> RunConfig:
    data = base_config(periods=6).model_dump(mode="json")
    data["ticks_per_period"] = 10
    data["clock"] = {"mode": "sync", "epoch_ticks": 5}
    data["swarm"] = {"mode": "agents", "n_agents": 500, "arrivals": {"lam_mean": 0.3}}
    data["market"]["demand"] = {"kind": "swarm"}
    data["market"]["fringe"] = {"share": fringe, "horizon_ticks": 20}
    return RunConfig.model_validate(data)


@pytest.mark.parametrize("fringe", [0.0, 0.3])
def test_benchmark_valid_flag_and_delta_with_fringe(tmp_path, fringe) -> None:
    r = run_config(_cfg(fringe), out=tmp_path)
    m = load_run(r.run_dir).frame("metrics_per_period")
    assert m.height == 6
    if fringe > 0:
        assert not m["benchmark_valid"].any()
        assert m["delta"].is_nan().all()
        # F7: per-tick arrival counts on the dealer rows only, never on the fringe row
        flow = load_run(r.run_dir).frame("flow_per_tick")
        fr = flow.filter(pl.col("dealer_id") == -2)
        assert fr.height > 0 and (fr["n_arrivals"] == 0).all() and (fr["n_outside"] == 0).all()
        per_tick = (
            flow.filter(pl.col("dealer_id") >= 0)
            .group_by("tick")
            .agg(pl.col("n_arrivals").n_unique().alias("u"), pl.col("n_arrivals").first())
        )
        assert (per_tick["u"] == 1).all() and per_tick["n_arrivals"].sum() > 0
        assert (fr["sell_qty"].sum() > 0) and (fr["buy_qty"].sum() > 0)
    else:
        assert m["benchmark_valid"].all()
        assert not m["delta"].is_nan().any()
