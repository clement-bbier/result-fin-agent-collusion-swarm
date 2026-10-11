"""F4: ``quotes`` rows under the speed bump carry ``decision_tick`` and ``apply_period``."""

from __future__ import annotations

import polars as pl
import pytest
from fakes import base_config

from finswarm.config.schema import RunConfig
from finswarm.sim.run import run_config
from finswarm.store.reader import load_run

pytestmark = pytest.mark.smoke


@pytest.mark.parametrize("mode", ["sync", "poisson"])
def test_quotes_decision_tick_and_apply_period(tmp_path, mode) -> None:
    data = base_config(periods=12).model_dump(mode="json")
    data["ticks_per_period"] = 10
    data["clock"] = {"mode": mode, "epoch_ticks": 3, "speed_bump_max": 25}
    r = run_config(RunConfig.model_validate(data), out=tmp_path)
    data_ = load_run(r.run_dir)
    q = data_.frame("quotes")
    sent = q.filter(pl.col("decision_id") >= 0)
    held = q.filter(pl.col("decision_id") < 0)
    assert sent.height > 0
    assert ((sent["tick"] - sent["decision_tick"]) == 25).all()
    assert (sent["apply_period"] == sent["tick"] // 10).all()
    assert (sent["period"] == sent["decision_tick"] // 10).all()
    assert (sent["apply_period"] >= 12).any()  # last decisions never take effect in the run
    assert (held["tick"] == held["decision_tick"]).all()
    assert (held["apply_period"] == held["period"]).all()
    llm = data_.frame("llm_decisions").sort("decision_id")
    mine = sent.filter(pl.col("decision_id").is_in(llm["decision_id"].implode())).sort(
        "decision_id"
    )
    assert mine["decision_tick"].to_list() == llm["obs_tick"].to_list()
    ap = data_.frame("agent_period")
    assert ap["period"].max() == 11  # agent_period stays on the run's periods
