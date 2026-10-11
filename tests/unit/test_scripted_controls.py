"""Registered scripted controls: best_responder (SC1), cost_plus (SC2), leader_follower (SC4),
the single-dealer SC5, and the control/MQC files that use them."""

import asyncio
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest

from finswarm.agents.scripted import BestResponder, CostPlus, LeaderFollower, make_scripted
from finswarm.agents.scripted.competitive import demand_dict
from finswarm.config.load import load_campaign_config, load_yaml
from finswarm.config.schema import CampaignConfig, MarketConfig
from finswarm.core.types import Observation, PeriodRecord, Quote
from finswarm.market.demand import DemandParams, best_response_price, monopoly_price, nash_price
from finswarm.sim.campaign import run_campaign
from finswarm.store.reader import load_run

ROOT = Path(__file__).resolve().parents[2]
CONTROLS = ROOT / "configs" / "controls"


def obs(period: int = 1, standing=(), history=(), cost: float = 1.0, **extra: Any) -> Observation:
    hist = (
        [PeriodRecord(period - 1, 1.5, 0.0, 0.0, tuple(history), float("nan"))] if history else []
    )
    quotes = [
        Quote(agent_id=11 + i, ask_ticks=round((p - 1.0) / 0.01)) for i, p in enumerate(standing)
    ]
    return Observation(
        tick=period, period=period, agent_id=10, slot=0, role="dealer", inventory=0.0, cash=0.0,
        cost=cost, history=hist, rival_quotes=quotes,
        extra={"p_min": 1.0, "tick_size": 0.01, "price_max": 4.0, **extra},
    )  # fmt: skip


def decide(agent, o: Observation) -> float:
    return float(asyncio.run(agent.decide(o)).ask)


# ----------------------------------------------------------------------------- best_responder
def test_best_responder_is_the_static_best_response_and_reaches_nash() -> None:
    m = MarketConfig(n_dealers=2)
    br = make_scripted("best_responder", {}, slot=0, agent_id=10, market=m)
    assert isinstance(br, BestResponder)
    p_n = nash_price(m)
    near_nash = decide(br, obs(standing=[p_n]))  # quotes are on the 0.01 grid: 1.47
    assert near_nash == pytest.approx(best_response_price([1.47], m))
    assert abs(near_nash - p_n) < 0.005  # Nash is (up to the grid) a fixed point
    assert decide(br, obs(standing=[1.8])) == pytest.approx(best_response_price([1.8], m))
    # no standing quote visible (view hides them): the last period's rival price is used
    assert decide(br, obs(history=[1.8])) == pytest.approx(best_response_price([1.8], m))
    assert decide(br, obs()) == pytest.approx(br.start)  # nothing visible yet: start (Nash)
    # iterated best responses (on the grid) converge to the symmetric Nash price
    p = 1.9
    for _ in range(40):
        p = decide(br, obs(standing=[p]))
    assert abs(p - p_n) < 0.01


def test_best_responder_follows_live_routing_and_cost() -> None:
    m = MarketConfig(n_dealers=4)
    br = make_scripted("best_responder", {}, slot=0, agent_id=10, market=m)
    rivals = [1.5, 1.5, 1.5]
    base = decide(br, obs(standing=rivals))
    live = {**demand_dict(m), "tau": 0.0625}  # lever L1: s1 -> s3, more price-sensitive
    sharper = decide(br, obs(standing=rivals, demand_live=live))
    want = best_response_price(rivals, DemandParams(mu=0.0625))
    assert sharper == pytest.approx(want) and sharper < base
    dearer = decide(br, obs(standing=rivals, cost=1.3))
    assert dearer > base  # the live unit cost is the marginal cost


def test_single_dealer_best_response_is_the_monopoly_price() -> None:
    m = MarketConfig(n_dealers=1)
    br = make_scripted("best_responder", {}, slot=0, agent_id=10, market=m)
    assert decide(br, obs()) == pytest.approx(monopoly_price(m, 1), abs=1e-6)


# ---------------------------------------------------------------------------------- cost_plus
def test_cost_plus_shares_one_common_ar1_path_per_run_seed() -> None:
    a = make_scripted("cost_plus", {}, slot=0, agent_id=10, seed=[7, 0x5C1, 0])
    b = make_scripted("cost_plus", {}, slot=1, agent_id=11, seed=[7, 0x5C1, 1])
    c = make_scripted("cost_plus", {}, slot=0, agent_id=10, seed=[8, 0x5C1, 0])
    assert isinstance(a, CostPlus) and a.common_seed == b.common_seed == 7 != c.common_seed
    path_a = [a.common_shock(t) for t in range(60)]
    path_b = [b.common_shock(t) for t in (5, 59)]  # asynchronous callers agree
    assert path_b == [path_a[5], path_a[59]]
    assert [c.common_shock(t) for t in range(60)] != path_a
    x = np.asarray(path_a)
    assert abs(np.corrcoef(x[:-1], x[1:])[0, 1]) > 0.5  # persistent (rho = 0.9)
    pa = [decide(a, obs(period=t)) for t in range(60, 80)]
    pb = [decide(b, obs(period=t)) for t in range(60, 80)]
    assert np.corrcoef(pa, pb)[0, 1] > 0.8 and pa != pb  # parallel moves, own noise
    assert min(pa + pb) >= 1.0  # never below the true unit cost


def test_cost_plus_state_roundtrip() -> None:
    a = make_scripted("cost_plus", {"idio_sd": 0.05}, slot=0, seed=[3, 1, 0])
    for t in range(10):
        decide(a, obs(period=t))
    b = make_scripted("cost_plus", {"idio_sd": 0.05}, slot=0, seed=[3, 1, 0])
    b.restore(a.state())
    assert [decide(a, obs(period=t)) for t in range(10, 20)] == [
        decide(b, obs(period=t)) for t in range(10, 20)
    ]


# ---------------------------------------------------------------------------- leader_follower
def test_leader_moves_on_schedule_and_followers_match() -> None:
    leader = make_scripted("leader_follower", {"position": "leader"}, slot=0, agent_id=10, seed=1)
    assert isinstance(leader, LeaderFollower) and leader.price == leader.hi
    prices = [decide(leader, obs(period=t)) for t in range(0, 60)]
    changes = {t for t in range(1, 60) if prices[t] != prices[t - 1]}
    assert changes <= {10, 20, 30, 40, 50} and changes
    assert all(leader.lo - 1e-9 <= p <= leader.hi + 1e-9 for p in prices)
    f = make_scripted("leader_follower", {"leader_slot": 0}, slot=2, agent_id=12)
    o = obs()  # runner ids: agent_id = offset + slot, so the leader (slot 0) is agent 10
    o.rival_quotes = [Quote(agent_id=10, ask_ticks=75), Quote(agent_id=13, ask_ticks=90)]
    assert decide(f, o) == pytest.approx(1.75)  # matches the leader (id 10), not the max
    o.rival_quotes = [Quote(agent_id=901, ask_ticks=75), Quote(agent_id=902, ask_ticks=90)]
    assert decide(f, o) == pytest.approx(1.90)  # ids hidden: the highest price leads


# ------------------------------------------------------------------------------- the configs
def test_control_files_use_the_registered_policies() -> None:
    def policies(name: str) -> list[str]:
        return [s.policy for s in load_campaign_config(CONTROLS / f"{name}.yaml").base.llm.slots]

    for name in ["SC-1", "S1c-SC", *(p.stem for p in CONTROLS.glob("MQC-*.yaml"))]:
        assert policies(name) == ["best_responder"] * 4, name
    assert policies("SC-2") == ["cost_plus"] * 4
    assert policies("SC-4") == ["leader_follower"] * 4
    assert policies("SC-5") == ["best_responder"]
    sc4 = load_campaign_config(CONTROLS / "SC-4.yaml").base.llm.slots
    assert [s.params.get("position") for s in sc4] == ["leader"] + ["follower"] * 3


def shrink(path: Path, periods: int = 12) -> CampaignConfig:
    data = load_yaml(path)
    data["seeds"] = [0]
    base = data["base"]
    base.update(periods=periods, ticks_per_period=10)
    base["swarm"].update(n_agents=500, arrivals={"lam_mean": 0.02, "lam_sd_log": 0.5})
    base["clock"]["epoch_ticks"] = 10
    data["branches"] = []
    return CampaignConfig.model_validate(data)


def test_sc1_runs_and_settles_at_the_static_nash_price(tmp_path) -> None:
    (res,) = run_campaign(
        shrink(CONTROLS / "SC-1.yaml"), provider="mock", out=tmp_path, dashboard=False
    )
    assert res.status == "ok"
    q = load_run(res.run_dir).frame("quotes").filter(pl.col("period") >= 8)
    p_n = nash_price(MarketConfig(n_dealers=4))
    assert q["ask"].min() >= p_n - 0.051 and q["ask"].max() <= p_n + 0.051  # within one tick
