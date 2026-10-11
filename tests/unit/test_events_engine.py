import asyncio
import json
from typing import Any

import polars as pl
import pytest
from fakes import FakeMarket, base_config, fake_market_factory

from finswarm.core.rng import RNGRegistry
from finswarm.perturb.conditions import ConditionSyntaxError, parse_condition
from finswarm.perturb.engine import PerturbationEngine
from finswarm.perturb.events import ForcedDeviation, InjectNotice, Revert, SetParam, When
from finswarm.sim.checkpoints import find_checkpoint
from finswarm.sim.online_metrics import OnlineMetrics
from finswarm.sim.run import Runner, run_config
from finswarm.store.reader import load_run


def metrics_with(deltas: list[float]) -> OnlineMetrics:
    m = OnlineMetrics()
    for t, d in enumerate(deltas):
        m.update(t, {"delta": d, "price_mean": 1.5 + d})
    return m


def test_condition_grammar() -> None:
    c = parse_condition("delta_ma20 >= 0.5 for 20")
    (term,) = c.terms
    (cl,) = term
    assert (cl.metric, cl.op, cl.threshold, cl.for_n) == ("delta_ma20", ">=", 0.5, 20)
    c2 = parse_condition("delta > 0.3 AND price_mean < 2 or period >= 99")
    assert len(c2.terms) == 2 and len(c2.terms[0]) == 2
    for bad in ("", "delta >> 1", "delta >= x", "delta >= 1 for 0"):
        with pytest.raises(ConditionSyntaxError):
            parse_condition(bad)


def test_moving_average_and_for_window() -> None:
    m = metrics_with([0.0] * 5 + [1.0] * 25)
    assert m.value("delta_ma20") == pytest.approx(1.0)
    assert m.value("delta_ma20", offset=10) == pytest.approx(0.75)
    assert parse_condition("delta_ma20 >= 0.5 for 10").holds(m)
    assert not parse_condition("delta_ma20 >= 0.9 for 10").holds(m)
    assert not parse_condition("delta_ma50 >= 0.0").holds(m)  # not enough history -> NaN
    assert parse_condition("delta < 0 or price_mean > 2.4").holds(m)
    with pytest.raises(KeyError):
        m.value("nonsense")


class Ctx:
    """Minimal SimContext recording calls."""

    def __init__(self, metrics: OnlineMetrics) -> None:
        self.rngs = RNGRegistry(1)
        self.metrics = metrics
        self.params: dict[str, Any] = {"market.price_grid.tick": 0.01}
        self.rows: list[tuple] = []
        self.notices: dict[int, Any] = {}
        self.respawned = 0
        orig = self.rngs.respawn

        def respawn(name: str, *salts: int) -> None:
            self.respawned += 1
            orig(name, *salts)

        self.rngs.respawn = respawn  # type: ignore[method-assign]

    def n_slots(self) -> int:
        return 2

    def set_param(self, path: str, value: Any) -> Any:
        old = self.params.get(path)
        self.params[path] = value
        return old

    def add_notice(self, key, slots, text, until) -> None:
        self.notices[key] = (slots, text, until)

    def remove_notice(self, key) -> None:
        self.notices.pop(key)

    def wrap_forced(self, slot, delta, periods, event_id) -> None:
        self.rows.append(("wrap", slot))

    def unwrap(self, slot) -> None:
        self.rows.append(("unwrap", slot))

    def market_call(self, hook: str, **kw: Any) -> bool:
        return False

    def log_event(self, event_id, event_type, payload, source) -> None:
        self.rows.append((event_id, event_type, source, json.dumps(payload)))


def test_engine_triggers_fire_once_at_boundaries() -> None:
    events = [
        SetParam(path="market.price_grid.tick", value=0.05, when=When(at_period=3)),
        InjectNotice(template="hello {slot}", when=When(at_tick=5), persist_periods=2),
        SetParam(
            path="tags.m", value="x", when=When(on_condition="delta >= 0.5 for 2"), note="named"
        ),
        Revert(event_id=0, when=When(at_period=6)),
    ]
    eng = PerturbationEngine(events)
    m = OnlineMetrics()
    ctx = Ctx(m)
    fired: dict[int, list[str]] = {}
    for b in range(8):
        fired[b] = eng.boundary(ctx, b, tick=b)
        m.update(b, {"delta": 1.0 if b >= 2 else 0.0})
    assert fired[3] == ["event0_set_param"]
    assert fired[4] == ["named"]  # delta >= 0.5 in periods 2 and 3
    assert fired[5] == ["event1_inject_notice"]
    assert fired[6] == ["event3_revert"]
    assert sum(len(v) for v in fired.values()) == 4  # each event fires exactly once
    assert ctx.params["market.price_grid.tick"] == 0.01  # reverted
    sources = {r[1]: r[2] for r in ctx.rows if isinstance(r[0], int)}
    assert sources["set_param"] in ("config", "condition") and sources["inject_notice"] == "config"
    assert ctx.notices[1][2] == 7  # until = boundary 5 + 2 periods
    st = eng.state()
    eng2 = PerturbationEngine(events)
    eng2.restore(st)
    assert eng2.boundary(ctx, 7, tick=7) == []  # nothing re-applied after restore


def test_engine_branch_phase_respawns_once_after_config_events() -> None:
    events = [
        SetParam(path="a", value=1, when=When(at_period=5)),
        ForcedDeviation(slot=1, delta=0.1, source="branch"),  # no trigger -> branch period
    ]
    eng = PerturbationEngine(events, branch_period=5, branch_salt=42)
    ctx = Ctx(OnlineMetrics())
    for b in range(5):
        assert eng.boundary(ctx, b, tick=b) == []
    assert eng.boundary(ctx, 5, tick=5) == ["event0_set_param", "event1_forced_deviation"]
    assert ctx.respawned == 1
    eng.boundary(ctx, 6, tick=6)
    assert ctx.respawned == 1
    # a parent checkpoint restored into a child replays the branch phase
    eng3 = PerturbationEngine(events, branch_period=5, branch_salt=42)
    eng3.restore({**eng.state(), "fired": {0: 5}}, adopt_branch=False)
    assert eng3.boundary(ctx, 5, tick=5) == ["event1_forced_deviation"]


def _run(tmp_path, events, **over):
    cfg = base_config(events=events, **over)
    return run_config(cfg, out=tmp_path, market_factory=fake_market_factory)


def test_runner_applies_set_param_with_regrid_and_logs_event(tmp_path) -> None:
    r = _run(
        tmp_path,
        [
            {
                "type": "set_param",
                "path": "market.price_grid.tick",
                "value": 0.05,
                "when": {"at_period": 10},
            }
        ],
    )
    assert r.status == "ok"
    data = load_run(r.run_dir)
    ev = data.frame("events")
    assert ev.select(["event_id", "period", "event_type", "source"]).rows() == [
        (0, 10, "set_param", "config")
    ]
    assert json.loads(ev["payload"][0])["applied"] == {"old": 0.01, "new": 0.05}
    q = data.frame("quotes").filter(pl.col("period") >= 10)
    off_grid = q.filter(
        ((pl.col("ask") - 1.0) / 0.05 - ((pl.col("ask") - 1.0) / 0.05).round()).abs() > 1e-6
    )
    assert off_grid.height == 0
    assert find_checkpoint(r.run_dir, name="event0_set_param").period == 10


def test_runner_forced_deviation_sting_and_entry(tmp_path) -> None:
    events = [
        {
            "type": "forced_deviation",
            "slot": 0,
            "delta": 0.3,
            "periods": 2,
            "when": {"at_period": 6},
        },
        {"type": "add_scripted_agent", "policy": "static_nash", "when": {"at_period": 12}},
        {
            "type": "sting",
            "slot": 5,
            "delta": 0.2,
            "shadow_epochs": 2,
            "hold_epochs": 2,
            "when": {"at_period": 15},
        },
        {"type": "revert", "event_id": 1, "when": {"at_period": 25}},
    ]
    r = _run(tmp_path, events)
    data = load_run(r.run_dir)
    ap = data.frame("agent_period")
    p5 = ap.filter((pl.col("period") == 5) & (pl.col("slot") == 0))["price"][0]
    forced = ap.filter((pl.col("period").is_in([6, 7])) & (pl.col("slot") == 0))["price"].to_list()
    assert forced == pytest.approx([round(p5 - 0.3, 2)] * 2)
    assert set(ap.filter(pl.col("period") == 20)["slot"].to_list()) == {0, 1, 2, 3}
    assert set(ap.filter(pl.col("period") == 26)["slot"].to_list()) == {0, 1, 3}  # entrant reverted
    ev = data.frame("events")
    types = ev["event_type"].to_list()
    assert types.count("forced_deviation") == 1 and types.count("forced_deviation_end") == 1
    sting = ev.filter(pl.col("event_type") == "sting_phase")
    assert [json.loads(p)["phase"] for p in sting["payload"]] == ["shadow", "raise", "done"]
    assert set(sting["source"].to_list()) == {"derived"}


def test_runner_notice_reaches_llm_observations_only(tmp_path) -> None:
    seen: dict[int, list[str]] = {}

    def spy_view(obs, policy, rng):
        if obs.notices:
            seen.setdefault(obs.slot, []).extend(obs.notices)
        return obs

    cfg = base_config(
        events=[
            {
                "type": "inject_notice",
                "template": "audit at {period}",
                "persist_periods": 2,
                "when": {"at_period": 4},
            }
        ],
    )
    data = cfg.model_dump(mode="json")
    data["llm"]["slots"][1]["policy"] = "mock_llm"
    from finswarm.config.schema import RunConfig

    run_config(
        RunConfig.model_validate(data),
        out=tmp_path,
        market_factory=fake_market_factory,
        view_fn=spy_view,
    )
    assert seen == {1: ["audit at 4", "audit at 5"]}


class _Swarm:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def info_shock(self, d_fundamental, diffusion, rng) -> None:
        self.calls.append((d_fundamental, diffusion, float(rng.random())))


class _SwarmMarket(FakeMarket):
    def __init__(self, cfg, rngs=None) -> None:
        super().__init__(cfg, rngs)
        self.swarm = _Swarm()


def test_info_shock_delegates_to_swarm_with_interventions_rng(tmp_path) -> None:
    events = [
        {"type": "info_shock", "d_fundamental": 0.2, "when": {"at_period": 3}},
        {"type": "remove_liquidity", "share": 0.5, "when": {"at_period": 4}},
    ]
    runner = Runner(
        base_config(events=events, periods=6), out=tmp_path, market_factory=_SwarmMarket
    )
    res = asyncio.run(runner.run())
    # remove_liquidity has no hook on this market: the run fails loudly at period 4
    assert res.status == "failed" and res.periods_completed == 4
    assert "UnsupportedIntervention" in res.error and "remove_liquidity" in res.error
    (call,) = runner.market.swarm.calls
    assert call[:2] == (0.2, "attention")
    ev = load_run(res.run_dir).frame("events")
    supported = [json.loads(p)["applied"]["supported"] for p in ev["payload"]]
    assert supported == [True]
