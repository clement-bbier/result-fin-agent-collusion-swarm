"""F3: a lever branch and its sham share the ``interventions`` draws of their common events."""

from __future__ import annotations

from typing import Any

from fakes import base_config

from finswarm.config.schema import RunConfig
from finswarm.core.rng import RNGRegistry
from finswarm.perturb.engine import PerturbationEngine
from finswarm.perturb.events import InfoShock, SetParam
from finswarm.sim.run import branch_salt


class _Ctx:
    """SimContext double: ``info_shock`` draws its shock from the ``interventions`` stream."""

    def __init__(self) -> None:
        self.rngs = RNGRegistry(11)
        self.metrics = None
        self.shocks: list[float] = []

    def set_param(self, path: str, value: Any) -> Any:
        self.rngs["interventions"].random(3)  # a lever that consumes draws
        return 0.05

    def market_call(self, hook: str, **kw: Any) -> bool:
        self.shocks.append(float(self.rngs["interventions"].normal()))
        return True

    def log_event(self, *a: Any, **k: Any) -> None:
        pass


def _branch_cfg(name: str, events: list[dict[str, Any]]) -> RunConfig:
    data = base_config(periods=10).model_dump(mode="json")
    data.update(name=name, parent_run_id="trunk-abc", branch_period=5, events=events)
    return RunConfig.model_validate(data)


def test_branch_salt_ignores_name_and_events() -> None:
    shock = {"type": "info_shock", "d_fundamental": 0.1, "source": "branch"}
    lever = {"type": "set_param", "path": "market.price_grid.tick", "value": 0.01}
    a = _branch_cfg("L3-lever", [{**lever, "source": "branch"}, shock])
    b = _branch_cfg("L3-sham", [shock])
    assert branch_salt(a) == branch_salt(b) is not None
    c = _branch_cfg("L3-sham", [shock]).model_copy(update={"branch_period": 6})
    assert branch_salt(c) != branch_salt(b)


def test_lever_and_sham_draw_the_same_info_shock() -> None:
    def run(events: list[Any]) -> tuple[list[float], float]:
        eng = PerturbationEngine(events, branch_period=5, branch_salt=1234)
        ctx = _Ctx()
        eng.boundary(ctx, 5, tick=50)
        return ctx.shocks, float(ctx.rngs["interventions"].random())

    shock = InfoShock(d_fundamental=0.1, diffusion="instant", source="branch")
    lever = SetParam(path="market.price_grid.tick", value=0.01, source="branch")
    shocks_lever, after_lever = run([lever, shock])
    shocks_sham, after_sham = run([shock])
    assert len(shocks_lever) == 1 and shocks_lever == shocks_sham
    assert after_lever == after_sham  # the shared stream is not shifted by the lever either
    shocks_other, _ = run([lever, shock.model_copy(), shock])
    assert shocks_other[0] == shocks_sham[0] and shocks_other[1] != shocks_sham[0]
