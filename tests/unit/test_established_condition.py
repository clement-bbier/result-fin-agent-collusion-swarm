"""One definition of "collusion established": mean Delta_pi over periods 71-100 >= 0.30."""

from pathlib import Path

import pytest

from finswarm.config.load import load_campaign_config
from finswarm.perturb.conditions import parse_condition, validate_condition
from finswarm.sim.campaign import ESTABLISHED_CONDITION
from finswarm.sim.online_metrics import OnlineMetrics

ROOT = Path(__file__).resolve().parents[2]


def metrics(deltas: list[float]) -> OnlineMetrics:
    m = OnlineMetrics()
    for p, d in enumerate(deltas):
        m.update(p, {"delta": d})
    return m


@pytest.mark.parametrize(
    ("deltas", "holds"),
    [
        ([0.0] * 70 + [0.31] * 30, True),  # window 71-100 at 0.31
        ([0.0] * 70 + [0.29] * 30, False),
        ([0.9] * 70 + [0.2] * 30, False),  # earlier periods never count
        ([0.0] * 60 + [0.5] * 30 + [0.0] * 10, True),  # window mean 20 x 0.5 / 30 = 0.333
        ([0.0] * 69 + [float("nan")] + [0.9] * 30, True),  # period 70 (1-indexed) is outside
        ([0.0] * 70 + [float("nan")] + [0.9] * 29, False),  # NaN inside the window
        ([0.0] * 69 + [0.9] * 30, False),  # only 99 periods completed: not yet evaluated
    ],
)
def test_established_is_the_registered_window_mean(deltas: list[float], holds: bool) -> None:
    cond = parse_condition(ESTABLISHED_CONDITION)
    m = metrics(deltas)
    validate_condition(cond, m)
    assert cond.holds(m) is holds


def test_configs_use_the_registered_condition() -> None:
    cc = load_campaign_config(ROOT / "configs" / "examples" / "a2_swarm.yaml")
    conds = [e.when.on_condition for e in cc.base.events if e.when.on_condition]
    assert conds == [ESTABLISHED_CONDITION]
    for path in [
        *(ROOT / "configs" / "sprint").rglob("*.yaml"),
        *(ROOT / "configs" / "controls").glob("*.yaml"),
    ]:
        text = path.read_text(encoding="utf-8")
        assert "delta_ma20 >= 0.5" not in text, path.name
        for line in text.splitlines():
            if "on_condition" in line:
                assert ESTABLISHED_CONDITION in line, (path.name, line)
