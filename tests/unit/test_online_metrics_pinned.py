"""Online benchmarks of fringe markets come from the pinned replicated sweep."""

import json
import warnings
from pathlib import Path

import pytest

from finswarm.config.load import load_campaign_config
from finswarm.metrics.benchmarks import MarketModel
from finswarm.sim.online_metrics import (
    UnpinnedBenchmarksWarning,
    pinned_fringe_benchmarks,
    resolve_benchmarks,
)

ROOT = Path(__file__).resolve().parents[2]
BASE = load_campaign_config(ROOT / "configs/sprint/tier0/TR-H.yaml").base.market
FRINGE = BASE.model_copy(update={"fringe": BASE.fringe.model_copy(update={"share": 0.3})})
PINNED = {"p_nash": 1.111, "pi_nash": 3.0, "p_mono": 1.999, "pi_mono": 5.0}


def pin(tmp_path: Path, method: str = "sweep:market.sweep.profit_fn", provisional=False) -> Path:
    entry = {
        "model": MarketModel.from_config(FRINGE, 4, 0).to_dict(),
        "benchmarks": PINNED,
        "method": method,
        "provisional": provisional,
    }
    (tmp_path / "X.json").write_text(json.dumps({"markets": {"L2": entry}}), encoding="utf-8")
    return tmp_path


def test_fringe_market_returns_the_pinned_sweep(tmp_path: Path) -> None:
    d = pin(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        b = resolve_benchmarks(FRINGE, None, 4, pin_dir=d)
    assert b.to_dict() == PINNED
    assert pinned_fringe_benchmarks(FRINGE, 3, d) is None  # another number of sellers
    other = FRINGE.model_copy(update={"fringe": FRINGE.fringe.model_copy(update={"share": 0.2})})
    assert pinned_fringe_benchmarks(other, 4, d) is None


def test_unpinned_fringe_market_is_flagged_or_refused(tmp_path: Path) -> None:
    d = pin(tmp_path, method="closed_form:flow_diversion", provisional=True)  # not a sweep pin
    closed = resolve_benchmarks(BASE, None, 4)
    with pytest.warns(UnpinnedBenchmarksWarning, match="no pinned sweep"):
        b = resolve_benchmarks(FRINGE, None, 4, pin_dir=d)
    assert b.p_nash == pytest.approx(closed.p_nash)
    with pytest.raises(LookupError, match="make-benchmarks"):
        resolve_benchmarks(FRINGE, None, 4, pin_dir=tmp_path / "missing", strict=True)


def test_no_fringe_keeps_the_closed_form(tmp_path: Path) -> None:
    d = pin(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        b = resolve_benchmarks(BASE, None, 4, pin_dir=d)
    assert b.to_dict() != PINNED and b.p_nash < b.p_mono
