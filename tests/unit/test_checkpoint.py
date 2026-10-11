import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from finswarm.config.schema import RunConfig
from finswarm.core.checkpoint import (
    checkpoint_path,
    latest_checkpoint,
    list_checkpoints,
    load_checkpoint,
    save_checkpoint,
)
from finswarm.core.rng import RNGRegistry
from finswarm.core.types import Order, OrderType, Side
from finswarm.market.dealer import DealerMarket
from finswarm.swarm.state import SwarmState


def test_roundtrip_types_and_layout(tmp_path: Path) -> None:
    state: dict[str, Any] = {
        "meta": {"run_id": "r1", "period": 3, "tick": 30, "x": None, "nan": float("nan")},
        "rng": RNGRegistry(1).capture(),
        "market": {
            "tick": 30,
            "asks": np.array([1, -1], dtype=np.int64),
            "nested": {"f": np.arange(6.0).reshape(2, 3), "label": "a", "empty": {}},
            "params": {"tau": None, "u0": 0.0},
            "orders": [{"id": 1, "side": -1}],
        },
        "swarm": {"stype": np.array([0, 2], dtype=np.int8), "active": np.array([True, False])},
        "accounts": np.array([1.5, -1.5]),
        "agents": [{"slot": 0, "plans": "é"}],
        "pending_orders": [],
    }
    path = save_checkpoint(checkpoint_path(tmp_path, 3), state)
    names = sorted(p.name for p in path.iterdir())
    assert names == [
        "_manifest.json",
        "accounts.npz",
        "agents.json",
        "market.npz",
        "meta.json",
        "pending_orders.json",
        "rng.json",
        "swarm.npz",
    ]
    back = load_checkpoint(path)
    assert back["meta"]["run_id"] == "r1" and back["meta"]["x"] is None
    assert np.isnan(back["meta"]["nan"])
    assert back["rng"] == json.loads(json.dumps(state["rng"]))
    m = back["market"]
    assert m["tick"] == 30 and m["params"] == {"tau": None, "u0": 0.0}
    assert m["nested"]["label"] == "a" and m["nested"]["empty"] == {}
    assert m["asks"].dtype == np.int64 and np.array_equal(m["asks"], [1, -1])
    assert np.array_equal(m["nested"]["f"], state["market"]["nested"]["f"])
    assert m["orders"] == [{"id": 1, "side": -1}]
    assert back["swarm"]["stype"].dtype == np.int8 and back["swarm"]["active"].dtype == bool
    assert np.array_equal(back["accounts"], [1.5, -1.5])
    assert back["agents"][0]["plans"] == "é"


def test_atomic_overwrite_prune_and_corruption(tmp_path: Path) -> None:
    for p in range(5):
        save_checkpoint(checkpoint_path(tmp_path, p), {"meta": {"p": p}}, keep_last=2)
    cps = list_checkpoints(tmp_path)
    assert [c.name for c in cps] == ["period_0000003", "period_0000004"]
    assert latest_checkpoint(tmp_path) == cps[-1]
    save_checkpoint(cps[-1], {"meta": {"p": 44}})
    assert load_checkpoint(cps[-1])["meta"]["p"] == 44
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".")]
    with pytest.raises(TypeError):
        save_checkpoint(tmp_path / "bad", {"meta": {"x": object()}})
    assert not (tmp_path / "bad").exists()
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".")]
    (cps[-1] / "meta.json").write_text('{"p": 45}', encoding="utf-8")
    with pytest.raises(ValueError, match="corrupt"):
        load_checkpoint(cps[-1])


def _cfg(mode: str) -> RunConfig:
    return RunConfig(
        name="ck",
        ticks_per_period=10,
        swarm={"mode": mode, "n_agents": 400, "arrivals": {"lam_mean": 6.0 / 400}},
    )


def _quote(m: DealerMarket, slot: int, ticks: int, oid: int, delay: int = 0) -> None:
    ack = m.submit(
        Order(
            oid,
            m.dealer_agent_id(slot),
            Side.SELL,
            OrderType.QUOTE,
            ticks,
            1.0,
            -1,
            m.tick,
            m.tick + delay,
        )
    )
    assert ack.accepted


def _continue(m: DealerMarket) -> dict[str, Any]:
    m.advance(55)
    _quote(m, 1, 66, 100, delay=4)
    m.advance(148)
    out: dict[str, Any] = {f"t_{k}": v for k, v in m.drain_trades_arrays().items()}
    out.update({f"f_{k}": v for k, v in m.drain_flow().items()})
    out["periods"] = np.array([np.r_[p.qty, p.profit, p.ask] for p in m.drain_periods()])
    out["cash"] = m.dealer_cash.copy()
    if m.swarm is not None:
        out.update({f"s_{k}": v for k, v in m.swarm.state().items()})
    return out


@pytest.mark.parametrize("mode", ["agents", "closed_form"])
def test_save_load_continue_is_bit_identical(tmp_path: Path, mode: str) -> None:
    cfg = _cfg(mode)
    rng = RNGRegistry(7)
    m = DealerMarket(cfg, rng)
    _quote(m, 0, 55, 1)
    _quote(m, 1, 60, 2)
    m.advance(137)  # mid-period
    _quote(m, 0, 70, 3, delay=6)  # pending across the checkpoint
    state: dict[str, Any] = {"meta": {"tick": m.tick}, "rng": rng.capture(), "market": m.state()}
    if m.swarm is not None:
        state["swarm"] = m.swarm.state()
    path = save_checkpoint(checkpoint_path(tmp_path, m.tick // 10), state)
    reference = _continue(m)

    loaded = load_checkpoint(path)
    rng2 = RNGRegistry(12345)
    rng2.restore(loaded["rng"])
    swarm2 = SwarmState.from_state(loaded["swarm"]) if "swarm" in loaded else None
    m2 = DealerMarket(cfg, rng2, swarm2)
    m2.restore(loaded["market"])
    resumed = _continue(m2)
    assert reference.keys() == resumed.keys()
    for key, arr in reference.items():
        np.testing.assert_array_equal(arr, resumed[key], err_msg=key)
    assert reference["t_tick"].size > 0
