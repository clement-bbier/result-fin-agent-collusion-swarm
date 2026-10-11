"""Branch children record their own arm and label, not the parent's (``arm: trunk``).

Red-team finding: branch manifests took ``arm``/``label`` from the parent's tags because the
branch events (``set_param tags.arm``) fire after the manifest is first written; every lever
branch was recorded as the trunk arm and the paired analysis found no sham.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from finswarm.config.schema import RunConfig
from finswarm.perturb.branch import branch
from finswarm.perturb.events import SetParam
from finswarm.sim.manifest import read_manifest
from finswarm.sim.run import run_config

pytestmark = pytest.mark.smoke
PERIODS, BP = 6, 3


def _config() -> RunConfig:
    return RunConfig.model_validate(
        {
            "name": "trunk_mani",
            "periods": PERIODS,
            "ticks_per_period": 1,
            "seed": 3,
            "tags": {"arm": "trunk", "label": "TR-test"},
            "market": {"demand": {"beta": 1.0}},
            "swarm": {"mode": "closed_form"},
            "llm": {
                "mode": "mock",
                "lane": "mock",
                "slots": [
                    {"name": "A", "model": "mock-v1", "family": "mock", "policy": "undercutter"},
                    {"name": "B", "model": "mock-v1", "family": "mock", "policy": "undercutter"},
                ],
            },
            "clock": {"mode": "sync", "epoch_ticks": 1},
            "checkpoint": {"every_periods": 1, "keep_last": 3},
        }
    )


def test_branch_manifest_uses_branch_tags_and_name(tmp_path: Path) -> None:
    parent = run_config(_config(), out=tmp_path / "p")
    assert parent.status == "ok"
    assert read_manifest(parent.run_dir)["arm"] == "trunk"

    # arm/label set by the branch events
    evs = [SetParam(path="tags.arm", value="L1"), SetParam(path="tags.label", value="L1-test")]
    child = branch(parent.run_dir, BP, evs, "L1", out=tmp_path / "c1")
    m = read_manifest(child)
    assert m["status"] == "ok"
    assert (m["arm"], m["label"]) == ("L1", "L1-test")
    assert m["trunk_id"] == parent.run_id and m["branch_name"] == "L1"

    # no tag events: the branch name is the arm (never the inherited "trunk")
    sham = branch(parent.run_dir, BP, [], "sham", out=tmp_path / "c2")
    assert read_manifest(sham)["arm"] == "sham"
