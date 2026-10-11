"""A ``ReplayMiss`` (``llm.mode = replay`` with a request missing from the response cache)
pauses the run instead of failing it: the checkpoint is kept and the manifest says ``paused``,
so the run resumes once the cache holds the answers (BACKLOG: known unsupported until now)."""

from __future__ import annotations

from pathlib import Path

import pytest

from finswarm.config.schema import RunConfig
from finswarm.sim.manifest import read_manifest
from finswarm.sim.run import _is_pause, run_config


def _replay_config() -> RunConfig:
    return RunConfig.model_validate(
        {
            "name": "replay_pause",
            "periods": 4,
            "seed": 5,
            "llm": {
                "mode": "replay",
                "lane": "mock",
                "slots": [
                    {"name": "A", "model": "mock-v1", "family": "mock", "policy": "random"},
                    {"name": "B", "model": "mock-v1", "family": "mock", "policy": "random"},
                ],
            },
            "checkpoint": {"every_periods": 2},
        }
    )


def test_replay_miss_is_a_pause() -> None:
    from finswarm.llm.errors import LaneExhausted, ReplayMiss

    assert _is_pause(ReplayMiss("abcdef0123456789"))
    assert _is_pause(LaneExhausted("mock", "budget"))
    assert not _is_pause(RuntimeError("boom"))


def test_replay_miss_pauses_the_run_with_its_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FINSWARM_CACHE_DIR", str(tmp_path / "empty_cache"))
    res = run_config(_replay_config(), out=tmp_path / "runs")
    assert res.status == "paused", res.error
    assert res.error is not None and "ReplayMiss" in res.error
    manifest = read_manifest(res.run_dir)
    assert manifest["status"] == "paused"
    assert manifest.get("finished_at")  # the pause is recorded like any other stop
