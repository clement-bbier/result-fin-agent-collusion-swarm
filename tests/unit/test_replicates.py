"""Replicate salting: branches (sham, sham2) and ghosts draw fresh LLM samples after the
branch point, while the replay of the parent before it still hits the response cache.

The runs use the ``live`` cache mode on the free ``mock`` lane (MockLLM ``random`` policy, whose
price is a function of the request seed), with the SQLite cache in a temporary directory.
"""

import polars as pl
import pytest

from finswarm.agents.llm.factory import replicate_of
from finswarm.agents.llm.llm_agent import LLMAgent
from finswarm.config.schema import RunConfig
from finswarm.core.types import Observation
from finswarm.perturb.branch import branch, child_config
from finswarm.sim.ghost import ghost_config, run_ghosts
from finswarm.sim.run import run_config
from finswarm.store.reader import load_run

BRANCH_AT = 6  # trunk checkpoints at 0, 4, 8: branches restore period 4 and replay 4-5
HORIZON = [{"type": "set_param", "path": "periods", "value": 10}]


def live_config() -> RunConfig:
    return RunConfig.model_validate(
        {
            "name": "rep_trunk",
            "periods": 8,
            "seed": 3,
            "llm": {
                "mode": "live",
                "lane": "mock",
                "slots": [
                    {"name": "A", "model": "mock-v1", "family": "mock", "policy": "random"},
                    {"name": "B", "model": "mock-v1", "family": "mock", "policy": "random"},
                ],
            },
            "checkpoint": {"every_periods": 4},
        }
    )


def rows(run_dir) -> pl.DataFrame:
    return load_run(run_dir).frame("llm_decisions").sort(["period", "slot"])


def window(df: pl.DataFrame, lo: int, hi: int | None = None) -> pl.DataFrame:
    out = df.filter(pl.col("period") >= lo)
    return out if hi is None else out.filter(pl.col("period") < hi)


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    d = tmp_path / "cache"
    monkeypatch.setenv("FINSWARM_CACHE_DIR", str(d))
    return d


def _obs(period: int) -> Observation:
    return Observation(
        tick=period, period=period, agent_id=0, slot=0, role="dealer", inventory=0.0,
        cash=0.0, cost=1.0, extra={"max_wtp": 4.51},
    )  # fmt: skip


def _agent(**kw) -> LLMAgent:
    return LLMAgent(
        agent_id=0, slot=0, name="A", model="mock-v1", provider="mock", lane="mock",
        scheduler=None, prompt="fish_v1_json", base_seed=3, **kw,  # type: ignore[arg-type]
    )  # fmt: skip


def test_seed_salt_starts_at_the_branch_period() -> None:
    trunk = _agent()
    sham = _agent(replicate_id="sham", replicate_from=6)
    sham2 = _agent(replicate_id="sham2", replicate_from=6)
    for p in (0, 5):  # replay segment: the parent's seeds
        assert sham._seed(_obs(p), 0) == trunk._seed(_obs(p), 0) == sham2._seed(_obs(p), 0)
        assert "replicate_id" not in sham._tags(_obs(p))
    for p in (6, 9):
        seeds = {a._seed(_obs(p), 0) for a in (trunk, sham, sham2)}
        assert len(seeds) == 3
        assert sham._tags(_obs(p))["replicate_id"] == "sham"
    assert sham._seed(_obs(6), 1) != sham._seed(_obs(6), 0)  # repairs stay distinct


def test_replicate_ids_of_branches_and_ghosts() -> None:
    trunk = live_config()
    assert replicate_of(trunk) == (None, 0)
    sham = child_config(trunk, "rid", BRANCH_AT, [], "sham")
    assert replicate_of(sham) == ("sham", BRANCH_AT)
    nested = child_config(sham, "rid2", 7, [], "hyst")
    assert replicate_of(nested) == ("sham/hyst", 7)
    assert replicate_of(ghost_config(sham, "sid", 1, "replay_parent")) == ("sham/ghost:1", 6)
    assert replicate_of(ghost_config(trunk, "tid", 0, "replay_parent")) == ("ghost:0", 0)


def test_sham2_and_ghosts_draw_fresh_samples_after_the_branch(tmp_path, cache_dir) -> None:
    out = tmp_path / "runs"
    trunk = run_config(live_config(), out=out)
    assert trunk.status == "ok"
    assert not rows(trunk.run_dir)["cache_hit"].any()  # first run: every call is new

    sham_dir = branch(trunk.run_dir, BRANCH_AT, HORIZON, "sham")
    sham2_dir = branch(trunk.run_dir, BRANCH_AT, HORIZON, "sham2")
    sham, sham2 = rows(sham_dir), rows(sham2_dir)
    for df in (sham, sham2):
        # (b) the replay of the trunk up to the branch point is served by the cache
        pre = window(df, 4, BRANCH_AT)
        assert pre.height == 4 and pre["cache_hit"].all()
        # (a) after the branch point every call is a miss, although period 6-7 prompts of the
        # sham reproduce the trunk's and sham2's reproduce the sham's
        post = window(df, BRANCH_AT)
        assert post.height == 8 and not post["cache_hit"].any()
    # (c) different replicate ids diverge; the shared pre-branch segment does not
    assert window(sham, 4, BRANCH_AT)["price"].equals(window(sham2, 4, BRANCH_AT)["price"])
    assert window(sham, BRANCH_AT)["price"].to_list() != window(sham2, BRANCH_AT)["price"].to_list()

    ghosts = run_ghosts(sham_dir)
    assert [g.status for g in ghosts] == ["ok", "ok"]
    for k, g in enumerate(ghosts):
        assert g.manifest["resolved_config"]["tags"]["replicate_id"] == f"sham/ghost:{k}"
        df = rows(g.run_dir)
        pre = window(df, 4, BRANCH_AT)
        assert pre.height == 4 and pre["cache_hit"].all()  # replay of the trunk
        post = window(df, BRANCH_AT)
        assert post["slot"].unique().to_list() == [k]  # rivals replay the sham's quotes
        assert post.height == 4 and not post["cache_hit"].any()  # (a) fresh focal samples
        focal_sham = window(sham, BRANCH_AT).filter(pl.col("slot") == k)["price"].to_list()
        assert post["price"].to_list() != focal_sham  # not a copy of the sham

    # determinism: the same replicate re-run elsewhere replays its own samples from the cache
    again = rows(branch(trunk.run_dir, BRANCH_AT, HORIZON, "sham2", out=tmp_path / "again"))
    assert window(again, 4)["cache_hit"].all()
    assert window(again, 4)["price"].equals(window(sham2, 4)["price"])


def test_ghosts_of_a_trunk_are_salted_from_period_zero(tmp_path, cache_dir) -> None:
    trunk = run_config(live_config(), out=tmp_path / "runs")
    ghosts = run_ghosts(trunk.run_dir)
    for k, g in enumerate(ghosts):
        df = rows(g.run_dir)
        assert df["slot"].unique().to_list() == [k] and df.height == 8
        assert not df["cache_hit"].any()
