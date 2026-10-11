"""Runner-facing factory: ``make_llm_agent(slot_cfg, *, slot, agent_id, run_cfg, run_id, rngs,
provider, lane, mode) -> RunnerLLMAgent``.

* Model: ``slot_cfg.model`` resolved through ``configs/models.yaml`` (ids or ``aliases``), so
  requests, cache keys and logs carry the verbatim catalogue id.
* Lane: ``mode == "mock"`` -> the ``mock`` lane; else ``lane`` (event override) or
  ``run_cfg.llm.lane`` when that catalogue lane serves the model, otherwise the catalogue lane
  that serves it (``haiku`` -> ``anthropic_haiku``).
* Scheduler: one ``LaneScheduler`` per cache mode per process, shared by every agent of every
  run (lane rate limits and budgets are global), reference counted and closed when the last
  agent using it is closed. ``live``/``replay`` schedulers use the SQLite cache and ledger under
  ``$FINSWARM_CACHE_DIR`` (default ``<repo>/cache``); ``mock`` uses an in-memory cache (cache
  disabled) and answers every request with ``MockLLM(slot policy)``.
* Seeds: ``base_seed = run_cfg.seed`` (branches keep it, so ``replay`` hits the cache).
  Replicates (branch children, ghosts) carry ``tags.replicate_id``; from their branch period
  (``run_cfg.branch_period``, 0 for a ghost of a trunk) every request seed is salted with it,
  so they draw fresh samples instead of replaying the trunk or the sham (``replicate_of``).

The returned agent adds what the runner reads: ``last_call`` (the latest ``drain_logs()``-style
row, without draining), ``last_prompt`` (system/user text of the latest decision),
``set_llm_mode(mode)`` and ``async aclose()``.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

from finswarm.agents.llm.llm_agent import LLMAgent
from finswarm.agents.llm.templates import PromptParts
from finswarm.config.schema import LLMSlotConfig, RunConfig
from finswarm.core.rng import RNGRegistry
from finswarm.core.types import Decision, Observation
from finswarm.llm.pricing import Catalog, load_catalog, repo_root

MODES = ("live", "replay", "mock")
REPLICATE_TAG = "replicate_id"


def replicate_of(run_cfg: RunConfig) -> tuple[str | None, int]:
    """``(replicate_id, first salted period)`` of a run: ``tags.replicate_id`` (set by
    ``perturb.branch.child_config`` and ``sim.ghost.ghost_config``; absent on trunks) and
    ``branch_period`` (0 when the run is not a branch, e.g. a ghost of a trunk)."""
    rep = run_cfg.tags.get(REPLICATE_TAG) or None
    return rep, int(run_cfg.branch_period or 0)


def cache_dir() -> Path:
    env = os.environ.get("FINSWARM_CACHE_DIR")
    return Path(env) if env else repo_root() / "cache"


class SchedulerPool:
    """Process-wide ``LaneScheduler`` per cache mode, reference counted."""

    def __init__(self) -> None:
        self._entries: dict[str, list[Any]] = {}  # mode -> [scheduler, refcount]

    def _build(self, mode: str, cat: Catalog) -> Any:
        from finswarm.llm.scheduler import LaneScheduler, load_budget_policy

        if mode == "mock":
            from finswarm.llm.cache import ResponseCache
            from finswarm.llm.ledger import Ledger

            return LaneScheduler(
                {"mock": cat.lanes["mock"]},
                catalog=cat,
                cache=ResponseCache(":memory:", mode="mock"),
                ledger=Ledger(":memory:"),
                budget=load_budget_policy(),
            )
        d = cache_dir()
        return LaneScheduler.from_catalog(
            cat,
            mode=mode,
            cache_path=d / "llm_cache.sqlite",
            ledger_path=d / "llm_ledger.sqlite",
        )

    def acquire(self, mode: str, cat: Catalog | None = None) -> Any:
        if mode not in MODES:
            raise ValueError(f"unknown llm mode {mode!r}; expected one of {MODES}")
        entry = self._entries.get(mode)
        if entry is None:
            entry = self._entries[mode] = [self._build(mode, cat or load_catalog()), 0]
        entry[1] += 1
        return entry[0]

    def _drop(self, mode: str, sched: Any) -> Any | None:
        entry = self._entries.get(mode)
        if entry is None or entry[0] is not sched:
            return None
        entry[1] -= 1
        if entry[1] > 0:
            return None
        del self._entries[mode]
        return sched

    async def release(self, mode: str, sched: Any) -> None:
        dead = self._drop(mode, sched)
        if dead is not None:
            await dead.aclose()

    def release_nowait(self, mode: str, sched: Any) -> None:
        dead = self._drop(mode, sched)
        if dead is None:
            return
        try:
            asyncio.get_running_loop().create_task(dead.aclose())
        except RuntimeError:  # no running loop: workers (if any) die with their loop
            pass


POOL = SchedulerPool()


def resolve_lane(cat: Catalog, model: str, lane: str | None, mode: str) -> str:
    if mode == "mock":
        return "mock"
    if lane is not None and lane in cat.lanes and model in cat.lanes[lane].models:
        return lane
    found = cat.lane_for_model(model)
    if found is None:
        raise ValueError(
            f"no lane in configs/models.yaml serves model {model!r} (requested lane {lane!r})"
        )
    return found


class RunnerLLMAgent(LLMAgent):
    """``LLMAgent`` plus the runner hooks (``last_call``, ``last_prompt``, mode, close)."""

    def __init__(self, *args: Any, llm_mode: str = "mock", **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.llm_mode = llm_mode
        self.last_prompt: dict[str, Any] | None = None
        self._last_row: dict[str, Any] | None = None

    @property
    def last_call(self) -> dict[str, Any] | None:
        return self._last_row

    def render_parts(self, obs: Observation) -> PromptParts:
        parts = super().render_parts(obs)
        self.last_prompt = {
            "prompt_version": self.template.version,
            "system": parts.system,
            "user": parts.user_text(),
        }
        return parts

    async def decide(self, obs: Observation) -> Decision:
        d = await super().decide(obs)
        self._last_row = dict(self.decision_log[-1]) if self.decision_log else None
        return d

    def set_llm_mode(self, mode: str) -> None:
        """Switch cache mode (``replay`` before a branch point, the parent's mode after)."""
        if mode == self.llm_mode:
            return
        new = POOL.acquire(mode)
        POOL.release_nowait(self.llm_mode, self.scheduler)
        self.scheduler = new
        self.llm_mode = mode
        if mode == "mock":
            self.lane = "mock"

    async def aclose(self) -> None:
        sched, self.scheduler = self.scheduler, None  # type: ignore[assignment]
        if sched is not None:
            await POOL.release(self.llm_mode, sched)

    def close_nowait(self) -> None:
        """Release the scheduler without awaiting (the runner replaced this agent mid-run)."""
        sched, self.scheduler = self.scheduler, None  # type: ignore[assignment]
        if sched is not None:
            POOL.release_nowait(self.llm_mode, sched)


def make_llm_agent(
    slot_cfg: LLMSlotConfig,
    *,
    slot: int,
    agent_id: int,
    run_cfg: RunConfig,
    run_id: str,
    rngs: RNGRegistry,
    provider: str | None,
    lane: str | None,
    mode: str,
) -> RunnerLLMAgent:
    """Build the LLM agent of one slot (``rngs``/``provider`` are accepted for the runner
    contract; request seeds derive from ``run_cfg.seed`` and the provider from the lane)."""
    del rngs, provider
    cat = load_catalog()
    mode = mode or run_cfg.llm.mode
    model = cat.resolve_model(slot_cfg.model)
    lane_name = resolve_lane(cat, model, lane or run_cfg.llm.lane, mode)
    family = slot_cfg.family
    if mode != "mock" and model in cat.models:
        family = cat.models[model].provider  # the lane's provider must match the request
    sc = slot_cfg.model_copy(update={"model": model, "family": family})
    tier = run_cfg.llm.model_copy(update={"lane": lane_name})
    sched = POOL.acquire(mode, cat)
    replicate_id, replicate_from = replicate_of(run_cfg)
    return RunnerLLMAgent.from_slot(  # type: ignore[return-value]
        sc,
        agent_id=agent_id,
        slot=slot,
        scheduler=sched,
        tier=tier,
        run_id=run_id,
        base_seed=run_cfg.seed,
        n_rounds=run_cfg.periods,
        llm_mode=mode,
        replicate_id=replicate_id,
        replicate_from=replicate_from,
    )
