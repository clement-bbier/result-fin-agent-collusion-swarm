"""The epoch loop: one run of one configuration, resumable from any checkpoint.

Per period ``p`` (``ticks_per_period`` ticks):

1. boundary ``p``: perturbation events (``finswarm.perturb.engine``; ``set_param clock.*``
   reconfigures the decision clock from this boundary on), ghost overrides, named and periodic
   checkpoints (each checkpoint first flushes the Parquet buffers);
2. per decision epoch (a batch popped from ``finswarm.core.clock.EpochClock``: ``sync`` =
   every slot each ``clock.epoch_ticks`` ticks, ``ceil(ticks_per_period / epoch_ticks)``
   epochs per period; ``poisson`` = staggered per-slot revisions, mean interval
   ``epoch_ticks``): ``market.advance`` up to the batch tick, raw observations as of that tick
   -> public view (``finswarm.market.view.apply_view``) -> ``await asyncio.gather(
   agent.decide(obs))`` -> orders with ``apply_tick = decision_tick + speed_bump_max`` (a FIFO
   constant delay, optional order-preserving per-slot jitter ``tags.speed_bump_jitter``) ->
   ``market.submit`` (the market applies them at ``apply_tick``) -> ``market.advance`` to
   the next batch or the period end -> ``market.drain_trades_arrays`` (columnar, written to
   ``trades`` without per-trade Python rows) and ``market.drain_flow`` (``flow_per_tick``);
3. accounting: one ``PeriodRecord`` per agent (``record_period``) from the market's
   ``drain_periods()`` summary when available, ``agent_period``, ``metrics_per_period`` and
   online metrics.

Quotes: ``SlotState.bid/ask`` is the quote a slot last *sent* (its target); ``live_bid/ask``
is the quote *in force* in the market (pending orders are mirrored in ``pending_quotes`` and
take effect at their ``apply_tick``). Rivals observe in-force quotes; period prices are the
in-force quotes at the period's last tick. ``quotes`` has one row per active slot per epoch:
a slot that decided carries its ``decision_id``, the quote it sent, ``tick = apply_tick``
(when the change takes effect), ``decision_tick`` and ``apply_period`` (the period containing
``tick``); ``period`` is the decision period. The others carry their in-force quote at the
epoch tick (``decision_tick = tick``, ``apply_period = period``).

Determinism: all randomness comes from ``RNGRegistry`` streams (``matching`` for the clock's
revision gaps and speed-bump jitter, ``interventions`` for event-created agents, ``view`` for
pseudonyms) and from agents' own seeded generators; every piece of state is in the checkpoint
(``market.state()``, ``market.swarm.state()`` and ``rng.capture()`` taken at the same moment,
agent states, the public view, the clock schedule and the pending quote changes), so a run
killed at any point and resumed writes byte-identical ``trades``/``quotes`` parts. Restore
order: fresh ``RNGRegistry(seed)``, ``SwarmState.from_state``, ``make_market(cfg, rng, swarm)``,
``market.restore``, agents, clock, view, then ``rng.restore``.

Failures: an :class:`~finswarm.sim.integration.UnsupportedIntervention` (a path the market
cannot set live, two-sided quotes, a missing event hook, an unimplemented mechanism, the
fringe outside ``agents`` mode) marks the run ``failed`` with the
reason in ``manifest.error``; it never raises out of ``run()``.

Accounting conventions: one-sided dealers/sellers produce to order at unit cost
``market.cost * market.alpha`` (Calvano), so ``cash`` accumulates profit; buyers accumulate
surplus ``value - price``; two-sided dealers carry inventory and are marked to the period's
VWAP. ``undercut`` = own price strictly below the lowest rival price of the previous period.
"""

from __future__ import annotations

import asyncio
import bisect
import dataclasses
import hashlib
import json
import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from finswarm.agents.base import AgentMemory
from finswarm.agents.scripted.competitive import demand_dict
from finswarm.agents.scripted.forced import ForcedDeviationWrapper
from finswarm.agents.scripted.registry import (
    canonical_policy,
    make_scripted,
    parse_policy,
)
from finswarm.config.load import canonical_json, config_hash, load_yaml
from finswarm.config.load import run_id as make_run_id
from finswarm.config.schema import CampaignConfig, LLMSlotConfig, MarketConfig, RunConfig
from finswarm.core.clock import Epoch, EpochClock, jitter_of
from finswarm.core.rng import RNGRegistry
from finswarm.core.types import (
    AgentKind,
    Decision,
    Observation,
    Order,
    OrderAck,
    OrderType,
    PeriodRecord,
    Quote,
    Side,
    Trade,
)
from finswarm.market.base import price_from_ticks, ticks_from_price
from finswarm.market.view import RAW_BOOK_KEY, RAW_QUOTES_KEY, PublicView
from finswarm.perturb.engine import PerturbationEngine, events_hash
from finswarm.perturb.events import SetParam
from finswarm.sim import checkpoints as ck
from finswarm.sim.integration import (
    JITTER_TAG,
    UnsupportedIntervention,
    after_market_param,
    apply_clock_param,
    build_market,
    check_clock_param,
    market_hook,
    resolve_llm_factory,
    resolve_market_factory,
    resolve_view_fn,
    settable_paths,
    slot_is_scripted,
    speed_bump_jitter,
    swarm_from_state,
    swarm_state_of,
)
from finswarm.sim.manifest import (
    DONE_STATUSES,
    empty_llm_totals,
    environment,
    git_info,
    package_versions,
    read_manifest,
    seeds_info,
    utc_now,
    write_manifest,
)
from finswarm.sim.online_metrics import Benchmarks, OnlineMetrics, resolve_benchmarks
from finswarm.store.writer import (
    EpochBuffers,
    PartWriter,
    atomic_write_text,
    copy_parts_rewriting_run_id,
)

log = logging.getLogger(__name__)

NAN = float("nan")
IDENTITY_FIELDS = ("name", "seed", "periods", "events", "parent_run_id", "branch_period", "ghost")
#: identity tags: the run's own value wins over the restored checkpoint's (a branch or a ghost
#: restores its parent's live config, whose other tags stay settable live)
IDENTITY_TAGS = ("replicate_id",)
FREE_DICTS = ("tags", "swarm.type_weights")
SLOT_SEED_SALT = 0x5C1
EVENT_SEED_SALT = 0x5C2
DEFAULT_QUOTE_SIZE = 1.0e6
DEFAULT_ORDER_SIZE = 1.0
SELL_ROLES = ("dealer", "seller", "trader")
MAX_SLOTS = 127  # quotes.slot is Int8
CANCEL_SIDE = 0  # pending_quotes entry of a delayed cancel (Side.BUY = 1, Side.SELL = -1)
DEFAULT_MAX_WTP_OVER_ALPHA = 4.51  # Fish et al. (2024) prompt: maximum willingness to pay
#: live changes no market supports yet: they fail the run, they are never ignored
UNSUPPORTED_LIVE = {
    "market.price_grid.tick": "tick-size change mid-run is not supported yet",
    "market.price_grid.min": "price-grid change mid-run is not supported yet",
    "market.price_grid.max": "price-grid change mid-run is not supported yet",
    "market.two_sided": "two-sided quotes are not supported yet",
    "market.mechanism": "mechanism switch mid-run is not supported yet",
    "market.fringe.share": "the public limit-order fringe needs swarm.mode='agents'",
}


class SimulatedCrash(RuntimeError):
    """Raised by the ``halt_after_period`` test hook: stop without flushing (like a kill)."""


class PauseRequested(RuntimeError):
    """Raise from an agent/LLM layer to pause the run (resume restarts at the last checkpoint)."""


def _is_pause(exc: BaseException) -> bool:
    """Errors that pause the run (checkpoint kept, ``status = paused``) instead of failing it:
    an explicit pause, an exhausted lane or budget, and a ``ReplayMiss`` (``llm.mode = replay``
    with a request missing from the response cache: the run resumes once the cache has it)."""
    name = type(exc).__name__.lower()
    return (
        isinstance(exc, PauseRequested)
        or "exhausted" in name
        or "budget" in name
        or name == "replaymiss"
    )


@dataclass
class RunResult:
    run_id: str
    run_dir: Path
    status: str
    periods_completed: int
    manifest: dict[str, Any]
    skipped: bool = False
    error: str | None = None


@dataclass
class SlotState:
    slot: int
    agent_id: int
    spec: dict[str, Any]
    agent: Any = None
    active: bool = True
    bid: float = NAN
    ask: float = NAN
    bid_size: float = 0.0
    ask_size: float = 0.0
    inventory: float = 0.0
    cash: float = 0.0
    quote_orders: dict[int, int] = field(default_factory=dict)  # side -> order id
    open_orders: list[int] = field(default_factory=list)
    history: deque[PeriodRecord] = field(default_factory=lambda: deque(maxlen=1000))
    prev_rivals: tuple[float, ...] = ()
    n_decisions: int = 0
    last_decision_id: int = -1
    cash0: float = 0.0
    inv0: float = 0.0
    p_qty: float = 0.0
    # the quote in force in the market (``bid``/``ask`` above is the quote last sent, which
    # takes effect at its ``apply_tick``; they differ only while a speed bump delays it)
    live_bid: float = NAN
    live_ask: float = NAN
    live_bid_size: float = 0.0
    live_ask_size: float = 0.0
    live_orders: dict[int, int] = field(default_factory=dict)  # side -> order id in force
    # absolute index (``Runner.trades_base`` + position in ``Runner.last_trades``) of the first
    # trade not yet shown to this slot: its order flow since its own last decision. ``None``
    # (never decided, or deactivated) falls back to ``Runner.batch_cursor``.
    trade_cursor: int | None = None

    _RT = (
        "active",
        "bid",
        "ask",
        "bid_size",
        "ask_size",
        "inventory",
        "cash",
        "prev_rivals",
        "n_decisions",
        "last_decision_id",
        "cash0",
        "inv0",
        "p_qty",
        "trade_cursor",
    )

    def runtime_state(self) -> dict[str, Any]:
        st = {k: getattr(self, k) for k in self._RT}
        st["quote_orders"] = dict(self.quote_orders)
        st["open_orders"] = list(self.open_orders)
        st["history"] = [dataclasses.asdict(r) for r in self.history]
        st["live"] = [self.live_bid, self.live_ask, self.live_bid_size, self.live_ask_size]
        st["live_orders"] = dict(self.live_orders)
        return st

    def load_runtime(self, st: dict[str, Any]) -> None:
        for k in self._RT:
            setattr(self, k, st.get(k) if k == "trade_cursor" else st[k])
        self.prev_rivals = tuple(st["prev_rivals"])
        self.quote_orders = {int(k): int(v) for k, v in st["quote_orders"].items()}
        self.open_orders = list(st["open_orders"])
        # checkpoints written before the clock refactor: no delayed quotes, live == sent
        live = st.get("live") or [self.bid, self.ask, self.bid_size, self.ask_size]
        self.live_bid, self.live_ask = float(live[0]), float(live[1])
        self.live_bid_size, self.live_ask_size = float(live[2]), float(live[3])
        lo = st.get("live_orders", st["quote_orders"])
        self.live_orders = {int(k): int(v) for k, v in lo.items()}
        self.history = deque(
            (
                PeriodRecord(**{**r, "rival_prices": tuple(r["rival_prices"])})
                for r in st["history"]
            ),
            maxlen=1000,
        )

    @property
    def base_agent(self) -> Any:
        a = self.agent
        return a.inner if isinstance(a, ForcedDeviationWrapper) else a


class _SafeDict(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _finite(x: float | None) -> bool:
    return x is not None and math.isfinite(x)


def _trades_from_columns(cols: dict[str, Any]) -> list[Trade]:
    """``Trade`` objects for the next observations (``trades_since_epoch``)."""
    k = int(np.asarray(cols["tick"]).shape[0])
    if k == 0:
        return []
    mech = cols["mechanism"]
    mechs = [mech] * k if isinstance(mech, str) else list(mech)
    lists = {
        name: np.asarray(cols[name]).tolist()
        for name in (
            "trade_id",
            "tick",
            "price_ticks",
            "size",
            "buyer_id",
            "seller_id",
            "aggressor",
            "batch_id",
            "buyer_value",
            "seller_cost",
        )
    }
    return [
        Trade(
            trade_id=int(lists["trade_id"][i]),
            tick=int(lists["tick"][i]),
            price_ticks=int(lists["price_ticks"][i]),
            size=float(lists["size"][i]),
            buyer_id=int(lists["buyer_id"][i]),
            seller_id=int(lists["seller_id"][i]),
            aggressor=int(lists["aggressor"][i]),
            mechanism=str(mechs[i]),
            batch_id=int(lists["batch_id"][i]),
            buyer_value=float(lists["buyer_value"][i]),
            seller_cost=float(lists["seller_cost"][i]),
        )
        for i in range(k)
    ]


def _get_path(data: Any, parts: list[str]) -> Any:
    for k in parts:
        data = data[int(k)] if isinstance(data, list) else data[k]
    return data


def apply_provider(cfg: RunConfig, provider: str | None) -> RunConfig:
    """``--provider mock`` forces mock mode (stand-ins or MockLLM); otherwise sets the lane."""
    if provider is None:
        return cfg
    data = cfg.model_dump(mode="json")
    if provider == "mock":
        data["llm"]["mode"] = "mock"
        data["llm"]["lane"] = "mock"
    else:
        data["llm"]["lane"] = provider
    return RunConfig.model_validate(data)


def branch_events_of(cfg: RunConfig) -> list[Any]:
    return [e for e in cfg.events if e.source == "branch"]


def branch_salt(cfg: RunConfig) -> int | None:
    """Salt of the re-spawned ``interventions`` stream: ``(parent_run_id, branch_period)``
    only, so a lever branch and its sham (different names and events) share it; each branch
    event then draws from its own child stream (``PerturbationEngine.branch_stream_key``)."""
    if cfg.branch_period is None:
        return None
    payload = canonical_json({"parent": cfg.parent_run_id, "period": cfg.branch_period})
    return int(hashlib.sha256(payload.encode()).hexdigest()[:15], 16)


def _per_run_cap() -> float | None:
    """``per_run_max_usd`` from ``configs/budget.yaml`` (None when absent)."""
    try:
        from finswarm.llm.scheduler import load_budget_policy

        return load_budget_policy().per_run_max_usd
    except (ImportError, OSError, ValueError):  # pragma: no cover - unreadable budget file
        return None


def _retire(agent: Any) -> None:
    """Release what a replaced agent holds (an LLM agent's shared scheduler reference)."""
    fn = getattr(agent, "close_nowait", None)
    if fn is not None:
        fn()


@dataclass(frozen=True)
class BranchOrigin:
    """Where a branch child starts: the parent run directory (checkpoint <= branch period)."""

    parent_dir: Path


class Runner:
    def __init__(
        self,
        cfg: RunConfig,
        *,
        out: str | Path | None = None,
        run_dir: str | Path | None = None,
        provider: str | None = None,
        market_factory: Any = None,
        llm_factory: Any = None,
        view_fn: Any = None,
        benchmarks: Benchmarks | None = None,
        progress: Any = None,
        halt_after_period: int | None = None,
        branch_origin: BranchOrigin | None = None,
        agent_overrides: dict[int, dict[str, Any]] | None = None,
        overrides_at: int = 0,
        replay_source: str | Path | None = None,
        extra_manifest: dict[str, Any] | None = None,
        mirror_dir: str | Path | None = None,
    ) -> None:
        self.cfg = apply_provider(cfg, provider)
        self.provider = provider
        self.run_id = make_run_id(self.cfg)
        if run_dir is None and out is None:
            raise ValueError("give out= or run_dir=")
        self.run_dir = Path(run_dir) if run_dir is not None else Path(out) / self.run_id
        self.market_factory, self.market_impl = resolve_market_factory(market_factory)
        self.view_fn, self.view_impl = resolve_view_fn(view_fn)
        self.llm_factory, self.llm_impl = resolve_llm_factory(llm_factory)
        self.bench_override = benchmarks
        self.progress = progress
        self.halt_after_period = halt_after_period
        self.branch_origin = branch_origin
        self.agent_overrides = {int(k): dict(v) for k, v in (agent_overrides or {}).items()}
        self.overrides_at = int(overrides_at)
        self.replay_source = str(replay_source) if replay_source is not None else None
        self.extra_manifest = dict(extra_manifest or {})
        self.mirror_dir = Path(mirror_dir) if mirror_dir is not None else None
        self.n_swarm = self.cfg.swarm.n_agents if self.cfg.swarm.mode == "agents" else 0
        self.salt = branch_salt(self.cfg)

        self.live: RunConfig = self.cfg
        self.rngs = RNGRegistry(self.cfg.seed)
        self.metrics = OnlineMetrics()
        self.engine = PerturbationEngine(
            self.cfg.events, branch_period=self.cfg.branch_period, branch_salt=self.salt
        )
        self.buffers = EpochBuffers()
        self.writer = PartWriter(self.run_dir, store_prompts=self.cfg.logging.store_prompts)
        self.slots: list[SlotState] = []
        self._by_id: dict[int, SlotState] = {}
        self.market: Any = None
        self.benchmarks: Benchmarks | None = None
        # benchmarks frozen at run start (trunk / pre-lever) for the online delta_p
        self.online_benchmarks: Benchmarks | None = None
        self.period = 0
        self.epoch = 0
        self.next_order_id = 0
        self.next_decision_id = 0
        self.notices: dict[int, dict[str, Any]] = {}
        self.llm_totals = empty_llm_totals()
        # trades not yet shown to every slot (per-slot cursors, ``SlotState.trade_cursor``);
        # ``trades_base`` is the absolute index of ``last_trades[0]``, ``batch_cursor`` the
        # absolute index at the last decision batch (the default cursor)
        self.last_trades: list[Trade] = []
        self.trades_base = 0
        self.batch_cursor = 0
        self.prev_vwap = NAN
        self.overrides_applied = False
        self.restored_at: int | None = None
        self.manifest: dict[str, Any] = {}
        self._t0 = 0.0
        self._wall_prev = 0.0
        self.public_view: PublicView | None = None
        self.removed_liquidity: list[list[int]] = []  # swarm ids per remove_liquidity event
        # time-stamped public inputs of the view (needed for view.delay_ticks > 0)
        self.quote_log: list[tuple[int, int, int, int, float, float]] = []
        self.book_log: list[tuple[int, list[tuple[float, float, float]]]] = []
        self.per_run_max_usd = _per_run_cap()
        self.clock: EpochClock | None = None  # decision schedule (finswarm.core.clock)
        # quote changes sent but not in force yet (speed bump / staleness window), applied to
        # ``SlotState.live_*`` in ``(apply_tick, order_id)`` order, as the market applies them:
        # (apply_tick, order_id, slot, side: Side value or CANCEL_SIDE, price, size)
        self.pending_quotes: list[tuple[int, int, int, int, float, float]] = []

    # =========================================================================================
    # lifecycle
    # =========================================================================================
    async def run(self) -> RunResult:
        self._t0 = time.perf_counter()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        prev = read_manifest(self.run_dir)
        if prev.get("status") in DONE_STATUSES:
            return RunResult(
                self.run_id,
                self.run_dir,
                str(prev["status"]),
                int(prev.get("periods_completed", 0)),
                prev,
                skipped=True,
            )
        ckpts = ck.list_checkpoints(self.run_dir)
        resumed_from: int | None = None
        try:
            if ckpts:
                self._restore(ck.load_payload(ckpts[-1]), adopt_branch=True)
                self.restored_at = resumed_from = ckpts[-1].period
                self._wall_prev = float(prev.get("wall_time_s") or 0.0)
            elif self.branch_origin is not None:
                self._init_branch()
            else:
                self._init_fresh()
        except UnsupportedIntervention as exc:
            self.manifest = self._new_manifest(prev, resumed_from)
            error = f"UnsupportedIntervention: {exc}"
            await self._close_agents()
            self._finish("failed", error)
            return RunResult(
                self.run_id, self.run_dir, "failed", self.period, self.manifest, error=error
            )
        self.manifest = self._new_manifest(prev, resumed_from)
        write_manifest(self.run_dir, self.manifest)

        status, error = "ok", None
        try:
            while self.period < self.cfg.periods:
                await self._boundary(self.period)
                await self._run_period(self.period)
                self.period += 1
                self._check_run_budget()
                if self.halt_after_period is not None and self.period > self.halt_after_period:
                    raise SimulatedCrash(f"halted after period {self.period - 1}")
            await self._boundary(self.period, final=True)
        except SimulatedCrash as exc:
            status, error = "failed", str(exc)
        except UnsupportedIntervention as exc:
            status, error = "failed", f"UnsupportedIntervention: {exc}"
        except Exception as exc:
            if _is_pause(exc):
                status, error = "paused", repr(exc)
            else:
                self._finish("failed", repr(exc))
                await self._close_agents()
                raise
        await self._close_agents()
        self._finish(status, error)
        return RunResult(self.run_id, self.run_dir, status, self.period, self.manifest, error=error)

    def _arm_label(self) -> tuple[str, str]:
        """``(arm, label)`` of this run for the analysis layer.

        A branch child inherits the parent's ``tags`` (``arm: trunk``) and only changes them
        through its branch events (``set_param tags.arm``), which fire after the manifest is
        first written; so the planned branch-event tags are applied here, then any tag changed
        by an event that already fired (``self.live``), and the branch name is the fallback.
        """
        tags: dict[str, Any] = dict(self.cfg.tags)
        if self.cfg.branch_period is not None:  # a child never keeps the parent's arm/label
            tags.pop("arm", None)
            tags.pop("label", None)
        for e in branch_events_of(self.cfg):
            path = str(getattr(e, "path", "") or "")
            if getattr(e, "type", None) == "set_param" and path.startswith("tags."):
                tags[path[len("tags.") :]] = getattr(e, "value", None)
        live = getattr(self, "live", None)
        if live is not None:
            for k in ("arm", "label"):
                if k in live.tags and live.tags.get(k) != self.cfg.tags.get(k):
                    tags[k] = live.tags[k]
        default_arm = self.extra_manifest.get("branch_name") or self.cfg.name
        arm = tags.get("arm", default_arm)
        return str(default_arm if arm is None else arm), str(tags.get("label", "") or "")

    def _finish(self, status: str, error: str | None) -> None:
        self.manifest.update(self._progress_fields())
        self.manifest["arm"], self.manifest["label"] = self._arm_label()
        self.manifest["status"] = status
        self.manifest["error"] = error
        self.manifest["finished_at"] = utc_now() if status in ("ok", "failed", "paused") else None
        write_manifest(self.run_dir, self.manifest)
        if self.mirror_dir is not None:  # tracked copy: <results>/manifests/<run_id>.json
            self.mirror_dir.mkdir(parents=True, exist_ok=True)
            atomic_write_text(
                self.mirror_dir / f"{self.run_id}.json",
                json.dumps(self.manifest, sort_keys=True, indent=1, default=str),
            )

    def _check_run_budget(self) -> None:
        cap = self.per_run_max_usd
        if cap is not None and self.llm_totals["cost_usd"] > cap:
            raise PauseRequested(
                f"per-run budget exceeded: {self.llm_totals['cost_usd']:.4f} USD > "
                f"per_run_max_usd {cap:.2f} (configs/budget.yaml)"
            )

    async def _close_agents(self) -> None:
        for s in self.slots:
            fn = getattr(s.base_agent, "aclose", None)
            if fn is not None:
                try:
                    await fn()
                except Exception:  # pragma: no cover - best effort
                    log.exception("agent close failed")

    # =========================================================================================
    # initialisation, checkpoints, restore
    # =========================================================================================
    def _init_fresh(self) -> None:
        self.rngs = RNGRegistry(self.cfg.seed)
        self.live = self.cfg
        self.writer.restore({"next_index": {}, "prompts_bytes": 0})  # drop stale parts
        self.market = self._new_market(self.live)
        self._check_event_support()
        self._refresh_benchmarks(n_firms=self._n_seller_cfg_slots())
        self.online_benchmarks = self.benchmarks
        self.slots = []
        for i, sc in enumerate(self.live.llm.slots):
            spec = self._initial_spec(i, sc)
            s = SlotState(slot=i, agent_id=self.n_swarm + i, spec=spec)
            s.agent = self._build_agent(spec, i, s.agent_id)
            self.slots.append(s)
        self._reindex()
        self._refresh_benchmarks()
        self.removed_liquidity, self.quote_log, self.book_log = [], [], []
        self._new_clock(None)
        self._new_public_view(None)
        self.engine.validate(self.metrics)

    def _new_market(self, cfg: RunConfig, swarm: Any = None) -> Any:
        return build_market(self.market_factory, cfg, self.rngs, swarm)

    def _new_clock(self, state: dict[str, Any] | None) -> None:
        """The decision clock: every slot decides at the current tick, or the restored schedule.

        Building never draws. A checkpoint without clock state (written before the clock
        refactor) restarts the schedule at the checkpoint tick (exact in ``sync`` mode).
        """
        self.clock = EpochClock(
            self.live.clock, len(self.slots), self.rngs, start_tick=int(self.market.tick)
        )
        if state is not None:
            self.clock.restore(state)
            if self.clock.cfg != self.live.clock:
                raise ValueError("checkpoint clock state disagrees with the live clock config")
        else:  # the ClockConfig field wins; the legacy tag is read when the field is 0
            jitter = jitter_of(self.live.clock) or speed_bump_jitter(self.live.tags.get(JITTER_TAG))
            if jitter:
                self.clock.set_jitter(jitter)

    def _new_public_view(self, state: dict[str, Any] | None) -> None:
        """The run's stateful public view (pseudonyms); only with the real ``apply_view``."""
        if self.view_impl != "market.view":
            self.public_view = None
            return
        self.public_view = PublicView(
            self.live.view,
            self.rngs,
            p_min=float(self.market.p_min),
            tick_size=float(self.market.tick_size),
            tagged_ids=[s.agent_id for s in self.slots],
        )
        if state is not None:
            self.public_view.restore(state)

    def _check_event_support(self) -> None:
        """Fail fast on ``set_param`` events the market or the clock cannot apply live (before
        period 0)."""
        paths = settable_paths(self.market)
        for i, ev in enumerate(self.cfg.events):
            if not isinstance(ev, SetParam):
                continue
            top = ev.path.split(".")[0]
            if top == "clock" or ev.path == f"tags.{JITTER_TAG}":
                try:
                    if top == "clock":
                        check_clock_param(ev.path, ev.value, self.live.clock)
                    else:
                        speed_bump_jitter(ev.value)
                except UnsupportedIntervention as exc:
                    raise UnsupportedIntervention(f"event {i}: {exc}") from exc
                continue
            if paths is None or top not in ("market", "swarm"):
                continue  # test doubles: changes go through set_tick / migrate_from
            if ev.path not in paths:
                why = UNSUPPORTED_LIVE.get(ev.path, "not settable live")
                raise UnsupportedIntervention(
                    f"event {i} set_param {ev.path!r}: {why} in the "
                    f"{getattr(self.market, 'kind', '?')} market (settable live: {list(paths)})"
                )

    def _n_seller_cfg_slots(self) -> int | None:
        n = sum(1 for sc in self.live.llm.slots if sc.role in SELL_ROLES)
        return n or None

    def _initial_spec(self, i: int, sc: LLMSlotConfig) -> dict[str, Any]:
        seed = [self.cfg.seed, SLOT_SEED_SALT, i]
        if slot_is_scripted(sc):
            return {
                "type": "scripted",
                "policy": sc.policy,
                "params": dict(sc.params),
                "seed": seed,
                "name": sc.name,
                "role": sc.role,
                "memory_periods": sc.memory_periods,
            }
        return {"type": "llm", "slot_cfg": sc.model_dump(mode="json"), "lane": None, "seed": seed}

    def _init_branch(self) -> None:
        assert self.branch_origin is not None and self.cfg.branch_period is not None
        parent = self.branch_origin.parent_dir
        src = ck.find_checkpoint(parent, at_or_before=self.cfg.branch_period)
        if src is None:
            raise FileNotFoundError(f"no checkpoint <= {self.cfg.branch_period} in {parent}")
        payload = ck.load_payload(src)
        copy_parts_rewriting_run_id(
            parent, self.run_dir, payload["writer"]["next_index"], self.run_id
        )
        nbytes = int(payload["writer"].get("prompts_bytes", 0))
        src_prompts = parent / "prompts.jsonl"
        if nbytes and src_prompts.exists():
            with open(src_prompts, "rb") as fh:
                (self.run_dir / "prompts.jsonl").write_bytes(fh.read(nbytes))
        self._restore(payload, adopt_branch=False)
        self.restored_at = src.period
        self._checkpoint(src.period, ["branch_origin"], update_manifest=False)

    def _restore(self, payload: dict[str, Any], *, adopt_branch: bool) -> None:
        live = dict(payload["live"])
        own = self.cfg.model_dump(mode="json")
        for k in IDENTITY_FIELDS:
            live[k] = own[k]
        tags = {k: v for k, v in (live.get("tags") or {}).items() if k not in IDENTITY_TAGS}
        tags.update({k: own["tags"][k] for k in IDENTITY_TAGS if k in own["tags"]})
        live["tags"] = tags
        self.live = RunConfig.model_validate(live)
        # order matters: fresh streams, swarm from its own state, market, market state, ...,
        # and the streams captured with them last (building never consumes a stream then)
        self.rngs = RNGRegistry(self.cfg.seed)
        self.market = self._new_market(self.live, swarm_from_state(payload.get("swarm")))
        self.market.restore(payload["market"])
        self._check_event_support()
        self.removed_liquidity = [list(x) for x in payload.get("removed_liquidity", [])]
        self.quote_log = [tuple(x) for x in payload.get("quote_log", [])]  # type: ignore[misc]
        self.book_log = [(int(t), list(lv)) for t, lv in payload.get("book_log", [])]
        self.period = int(payload["period"])
        self.epoch = int(payload["epoch"])
        self.next_order_id = int(payload["next_order_id"])
        self.next_decision_id = int(payload["next_decision_id"])
        self.metrics.restore(payload["metrics"])
        self.notices = {int(k): dict(v) for k, v in payload["notices"].items()}
        self.llm_totals = {**empty_llm_totals(), **payload["llm_totals"]}
        self.last_trades = [Trade(**t) for t in payload["last_trades"]]
        self.trades_base = int(payload.get("trades_base", 0))
        self.batch_cursor = int(payload.get("batch_cursor", self.trades_base))
        self.prev_vwap = float(payload["prev_vwap"])
        ob = payload.get("online_benchmarks", "missing")
        self.overrides_applied = bool(payload["overrides_applied"])
        self.engine.restore(payload["engine"], adopt_branch=adopt_branch)
        self.slots = []
        for sd in payload["slots"]:
            s = SlotState(slot=int(sd["slot"]), agent_id=int(sd["agent_id"]), spec=sd["spec"])
            s.load_runtime(sd["rt"])
            s.agent = self._agent_from_snapshot(sd, s.slot, s.agent_id)
            self.slots.append(s)
        self._reindex()
        self._new_clock(payload.get("clock"))
        self.pending_quotes = [
            (int(a), int(o), int(sl), int(sd), float(px), float(sz))
            for a, o, sl, sd, px, sz in payload.get("pending_quotes", [])
        ]
        self._new_public_view(payload.get("public_view"))
        self.rngs.restore(payload["rng"])
        self._refresh_benchmarks()
        self.writer.restore(payload["writer"])
        self.online_benchmarks = (
            self.benchmarks if ob == "missing" else (Benchmarks(**ob) if ob else None)
        )

    def _snapshot_slot(self, s: SlotState) -> dict[str, Any]:
        agent = s.agent
        forced = None
        if isinstance(agent, ForcedDeviationWrapper):
            forced = {
                "delta": agent.delta,
                "periods": agent.periods,
                "event_id": agent.event_id,
                "left": agent.left,
                "forced_price": agent.forced_price,
            }
            agent = agent.inner
        return {
            "slot": s.slot,
            "agent_id": s.agent_id,
            "spec": s.spec,
            "agent_state": agent.state(),
            "name": agent.name,
            "forced": forced,
            "active": s.active,
            "rt": s.runtime_state(),
        }

    def _agent_from_snapshot(self, sd: dict[str, Any], slot: int, agent_id: int) -> Any:
        agent = self._build_agent(sd["spec"], slot, agent_id)
        agent.restore(sd["agent_state"])
        try:
            agent.name = sd["name"]
        except AttributeError:  # pragma: no cover - read-only names
            pass
        f = sd.get("forced")
        if f:
            w = ForcedDeviationWrapper(
                agent, delta=f["delta"], periods=f["periods"], event_id=f["event_id"]
            )
            w.left, w.forced_price = int(f["left"]), float(f["forced_price"])
            return w
        return agent

    def _capture(self, period: int) -> dict[str, Any]:
        return {
            "version": 1,
            "run_id": self.run_id,
            "period": period,
            "epoch": self.epoch,
            "next_order_id": self.next_order_id,
            "next_decision_id": self.next_decision_id,
            # captured together, nothing draws in between (checkpoint composition contract)
            "rng": self.rngs.capture(),
            "market": self.market.state(),
            "swarm": swarm_state_of(self.market),
            "public_view": self.public_view.state() if self.public_view is not None else None,
            # decision schedule (next decision tick per slot) and quote changes not in force
            "clock": self.clock.state() if self.clock is not None else None,
            "pending_quotes": [list(x) for x in self.pending_quotes],
            "removed_liquidity": [list(x) for x in self.removed_liquidity],
            "quote_log": [list(x) for x in self.quote_log],
            "book_log": [[t, list(lv)] for t, lv in self.book_log],
            "live": self.live.model_dump(mode="json"),
            "slots": [self._snapshot_slot(s) for s in self.slots],
            "engine": self.engine.state(),
            "metrics": self.metrics.state(),
            "notices": {k: dict(v) for k, v in self.notices.items()},
            "llm_totals": dict(self.llm_totals),
            "last_trades": [dataclasses.asdict(t) for t in self.last_trades],
            "trades_base": self.trades_base,
            "batch_cursor": self.batch_cursor,
            "prev_vwap": self.prev_vwap,
            "online_benchmarks": (
                self.online_benchmarks.to_dict() if self.online_benchmarks else None
            ),
            "writer": self.writer.state(),
            "overrides_applied": self.overrides_applied,
        }

    def _checkpoint(self, period: int, names: list[str], *, update_manifest: bool = True) -> None:
        self.writer.flush(self.buffers)
        payload = self._capture(period)
        ck.write_checkpoint(
            self.run_dir,
            period,
            payload,
            names,
            {"run_id": self.run_id, "tick": int(self.market.tick), "writer": self.writer.state()},
        )
        ck.prune_checkpoints(self.run_dir, self.live.checkpoint.keep_last)
        if update_manifest and self.manifest:
            self.manifest.update(self._progress_fields())
            write_manifest(self.run_dir, self.manifest)

    # =========================================================================================
    # agents
    # =========================================================================================
    def _reindex(self) -> None:
        self._by_id = {s.agent_id: s for s in self.slots}

    def _llm_mode(self) -> str:
        bp = self.cfg.branch_period
        if bp is not None and self.period < bp and self.live.llm.mode == "live":
            return "replay"
        return self.live.llm.mode

    def _build_agent(self, spec: dict[str, Any], slot: int, agent_id: int) -> Any:
        if "bench" not in spec:  # freeze defaults at first build so rebuilds are identical
            spec["bench"] = self.benchmarks.to_dict() if self.benchmarks else None
            spec["market"] = self.live.market.model_dump(mode="json")
        bench = Benchmarks(**spec["bench"]) if spec.get("bench") else None
        market = MarketConfig.model_validate(spec["market"])
        if spec["type"] == "scripted":
            return make_scripted(
                spec["policy"],
                spec.get("params"),
                slot,
                agent_id,
                name=spec.get("name"),
                role=spec.get("role", "dealer"),
                memory_periods=int(spec.get("memory_periods", 100)),
                seed=spec.get("seed"),
                benchmarks=bench,
                market=market,
                replay_source=(spec.get("params") or {}).get("source") or self.replay_source,
            )
        if spec["type"] != "llm":
            raise ValueError(f"unknown agent spec type {spec['type']!r}")
        sc = LLMSlotConfig.model_validate(spec["slot_cfg"])
        if self.llm_factory is not None:
            return self.llm_factory(
                sc,
                slot=slot,
                agent_id=agent_id,
                run_cfg=self.live,
                run_id=self.run_id,
                rngs=self.rngs,
                provider=self.provider,
                lane=spec.get("lane"),
                mode=self._llm_mode(),
            )
        if self.live.llm.mode != "mock" and sc.family != "mock":
            raise RuntimeError(
                "LLM agents are not available (finswarm.agents.llm is not importable); "
                "use --provider mock to run with the mock stand-in"
            )
        params: dict[str, Any] = {}
        if sc.policy and canonical_policy(sc.policy) == "mock_llm":
            params = parse_policy(sc.policy)[1]
        params = {
            "model": sc.model,
            "lane": spec.get("lane") or self.live.llm.lane,
            "prompt_version": sc.prompt,
            **params,
        }
        return make_scripted(
            "mock_llm",
            params,
            slot,
            agent_id,
            name=sc.name,
            role=sc.role,
            memory_periods=sc.memory_periods,
            seed=spec.get("seed"),
            benchmarks=bench,
            market=market,
        )

    def _event_seed(self) -> list[int]:
        return [
            self.cfg.seed,
            EVENT_SEED_SALT,
            int(self.rngs["interventions"].integers(0, 2**31 - 1)),
        ]

    def _set_llm_mode(self, mode: str) -> None:
        for s in self.slots:
            fn = getattr(s.base_agent, "set_llm_mode", None)
            if fn is not None:
                fn(mode)

    # =========================================================================================
    # SimContext (used by the perturbation engine)
    # =========================================================================================
    def n_slots(self) -> int:
        return len(self.slots)

    def set_param(self, path: str, value: Any) -> Any:
        parts = path.split(".")
        if parts[0] in IDENTITY_FIELDS:
            raise ValueError(f"set_param: {parts[0]!r} cannot change during a run")
        if parts[0] == "clock":  # validated before anything changes
            check_clock_param(path, value, self.live.clock)
        if path == f"tags.{JITTER_TAG}":
            speed_bump_jitter(value)
        data = self.live.model_dump(mode="json")
        parent = _get_path(data, parts[:-1]) if len(parts) > 1 else data
        key = parts[-1]
        if isinstance(parent, list):
            old = parent[int(key)]
            parent[int(key)] = value
        else:
            if key not in parent and ".".join(parts[:-1]) not in FREE_DICTS:
                raise KeyError(f"set_param: unknown config path {path!r}")
            old = parent.get(key)
            parent[key] = value
        self.live = RunConfig.model_validate(data)
        self._propagate(path, value)
        return old

    def _propagate(self, path: str, value: Any) -> None:
        top = path.split(".")[0]
        if top == "clock":  # events fire at boundaries: in force from the period that starts
            if self.clock is not None:
                apply_clock_param(self.clock, self.live.clock, int(self.market.tick))
            return
        if path == f"tags.{JITTER_TAG}":
            if self.clock is not None:
                self.clock.set_jitter(speed_bump_jitter(value))
            return
        if top not in ("market", "swarm"):
            return  # view.*, logging.*, tags.* are read live by the runner
        m = self.market
        paths = settable_paths(m)
        if paths is not None:  # the market says what it can change: anything else fails
            if path not in paths:
                why = UNSUPPORTED_LIVE.get(path, "not settable live")
                raise UnsupportedIntervention(
                    f"set_param {path!r}: {why} in the {getattr(m, 'kind', '?')} market "
                    f"(settable live: {list(paths)})"
                )
            m.set_param(path, value)
            after_market_param(self, path)  # re-grid: runner quotes on the new grid; log
        elif path == "market.price_grid.tick" and hasattr(m, "set_tick"):
            m.set_tick(float(value))
        elif hasattr(m, "set_param"):
            try:
                m.set_param(path, value)
            except KeyError as exc:
                raise UnsupportedIntervention(f"set_param {path!r}: {exc}") from exc
        else:  # hook-less test doubles: rebuild from the live config and migrate the state
            new = self._new_market(self.live)
            new.migrate_from(m)
            self.market = new
        self._refresh_benchmarks()

    def _refresh_benchmarks(self, n_firms: int | None = None) -> None:
        if n_firms is None:
            n = sum(1 for s in self.slots if s.active and s.agent.role in SELL_ROLES)
            n_firms = n or None
        self.benchmarks = resolve_benchmarks(self.live.market, self.bench_override, n_firms)

    def spec_for_model(
        self, slot: int, model: str, lane: str, prompt: str | None
    ) -> dict[str, Any]:
        cur = self.slots[slot]
        role = cur.base_agent.role
        mem = int(getattr(cur.base_agent.memory, "memory_periods", 100))
        if model.startswith("scripted:"):
            return {
                "type": "scripted",
                "policy": model.split(":", 1)[1],
                "params": {},
                "seed": self._event_seed(),
                "role": role,
                "memory_periods": mem,
            }
        if model.startswith("replay"):
            src_slot = int(model.split(":", 1)[1]) if ":" in model else slot
            return {
                "type": "scripted",
                "policy": "replay_rival",
                "params": {"slot": src_slot, "source": self.replay_source},
                "seed": self._event_seed(),
                "role": role,
                "memory_periods": mem,
            }
        if cur.spec.get("type") == "llm":
            base = dict(cur.spec["slot_cfg"])
        elif slot < len(self.live.llm.slots):
            base = self.live.llm.slots[slot].model_dump(mode="json")
        else:
            base = {"name": f"slot{slot}", "role": role, "family": lane}
        base["model"] = model
        base["policy"] = None
        if prompt:
            base["prompt"] = prompt
        base.setdefault("family", lane)
        return {"type": "llm", "slot_cfg": base, "lane": lane, "seed": self._event_seed()}

    def snapshot_agent(self, slot: int) -> dict[str, Any]:
        return self._snapshot_slot(self.slots[slot])

    def replace_agent(
        self, slot: int, spec: dict[str, Any], *, keep_memory: bool, keep_name: bool
    ) -> None:
        s = self.slots[slot]
        old = s.base_agent
        new = self._build_agent(spec, slot, s.agent_id)
        if keep_memory:
            new.memory = AgentMemory.from_state(old.memory.to_state())
        if keep_name:
            new.name = old.name
            spec["name"] = old.name
        s.spec = spec
        s.agent = new
        _retire(old)
        self._refresh_benchmarks()

    def restore_agent(self, slot: int, snapshot: dict[str, Any]) -> None:
        s = self.slots[slot]
        old = s.base_agent
        s.spec = snapshot["spec"]
        s.agent = self._agent_from_snapshot(snapshot, slot, s.agent_id)
        s.active = bool(snapshot.get("active", True))
        _retire(old)
        self._refresh_benchmarks()

    def add_agent(self, spec: dict[str, Any]) -> int:
        slot = len(self.slots)
        if slot >= MAX_SLOTS:
            raise ValueError("too many tagged agents")
        seats = getattr(self.market, "n_dealers", None)
        if seats is not None and slot >= int(seats):
            raise UnsupportedIntervention(
                f"no free dealer seat for a new slot {slot}: the market has {seats} seats "
                "(integration.ENTRANT_RESERVE entrants per run)"
            )
        spec.setdefault("seed", self._event_seed())
        spec.setdefault("role", "dealer")
        s = SlotState(slot=slot, agent_id=self.n_swarm + slot, spec=spec)
        s.agent = self._build_agent(spec, slot, s.agent_id)
        self.slots.append(s)
        self._reindex()
        self._refresh_benchmarks()
        if self.clock is not None:  # an entrant posts its first quote at the boundary
            self.clock.add_slot(int(self.market.tick))
        if self.public_view is not None:
            self.public_view.register([s.agent_id])
        return slot

    def deactivate_agent(self, slot: int) -> None:
        s = self.slots[slot]
        self._cancel_all(s)
        s.active = False
        s.trade_cursor = None
        self._refresh_benchmarks()

    def wrap_forced(self, slot: int, delta: float, periods: int, event_id: int) -> None:
        s = self.slots[slot]
        s.agent = ForcedDeviationWrapper(
            s.base_agent, delta=delta, periods=periods, event_id=event_id
        )

    def unwrap(self, slot: int) -> None:
        s = self.slots[slot]
        s.agent = s.base_agent

    def add_notice(self, key: int, slots: list[int] | None, text: str, until: int | None) -> None:
        self.notices[int(key)] = {"slots": slots, "text": text, "until": until}

    def remove_notice(self, key: int) -> None:
        self.notices.pop(int(key), None)

    def market_call(self, hook: str, **kwargs: Any) -> bool:
        """Run an event hook (see ``integration.market_hook``); a missing hook raises
        ``UnsupportedIntervention`` so the run fails with a reason instead of ignoring it."""
        market_hook(self.market, hook, self.rngs, self.removed_liquidity, **kwargs)
        return True

    def log_event(
        self, event_id: int, event_type: str, payload: dict[str, Any], source: str
    ) -> None:
        self.buffers.add(
            "events",
            {
                "run_id": self.run_id,
                "event_id": event_id,
                "tick": int(self.market.tick),
                "period": self.period,
                "event_type": event_type,
                "payload": json.dumps(payload, sort_keys=True, default=str),
                "source": source,
            },
        )

    # =========================================================================================
    # boundary and period
    # =========================================================================================
    async def _boundary(self, b: int, *, final: bool = False) -> None:
        for s in self.slots:
            if isinstance(s.agent, ForcedDeviationWrapper) and s.agent.done:
                eid = s.agent.event_id
                s.agent = s.agent.inner
                self.log_event(eid, "forced_deviation_end", {"slot": s.slot}, "derived")
        names = self.engine.boundary(self, b, int(self.market.tick))
        if self.agent_overrides and not self.overrides_applied and b >= self.overrides_at:
            for slot, spec in sorted(self.agent_overrides.items()):
                self.replace_agent(
                    slot, json.loads(json.dumps(spec)), keep_memory=False, keep_name=True
                )
            self.overrides_applied = True
            self.log_event(
                -1, "agent_overrides", {"slots": sorted(self.agent_overrides)}, "derived"
            )
        if self.cfg.branch_period is not None and b == self.cfg.branch_period:
            self._set_llm_mode(self.live.llm.mode)
        self.notices = {
            k: v for k, v in self.notices.items() if v["until"] is None or v["until"] > b
        }
        every = self.live.checkpoint.every_periods
        extra: list[str] = []
        if b == 0:
            extra.append("initial")
        if final:
            extra.append("final")
        if not extra and every > 0 and b % every == 0:
            extra.append("periodic")
        if (names or extra) and not (b == self.restored_at and not names):
            self._checkpoint(b, names + extra)

    async def _run_period(self, p: int) -> None:
        """Decision batches popped from the clock, the market advanced between them."""
        assert self.clock is not None
        tpp = max(1, self.live.ticks_per_period)
        tick_start = int(self.market.tick)
        tick_end = tick_start + tpp
        self._period_span = (tick_start, tpp)  # for ``quotes.apply_period``
        for s in self.slots:
            s.cash0, s.inv0, s.p_qty = s.cash, s.inventory, 0.0
        acc = {"vol": 0.0, "notional": 0.0, "n": 0, "surplus": 0.0, "n_surplus": 0}
        mids: list[float] = []
        n_fail = 0
        e = 0
        while (batch := self.clock.next_batch(tick_end)) is not None:
            self._advance_to(batch.tick, p, acc, mids)  # staggered clocks: run to the batch
            n_fail += await self._run_epoch(batch, p, e)
            nxt = self.clock.peek()
            self._advance_to(tick_end if nxt is None else min(nxt, tick_end), p, acc, mids)
            self.epoch += 1
            e += 1
        self._advance_to(tick_end, p, acc, mids)
        self._close_period(p, tick_start, acc, mids, n_fail, self._period_summary(p))

    async def _run_epoch(self, batch: Epoch, p: int, e: int) -> int:
        """One decision batch: observations as of the batch tick, concurrent decisions, orders
        applied at ``decision_tick + speed bump``, one ``quotes`` row per active slot."""
        t = int(self.market.tick)
        self._apply_pending(t)  # changes the market applied at the end of the last advance
        due = [self.slots[i] for i in batch.slots if self.slots[i].active]
        book = self.market.snapshot(self.live.logging.snapshot_levels)
        self._log_public_state(book)
        obs = [self._view(self._observe(s, p, e, book)) for s in due]
        self._consume_trades(due)  # each deciding slot sees later trades from now on
        decisions = await asyncio.gather(
            *(s.agent.decide(o) for s, o in zip(due, obs, strict=True))
        )
        n_fail = 0
        applied: dict[int, int] = {}  # slot -> apply_tick of this epoch's decision
        decided_at: dict[int, int] = {}  # slot -> its decision tick
        for s, o, d in zip(due, obs, decisions, strict=True):
            decided_at[s.slot] = batch.decision_tick(s.slot)
            fail, applied[s.slot] = self._apply_decision(s, o, d, p, decided_at[s.slot])
            n_fail += fail
            drain = getattr(s.base_agent, "drain_events", None)
            if drain is not None:
                for rec in drain():
                    self.log_event(int(rec.get("event_id", -1)), "sting_phase", rec, "derived")
        for s in self.slots:
            if not s.active:
                continue
            if s.slot in applied:  # the quote it sent, from the tick it takes effect
                self._quote_row(
                    s,
                    p,
                    s.last_decision_id,
                    applied[s.slot],
                    sent=True,
                    decision_tick=decided_at[s.slot],
                )
            else:  # its quote in force now
                self._quote_row(s, p, -1, t, sent=False, decision_tick=t)
        return n_fail

    def _advance_to(self, tick: int, p: int, acc: dict[str, Any], mids: list[float]) -> None:
        """Run the market to ``tick``: drain trades and flow, account, record the mid; quote
        changes up to ``tick - 1`` are now in force on the runner's side too."""
        n = int(tick) - int(self.market.tick)
        if n <= 0:
            return
        self.market.advance(n)
        cols = self._drain_trade_columns()
        self._account(cols, p, acc)
        self._drain_market_extras(p)
        mid = self.market.mid()
        mids.append(float(mid) if mid is not None else NAN)
        self.last_trades.extend(_trades_from_columns(cols))
        self._apply_pending(int(self.market.tick) - 1)

    # -- order flow shown to agents (per-slot cursors) -------------------------------------------
    def _trades_since(self, s: SlotState) -> list[Trade]:
        """Trades since slot ``s``'s own last decision (since the last batch if it has none)."""
        c = self.batch_cursor if s.trade_cursor is None else s.trade_cursor
        return self.last_trades[max(0, c - self.trades_base) :]

    def _consume_trades(self, due: list[SlotState]) -> None:
        """The slots in ``due`` just observed: move their cursors to now, prune the log.

        Under the ``sync`` clock every active slot decides in every batch, so the log is
        emptied at each batch exactly as before (byte-identical observations)."""
        total = self.trades_base + len(self.last_trades)
        for s in due:
            s.trade_cursor = total
        self.batch_cursor = total
        lo = min(
            [total]
            + [x.trade_cursor for x in self.slots if x.active and x.trade_cursor is not None]
        )
        if lo > self.trades_base:
            del self.last_trades[: lo - self.trades_base]
            self.trades_base = lo

    # -- quote changes in flight (speed bump, staleness window) ---------------------------------
    def _set_live(self, s: SlotState, side: int, price: float, size: float, order_id: int) -> None:
        if side == CANCEL_SIDE:  # a cancel withdraws both sides
            s.live_bid = s.live_ask = NAN
            s.live_bid_size = s.live_ask_size = 0.0
            s.live_orders.clear()
        elif side == int(Side.SELL):
            s.live_ask, s.live_ask_size = price, size
            s.live_orders[side] = order_id
        else:
            s.live_bid, s.live_bid_size = price, size
            s.live_orders[side] = order_id

    def _schedule_live(
        self, s: SlotState, side: int, price: float, size: float, order_id: int, apply_tick: int
    ) -> None:
        """A quote change takes effect now, or at ``apply_tick`` (kept in ``pending_quotes``)."""
        if apply_tick <= int(self.market.tick):
            self._set_live(s, side, price, size, order_id)
        else:
            bisect.insort(self.pending_quotes, (apply_tick, order_id, s.slot, side, price, size))

    def _apply_pending(self, upto: int) -> None:
        """Quote changes with ``apply_tick <= upto`` take effect, in the market's order."""
        pq = self.pending_quotes
        k = 0
        while k < len(pq) and pq[k][0] <= upto:
            _, oid, slot, side, price, size = pq[k]
            self._set_live(self.slots[slot], side, price, size, oid)
            k += 1
        if k:
            del pq[:k]

    def _view(self, obs: Observation) -> Observation:
        if self.public_view is not None:
            out = self.view_fn(obs, self.live.view, self.rngs["view"], view=self.public_view)
        else:
            out = self.view_fn(obs, self.live.view, self.rngs["view"])
        for k in [k for k in out.extra if k.startswith("_raw")]:
            del out.extra[k]  # raw view inputs never reach an agent
        return out

    def _drain_trade_columns(self) -> dict[str, Any]:
        """Trades since the last drain as columns (``drain_trades_arrays`` fast path)."""
        fn = getattr(self.market, "drain_trades_arrays", None)
        if fn is not None:
            cols = dict(fn())
            cols["mechanism"] = str(getattr(self.market.kind, "value", self.market.kind))
            return cols
        trades = self.market.drain_trades()
        pt = np.array([t.price_ticks for t in trades], dtype=np.int64)
        mech = {t.mechanism for t in trades}
        return {
            "trade_id": np.array([t.trade_id for t in trades], dtype=np.int64),
            "tick": np.array([t.tick for t in trades], dtype=np.int64),
            "price_ticks": pt,
            "price": float(self.market.p_min) + pt * float(self.market.tick_size),
            "size": np.array([t.size for t in trades], dtype=np.float64),
            "buyer_id": np.array([t.buyer_id for t in trades], dtype=np.int64),
            "seller_id": np.array([t.seller_id for t in trades], dtype=np.int64),
            "aggressor": np.array([t.aggressor for t in trades], dtype=np.int64),
            "batch_id": np.array([t.batch_id for t in trades], dtype=np.int64),
            "buyer_value": np.array([t.buyer_value for t in trades], dtype=np.float64),
            "seller_cost": np.array([t.seller_cost for t in trades], dtype=np.float64),
            "mechanism": mech.pop() if len(mech) == 1 else [t.mechanism for t in trades],
        }

    def _drain_market_extras(self, p: int) -> None:
        """``drain_flow`` -> ``flow_per_tick`` and ``drain_orders`` -> ``orders``: columnar
        dicts go straight into the buffers (no per-tick Python loop); row lists are kept for
        markets that return them."""
        for hook, table in (("drain_flow", "flow_per_tick"), ("drain_orders", "orders")):
            fn = getattr(self.market, hook, None)
            if fn is None:
                continue
            data = fn()
            if isinstance(data, dict):
                cols = {k: np.asarray(v) for k, v in data.items()}
                if not cols or len(next(iter(cols.values()))) == 0:
                    continue
                if table == "flow_per_tick" and "dealer_id" in cols:
                    rel = cols["dealer_id"].astype(np.int64) - self.n_swarm
                    # drop unused entrant seats; keep the public fringe (dealer_id -2)
                    keep = ((rel >= 0) & (rel < len(self.slots))) | (cols["dealer_id"] < 0)
                    if not keep.all():
                        cols = {k: v[keep] for k, v in cols.items()}
                self.buffers.add_columns(table, {"run_id": self.run_id, **cols})
            else:
                for row in data:
                    self.buffers.add(table, {"run_id": self.run_id, "period": p, **row})

    def _period_summary(self, p: int) -> Any:
        fn = getattr(self.market, "drain_periods", None)
        if fn is None:
            return None
        out = None
        for ps in fn():
            if int(ps.period) == p:
                out = ps
        return out

    def _unit_cost(self) -> float:
        return self.live.market.cost * self.live.market.alpha

    def _ticks(self, price: float) -> int:
        if not _finite(price):
            return -1
        return ticks_from_price(price, self.market.p_min, self.market.tick_size)

    # -- public state seen through the view -----------------------------------------------------
    def _log_quote(self, s: SlotState, tick: int) -> None:
        self.quote_log.append(
            (
                int(tick),
                s.agent_id,
                self._ticks(s.bid),
                self._ticks(s.ask),
                float(s.bid_size),
                float(s.ask_size),
            )
        )

    def _log_public_state(self, book: list[Any]) -> None:
        """Record this epoch's book and prune both logs to what ``view.delay_ticks`` needs."""
        now = int(self.market.tick)
        self.book_log.append((now, list(book)))
        cutoff = now - max(0, int(self.live.view.delay_ticks))
        last: dict[int, int] = {}
        for i, q in enumerate(self.quote_log):
            if q[0] <= cutoff:
                last[q[1]] = i
        self.quote_log = [
            q for i, q in enumerate(self.quote_log) if q[0] > cutoff or last.get(q[1]) == i
        ]
        first = max((i for i, (t, _) in enumerate(self.book_log) if t <= cutoff), default=0)
        self.book_log = self.book_log[first:]

    def _max_wtp(self) -> float:
        tag = self.live.tags.get("max_wtp")
        if tag is not None:
            return float(tag)
        return DEFAULT_MAX_WTP_OVER_ALPHA * float(self.live.market.alpha)

    def _observe(self, s: SlotState, p: int, e: int, book: list[Any]) -> Observation:
        live = self.live
        h = live.view.history_periods
        hist = list(s.history)[-h:] if h > 0 else []
        active = {x.agent_id for x in self.slots if x.active and x.slot != s.slot}
        rivals = [  # quotes in force (a quote delayed by the speed bump is not visible yet)
            Quote(
                agent_id=x.agent_id,
                bid_ticks=self._ticks(x.live_bid),
                ask_ticks=self._ticks(x.live_ask),
                bid_size=x.live_bid_size,
                ask_size=x.live_ask_size,
            )
            for x in self.slots
            if x.agent_id in active
        ]
        notices = []
        for k in sorted(self.notices):
            n = self.notices[k]
            if n["slots"] is None and s.agent.kind != AgentKind.LLM:
                continue
            if n["slots"] is not None and s.slot not in n["slots"]:
                continue
            notices.append(
                n["text"].format_map(_SafeDict(period=p, slot=s.slot, name=s.agent.name))
            )
        b = self.benchmarks
        extra: dict[str, Any] = {
            "p_min": float(self.market.p_min),
            "tick_size": float(self.market.tick_size),
            "price_max": float(live.market.price_grid.max),
            "epoch": self.epoch,
            "epoch_in_period": e,
            "alpha": float(live.market.alpha),
            "two_sided": bool(live.market.two_sided),
            "max_wtp": self._max_wtp(),
            "n_rounds": int(self.cfg.periods),
            RAW_QUOTES_KEY: [
                (t, Quote(agent_id=aid, bid_ticks=bt, ask_ticks=at, bid_size=bs, ask_size=az))
                for t, aid, bt, at, bs, az in self.quote_log
                if aid in active
            ],
            RAW_BOOK_KEY: list(self.book_log),
            # live demand and routing (levers move them): the scripted best responder reads it
            "demand_live": demand_dict(live.market),
        }
        if b is not None:
            extra["nash_price"] = float(b.p_nash)
            extra["monopoly_price"] = float(b.p_mono)
            extra["default_price"] = float(b.p_nash)
        if "default_price" in live.tags:
            extra["default_price"] = float(live.tags["default_price"])
        return Observation(
            tick=int(self.market.tick),
            period=p,
            agent_id=s.agent_id,
            slot=s.slot,
            role=s.agent.role,
            inventory=s.inventory,
            cash=s.cash,
            cost=self._unit_cost(),
            history=hist,
            rival_quotes=rivals,
            trades_since_epoch=self._trades_since(s),
            book_levels=list(book),
            notices=notices,
            view_policy_id=live.view.id,
            extra=extra,
        )

    # -- decisions -> orders --------------------------------------------------------------------
    def _snap(self, price: float) -> int:
        idx = ticks_from_price(price, self.market.p_min, self.market.tick_size)
        hi = ticks_from_price(
            self.live.market.price_grid.max, self.market.p_min, self.market.tick_size
        )
        return min(max(idx, 0), hi)

    def _apply_decision(
        self, s: SlotState, obs: Observation, d: Decision, p: int, decision_tick: int
    ) -> tuple[int, int]:
        """Orders of one decision; returns ``(parse failure, apply_tick)``.

        ``apply_tick = decision_tick + speed bump`` (``EpochClock.apply_tick``: constant delay,
        FIFO per slot) is fixed for every decision, holds included.
        """
        assert self.clock is not None
        did = self.next_decision_id
        self.next_decision_id += 1
        s.last_decision_id = did
        s.n_decisions += 1
        apply_tick = self.clock.apply_tick(decision_tick, s.slot)
        flagged = (not d.parse_ok) or d.refusal
        action = "hold" if flagged else (d.action or "hold").lower()
        if action == "quote":
            self._quote(s, d, p, apply_tick)
        elif action in ("limit", "market"):
            self._order(
                s, d, p, apply_tick, OrderType.LIMIT if action == "limit" else OrderType.MARKET
            )
        elif action == "cancel":
            self._cancel(s, p, apply_tick)
        if s.agent.kind == AgentKind.LLM and not getattr(s.agent, "skip_llm_log", False):
            drain = getattr(s.base_agent, "drain_logs", None)
            if drain is not None:  # real LLM agents: one row per call, as logged by the agent
                for row in drain():
                    self._llm_log_row(s, obs, d, row, did, p, apply_tick)
            else:  # stand-ins exposing ``last_call``
                self._llm_row(s, obs, d, did, p, apply_tick)
        return (0 if d.parse_ok else 1), apply_tick

    def _quote(self, s: SlotState, d: Decision, p: int, apply_tick: int) -> None:
        bid, ask = d.bid, d.ask
        role = s.agent.role
        if bid is None and ask is None and d.price is not None:
            if role == "buyer":
                bid = d.price
            else:
                ask = d.price
        if not self.live.market.two_sided:
            if role == "buyer":
                ask = None
            else:
                bid = None
        size = float(d.size) if d.size is not None else DEFAULT_QUOTE_SIZE
        if _finite(ask):
            self._submit_quote(s, Side.SELL, float(ask), size, p, apply_tick)
        if _finite(bid):
            self._submit_quote(s, Side.BUY, float(bid), size, p, apply_tick)

    def _new_order(
        self, s: SlotState, side: Side, otype: OrderType, idx: int, size: float, apply_tick: int
    ) -> Order:
        o = Order(
            order_id=self.next_order_id,
            agent_id=s.agent_id,
            side=side,
            otype=otype,
            price_ticks=idx,
            size=size,
            tif_ticks=-1,
            submit_tick=int(self.market.tick),
            apply_tick=apply_tick,
        )
        self.next_order_id += 1
        return o

    def _submit_quote(
        self, s: SlotState, side: Side, price: float, size: float, p: int, apply_tick: int
    ) -> None:
        idx = self._snap(price)
        order = self._new_order(s, side, OrderType.QUOTE, idx, size, apply_tick)
        ack = self.market.submit(order)
        self._order_row(s, order, ack, p)
        if ack.accepted:  # a QUOTE replaces the agent's standing quote on that side
            px = price_from_ticks(idx, self.market.p_min, self.market.tick_size)
            if side == Side.SELL:
                s.ask, s.ask_size = px, size
            else:
                s.bid, s.bid_size = px, size
            s.quote_orders[int(side)] = order.order_id
            self._log_quote(s, apply_tick)  # public once in force
            self._schedule_live(s, int(side), px, size, order.order_id, apply_tick)

    def _order(self, s: SlotState, d: Decision, p: int, apply_tick: int, otype: OrderType) -> None:
        side = (
            d.side if d.side is not None else (Side.BUY if s.agent.role == "buyer" else Side.SELL)
        )
        if otype == OrderType.LIMIT:
            if d.price is None or not _finite(d.price):
                return
            idx = self._snap(float(d.price))
        else:
            idx = -1
        size = float(d.size) if d.size is not None else DEFAULT_ORDER_SIZE
        order = self._new_order(s, Side(side), otype, idx, size, apply_tick)
        ack = self.market.submit(order)
        self._order_row(s, order, ack, p)
        if ack.accepted and otype == OrderType.LIMIT:
            s.open_orders.append(order.order_id)

    def _cancel(self, s: SlotState, p: int, apply_tick: int) -> None:
        """A ``cancel`` decision: at once, or a ``CANCEL`` order applied at ``apply_tick``."""
        if apply_tick <= int(self.market.tick):
            self._cancel_all(s)
            return
        order = self._new_order(s, Side.SELL, OrderType.CANCEL, -1, 0.0, apply_tick)
        ack = self.market.submit(order)
        self._order_row(s, order, ack, p)
        if not ack.accepted:  # markets without delayed cancels withdraw at once
            self._cancel_all(s)
            return
        s.bid = s.ask = NAN
        s.bid_size = s.ask_size = 0.0
        s.quote_orders.clear()
        self._log_quote(s, apply_tick)
        self._schedule_live(s, CANCEL_SIDE, NAN, 0.0, order.order_id, apply_tick)

    def _cancel_all(self, s: SlotState) -> None:
        """Withdraw everything ``s`` has in the market now, in force or still in flight."""
        in_flight = [q[1] for q in self.pending_quotes if q[2] == s.slot]
        had_quote = bool(s.quote_orders) or bool(s.live_orders) or bool(in_flight)
        ids = [*s.quote_orders.values(), *s.live_orders.values(), *in_flight, *s.open_orders]
        for oid in dict.fromkeys(ids):
            self.market.cancel(oid)
        s.quote_orders.clear()
        s.open_orders.clear()
        s.live_orders.clear()
        s.bid = s.ask = s.live_bid = s.live_ask = NAN
        s.bid_size = s.ask_size = s.live_bid_size = s.live_ask_size = 0.0
        if in_flight:
            now = int(self.market.tick)
            self.pending_quotes = [q for q in self.pending_quotes if q[2] != s.slot]
            self.quote_log = [q for q in self.quote_log if q[1] != s.agent_id or q[0] <= now]
        if had_quote:
            self._log_quote(s, int(self.market.tick))

    def _order_row(self, s: SlotState, o: Order, ack: OrderAck, p: int) -> None:
        price = (
            price_from_ticks(o.price_ticks, self.market.p_min, self.market.tick_size)
            if o.price_ticks >= 0
            else NAN
        )
        self.buffers.add(
            "orders",
            {
                "run_id": self.run_id,
                "tick": o.submit_tick,
                "period": p,
                "order_id": o.order_id,
                "agent_id": o.agent_id,
                "agent_kind": int(s.agent.kind),
                "swarm_type": -1,
                "side": int(o.side),
                "otype": int(o.otype),
                "price": price,
                "price_ticks": o.price_ticks,
                "size": o.size,
                "tif_ticks": o.tif_ticks,
                "status": 1 if ack.accepted else 0,
                "reason": ack.reason,
            },
        )

    def _llm_row(
        self, s: SlotState, obs: Observation, d: Decision, did: int, p: int, apply_tick: int
    ) -> None:
        info = getattr(s.agent, "last_call", None) or {}
        view_hash = self._view_hash(obs)
        parsed = {
            k: getattr(d, k)
            for k in ("action", "price", "size", "side", "bid", "ask")
            if getattr(d, k) is not None
        }
        price = d.price if d.price is not None else (d.ask if d.ask is not None else d.bid)
        sc = s.spec.get("slot_cfg") or {}
        row = {
            "run_id": self.run_id,
            "decision_id": did,
            "period": p,
            "epoch": self.epoch,
            "obs_tick": obs.tick,
            "apply_tick": apply_tick,
            "agent_id": s.agent_id,
            "slot": s.slot,
            "provider": info.get("provider", self.provider or "unknown"),
            "model": info.get("model", sc.get("model")),
            "lane": info.get("lane", s.spec.get("lane") or self.live.llm.lane),
            "prompt_version": info.get("prompt_version", sc.get("prompt")),
            "prompt_hash": info.get("prompt_hash"),
            "view_hash": view_hash,
            "view_policy_id": obs.view_policy_id,
            "cache_hit": bool(info.get("cache_hit", False)),
            "raw_output": d.raw,
            "parsed_action": json.dumps(parsed, sort_keys=True, default=int),
            "action_type": d.action,
            "price": price if price is not None else NAN,
            "size": d.size if d.size is not None else NAN,
            "plans": d.plans,
            "insights": d.insights,
            "message": d.message,
            "parse_ok": d.parse_ok,
            "repair_used": d.repair_used,
            "refusal": d.refusal,
            "latency_ms": float(info.get("latency_ms", 0.0)),
            "input_tokens": int(info.get("input_tokens", 0)),
            "output_tokens": int(info.get("output_tokens", 0)),
            "cached_tokens": int(info.get("cached_tokens", 0)),
            "cost_usd": float(info.get("cost_usd", 0.0)),
            "request_id": info.get("request_id"),
            "error": info.get("error"),
        }
        self._emit_llm_row(s, row, did, p)

    @staticmethod
    def _view_hash(obs: Observation) -> str:
        vh = obs.extra.get("view_hash")
        if vh:
            return str(vh)
        blob = json.dumps(dataclasses.asdict(obs), sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def _llm_log_row(
        self,
        s: SlotState,
        obs: Observation,
        d: Decision,
        row: dict[str, Any],
        did: int,
        p: int,
        apply_tick: int,
    ) -> None:
        """A row from ``agent.drain_logs()`` with the runner's ids, epoch and ``apply_tick``."""
        out = dict(row)
        out.update(
            run_id=self.run_id,
            decision_id=did,
            period=p,
            epoch=self.epoch,
            obs_tick=int(out.get("obs_tick") if out.get("obs_tick") is not None else obs.tick),
            apply_tick=apply_tick,
            agent_id=s.agent_id,
            slot=s.slot,
            view_hash=out.get("view_hash") or self._view_hash(obs),
            view_policy_id=out.get("view_policy_id") or obs.view_policy_id,
        )
        for k in ("price", "size"):
            if out.get(k) is None:
                out[k] = NAN
        for k in ("latency_ms", "cost_usd"):
            out[k] = float(out.get(k) or 0.0)
        for k in ("input_tokens", "output_tokens", "cached_tokens"):
            out[k] = int(out.get(k) or 0)
        out["cache_hit"] = bool(out.get("cache_hit", False))
        self._emit_llm_row(s, out, did, p)

    def _emit_llm_row(self, s: SlotState, row: dict[str, Any], did: int, p: int) -> None:
        self.buffers.add("llm_decisions", row)
        t = self.llm_totals
        t["calls"] += 1
        t["cache_hits"] += int(row["cache_hit"])
        t["input_tokens"] += row["input_tokens"]
        t["output_tokens"] += row["output_tokens"]
        t["cached_tokens"] += row["cached_tokens"]
        t["cost_usd"] += row["cost_usd"]
        t["parse_failures"] += int(not row["parse_ok"])
        t["refusals"] += int(bool(row["refusal"]))
        prompt = getattr(s.base_agent, "last_prompt", None)
        if self.live.logging.store_prompts and prompt:
            self.buffers.add_prompt(
                {"run_id": self.run_id, "decision_id": did, "slot": s.slot, "period": p, **prompt}
            )

    def _quote_row(
        self, s: SlotState, p: int, decision_id: int, tick: int, *, sent: bool, decision_tick: int
    ) -> None:
        """``sent``: the quote ``s`` last sent, stamped with the tick it takes effect; else the
        quote in force at ``tick``. ``period``/``decision_tick`` are when it was decided (or
        observed); ``apply_period`` is the period containing ``tick`` at the current
        ``ticks_per_period`` (``>= periods`` when it never takes effect within the run)."""
        t0, tpp = getattr(self, "_period_span", (int(tick), 1))
        apply_period = p + (int(tick) - t0) // tpp
        if sent:
            bid, ask, bsz, asz = s.bid, s.ask, s.bid_size, s.ask_size
        else:
            bid, ask, bsz, asz = s.live_bid, s.live_ask, s.live_bid_size, s.live_ask_size
        self.buffers.add(
            "quotes",
            {
                "run_id": self.run_id,
                "period": p,
                "epoch": self.epoch,
                "tick": int(tick),
                "agent_id": s.agent_id,
                "slot": s.slot,
                "is_llm": s.agent.kind == AgentKind.LLM,
                "bid": bid,
                "ask": ask,
                "bid_size": bsz,
                "ask_size": asz,
                "inventory": s.inventory,
                "cash": s.cash,
                "decision_id": decision_id,
                "decision_tick": int(decision_tick),
                "apply_period": int(apply_period),
            },
        )

    # -- accounting -----------------------------------------------------------------------------
    def _kind_lookup(self, ids: np.ndarray) -> np.ndarray:
        """``AgentKind`` per agent id (swarm ids and the aggregate buyer -1 are SWARM)."""
        out = np.full(ids.shape[0], int(AgentKind.SWARM), dtype=np.int8)
        if not self.slots or ids.shape[0] == 0:
            return out
        kinds = np.array([int(s.agent.kind) for s in self.slots], dtype=np.int8)
        rel = ids.astype(np.int64) - self.n_swarm
        m = (rel >= 0) & (rel < len(self.slots))
        out[m] = kinds[rel[m]]
        return out

    def _account(self, cols: dict[str, Any], p: int, acc: dict[str, Any]) -> None:
        """Write the epoch's trades (one columnar chunk) and update tagged agents' accounts.

        One-sided sellers earn ``price - c*alpha`` per unit, buyers ``value - price``; two-sided
        dealers move cash and inventory. Per-slot sums use ``np.bincount`` (no per-trade loop).
        """
        k = int(np.asarray(cols["tick"]).shape[0])
        if k == 0:
            return
        price = np.asarray(cols["price"], dtype=np.float64)
        size = np.asarray(cols["size"], dtype=np.float64)
        buyer = np.asarray(cols["buyer_id"], dtype=np.int64)
        seller = np.asarray(cols["seller_id"], dtype=np.int64)
        self.buffers.add_columns(
            "trades",
            {
                "run_id": self.run_id,
                "trade_id": cols["trade_id"],
                "tick": cols["tick"],
                "period": p,
                "price": price,
                "size": size,
                "buyer_id": buyer,
                "seller_id": seller,
                "buyer_kind": self._kind_lookup(buyer),
                "seller_kind": self._kind_lookup(seller),
                "aggressor": cols["aggressor"],
                "mechanism": cols["mechanism"],
                "batch_id": cols["batch_id"],
                "buyer_value": cols["buyer_value"],
                "seller_cost": cols["seller_cost"],
            },
        )
        notional = price * size
        two = self.live.market.two_sided
        cost = self._unit_cost()
        ns = len(self.slots)
        for ids, side in ((seller, Side.SELL), (buyer, Side.BUY)):
            rel = ids - self.n_swarm
            m = (rel >= 0) & (rel < ns)
            if not m.any():
                continue
            hits = np.bincount(rel[m], minlength=ns)
            qty = np.bincount(rel[m], weights=size[m], minlength=ns)
            val = np.bincount(rel[m], weights=notional[m], minlength=ns)
            for s in self.slots:
                if hits[s.slot] == 0:
                    continue
                q, v = float(qty[s.slot]), float(val[s.slot])
                s.p_qty += q
                if two:
                    sign = 1.0 if side == Side.SELL else -1.0
                    s.cash += sign * v
                    s.inventory -= sign * q
                elif side == Side.SELL:
                    s.cash += v - cost * q
                else:
                    s.cash += cost * q - v  # buyer surplus; the cost field holds the value
        acc["vol"] += float(size.sum())
        acc["notional"] += float(notional.sum())
        acc["n"] += k
        bv = np.asarray(cols["buyer_value"], dtype=np.float64)
        fin = np.isfinite(bv)
        if fin.any():
            acc["surplus"] += float(((bv[fin] - price[fin]) * size[fin]).sum())
            acc["n_surplus"] += int(fin.sum())

    def _own_price(self, s: SlotState) -> float:
        """The quote in force (at period close: during the period's last tick)."""
        bid, ask = s.live_bid, s.live_ask
        if self.live.market.two_sided and _finite(bid) and _finite(ask):
            return 0.5 * (bid + ask)
        if s.agent.role == "buyer":
            return bid
        return ask

    def _close_period(
        self,
        p: int,
        tick_start: int,
        acc: dict[str, Any],
        mids: list[float],
        n_fail: int,
        summary: Any = None,
    ) -> None:
        live = self.live
        vol = acc["vol"]
        vwap = acc["notional"] / vol if vol > 0 else NAN
        active = [s for s in self.slots if s.active]
        prices = {s.slot: self._own_price(s) for s in active}
        profits: dict[int, float] = {}
        for s in active:
            profit = s.cash - s.cash0
            quantity = s.p_qty
            if live.market.two_sided and _finite(vwap):
                profit += vwap * (s.inventory - s.inv0)
            elif (
                summary is not None and s.agent.role in SELL_ROLES and s.slot < len(summary.qty)
            ):  # the market's own per-dealer period accounting (drain_periods)
                quantity = float(summary.qty[s.slot])
                profit = float(summary.profit[s.slot])
            profits[s.slot] = profit
            rivals = tuple(prices[x.slot] for x in active if x.slot != s.slot)
            rec = PeriodRecord(
                period=p,
                own_price=prices[s.slot],
                own_quantity=quantity,
                own_profit=profit,
                rival_prices=rivals,
                market_price=vwap,
            )
            s.history.append(rec)
            s.agent.record_period(rec)
            prev = [x for x in s.prev_rivals if _finite(x)]
            undercut = bool(prev) and _finite(prices[s.slot]) and prices[s.slot] < min(prev) - 1e-12
            s.prev_rivals = rivals
            self.buffers.add(
                "agent_period",
                {
                    "run_id": self.run_id,
                    "period": p,
                    "agent_id": s.agent_id,
                    "slot": s.slot,
                    "price": prices[s.slot],
                    "quantity": quantity,
                    "profit": profit,
                    "inventory": s.inventory,
                    "cash": s.cash,
                    "undercut": undercut,
                },
            )
        study = [s for s in active if s.agent.role in SELL_ROLES]
        focus = [s for s in study if s.agent.kind == AgentKind.LLM] or study
        mean_profit = sum(profits[s.slot] for s in focus) / len(focus) if focus else NAN
        fp = [prices[s.slot] for s in focus if _finite(prices[s.slot])]
        mean_price = sum(fp) / len(fp) if fp else NAN
        # registered semantics: delta_p against the benchmarks frozen at run start (trunk /
        # pre-lever) in every window; with the fringe on the closed-form profit benchmarks do
        # not apply, so delta (delta_pi) is NaN and the row says ``benchmark_valid = False``
        b, b0 = self.benchmarks, self.online_benchmarks
        fringe = getattr(self.live.market, "fringe", None)
        benchmark_valid = float(getattr(fringe, "share", 0.0) or 0.0) <= 0.0
        delta = b.delta(mean_profit) if b and benchmark_valid else NAN
        delta_p = b0.delta_p(mean_price) if b0 else NAN
        llm_profit = sum(profits[s.slot] for s in active if s.agent.kind == AgentKind.LLM)
        fm = [m for m in mids if _finite(m)]
        mid_mean = sum(fm) / len(fm) if fm else NAN
        bb, ba = self.market.best_bid(), self.market.best_ask()
        spread = ba - bb if bb is not None and ba is not None else NAN
        top = self.market.snapshot(1)
        depth_top = sum(lv[1] + lv[2] for lv in top) if top else NAN
        if summary is not None:
            fundamental = float(summary.fundamental)
        else:
            f = getattr(self.market, "fundamental", None)
            fundamental = float(f) if f is not None else NAN
        rv = (
            abs(math.log(vwap / self.prev_vwap))
            if _finite(vwap) and _finite(self.prev_vwap) and vwap > 0 and self.prev_vwap > 0
            else NAN
        )
        self.buffers.add(
            "metrics_per_period",
            {
                "run_id": self.run_id,
                "period": p,
                "tick_start": tick_start,
                "tick_end": int(self.market.tick),
                "fundamental": fundamental,
                "mid_mean": mid_mean,
                "quoted_spread": spread,
                "effective_spread": 2.0 * abs(vwap - mid_mean)
                if _finite(vwap) and _finite(mid_mean)
                else NAN,
                "depth_top": depth_top,
                "volume": vol,
                "n_trades": int(acc["n"]),
                "realised_vol": rv,
                "pricing_error": abs(vwap - fundamental)
                if _finite(vwap) and _finite(fundamental)
                else NAN,
                "llm_profit_total": llm_profit,
                "swarm_surplus": acc["surplus"] if acc["n_surplus"] else NAN,
                "delta": delta,
                "parse_fail_count": n_fail,
                "benchmark_valid": benchmark_valid,
            },
        )
        if _finite(vwap):
            self.prev_vwap = vwap
        self.metrics.update(
            p,
            {
                "delta": delta,
                "delta_p": delta_p,
                "price_mean": mean_price,
                "profit_mean": mean_profit,
                "volume": vol,
                "n_trades": float(acc["n"]),
                "parse_fail_count": float(n_fail),
                "llm_profit_total": llm_profit,
                "mid_mean": mid_mean,
            },
        )
        se = live.logging.snapshot_every
        if se > 0 and p % se == 0:
            self._book_rows(p)
        if self.progress is not None:
            self.progress(
                self.run_id,
                {
                    "name": self.cfg.name,
                    "seed": self.cfg.seed,
                    "period": p,
                    "periods": self.cfg.periods,
                    "delta": delta,
                    "delta_ma20": self.metrics.value("delta_ma20"),
                    "calls": self.llm_totals["calls"],
                    "cost_usd": self.llm_totals["cost_usd"],
                    "status": "running",
                },
            )

    def _book_rows(self, p: int) -> None:
        levels = self.market.snapshot(self.live.logging.snapshot_levels)
        bids = sorted([lv for lv in levels if lv[1] > 0], key=lambda x: -x[0])
        asks = sorted([lv for lv in levels if lv[2] > 0], key=lambda x: x[0])
        for side, rows, col in ((Side.BUY, bids, 1), (Side.SELL, asks, 2)):
            for level, lv in enumerate(rows):
                n_tagged = sum(
                    1
                    for s in self.slots
                    if s.active
                    and _finite(s.live_bid if side == Side.BUY else s.live_ask)
                    and abs((s.live_bid if side == Side.BUY else s.live_ask) - lv[0]) < 1e-9
                )
                self.buffers.add(
                    "book_snapshots",
                    {
                        "run_id": self.run_id,
                        "tick": int(self.market.tick),
                        "period": p,
                        "side": int(side),
                        "level": level,
                        "price": lv[0],
                        "depth": lv[col],
                        "n_tagged": n_tagged,
                    },
                )

    # =========================================================================================
    # manifest
    # =========================================================================================
    def _slots_info(self) -> list[dict[str, Any]]:
        out = []
        for s in self.slots:
            a = s.base_agent
            sc = s.spec.get("slot_cfg") or {}
            out.append(
                {
                    "slot": s.slot,
                    "agent_id": s.agent_id,
                    "name": a.name,
                    "kind": AgentKind(a.kind).name,
                    "spec_type": s.spec.get("type"),
                    "policy": s.spec.get("policy") or getattr(a, "policy", None),
                    "model": sc.get("model") or getattr(a, "model", None),
                    "family": sc.get("family"),
                    "lane": s.spec.get("lane") or getattr(a, "lane", None) or self.live.llm.lane,
                    "prompt_version": sc.get("prompt") or getattr(a, "prompt_version", None),
                    "active": s.active,
                }
            )
        return out

    def _progress_fields(self) -> dict[str, Any]:
        wall = self._wall_prev + (time.perf_counter() - self._t0)
        return {
            "periods_completed": self.period,
            "wall_time_s": round(wall, 3),
            "llm": dict(self.llm_totals),
            "cost_usd": float(self.llm_totals["cost_usd"]),
            "slots": self._slots_info(),
            "online": {k: (v if _finite(v) else None) for k, v in self.metrics.latest().items()},
            "benchmarks": self.benchmarks.to_dict() if self.benchmarks else None,
            "checkpoints": [
                {"period": c.period, "names": c.names} for c in ck.list_checkpoints(self.run_dir)
            ],
        }

    def _new_manifest(self, prev: dict[str, Any], resumed_from: int | None) -> dict[str, Any]:
        resumes = list(prev.get("resumes", [])) if resumed_from is not None else []
        if resumed_from is not None:
            resumes.append({"at": utc_now(), "from_period": resumed_from})
        bev = branch_events_of(self.cfg)
        arm, label = self._arm_label()
        m = {
            "schema_version": 1,
            "run_id": self.run_id,
            "name": self.cfg.name,
            "seed": self.cfg.seed,
            # keys read by finswarm.analysis / scripts/verify.py
            "alpha": float(self.cfg.market.alpha),
            "arm": arm,
            "trunk_id": self.cfg.parent_run_id or self.run_id,
            "condition": self.cfg.name,
            "label": label,
            "config_hash": config_hash(self.cfg),
            "resolved_config": self.cfg.model_dump(mode="json"),
            "git": git_info(),
            "seeds": seeds_info(self.cfg.seed, self.salt),
            "packages": package_versions(),
            "environment": environment(),
            "impl": {
                "market": self.market_impl,
                "view": self.view_impl,
                "llm_agents": self.llm_impl,
            },
            "provider": self.provider,
            "llm_mode": self.cfg.llm.mode,
            "periods": self.cfg.periods,
            "started_at": prev.get("started_at") if resumed_from is not None else utc_now(),
            "finished_at": None,
            "resumes": resumes,
            "parent_run_id": self.cfg.parent_run_id,
            "branch_period": self.cfg.branch_period,
            "branch_events_hash": events_hash(bev) if self.cfg.branch_period is not None else None,
            "branch_events": [e.model_dump(mode="json") for e in bev],
            "status": "running",
            "error": None,
            **self.extra_manifest,
        }
        m.update(self._progress_fields())
        return m


# =============================================================================================
# public API
# =============================================================================================
async def run_config_async(
    cfg: RunConfig,
    *,
    out: str | Path = Path("results/runs"),
    provider: str | None = None,
    run_ghosts: bool = True,
    **kw: Any,
) -> RunResult:
    runner = Runner(cfg, out=out, provider=provider, **kw)
    result = await runner.run()
    if run_ghosts and result.status == "ok" and runner.cfg.ghost.enabled:
        from finswarm.sim.ghost import run_ghosts_async

        await run_ghosts_async(
            result.run_dir,
            rival_policy=runner.cfg.ghost.rival_policy,
            provider=provider,
            market_factory=kw.get("market_factory"),
            llm_factory=kw.get("llm_factory"),
        )
    return result


def run_config(cfg: RunConfig, **kw: Any) -> RunResult:
    """Run (or resume, or skip if already ``ok``) one configuration synchronously."""
    return asyncio.run(run_config_async(cfg, **kw))


def summarize(result: RunResult) -> str:
    on = result.manifest.get("online", {}) or {}
    d = on.get("delta_ma20")
    dtxt = f"{d:.3f}" if isinstance(d, int | float) else "n/a"
    return (
        f"{result.run_id}  status={result.status}  periods={result.periods_completed}  "
        f"delta_ma20={dtxt}  cost_usd={result.manifest.get('cost_usd', 0.0):.4f}  "
        f"dir={result.run_dir}"
    )


def run_from_path(
    config: str | Path, provider: str | None = None, out: str | Path = Path("results/runs")
) -> RunResult | list[RunResult]:
    """CLI entry point. A YAML with a ``base`` key is a campaign (grid x seeds); else a run."""
    data = load_yaml(config)
    if "base" in data:
        from finswarm.sim.campaign import check_prereg, run_campaign

        cc = CampaignConfig.model_validate(data)
        check_prereg(cc)
        results = run_campaign(
            cc,
            provider=provider,
            out=out,
            dashboard=False,
            mirror_dir=Path(out).parent / "manifests",
        )
        for r in results:
            print(summarize(r))
        return results
    res = run_config(
        RunConfig.model_validate(data),
        out=out,
        provider=provider,
        mirror_dir=Path(out).parent / "manifests",
    )
    print(summarize(res))
    return res
