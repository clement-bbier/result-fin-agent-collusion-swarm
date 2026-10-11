"""Integration points between the runner and the other workstreams (one place, one contract).

* Market: :func:`real_market_factory` builds ``finswarm.market.registry.make_market(cfg, rng,
  swarm)``. In ``swarm.mode == "agents"`` the swarm population is drawn here from the
  ``population`` stream (``finswarm.swarm.build_population_for_run``) and handed to the market;
  on restore the runner passes ``SwarmState.from_state(...)`` instead. The market is built with
  ``n_dealers = max(market.n_dealers, len(llm.slots)) + ENTRANT_RESERVE`` so that entrants
  (``add_scripted_agent``, ``sting`` on a new slot) get a dealer seat; empty seats never quote
  and never receive flow. Mechanisms the registry does not implement, two-sided quotes and other
  unsupported requests raise :class:`UnsupportedIntervention`, which fails the run with a reason.
* Injected factories (test doubles) keep the two-argument signature ``factory(cfg, rngs)``.
* Market hooks used by events (see :func:`market_hook`): ``info_shock`` ->
  ``market.apply_info_shock(d_fundamental, diffusion)``; ``remove_liquidity`` ->
  ``finswarm.swarm.shocks.remove_liquidity(market.swarm, share, swarm_types, rng)``;
  ``restore_liquidity`` re-activates the agents removed by the reverted event; any other hook is
  looked up on the market, then on ``market.swarm``. A missing hook raises
  :class:`UnsupportedIntervention` (events never fail silently).
* Clock: ``set_param`` on :data:`CLOCK_SETTABLE` (``clock.mode``, ``clock.epoch_ticks``,
  ``clock.speed_bump_max``, ``clock.staleness_window``) -> :func:`apply_clock_param` ->
  ``EpochClock.reconfigure(cfg, boundary_tick)``; other ``clock.*`` paths or invalid values
  raise :class:`UnsupportedIntervention` (config events are checked before period 0).
  ``tags.speed_bump_jitter`` (:data:`JITTER_TAG`) sets the optional per-slot jitter.
* Live ``set_param`` on ``market.*`` paths: ``market.set_param(path, value)`` for paths in
  ``market.settable_params`` (the dealer market: routing, cost, demand, ``price_grid.tick``
  everywhere, ``fringe.share`` in ``agents`` mode), then :func:`after_market_param` brings the
  runner's own copies in line: after a re-grid (lever L3) the tagged agents' standing
  quotes, the public quote log and the last trades are re-expressed on the new grid (asks up,
  bids down, trades nearest) and the market's report (old/new tick, per-dealer re-quantised
  prices) is logged as a ``derived`` ``regrid`` event; a fringe switch (lever L2) is logged
  as a ``derived`` ``fringe_share`` event.
* View: ``finswarm.market.view.apply_view(obs, policy, rng, *, view=None) -> Observation``.
* LLM agents: ``finswarm.agents.llm.make_llm_agent(slot_cfg, *, slot, agent_id, run_cfg,
  run_id, rngs, provider, lane, mode) -> Agent``.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable
from functools import lru_cache
from typing import Any

from finswarm.config.schema import ClockConfig, LLMSlotConfig, RunConfig
from finswarm.core.clock import CLOCK_PARAMS, EpochClock, check_clock_config
from finswarm.core.rng import RNGRegistry
from finswarm.core.types import Observation

log = logging.getLogger(__name__)

MarketFactory = Callable[..., Any]
ViewFn = Callable[..., Observation]

#: spare dealer seats for agents added during a run (entry, sting on a new slot)
ENTRANT_RESERVE = 2


class UnsupportedIntervention(RuntimeError):
    """The configuration or an event asks for something the market cannot do.

    The runner marks the run ``failed`` with this message as the reason (never a silent no-op).
    """


# ---------------------------------------------------------------------------------- slots
@lru_cache(maxsize=1)
def _mock_llm_models() -> frozenset[str]:
    """Catalogue ids and aliases served by the ``mock`` provider (e.g. ``mock-v1``)."""
    try:
        from finswarm.llm.pricing import load_catalog

        cat = load_catalog()
    except (FileNotFoundError, ImportError, ValueError):  # pragma: no cover - no catalogue
        return frozenset({"mock-v1"})
    ids = {m for m, spec in cat.models.items() if cat.providers[spec.provider].kind == "mock"}
    return frozenset(ids | {a for a, m in cat.aliases.items() if m in ids})


def slot_is_scripted(sc: LLMSlotConfig) -> bool:
    """A ``mock``-family slot whose ``policy`` names a scripted strategy runs as a scripted agent
    unless its ``model`` is a mock LLM model (``mock-v1``): then it is a real ``LLMAgent`` whose
    answers come from ``MockLLM(policy)`` (prompt rendering, parsing and logging exercised)."""
    from finswarm.agents.scripted.registry import is_scripted_policy

    return (
        sc.family == "mock" and is_scripted_policy(sc.policy) and sc.model not in _mock_llm_models()
    )


# --------------------------------------------------------------------------------- market
def market_config(cfg: RunConfig) -> RunConfig:
    """The config handed to the market: dealer seats for every slot plus the entrant reserve."""
    seats = max(int(cfg.market.n_dealers), len(cfg.llm.slots)) + ENTRANT_RESERVE
    if seats == cfg.market.n_dealers:
        return cfg
    return cfg.model_copy(
        update={"market": cfg.market.model_copy(update={"n_dealers": seats})}, deep=True
    )


def real_market_factory(cfg: RunConfig, rngs: RNGRegistry, swarm: Any = None) -> Any:
    """``make_market(cfg, rng, swarm)``; draws the swarm (``population`` stream) if needed."""
    from finswarm.market.registry import make_market

    if cfg.market.two_sided:
        raise UnsupportedIntervention(
            "market.two_sided=true: two-sided dealer quotes are not implemented yet"
        )
    mcfg = market_config(cfg)
    if swarm is None and cfg.swarm.mode == "agents":
        from finswarm.swarm.population import build_population_for_run

        swarm = build_population_for_run(mcfg, rngs)
    try:
        return make_market(mcfg, rngs, swarm)
    except NotImplementedError as exc:
        raise UnsupportedIntervention(str(exc)) from exc


def resolve_market_factory(injected: MarketFactory | None) -> tuple[MarketFactory, str]:
    if injected is not None:
        return injected, "injected"
    return real_market_factory, "market.registry"


def build_market(factory: MarketFactory, cfg: RunConfig, rngs: RNGRegistry, swarm: Any) -> Any:
    """Call ``factory``; the swarm is passed only when there is one (restore in agents mode)."""
    if swarm is None:
        return factory(cfg, rngs)
    return factory(cfg, rngs, swarm)


def swarm_state_of(market: Any) -> dict[str, Any] | None:
    """``market.swarm.state()`` when the market carries a serialisable swarm, else ``None``."""
    sw = getattr(market, "swarm", None)
    fn = getattr(sw, "state", None)
    return fn() if callable(fn) else None


def swarm_from_state(state: dict[str, Any] | None) -> Any:
    if state is None:
        return None
    from finswarm.swarm.state import SwarmState

    return SwarmState.from_state(state)


def settable_paths(market: Any) -> tuple[str, ...] | None:
    """Config paths the market can change live, or ``None`` when it does not say."""
    paths = getattr(market, "settable_params", None)
    return tuple(paths) if paths is not None else None


# ---------------------------------------------------------------------------------- clock
#: ``set_param`` paths of the decision clock (``finswarm.core.clock``) settable live; a change
#: applies at the period boundary where the event fires (in force from that period on)
CLOCK_SETTABLE: tuple[str, ...] = tuple(f"clock.{k}" for k in CLOCK_PARAMS)


def check_clock_param(path: str, value: Any, current: ClockConfig) -> ClockConfig:
    """The clock config after ``set_param(path, value)`` on ``current``.

    Raises :class:`UnsupportedIntervention` when ``path`` is not a live clock parameter or the
    value is invalid (unknown mode, ``epoch_ticks < 1``, negative bump or window).
    """
    if path not in CLOCK_SETTABLE:
        raise UnsupportedIntervention(
            f"set_param {path!r}: not a live clock parameter (settable: {list(CLOCK_SETTABLE)})"
        )
    key = path.split(".", 1)[1]
    try:
        cfg = ClockConfig.model_validate({**current.model_dump(mode="json"), key: value})
        check_clock_config(cfg)
    except ValueError as exc:  # pydantic's ValidationError is a ValueError
        raise UnsupportedIntervention(f"set_param {path!r} = {value!r}: {exc}") from exc
    return cfg


#: ``RunConfig.tags`` key of the optional per-slot speed-bump jitter (ticks, ``ClockConfig`` has
#: no field for it): each slot gets a constant extra latency ``U{0..J}`` (order preserved)
JITTER_TAG = "speed_bump_jitter"


def speed_bump_jitter(value: Any) -> int:
    """The jitter ``J >= 0`` from its tag value (``None`` -> 0); else UnsupportedIntervention."""
    if value is None:
        return 0
    try:
        j = int(value)
    except (TypeError, ValueError):
        j = -1
    if j < 0:
        raise UnsupportedIntervention(f"tags.{JITTER_TAG} = {value!r}: an integer >= 0 expected")
    return j


def apply_clock_param(clock: EpochClock, cfg: ClockConfig, tick: int) -> None:
    """``set_param clock.*`` hook: the clock adopts ``cfg`` from boundary tick ``tick`` on."""
    clock.reconfigure(cfg, tick)


def after_market_param(runner: Any, path: str) -> dict[str, Any] | None:
    """Runner-side follow-up of a successful ``runner.market.set_param(path, value)``.

    Reads ``market.last_param_report`` (``None`` for plain parameter changes). After a re-grid
    the runner's tick-indexed copies are moved onto the new grid with the market's own rules
    (``finswarm.market.grid``): each slot's sent and in-force ``ask`` rounds up and ``bid``
    down, pending (speed-bumped) quotes take the market's new level or are dropped when the
    market withdrew them, the public quote log likewise, the trades not yet shown to every
    slot to the nearest level. The report is
    logged as one ``derived`` event row (``regrid`` / ``fringe_share``). Returns the report.
    """
    report = getattr(runner.market, "last_param_report", None)
    if not isinstance(report, dict) or report.get("path") != path:
        return None
    if report.get("kind") == "regrid":
        _sync_runner_after_regrid(runner, report)
    runner.log_event(-1, str(report.get("kind", "market_param")), report, "derived")
    return report


def _sync_runner_after_regrid(runner: Any, report: dict[str, Any]) -> None:
    import dataclasses

    import numpy as np

    from finswarm.market.grid import nearest_indices, regrid_indices, regrid_price

    old, new = float(report["old_tick"]), float(report["new_tick"])
    n_new = int(report["n_levels"][1])
    p_min = float(report["p_min"])
    from finswarm.core.types import Side

    # ``Side`` values (Side.SELL = -1, Side.BUY = 1; delayed cancels in ``pending_quotes``: 0)
    sell, buy = int(Side.SELL), int(Side.BUY)
    for s in runner.slots:
        s.ask = regrid_price(s.ask, p_min, old, new, n_new, up=True)
        s.bid = regrid_price(s.bid, p_min, old, new, n_new, up=False)
        if hasattr(s, "live_ask"):  # the quote in force, as the market re-quantised it
            s.live_ask = regrid_price(s.live_ask, p_min, old, new, n_new, up=True)
            s.live_bid = regrid_price(s.live_bid, p_min, old, new, n_new, up=False)
            if not np.isfinite(s.live_ask):  # withdrawn by the market
                s.live_ask_size = 0.0
                s.live_orders.pop(sell, None)
            if not np.isfinite(s.live_bid):
                s.live_bid_size = 0.0
                s.live_orders.pop(buy, None)
    if getattr(runner, "pending_quotes", None):
        # the market's own outcome per pending QUOTE (``new_ticks < 0``: withdrawn at re-grid)
        by_id = {int(x["order_id"]): int(x["new_ticks"]) for x in report.get("pending") or []}
        kept = []
        for apply_tick, oid, slot, side, price, size in runner.pending_quotes:
            if side in (sell, buy):
                k = by_id.get(int(oid))
                if k is None:  # not reported (test doubles): the same rule as the market
                    price = regrid_price(price, p_min, old, new, n_new, up=side == sell)
                elif k >= 0:
                    price = p_min + k * new
                if (k is not None and k < 0) or not np.isfinite(price):
                    continue
            kept.append((apply_tick, oid, slot, side, price, size))
        runner.pending_quotes = kept
    if runner.quote_log:
        ql = runner.quote_log
        bids = regrid_indices(np.array([q[2] for q in ql]), old, new, n_new, up=False)
        asks = regrid_indices(np.array([q[3] for q in ql]), old, new, n_new, up=True)
        runner.quote_log = [
            (q[0], q[1], int(b), int(a), q[4], q[5])
            for q, b, a in zip(ql, bids.tolist(), asks.tolist(), strict=True)
        ]
    if runner.last_trades:
        idx = nearest_indices(
            np.array([t.price_ticks for t in runner.last_trades]), old, new, n_new
        )
        runner.last_trades = [
            dataclasses.replace(t, price_ticks=int(k))
            for t, k in zip(runner.last_trades, idx.tolist(), strict=True)
        ]


# ---------------------------------------------------------------------------------- hooks
def _with_rng(fn: Callable[..., Any], kwargs: dict[str, Any], rng: Any) -> dict[str, Any]:
    try:
        if "rng" in inspect.signature(fn).parameters:
            return {**kwargs, "rng": rng}
    except (TypeError, ValueError):  # pragma: no cover
        pass
    return kwargs


def market_hook(market: Any, hook: str, rngs: RNGRegistry, removed: list[list[int]], **kw: Any):
    """Run event hook ``hook`` on ``market``; returns the hook's result.

    ``removed`` is the runner's (checkpointed) stack of swarm ids deactivated by
    ``remove_liquidity``; ``restore_liquidity`` pops it.
    """
    swarm = getattr(market, "swarm", None)
    real_swarm = swarm is not None and hasattr(swarm, "set_active") and hasattr(swarm, "stype")
    if hook == "info_shock" and hasattr(market, "apply_info_shock"):
        return market.apply_info_shock(float(kw["d_fundamental"]), kw.get("diffusion", "attention"))
    if hook == "remove_liquidity" and real_swarm:
        from finswarm.swarm.shocks import remove_liquidity

        ids = remove_liquidity(
            swarm, float(kw["share"]), list(kw["swarm_types"]), rngs["interventions"]
        )
        removed.append([int(i) for i in ids])
        return len(ids)
    if hook == "restore_liquidity" and real_swarm:
        import numpy as np

        ids = removed.pop() if removed else []
        mask = np.zeros(swarm.n, dtype=bool)
        mask[np.asarray(ids, dtype=np.int64)] = True
        swarm.set_active(mask, True)
        return len(ids)
    if hook in ("remove_liquidity", "restore_liquidity") and hasattr(market, "apply_info_shock"):
        raise UnsupportedIntervention(
            f"{hook}: the market has no background swarm (swarm.mode must be 'agents')"
        )
    fn = getattr(market, hook, None)
    if fn is None and swarm is not None:
        fn = getattr(swarm, hook, None)
    if fn is None:
        raise UnsupportedIntervention(
            f"event hook {hook!r} is not supported by market {type(market).__name__}"
        )
    return fn(**_with_rng(fn, kw, rngs["interventions"]))


# ----------------------------------------------------------------------------------- view
def identity_view(obs: Observation, policy: Any, rng: Any, **_: Any) -> Observation:
    return obs


def resolve_view_fn(injected: ViewFn | None = None) -> tuple[ViewFn, str]:
    if injected is not None:
        return injected, "injected"
    from finswarm.market.view import apply_view

    return apply_view, "market.view"


# ------------------------------------------------------------------------------------ llm
def resolve_llm_factory(injected: Callable[..., Any] | None = None) -> tuple[Any, str]:
    if injected is not None:
        return injected, "injected"
    from finswarm.agents.llm import make_llm_agent

    return make_llm_agent, "agents.llm"
