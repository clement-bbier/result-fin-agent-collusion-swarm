"""Perturbation engine: decide which events fire at a period boundary and apply them.

Semantics (0-indexed periods):

* A boundary ``B`` sits after period ``B - 1`` was matched and accounted and before the
  observations of period ``B``. Events apply only at boundaries.
* ``when.at_period = P`` fires at boundary ``P`` (in force from period ``P`` on);
  ``when.at_tick = T`` fires at the first boundary with ``market.tick >= T``;
  ``when.on_condition`` fires at the first boundary where the condition holds over the online
  metrics of completed periods. Several trigger fields combine with AND. An event with no
  trigger fires at boundary 0 (config) or at the branch period (branch events). Each event
  fires at most once; ``revert`` undoes a fired event by its index in ``RunConfig.events``.
* At a boundary, config events (``source != "branch"``) are processed first. Then, in a branch
  run at ``B == branch_period``, the ``interventions`` RNG stream is re-spawned with
  ``(seed, branch_salt)`` (``branch_salt`` hashes ``(parent_run_id, branch_period)`` only) and
  the branch events are processed, each on its own child stream ``(seed, branch_salt, ordinal
  among branch events of its type, type hash)`` with the shared stream put back afterwards
  (common random numbers between a lever branch and its sham). The engine remembers which
  phases of which boundary are done, so a run restored from a checkpoint never re-applies.
* Every application writes one ``events`` row and returns a checkpoint name
  (``event.note`` if set, else ``event<id>_<type>``).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np

from finswarm.core.rng import RNGRegistry
from finswarm.perturb.conditions import Condition, MetricSource, parse_condition
from finswarm.perturb.events import (
    AddScriptedAgent,
    Event,
    ForcedDeviation,
    InfoShock,
    InjectNotice,
    RemoveLiquidity,
    ReplaceAgent,
    Revert,
    SetParam,
    Sting,
)


class SimContext(Protocol):
    """What the engine needs from the runner (implemented by ``finswarm.sim.run.Runner``)."""

    rngs: RNGRegistry
    metrics: MetricSource

    def n_slots(self) -> int: ...

    def set_param(self, path: str, value: Any) -> Any: ...

    def spec_for_model(self, slot: int, model: str, lane: str, prompt: str | None) -> dict: ...

    def snapshot_agent(self, slot: int) -> dict: ...

    def replace_agent(
        self, slot: int, spec: dict, *, keep_memory: bool, keep_name: bool
    ) -> None: ...

    def restore_agent(self, slot: int, snapshot: dict) -> None: ...

    def add_agent(self, spec: dict) -> int: ...

    def deactivate_agent(self, slot: int) -> None: ...

    def wrap_forced(self, slot: int, delta: float, periods: int, event_id: int) -> None: ...

    def unwrap(self, slot: int) -> None: ...

    def add_notice(
        self, key: int, slots: list[int] | None, text: str, until: int | None
    ) -> None: ...

    def remove_notice(self, key: int) -> None: ...

    def market_call(self, hook: str, **kwargs: Any) -> bool: ...

    def log_event(self, event_id: int, event_type: str, payload: dict, source: str) -> None: ...


def checkpoint_name(event_id: int, ev: Event) -> str:
    return ev.note.strip() if ev.note and ev.note.strip() else f"event{event_id}_{ev.type}"


def events_hash(events: Sequence[Event]) -> str:
    import hashlib

    blob = json.dumps(
        [e.model_dump(mode="json") for e in events], sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


class PerturbationEngine:
    def __init__(
        self,
        events: Sequence[Event],
        *,
        branch_period: int | None = None,
        branch_salt: int | None = None,
    ) -> None:
        self.events: list[Event] = list(events)
        self.branch_period = branch_period
        self.branch_salt = branch_salt
        self.conditions: dict[int, Condition] = {
            i: parse_condition(ev.when.on_condition)
            for i, ev in enumerate(self.events)
            if ev.when.on_condition
        }
        self.fired: dict[int, int] = {}  # event id -> boundary where it fired
        self.undo: dict[int, dict[str, Any]] = {}
        self.done_through: dict[str, int] = {"config": -1, "branch": -1}
        self.branch_respawned = False
        for i, ev in enumerate(self.events):
            if isinstance(ev, Revert) and not 0 <= ev.event_id < len(self.events):
                raise ValueError(f"event {i}: revert of unknown event_id {ev.event_id}")

    # ---------------------------------------------------------------------------------
    def validate(self, metrics: MetricSource) -> None:
        from finswarm.perturb.conditions import validate_condition

        for cond in self.conditions.values():
            validate_condition(cond, metrics)

    def _due(self, i: int, ev: Event, boundary: int, tick: int, metrics: MetricSource) -> bool:
        if i in self.fired:
            return False
        w = ev.when
        has_trigger = False
        if w.at_period is not None:
            has_trigger = True
            if boundary < w.at_period:
                return False
        if w.at_tick is not None:
            has_trigger = True
            if tick < w.at_tick:
                return False
        if w.on_condition:
            has_trigger = True
            if not self.conditions[i].holds(metrics):
                return False
        if not has_trigger:
            default = self.branch_period if ev.source == "branch" else 0
            return boundary >= (default or 0)
        return True

    def boundary(self, ctx: SimContext, boundary: int, tick: int) -> list[str]:
        """Process boundary ``boundary``; return names for the checkpoint written there."""
        names: list[str] = []
        if self.done_through["config"] < boundary:
            for i, ev in enumerate(self.events):
                if ev.source != "branch" and self._due(i, ev, boundary, tick, ctx.metrics):
                    names.append(self._fire(ctx, i, ev, boundary, tick))
            self.done_through["config"] = boundary
        bp = self.branch_period
        if bp is not None and boundary >= bp and self.done_through["branch"] < boundary:
            if not self.branch_respawned:
                ctx.rngs.respawn("interventions", int(self.branch_salt or 0))
                self.branch_respawned = True
            for i, ev in enumerate(self.events):
                if ev.source == "branch" and self._due(i, ev, boundary, tick, ctx.metrics):
                    names.append(self._fire_branch(ctx, i, ev, boundary, tick))
            self.done_through["branch"] = boundary
        return names

    def branch_stream_key(self, i: int) -> tuple[int, int]:
        """``(ordinal, type code)`` of branch event ``i``: its rank among the branch events of
        the same type and a stable hash of the type. A lever branch and its sham that share an
        event (e.g. the same post-T0 ``info_shock``) give it the same key whatever other
        (lever-specific) events they carry."""
        ev = self.events[i]
        ordinal = sum(1 for e in self.events[:i] if e.source == "branch" and e.type == ev.type)
        code = int(hashlib.sha256(str(ev.type).encode()).hexdigest()[:12], 16)
        return ordinal, code

    def _fire_branch(self, ctx: SimContext, i: int, ev: Event, boundary: int, tick: int) -> str:
        """Fire branch event ``i`` on its own ``interventions`` child stream, seeded with
        ``(seed, branch_salt, ordinal, type code)``; the shared stream is put back afterwards,
        so a lever-specific event never shifts the draws of the events it shares with its
        sham (common random numbers)."""
        base = ctx.rngs.capture()["interventions"]
        salts = [int(ctx.rngs.seed), int(self.branch_salt or 0), *self.branch_stream_key(i)]
        child = np.random.PCG64(np.random.SeedSequence(salts))  # = respawn(name, *salts[1:])
        ctx.rngs.restore({"interventions": child.state})
        try:
            return self._fire(ctx, i, ev, boundary, tick)
        finally:
            ctx.rngs.restore({"interventions": base})

    # ---------------------------------------------------------------------------------
    def _fire(self, ctx: SimContext, i: int, ev: Event, boundary: int, tick: int) -> str:
        applied = self.apply(ctx, i, ev, boundary)
        self.fired[i] = boundary
        payload = {"event": ev.model_dump(mode="json"), "applied": applied}
        source = "condition" if (ev.when.on_condition and ev.source == "config") else ev.source
        ctx.log_event(i, ev.type, payload, source)
        return checkpoint_name(i, ev)

    def apply(self, ctx: SimContext, i: int, ev: Event, boundary: int) -> dict[str, Any]:
        if isinstance(ev, SetParam):
            old = ctx.set_param(ev.path, ev.value)
            self.undo[i] = {"type": "set_param", "path": ev.path, "old": old}
            return {"old": old, "new": ev.value}
        if isinstance(ev, ReplaceAgent):
            snap = ctx.snapshot_agent(ev.slot)
            spec = ctx.spec_for_model(ev.slot, ev.model, ev.lane, ev.prompt)
            ctx.replace_agent(ev.slot, spec, keep_memory=ev.keep_memory, keep_name=ev.keep_name)
            self.undo[i] = {"type": "agent", "slot": ev.slot, "snapshot": snap}
            return {"spec": spec}
        if isinstance(ev, InjectNotice):
            until = None if ev.persist_periods < 0 else boundary + ev.persist_periods
            ctx.add_notice(i, ev.slots, ev.template, until)
            self.undo[i] = {"type": "notice"}
            return {"until": until}
        if isinstance(ev, InfoShock):
            ok = ctx.market_call(
                "info_shock", d_fundamental=ev.d_fundamental, diffusion=ev.diffusion
            )
            self.undo[i] = {"type": "info_shock", "d": ev.d_fundamental}
            return {"supported": ok}
        if isinstance(ev, RemoveLiquidity):
            ok = ctx.market_call(
                "remove_liquidity", share=ev.share, swarm_types=list(ev.swarm_types)
            )
            self.undo[i] = {"type": "remove_liquidity"}
            return {"supported": ok}
        if isinstance(ev, AddScriptedAgent):
            slot = ctx.add_agent({"type": "scripted", "policy": ev.policy, "params": ev.params})
            self.undo[i] = {"type": "added", "slot": slot}
            return {"slot": slot}
        if isinstance(ev, Sting):
            spec = {
                "type": "scripted",
                "policy": "sting_dealer",
                "params": {
                    "delta": ev.delta,
                    "hold_epochs": ev.hold_epochs,
                    "shadow_epochs": ev.shadow_epochs,
                    "probes": ev.probes,
                    "event_id": i,
                },
            }
            if 0 <= ev.slot < ctx.n_slots():
                snap = ctx.snapshot_agent(ev.slot)
                ctx.replace_agent(ev.slot, spec, keep_memory=False, keep_name=False)
                self.undo[i] = {"type": "agent", "slot": ev.slot, "snapshot": snap}
                return {"slot": ev.slot, "mode": "replace"}
            slot = ctx.add_agent(spec)
            self.undo[i] = {"type": "added", "slot": slot}
            return {"slot": slot, "mode": "add"}
        if isinstance(ev, ForcedDeviation):
            ctx.wrap_forced(ev.slot, ev.delta, ev.periods, i)
            self.undo[i] = {"type": "forced", "slot": ev.slot}
            return {"slot": ev.slot}
        if isinstance(ev, Revert):
            return self._revert(ctx, ev.event_id)
        raise TypeError(f"unsupported event {ev!r}")  # pragma: no cover

    def _revert(self, ctx: SimContext, target: int) -> dict[str, Any]:
        u = self.undo.pop(target, None)
        if u is None:
            return {"reverted": target, "status": "not_fired_or_already_reverted"}
        kind = u["type"]
        if kind == "set_param":
            ctx.set_param(u["path"], u["old"])
        elif kind == "agent":
            ctx.restore_agent(u["slot"], u["snapshot"])
        elif kind == "notice":
            ctx.remove_notice(target)
        elif kind == "info_shock":
            ctx.market_call("info_shock", d_fundamental=-u["d"], diffusion="instant")
        elif kind == "remove_liquidity":
            ctx.market_call("restore_liquidity")
        elif kind == "added":
            ctx.deactivate_agent(u["slot"])
        elif kind == "forced":
            ctx.unwrap(u["slot"])
        return {"reverted": target, "status": "ok", "kind": kind}

    # ---------------------------------------------------------------------------------
    def state(self) -> dict[str, Any]:
        return {
            "fired": dict(self.fired),
            "undo": dict(self.undo),
            "done_through": dict(self.done_through),
            "branch_respawned": self.branch_respawned,
        }

    def restore(self, st: dict[str, Any], *, adopt_branch: bool = True) -> None:
        self.fired = {int(k): int(v) for k, v in st["fired"].items()}
        self.undo = dict(st["undo"])
        self.done_through = dict(st["done_through"])
        if adopt_branch:
            self.branch_respawned = bool(st["branch_respawned"])
        else:  # restoring a parent checkpoint into a branch child
            self.branch_respawned = False
            self.done_through["branch"] = -1
