"""Decision clock of the tagged agents: when each slot decides, and when its orders apply.

One implementation shared by the runner (``finswarm.sim.run``) and the perturbation engine
(``set_param clock.*`` goes through :meth:`EpochClock.reconfigure`). Levers L5 (staggered
dealer updates) and L7 (speed bump) of the pre-registration are configured here.

Modes (``ClockConfig.mode``, ``E = epoch_ticks``):

``sync``
    Every slot decides at the same ticks ``start, start + E, ...``. When the runner passes the
    period end (``next_batch(end_tick)``) the grid re-aligns at every period start, so a period
    of ``T`` ticks has ``ceil(T / E)`` decision epochs, the last one possibly shorter.
``poisson``
    Each slot revises independently: after a decision at tick ``t`` its next decision tick is
    ``t + G`` with ``G = 1 + Geometric(1/E)`` on ``{1, 2, ...}`` (numpy's ``geometric(1/E)``:
    mean ``E``), drawn from the ``matching`` stream. Equivalently every tick is a revision tick
    of a slot with probability ``1/E``, independently across slots, so dealers revise at
    staggered times; the mean revision interval is ``E``. With ``E = 1`` both modes coincide
    (no draw is made).

A fresh clock (and a slot added mid-run) decides at its start tick: initial quotes are posted
at once, revisions are staggered afterwards.

``staleness_window`` ``w``: a decision *batch* starts at the earliest pending decision tick
``t``; every slot whose decision tick falls in ``[t, t + w)`` (cut at the period end) decides
in the same batch, concurrently, on observations as of ``t``. Each keeps its own decision tick.
Slots deciding at the same tick always share their observation (simultaneous moves), so
``w = 0`` and ``w = 1`` are the same: strictly sequential across ticks.

``speed_bump_max`` ``D``: a first-in-first-out speed bump. Every order of a decision taken at
``decision_tick`` applies at ``apply_tick = decision_tick + D`` (a constant delay, the same for
every slot; no draw). Optional per-slot jitter ``J`` (:meth:`EpochClock.set_jitter`; the runner
reads ``tags.speed_bump_jitter``): slot ``s`` gets a constant extra latency ``j_s ~ U{0..J}``,
drawn once from the ``matching`` stream (slot order) when the jitter is set or the slot joins,
so its delay ``D + j_s`` is constant and its orders keep their order. In every case
``apply_tick`` is also floored at the slot's previous ``apply_tick``: a later decision of a slot
never applies before an earlier one, even after ``D`` or ``J`` is lowered mid-run. Orders never
apply before their decision tick.

Draw order: the next-decision gaps of a batch's slots (slot order) when the batch is popped;
jitter offsets when the jitter is set or a slot joins. The speed bump itself never draws, so
the ``matching`` stream does not depend on what agents decide.

Changes (``reconfigure``) apply at a period boundary tick: a change of mode or ``epoch_ticks``
re-draws the schedule from that tick (``sync``: every slot decides at the boundary;
``poisson``: memoryless restart, next decision at ``tick + G - 1``); a change of ``D`` or ``w``
leaves the schedule untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from finswarm.config.schema import ClockConfig
from finswarm.core.rng import RNGRegistry

#: clock parameters a ``set_param`` event may change live (applied at a period boundary);
#: ``speed_bump_jitter`` is settable once ``ClockConfig`` carries the field (until then the
#: runner reads the ``tags.speed_bump_jitter`` tag and this entry is rejected by the schema)
CLOCK_PARAMS: tuple[str, ...] = (
    "mode",
    "epoch_ticks",
    "speed_bump_max",
    "staleness_window",
    "speed_bump_jitter",
)


def jitter_of(cfg: ClockConfig) -> int:
    """``cfg.speed_bump_jitter`` (0 when the frozen schema has no such field yet)."""
    return int(getattr(cfg, "speed_bump_jitter", 0) or 0)


@dataclass(frozen=True, slots=True)
class Epoch:
    """One decision batch: slots that decide on the same observation (taken at ``tick``)."""

    index: int
    tick: int
    slots: tuple[int, ...]
    decision_ticks: tuple[int, ...] = ()

    def decision_tick(self, slot: int) -> int:
        if not self.decision_ticks:
            return self.tick
        return self.decision_ticks[self.slots.index(slot)]


def check_clock_config(cfg: ClockConfig) -> None:
    if cfg.mode not in ("sync", "poisson"):
        raise ValueError(f"unknown clock mode {cfg.mode!r}")
    if cfg.epoch_ticks < 1:
        raise ValueError("clock.epoch_ticks must be >= 1")
    if cfg.speed_bump_max < 0 or cfg.staleness_window < 0:
        raise ValueError("clock.speed_bump_max and clock.staleness_window must be >= 0")
    if jitter_of(cfg) < 0:
        raise ValueError("clock.speed_bump_jitter must be >= 0")


class EpochClock:
    """Per-slot decision schedule, batching window and speed bump (see the module docstring).

    Building a clock never draws from the RNG (restore order of checkpoints); ``state()`` /
    ``restore()`` round-trip exactly.
    """

    def __init__(
        self,
        cfg: ClockConfig,
        n_slots: int,
        rng: RNGRegistry | np.random.Generator,
        start_tick: int = 0,
    ) -> None:
        check_clock_config(cfg)
        self.cfg = cfg
        self._gen = rng["matching"] if isinstance(rng, RNGRegistry) else rng
        self.epoch_index = 0
        self.sync_next = int(start_tick)
        self.next_ticks = np.full(int(n_slots), int(start_tick), dtype=np.int64)
        self.jitter_max = 0
        self.jitter = np.zeros(int(n_slots), dtype=np.int64)  # per-slot extra latency
        self.last_apply = np.full(int(n_slots), np.iinfo(np.int64).min, dtype=np.int64)

    # ------------------------------------------------------------------ properties
    @property
    def mode(self) -> str:
        return self.cfg.mode

    @property
    def n_slots(self) -> int:
        return int(self.next_ticks.shape[0])

    # ------------------------------------------------------------------ draws
    def _gaps(self, k: int) -> np.ndarray:
        p = 1.0 / self.cfg.epoch_ticks
        if p >= 1.0 or k == 0:
            return np.ones(k, dtype=np.int64)
        return self._gen.geometric(p, size=k).astype(np.int64)

    def delay(self, slot: int | None = None) -> int:
        """Speed bump of ``slot``: ``speed_bump_max`` plus the slot's jitter offset."""
        d = int(self.cfg.speed_bump_max)
        if slot is not None and 0 <= slot < self.jitter.shape[0]:
            d += int(self.jitter[slot])
        return d

    def apply_tick(self, decision_tick: int, slot: int | None = None) -> int:
        """Tick at which the orders of a decision taken at ``decision_tick`` take effect.

        ``decision_tick + delay(slot)``, floored at the slot's previous apply tick (FIFO).
        """
        at = int(decision_tick) + self.delay(slot)
        if slot is not None and 0 <= slot < self.last_apply.shape[0]:
            at = max(at, int(self.last_apply[slot]))
            self.last_apply[slot] = at
        return at

    def set_jitter(self, jitter_max: int) -> None:
        """Per-slot constant extra latency ``U{0..jitter_max}`` (re-drawn for every slot)."""
        j = int(jitter_max)
        if j < 0:
            raise ValueError("speed-bump jitter must be >= 0")
        self.jitter_max = j
        self.jitter = self._jitter_draws(self.n_slots)

    def _jitter_draws(self, k: int) -> np.ndarray:
        if self.jitter_max <= 0 or k == 0:
            return np.zeros(k, dtype=np.int64)
        return self._gen.integers(0, self.jitter_max + 1, size=k).astype(np.int64)

    # ------------------------------------------------------------------ schedule
    def peek(self) -> int | None:
        """Tick of the next decision batch (``None`` when there is none)."""
        if self.mode == "sync":
            return self.sync_next
        return int(self.next_ticks.min()) if self.n_slots else None

    def next_batch(self, end_tick: int | None = None) -> Epoch | None:
        """Pop the next decision batch strictly before ``end_tick`` (the period end).

        ``sync``: every slot, at the grid tick; the grid then moves to ``min(t + E, end_tick)``.
        ``poisson``: every slot whose decision tick is in ``[t, t + max(w, 1))`` (cut at
        ``end_tick``); each is rescheduled at ``decision_tick + G``.
        """
        t = self.peek()
        if t is None or (end_tick is not None and t >= end_tick):
            return None
        if self.mode == "sync":
            nxt = t + self.cfg.epoch_ticks
            if end_tick is not None:
                nxt = min(nxt, int(end_tick))
            slots = tuple(range(self.n_slots))
            dticks = (t,) * self.n_slots
            self.sync_next = nxt
            self.next_ticks[:] = nxt
        else:
            hi = t + max(1, self.cfg.staleness_window)
            if end_tick is not None:
                hi = min(hi, int(end_tick))
            idx = np.flatnonzero(self.next_ticks < hi)
            d = self.next_ticks[idx].copy()
            self.next_ticks[idx] = d + self._gaps(idx.size)
            slots = tuple(int(i) for i in idx)
            dticks = tuple(int(x) for x in d)
        ep = Epoch(self.epoch_index, int(t), slots, dticks)
        self.epoch_index += 1
        return ep

    def next_epoch(self) -> Epoch:
        """Pop the next decision batch (no period end)."""
        ep = self.next_batch()
        if ep is None:
            raise RuntimeError("clock has no slots")
        return ep

    def epochs_until(self, tick: int) -> list[Epoch]:
        """All batches with ``epoch.tick < tick`` (in order, no period alignment)."""
        out: list[Epoch] = []
        while True:
            t = self.peek()
            if t is None or t >= tick or (self.mode == "sync" and not self.n_slots):
                return out
            out.append(self.next_epoch())

    def add_slot(self, tick: int) -> int:
        """A new slot (entrant) that decides at ``tick`` (the boundary where it joins)."""
        first = self.sync_next if self.mode == "sync" else int(tick)
        self.next_ticks = np.append(self.next_ticks, np.int64(first))
        self.jitter = np.append(self.jitter, self._jitter_draws(1))
        self.last_apply = np.append(self.last_apply, np.int64(np.iinfo(np.int64).min))
        return self.n_slots - 1

    def reconfigure(self, cfg: ClockConfig, tick: int) -> None:
        """Adopt ``cfg`` from boundary tick ``tick`` on (see the module docstring)."""
        check_clock_config(cfg)
        old, self.cfg = self.cfg, cfg
        if jitter_of(cfg) != jitter_of(old):  # the field, when the schema has it; else the tag
            self.set_jitter(jitter_of(cfg))
        if old.mode == cfg.mode and old.epoch_ticks == cfg.epoch_ticks:
            return  # speed bump / staleness / jitter only: the schedule is unchanged
        tick = int(tick)
        self.sync_next = tick
        if cfg.mode == "sync":
            self.next_ticks[:] = tick
        else:  # memoryless restart: every tick from ``tick`` on is a revision w.p. 1/E
            self.next_ticks = tick + self._gaps(self.n_slots) - 1

    # ------------------------------------------------------------------ state
    def state(self) -> dict[str, Any]:
        return {
            "cfg": self.cfg.model_dump(mode="json"),
            "epoch_index": self.epoch_index,
            "sync_next": self.sync_next,
            "next_ticks": [int(x) for x in self.next_ticks],
            "jitter_max": self.jitter_max,
            "jitter": [int(x) for x in self.jitter],
            "last_apply": [int(x) for x in self.last_apply],
        }

    def restore(self, state: dict[str, Any]) -> None:
        if "cfg" in state:
            self.cfg = ClockConfig.model_validate(state["cfg"])
        elif state.get("mode") != self.mode:
            raise ValueError("clock mode mismatch on restore")
        self.epoch_index = int(state["epoch_index"])
        self.next_ticks = np.array(state["next_ticks"], dtype=np.int64)
        default = int(self.next_ticks.min()) if self.n_slots else 0
        self.sync_next = int(state.get("sync_next", default))
        n = self.n_slots
        self.jitter_max = int(state.get("jitter_max", 0))
        self.jitter = np.array(state.get("jitter", [0] * n), dtype=np.int64)
        lo = int(np.iinfo(np.int64).min)
        self.last_apply = np.array(state.get("last_apply", [lo] * n), dtype=np.int64)
