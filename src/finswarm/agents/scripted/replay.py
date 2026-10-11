"""Open-loop replay of a dealer's logged quotes (ghost-market rivals).

A ``ReplayRival`` re-posts, period by period and epoch by epoch, the quotes another run logged
for one slot. It never reacts to the live market: in a ghost copy where only slot ``k`` is the
LLM, any dependence of ``k`` on its rivals that survives is not strategic interaction.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from finswarm.agents.scripted.base import ScriptedAgent
from finswarm.core.types import Decision, Observation

Schedule = dict[int, list[tuple[int, float, float]]]  # period -> [(epoch, bid, ask)]


def load_schedule(source: str | Path, slot: int) -> Schedule:
    """Read the ``quotes`` table of ``source`` run for one slot (rows with a decision only)."""
    import polars as pl

    from finswarm.store.reader import load_run

    df = (
        load_run(source)
        .tables["quotes"]
        .filter((pl.col("slot") == slot) & (pl.col("decision_id") >= 0))
        .sort(["period", "epoch"])
        .select(["period", "epoch", "bid", "ask"])
        .collect()
    )
    sched: Schedule = {}
    for period, epoch, bid, ask in df.iter_rows():
        sched.setdefault(int(period), []).append((int(epoch), float(bid), float(ask)))
    return sched


class ReplayRival(ScriptedAgent):
    policy = "replay_rival"

    def __init__(
        self,
        agent_id: int,
        slot: int,
        *,
        schedule: Schedule,
        source: str = "",
        source_slot: int | None = None,
        **kw: Any,
    ) -> None:
        super().__init__(agent_id, slot, **kw)
        self.schedule = {
            int(k): [(int(e), float(b), float(a)) for e, b, a in v] for k, v in schedule.items()
        }
        self.source = source
        self.source_slot = slot if source_slot is None else int(source_slot)
        self._period = -1
        self._k = 0

    def act(self, obs: Observation) -> Decision:
        if obs.period != self._period:
            self._period, self._k = obs.period, 0
        rows = self.schedule.get(obs.period)
        self._k += 1
        if not rows:
            return self.hold("replay:missing")
        epoch = obs.extra.get("epoch")
        if epoch is not None:  # same global epoch numbering as the source run
            match = [r for r in rows if r[0] == int(epoch)]
            if not match:  # the source agent did not decide at this epoch: keep the quote
                return self.hold("replay:no_decision")
            _, bid, ask = match[0]
        else:
            _, bid, ask = rows[min(self._k - 1, len(rows) - 1)]
        bid_v = bid if math.isfinite(bid) else None
        ask_v = ask if math.isfinite(ask) else None
        if bid_v is None and ask_v is None:
            # The source logged a decision with no price on either side: it withdrew its quote
            # (a cancel, or a hold while unquoted). Replaying it as a hold would keep this
            # rival's previous quote in force, so the ghost would see a quote the sham never had.
            return Decision(action="cancel", plans="replay:cancel")
        price = ask_v if ask_v is not None else bid_v
        return Decision(action="quote", price=price, bid=bid_v, ask=ask_v, plans="replay")

    def _extra_state(self) -> dict[str, Any]:
        return {"period": self._period, "k": self._k}

    def _restore_extra(self, st: dict[str, Any]) -> None:
        self._period = int(st["period"])
        self._k = int(st["k"])
