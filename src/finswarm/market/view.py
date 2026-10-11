"""Public-view policy: what a tagged agent sees, and a hash of exactly what it saw.

Transformations are applied in a fixed order (``ViewPolicy``):

1. **visibility**   ``show_rival_quotes``, ``show_trades``, ``show_flow``, ``history_periods``;
2. **delay**        only information stamped ``<= tick - delay_ticks`` is visible (the latest
                    rival quote and book snapshot at or before the cutoff);
3. **aggregate**    book levels merged into buckets of ``aggregate_levels`` ticks;
4. **truncate**     top ``depth_truncation`` levels per side;
5. **canonicalise** prices rounded to ``price_round`` and sizes to ``size_round`` when set;
                    with ``canonicalise=True`` lists are also sorted canonically and floats
                    normalised (12 significant digits, no negative zero);
6. **anonymise**    tagged ids replaced by stable pseudonyms (re-drawn every
                    ``reshuffle_every`` periods from the ``view`` stream); swarm ids become -1;
                    the viewer keeps its own id.

Private fields of trades (``buyer_value``, ``seller_cost``) are always blanked.

``view_hash`` is the sha256 of the canonical JSON rendering of the transformed parts.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from finswarm.config.schema import ViewPolicy
from finswarm.core.rng import RNGRegistry
from finswarm.core.types import Observation, PeriodRecord, Quote, Trade

Level = tuple[float, float, float]  # (price, bid_depth, ask_depth)
PSEUDONYM_LOW = 1000
PSEUDONYM_HIGH = 10000


@dataclass(slots=True)
class RawView:
    """Untransformed observation parts, time-stamped so that delays can be applied."""

    tick: int
    period: int
    viewer_id: int
    rival_quotes: list[tuple[int, Quote]] = field(default_factory=list)  # (posted tick, quote)
    trades: list[Trade] = field(default_factory=list)
    book: list[tuple[int, list[Level]]] = field(default_factory=list)  # (snapshot tick, levels)
    history: list[PeriodRecord] = field(default_factory=list)
    flow: list[tuple[int, float, float]] = field(default_factory=list)  # (tick, buy, sell)


@dataclass(slots=True)
class ViewParts:
    rival_quotes: list[Quote]
    trades: list[Trade]
    book_levels: list[Level]
    history: list[PeriodRecord]
    flow: list[tuple[int, float, float]]


@dataclass(frozen=True, slots=True)
class ViewResult:
    parts: ViewParts
    view_hash: str
    policy_id: str


def _round_to(x: float, step: float | None) -> float:
    if step is None or step <= 0 or not math.isfinite(x):
        return x
    return round(round(x / step) * step, 12)


def _norm(x: float) -> float:
    if not math.isfinite(x):
        return x
    y = float(f"{x:.12g}")
    return 0.0 if y == 0 else y


class PublicView:
    def __init__(
        self,
        policy: ViewPolicy,
        rng: RNGRegistry | np.random.Generator,
        *,
        p_min: float,
        tick_size: float,
        tagged_ids: Iterable[int] = (),
    ) -> None:
        self.policy = policy
        self._gen = rng["view"] if isinstance(rng, RNGRegistry) else rng
        self.p_min = float(p_min)
        self.tick_size = float(tick_size)
        self.tagged_ids: list[int] = sorted({int(i) for i in tagged_ids})
        self._window: int | None = None
        self._pseudo: dict[int, int] = {}

    # ---------------------------------------------------------------- pseudonyms
    def _draw_pseudonyms(self, ids: Sequence[int], taken: set[int]) -> dict[int, int]:
        avail = np.setdiff1d(
            np.arange(PSEUDONYM_LOW, PSEUDONYM_HIGH), np.array(sorted(taken), dtype=np.int64)
        )
        if len(ids) > avail.size:
            raise ValueError("too many tagged ids for the pseudonym space")
        pick = self._gen.choice(avail, size=len(ids), replace=False)
        return {int(i): int(p) for i, p in zip(ids, pick, strict=True)}

    def _ensure_window(self, period: int) -> None:
        k = self.policy.reshuffle_every
        window = period // k if k > 0 else 0
        if window != self._window:
            self._pseudo = self._draw_pseudonyms(self.tagged_ids, set())
            self._window = window

    def register(self, ids: Iterable[int]) -> None:
        """Add tagged ids (e.g. an added scripted agent); they get fresh pseudonyms."""
        new = sorted({int(i) for i in ids} - set(self.tagged_ids))
        if not new:
            return
        self.tagged_ids = sorted(set(self.tagged_ids) | set(new))
        if self._window is not None:
            self._pseudo.update(self._draw_pseudonyms(new, set(self._pseudo.values())))

    def pseudonym(self, agent_id: int, viewer_id: int) -> int:
        if agent_id == viewer_id:
            return agent_id
        return self._pseudo.get(agent_id, -1)

    # ------------------------------------------------------------------- price utils
    def _price(self, ticks: int) -> float:
        return self.p_min + ticks * self.tick_size

    def _round_ticks(self, ticks: int) -> int:
        step = self.policy.price_round
        if ticks < 0 or step is None or step <= 0:
            return ticks
        p = _round_to(self._price(ticks), step)
        return int(round((p - self.p_min) / self.tick_size))

    # ---------------------------------------------------------------------- apply
    def apply(self, raw: RawView) -> ViewResult:
        pol = self.policy
        cutoff = raw.tick - max(0, pol.delay_ticks)

        # 1-2. visibility and delay
        quotes: list[Quote] = []
        if pol.show_rival_quotes:
            latest: dict[int, tuple[int, int, Quote]] = {}
            for k, (t, q) in enumerate(raw.rival_quotes):
                if q.agent_id == raw.viewer_id or t > cutoff:
                    continue
                prev = latest.get(q.agent_id)
                if prev is None or (t, k) >= (prev[0], prev[1]):
                    latest[q.agent_id] = (t, k, q)
            quotes = [replace(v[2]) for _, v in sorted(latest.items())]
        # private valuations and costs never reach an observation
        nan = float("nan")
        trades = (
            [replace(t, buyer_value=nan, seller_cost=nan) for t in raw.trades if t.tick <= cutoff]
            if pol.show_trades
            else []
        )
        flow = [f for f in raw.flow if f[0] <= cutoff] if pol.show_flow else []
        levels: list[Level] = []
        snaps = [(t, lv) for t, lv in raw.book if t <= cutoff]
        if snaps:
            levels = list(max(snaps, key=lambda s: s[0])[1])
        hist = list(raw.history)[-pol.history_periods :] if pol.history_periods > 0 else []

        # 3. aggregate
        if pol.aggregate_levels > 0:
            width = pol.aggregate_levels * self.tick_size
            buckets: dict[int, list[float]] = {}
            for p, b, a in levels:
                idx = int(math.floor((p - self.p_min) / width + 1e-9))
                acc = buckets.setdefault(idx, [0.0, 0.0])
                acc[0] += b
                acc[1] += a
            levels = [(self.p_min + i * width, v[0], v[1]) for i, v in sorted(buckets.items())]

        # 4. truncate
        if pol.depth_truncation > 0:
            d = pol.depth_truncation
            bids = sorted((lv for lv in levels if lv[1] > 0), key=lambda x: -x[0])[:d]
            asks = sorted((lv for lv in levels if lv[2] > 0), key=lambda x: x[0])[:d]
            merged: dict[float, list[float]] = {}
            for p, b, _ in bids:
                merged.setdefault(p, [0.0, 0.0])[0] = b
            for p, _, a in asks:
                merged.setdefault(p, [0.0, 0.0])[1] = a
            levels = [(p, v[0], v[1]) for p, v in sorted(merged.items())]

        # 5. canonicalise
        pr, sr = pol.price_round, pol.size_round
        if pr is not None or sr is not None or pol.canonicalise:
            quotes = [
                replace(
                    q,
                    bid_ticks=self._round_ticks(q.bid_ticks),
                    ask_ticks=self._round_ticks(q.ask_ticks),
                    bid_size=_round_to(q.bid_size, sr),
                    ask_size=_round_to(q.ask_size, sr),
                )
                for q in quotes
            ]
            trades = [
                replace(t, price_ticks=self._round_ticks(t.price_ticks), size=_round_to(t.size, sr))
                for t in trades
            ]
            levels = [(_round_to(p, pr), _round_to(b, sr), _round_to(a, sr)) for p, b, a in levels]
            hist = [
                replace(
                    h,
                    own_price=_round_to(h.own_price, pr),
                    own_quantity=_round_to(h.own_quantity, sr),
                    own_profit=_round_to(h.own_profit, pr),
                    rival_prices=tuple(_round_to(x, pr) for x in h.rival_prices),
                    market_price=_round_to(h.market_price, pr),
                )
                for h in hist
            ]
            flow = [(t, _round_to(b, sr), _round_to(s, sr)) for t, b, s in flow]
        if pol.canonicalise:
            levels = [(_norm(p), _norm(b), _norm(a)) for p, b, a in sorted(levels)]
            trades = sorted(trades, key=lambda t: (t.tick, t.trade_id))
            flow = sorted(flow)

        # 6. anonymise
        if pol.anonymise_ids:
            self._ensure_window(raw.period)
            v = raw.viewer_id
            quotes = [replace(q, agent_id=self.pseudonym(q.agent_id, v)) for q in quotes]
            trades = [
                replace(
                    t,
                    buyer_id=self.pseudonym(t.buyer_id, v),
                    seller_id=self.pseudonym(t.seller_id, v),
                )
                for t in trades
            ]
            if pol.canonicalise:
                quotes = sorted(quotes, key=lambda q: q.agent_id)

        parts = ViewParts(
            rival_quotes=quotes, trades=trades, book_levels=levels, history=hist, flow=flow
        )
        return ViewResult(parts=parts, view_hash=view_hash(parts, pol.id), policy_id=pol.id)

    # -------------------------------------------------------------------- state
    def state(self) -> dict[str, Any]:
        return {
            "window": self._window,
            "tagged_ids": list(self.tagged_ids),
            "pseudo": [[k, v] for k, v in sorted(self._pseudo.items())],
        }

    def restore(self, state: dict[str, Any]) -> None:
        self._window = None if state["window"] is None else int(state["window"])
        self.tagged_ids = [int(i) for i in state["tagged_ids"]]
        self._pseudo = {int(k): int(v) for k, v in state["pseudo"]}


def render(parts: ViewParts, policy_id: str = "") -> str:
    """Canonical JSON rendering of view parts (the input of ``view_hash``)."""

    def q(x: Quote) -> list[Any]:
        return [x.agent_id, x.bid_ticks, x.ask_ticks, x.bid_size, x.ask_size]

    def t(x: Trade) -> list[Any]:
        return [
            x.trade_id,
            x.tick,
            x.price_ticks,
            x.size,
            x.buyer_id,
            x.seller_id,
            x.aggressor,
            x.mechanism,
            x.batch_id,
        ]

    def h(x: PeriodRecord) -> list[Any]:
        return [
            x.period,
            x.own_price,
            x.own_quantity,
            x.own_profit,
            list(x.rival_prices),
            x.market_price,
        ]

    doc = {
        "policy": policy_id,
        "quotes": [q(x) for x in parts.rival_quotes],
        "trades": [t(x) for x in parts.trades],
        "book": [list(lv) for lv in parts.book_levels],
        "history": [h(x) for x in parts.history],
        "flow": [list(f) for f in parts.flow],
    }
    return json.dumps(doc, sort_keys=True, separators=(",", ":"), allow_nan=True)


def view_hash(parts: ViewParts, policy_id: str = "") -> str:
    return hashlib.sha256(render(parts, policy_id).encode("utf-8")).hexdigest()


# ------------------------------------------------------------------------------ observation
#: private ``Observation.extra`` keys the runner may fill with time-stamped raw inputs; they are
#: consumed (removed) by :func:`apply_view` and never reach an agent
RAW_QUOTES_KEY = "_raw_rival_quotes"  # list[(tick the quote took effect, Quote)]
RAW_BOOK_KEY = "_raw_book"  # list[(snapshot tick, levels)]


def apply_view(
    obs: Observation,
    policy: ViewPolicy,
    rng: RNGRegistry | np.random.Generator,
    *,
    view: PublicView | None = None,
) -> Observation:
    """Apply ``policy`` to a raw observation; return the observation the agent sees.

    Rival quotes, trades, book levels and history are replaced by their transformed versions
    (order: visibility, delay, aggregate, truncate, canonicalise, anonymise);
    ``obs.view_policy_id`` is set to ``policy.id`` and ``obs.extra["view_hash"]`` to the hash of
    exactly what is shown. ``obs.extra["flow"]`` is dropped unless ``policy.show_flow``.

    ``view`` is the run's stateful :class:`PublicView` (stable pseudonyms across epochs, state
    in checkpoints); without it a fresh one is built for this call. Time-stamped rival quotes
    and book snapshots can be passed in ``obs.extra[RAW_QUOTES_KEY]`` /
    ``obs.extra[RAW_BOOK_KEY]`` (needed for ``delay_ticks > 0``); otherwise the current ones
    are stamped at ``obs.tick``.
    """
    extra = dict(obs.extra)
    raw_quotes = extra.pop(RAW_QUOTES_KEY, None)
    raw_book = extra.pop(RAW_BOOK_KEY, None)
    p_min = float(extra.get("p_min", 0.0))
    tick_size = float(extra.get("tick_size", 1.0))
    if view is None:
        ids = [obs.agent_id, *(q.agent_id for q in obs.rival_quotes)]
        view = PublicView(policy, rng, p_min=p_min, tick_size=tick_size, tagged_ids=ids)
    else:
        view.policy = policy
        view.p_min, view.tick_size = p_min, tick_size
        view._gen = rng["view"] if isinstance(rng, RNGRegistry) else rng
    quotes = (
        list(raw_quotes) if raw_quotes is not None else [(obs.tick, q) for q in obs.rival_quotes]
    )
    book = list(raw_book) if raw_book is not None else [(obs.tick, list(obs.book_levels))]
    raw = RawView(
        tick=obs.tick,
        period=obs.period,
        viewer_id=obs.agent_id,
        rival_quotes=quotes,
        trades=list(obs.trades_since_epoch),
        book=book,
        history=list(obs.history),
    )
    res = view.apply(raw)
    if not policy.show_flow:
        extra.pop("flow", None)
    extra["view_hash"] = res.view_hash
    obs.rival_quotes = res.parts.rival_quotes
    obs.trades_since_epoch = res.parts.trades
    obs.book_levels = res.parts.book_levels
    obs.history = res.parts.history
    obs.view_policy_id = res.policy_id
    obs.extra = extra
    return obs
