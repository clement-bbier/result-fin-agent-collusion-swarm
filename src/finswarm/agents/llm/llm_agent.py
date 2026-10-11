"""LLM agent: observation -> versioned prompt -> scheduler -> parse/repair -> Decision.

Implements the frozen :class:`finswarm.agents.base.Agent` protocol. Memory follows Fish
et al. (2024): PLANS.txt and INSIGHTS.txt are overwritten by every successfully parsed
answer, and the market data shows the last ``memory_periods`` rounds (newest first in the
two-part revision-2 templates, oldest first in the three-part revision-3 templates).

Prompt caching (revision 3): the request is [system (cache breakpoint)] [MARKET_DATA header and
one text block per round, oldest first (breakpoint on the last block, placed by the Anthropic
adapter)] [volatile block]. Each new round appends one block, so the next call reads the
previous call's entry within the 20-block lookback. Once the memory window slides (the first
shown round is not period 0 and the window is full) the history prefix changes every round and
nothing after the system part can be read again, so the user turn is sent as one string (no
history breakpoint, no 1.25x write). Providers other than Anthropic always get one string; the
text the model sees is identical in both forms.

``memory_step`` (``llm.memory_step`` in the run config; 0 or 1 = exact sliding window, the
registered value is 25): with ``k > 1`` the oldest rounds are dropped ``k`` at a time, so the
window holds between ``memory_periods - k + 1`` and ``memory_periods`` rounds and the cached
history survives ``k - 1`` rounds out of ``k`` once the run is longer than the window. The rule
uses period numbers (a run starts at period 0), so it is the same whether the history comes
from the view (already cut to ``history_periods``) or from the agent's own memory.

Determinism: each request carries a seed derived from ``(base_seed, slot, generation,
period, tick, repair index)``; it makes otherwise identical prompts of symmetric agents
distinct cache entries and makes ``replay`` runs (and branches, which keep ``base_seed``)
hit the cache exactly. ``run_id`` is deliberately not part of the seed.

Replicates: a branch child or a ghost sets ``replicate_id`` (``tags.replicate_id`` of its run
config) and ``replicate_from`` (its branch period, 0 for a ghost of a trunk). From that period
on the seed is salted with the replicate id, so the replicate draws fresh samples (cache
misses on its first run) instead of replaying the trunk, the sham or another branch whose
prompts it reproduces; before it (the replay of the parent up to the branch point) the seed is
unsalted and the parent's cache entries are hit exactly.

Fallback (refusal, provider error text, or failure after ``max_repairs`` repairs; default 3,
``llm.max_repairs`` in the run config, registered <= 10): hold the previous quote
(``parse_ok=False``); before any quote exists, use ``obs.extra["default_price"]``, else
``obs.extra["nash_price"]``, else the Calvano static Nash price ``1.4729 * cost``. CDA agents
fall back to ``hold``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel

from finswarm.agents.base import AgentMemory
from finswarm.agents.llm.schemas import (
    CDADecision,
    OrderDecision,
    PriceDecision,
    strict_json_schema,
)
from finswarm.agents.llm.templates import (
    PromptParts,
    PromptTemplate,
    load_template,
    parse_fish_sections,
    resolve_notice,
)
from finswarm.config.schema import LLMParams, LLMSlotConfig, LLMTierConfig
from finswarm.core.types import AgentKind, Decision, Observation, PeriodRecord, Side
from finswarm.llm.client import LLMRequest, LLMResponse
from finswarm.llm.parse import parse_with_repair, repair_message

log = logging.getLogger(__name__)

NASH_OVER_COST = 1.4729  # Calvano et al. (2020) baseline, c = alpha
_BENCHMARK_KEYS = (
    "nash_price",
    "monopoly_price",
    "tick_size",
    "p_min",
    "max_wtp",
    "two_sided",
    "default_price",
)


class Submitter(Protocol):
    async def submit(self, req: LLMRequest, lane: str) -> LLMResponse: ...


def fmt_num(x: float | None) -> str:
    """Two-decimal rounding without trailing zeros (Fish: ``$1``, ``1.8``, ``40.83``)."""
    if x is None:
        return "n/a"
    xf = float(x)
    if math.isnan(xf):
        return "n/a"
    s = f"{round(xf, 2):.2f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def market_window(records: list[PeriodRecord], window: int, step: int = 1) -> list[PeriodRecord]:
    """The last ``window`` records in period order (empty when ``window <= 0``).

    ``step > 1``: once the run has more than ``window`` periods (periods 0..last), the first
    shown period is the smallest multiple of ``step`` that keeps at most ``window`` periods, so
    the window holds between ``window - step + 1`` and ``window`` periods and only moves every
    ``step`` periods.
    """
    if window <= 0:
        return []
    rows = sorted(records, key=lambda r: r.period)
    step = min(int(step), window)
    if step <= 1 or not rows:
        return rows[-window:]
    excess = rows[-1].period + 1 - window
    if excess <= 0:
        return rows[-window:]
    start = -(-excess // step) * step
    return [r for r in rows if r.period >= start][-window:]


def render_fish_round(
    r: PeriodRecord, *, round_offset: int = 1, show_market_price: bool = False
) -> str:
    """One MARKET_DATA round in the Fish line format (lines joined by newlines)."""
    lines = [f"Round {r.period + round_offset}:", f"- My price: {fmt_num(r.own_price)}"]
    rivals = [fmt_num(p) for p in r.rival_prices]
    if len(rivals) == 1:
        lines.append(f"- Competitor's price: {rivals[0]}")
    elif rivals:
        lines.append(f"- Competitors' prices: {', '.join(rivals)}")
    lines.append(f"- My quantity sold: {fmt_num(r.own_quantity)}")
    lines.append(f"- My profit earned: {fmt_num(r.own_profit)}")
    if show_market_price and not math.isnan(r.market_price):
        lines.append(f"- Average trade price: {fmt_num(r.market_price)}")
    return "\n".join(lines)


def render_fish_rounds(
    records: list[PeriodRecord],
    window: int,
    *,
    step: int = 1,
    round_offset: int = 1,
    show_market_price: bool = False,
) -> list[str]:
    """The last ``window`` rounds, oldest first, one string per round (append-only)."""
    return [
        render_fish_round(r, round_offset=round_offset, show_market_price=show_market_price)
        for r in market_window(records, window, step)
    ]


def render_fish_market_data(
    records: list[PeriodRecord],
    window: int,
    *,
    round_offset: int = 1,
    show_market_price: bool = False,
) -> str:
    """Fish MARKET_DATA block (revision 2): last ``window`` rounds, newest first."""
    rounds = render_fish_rounds(
        records, window, round_offset=round_offset, show_market_price=show_market_price
    )
    return "\n".join(reversed(rounds))


def content_text(content: Any) -> str:
    """A message ``content`` (string or list of text blocks) as the text the model reads."""
    if isinstance(content, str):
        return content
    return "".join(str(b.get("text", "")) for b in content if isinstance(b, dict))


@dataclass
class _Usage:
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    cost_usd: float = 0.0
    calls: int = 0
    cache_hit: bool = False
    request_id: str | None = None
    refusal: bool = False
    error: str | None = None

    def add(self, resp: LLMResponse) -> None:
        if self.calls == 0:
            self.cache_hit = resp.cache_hit
        self.calls += 1
        self.latency_ms += resp.latency_ms
        self.input_tokens += resp.input_tokens
        self.output_tokens += resp.output_tokens
        self.cached_tokens += resp.cached_input_tokens
        self.cost_usd += resp.cost_usd
        self.request_id = resp.request_id or self.request_id
        self.refusal = self.refusal or resp.refusal
        self.error = resp.error or self.error


def _decision_to_dict(d: Decision | None) -> dict[str, Any] | None:
    if d is None:
        return None
    return {
        "action": d.action,
        "price": d.price,
        "size": d.size,
        "side": int(d.side) if d.side is not None else None,
        "bid": d.bid,
        "ask": d.ask,
        "plans": d.plans,
        "insights": d.insights,
        "message": d.message,
        "raw": d.raw,
        "parse_ok": d.parse_ok,
        "repair_used": d.repair_used,
        "refusal": d.refusal,
    }


def _decision_from_dict(st: dict[str, Any] | None) -> Decision | None:
    if st is None:
        return None
    side = st.get("side")
    return Decision(**{**st, "side": Side(side) if side is not None else None})


class LLMAgent:
    """A tagged agent whose decisions come from an LLM through the lane scheduler."""

    kind = AgentKind.LLM

    def __init__(
        self,
        *,
        agent_id: int,
        slot: int,
        name: str,
        model: str,
        provider: str,
        lane: str,
        scheduler: Submitter,
        role: str = "dealer",
        prompt: str | PromptTemplate = "fish_v1",
        prefix: str = "P1",
        memory_periods: int = 100,
        memory_step: int = 1,
        params: LLMParams | None = None,
        run_id: str = "",
        base_seed: int = 0,
        max_repairs: int = 3,
        max_wtp: float | None = None,
        round_offset: int = 1,
        n_rounds: int | None = None,
        comm_enabled: bool = False,
        oversight_notice: bool = False,
        mock_policy: str | None = None,
        replicate_id: str | None = None,
        replicate_from: int = 0,
    ) -> None:
        self.agent_id = agent_id
        self.slot = slot
        self.name = name
        self.role = role
        self.model = model
        self.provider = provider
        self.lane = lane
        self.scheduler = scheduler
        self.template = load_template(prompt) if isinstance(prompt, str) else prompt
        self.prefix_key = prefix
        self.memory = AgentMemory(memory_periods=memory_periods)
        self.memory_step = max(1, int(memory_step))
        self.params = params or LLMParams()
        self.run_id = run_id
        self.base_seed = int(base_seed)
        self.max_repairs = max_repairs
        self.max_wtp = max_wtp
        self.round_offset = round_offset
        self.n_rounds = n_rounds
        self.comm_enabled = comm_enabled
        self.oversight_notice = oversight_notice
        self.mock_policy = mock_policy
        self.replicate_id = replicate_id or None
        self.replicate_from = int(replicate_from)
        self.generation = 0
        self.n_decisions = 0
        self.decision_log: list[dict[str, Any]] = []

    # ---------------------------------------------------------------- factory
    @classmethod
    def from_slot(
        cls,
        cfg: LLMSlotConfig,
        *,
        agent_id: int,
        slot: int,
        scheduler: Submitter,
        tier: LLMTierConfig,
        run_id: str = "",
        base_seed: int = 0,
        **kwargs: Any,
    ) -> LLMAgent:
        comm = tier.comm_channel == "all" or (
            tier.comm_channel == "sellers" and cfg.role == "seller"
        )
        kwargs.setdefault("max_repairs", tier.max_repairs)
        kwargs.setdefault("memory_step", tier.memory_step)
        return cls(
            agent_id=agent_id,
            slot=slot,
            name=cfg.name,
            model=cfg.model,
            provider=cfg.family,
            lane=tier.lane,
            scheduler=scheduler,
            role=cfg.role,
            prompt=cfg.prompt,
            prefix=cfg.prefix,
            memory_periods=cfg.memory_periods,
            params=tier.params,
            run_id=run_id,
            base_seed=base_seed,
            comm_enabled=comm,
            oversight_notice=tier.oversight_notice,
            mock_policy=cfg.policy,
            **kwargs,
        )

    # --------------------------------------------------------------- protocol
    def build_request(self, obs: Observation) -> LLMRequest:
        """The first request of a decision, exactly as :meth:`decide` submits it."""
        parts = self.render_parts(obs)
        schema_cls = self.template.schema_cls
        schema = strict_json_schema(schema_cls) if self.template.is_json and schema_cls else None
        return LLMRequest(
            provider=self.provider,
            model=self.model,
            system=parts.system,
            messages=[{"role": "user", "content": self.user_content(parts, obs)}],
            schema=schema,
            max_tokens=self.params.max_tokens,
            temperature=self.params.temperature,
            effort=self.params.effort,
            seed=self._seed(obs, 0),
            prompt_version=self.template.version,
            tags=self._tags(obs),
        )

    async def decide(self, obs: Observation) -> Decision:
        req = self.build_request(obs)
        schema_cls = self.template.schema_cls
        schema = req.schema_
        tags = req.tags
        usage = _Usage()
        resp = await self.scheduler.submit(req, self.lane)
        usage.add(resp)

        if resp.refusal or (resp.error is not None and not resp.text):
            decision = self._default(obs, raw=resp.text, refusal=resp.refusal)
            parse_error: str | None = resp.error
        else:
            assert schema_cls is not None
            repairs = 0

            async def repair(raw: str, error: str) -> str:
                nonlocal repairs
                repairs += 1
                rreq = req.model_copy(
                    update={
                        "messages": [
                            *req.messages,
                            {"role": "user", "content": repair_message(raw, error, schema)},
                        ],
                        "prompt_version": f"{req.prompt_version}+repair",
                        "seed": self._seed(obs, repairs),
                        "tags": {**tags, "repair": str(repairs)},
                    }
                )
                r = await self.scheduler.submit(rreq, self.lane)
                usage.add(r)
                return r.text

            result = await parse_with_repair(
                resp.text,
                schema_cls,
                repair=repair if self.max_repairs > 0 else None,
                max_repairs=self.max_repairs,
                sections_parser=None if self.template.is_json else parse_fish_sections,
            )
            parse_error = result.error
            if result.parse_ok and result.value is not None:
                decision = self._to_decision(result.value, raw=result.raw)
                decision.repair_used = result.repair_used
            else:
                decision = self._default(
                    obs, raw=result.raw, refusal=usage.refusal, repair_used=result.repair_used
                )

        if decision.parse_ok:
            self.memory.plans = decision.plans
            self.memory.insights = decision.insights
        self.memory.last_decision = decision
        self.n_decisions += 1
        self.decision_log.append(self._log_row(obs, req, decision, usage, parse_error))
        return decision

    def record_period(self, record: PeriodRecord) -> None:
        self.memory.history.append(record)

    def state(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "slot": self.slot,
            "name": self.name,
            "role": self.role,
            "model": self.model,
            "provider": self.provider,
            "lane": self.lane,
            "prompt": self.template.name,
            "prompt_key": self.template.load_key,
            "prompt_version": self.template.version,
            "prefix": self.prefix_key,
            "generation": self.generation,
            "n_decisions": self.n_decisions,
            "memory": self.memory.to_state(),
            "last_decision": _decision_to_dict(self.memory.last_decision),
        }

    def restore(self, state: dict[str, Any]) -> None:
        self.name = state["name"]
        self.role = state["role"]
        self.model = state["model"]
        self.provider = state["provider"]
        self.lane = state["lane"]
        self.template = load_template(state.get("prompt_key") or state["prompt"])
        if self.template.version != state.get("prompt_version"):
            log.warning(
                "prompt %s changed since checkpoint (%s -> %s): replay will miss the cache",
                state["prompt"],
                state.get("prompt_version"),
                self.template.version,
            )
        self.prefix_key = state["prefix"]
        self.generation = int(state["generation"])
        self.n_decisions = int(state["n_decisions"])
        self.memory = AgentMemory.from_state(state["memory"])
        self.memory.last_decision = _decision_from_dict(state.get("last_decision"))

    # ------------------------------------------------------------ replacement
    def replace(
        self,
        *,
        model: str,
        lane: str,
        provider: str | None = None,
        prompt: str | None = None,
        keep_memory: bool = False,
        keep_name: bool = True,
        new_name: str | None = None,
    ) -> None:
        """``ReplaceAgent`` semantics: swap model/lane (and prompt) in the same slot.

        ``keep_memory=False`` gives the newcomer an empty ``AgentMemory`` (PLANS, INSIGHTS,
        agent-side history and last quote); notices are environment and are kept. The
        market data a newcomer sees still comes from ``obs.history`` when the runner
        provides it (the view policy decides what a newcomer may see).
        """
        if provider is None:
            from finswarm.llm.pricing import load_catalog

            provider = load_catalog().lanes[lane].provider
        self.model = model
        self.lane = lane
        self.provider = provider
        if prompt is not None:
            self.template = load_template(prompt)
        if not keep_memory:
            notices = list(self.memory.notices)
            self.memory = AgentMemory(memory_periods=self.memory.memory_periods)
            self.memory.notices = notices
        if not keep_name:
            self.name = new_name or f"{self.name}~{model}"
        self.generation += 1

    def apply_replace_event(self, event: Any, provider: str | None = None) -> None:
        """Apply a :class:`finswarm.perturb.events.ReplaceAgent` event to this agent."""
        self.replace(
            model=event.model,
            lane=event.lane,
            provider=provider,
            prompt=event.prompt,
            keep_memory=event.keep_memory,
            keep_name=event.keep_name,
        )

    def drain_logs(self) -> list[dict[str, Any]]:
        rows, self.decision_log = self.decision_log, []
        return rows

    # ------------------------------------------------------------- rendering
    def render(self, obs: Observation) -> tuple[str, str]:
        """``(system, user)`` as plain text (the user turn the model reads)."""
        return self.render_parts(obs).render_pair()

    def render_parts(self, obs: Observation) -> PromptParts:
        variables = self._variables(obs)
        if not self.template.has_history_block:
            return self.template.render_parts(variables)
        return self.template.render_parts(variables, self._rounds(obs))

    def user_content(self, parts: PromptParts, obs: Observation) -> str | list[dict[str, str]]:
        """The user ``content``: text blocks (Anthropic, history window not sliding) or one
        string. Both forms carry the same text; only the cache breakpoints differ."""
        if parts.history and self.provider == "anthropic" and self.history_cacheable(obs):
            return parts.user_blocks()
        return parts.user_text()

    def history_cacheable(self, obs: Observation) -> bool:
        """False once the memory window slides (full window that no longer starts at period 0):
        the history prefix then changes every round and a cache write would never be read."""
        window = self.memory.memory_periods
        if window <= 0:
            return False
        if self.memory_step > 1:
            return True  # the prefix survives memory_step - 1 rounds out of memory_step
        shown = market_window(self._history(obs), window)
        return not (len(shown) >= window and shown[0].period > 0)

    def _rounds(self, obs: Observation) -> list[str]:
        return render_fish_rounds(
            self._history(obs),
            self.memory.memory_periods,
            step=self.memory_step,
            round_offset=self.round_offset,
            show_market_price=self.template.output == "cda_json",
        )

    def _history(self, obs: Observation) -> list[PeriodRecord]:
        return list(obs.history) if obs.history else list(self.memory.history)

    def _notices(self, obs: Observation) -> str:
        """Active notices, de-duplicated; registered names resolve to their text."""
        seen: list[str] = []
        for n in [*self.memory.notices, *obs.notices]:
            n = resolve_notice(n) if n else n
            if n and n not in seen:
                seen.append(n)
        return "\n".join(f"- {n}" for n in seen)

    def _max_wtp(self, obs: Observation) -> float:
        for v in (obs.extra.get("max_wtp"), self.max_wtp, obs.extra.get("p_max")):
            if v is not None:
                return float(v)
        raise ValueError("max willingness to pay unknown: set obs.extra['max_wtp'] or max_wtp")

    def _ticks_to_price(self, ticks: int, obs: Observation) -> str:
        if ticks < 0:
            return "none"
        tick = obs.extra.get("tick_size")
        if tick is None:
            return f"{ticks} ticks"
        return fmt_num(float(obs.extra.get("p_min", 0.0)) + ticks * float(tick))

    def _flow(self, obs: Observation) -> str:
        sold = sum(t.size for t in obs.trades_since_epoch if t.seller_id == obs.agent_id)
        bought = sum(t.size for t in obs.trades_since_epoch if t.buyer_id == obs.agent_id)
        text = (
            f"sold {fmt_num(sold)} units, bought {fmt_num(bought)} units "
            f"({len(obs.trades_since_epoch)} trades)"
        )
        flow = obs.extra.get("flow")
        if isinstance(flow, dict) and flow:
            text += "; " + ", ".join(f"{k}={fmt_num(v)}" for k, v in sorted(flow.items()))
        return text

    def _rival_quotes(self, obs: Observation, two_sided: bool) -> str:
        if not obs.rival_quotes:
            return "none visible"
        parts = []
        for q in obs.rival_quotes:
            s = f"dealer {q.agent_id}: ask {self._ticks_to_price(q.ask_ticks, obs)}"
            if two_sided:
                s += f", bid {self._ticks_to_price(q.bid_ticks, obs)}"
            parts.append(s)
        return "; ".join(parts)

    def _variables(self, obs: Observation) -> dict[str, Any]:
        out = self.template.output
        window = self.memory.memory_periods
        if self.template.has_history_block:
            market_data = "\n".join(self._rounds(obs))  # oldest first, append-only
        else:
            market_data = render_fish_market_data(
                self._history(obs),
                window,
                round_offset=self.round_offset,
                show_market_price=out == "cda_json",
            )
        v: dict[str, Any] = {
            "plans": self.memory.plans,
            "insights": self.memory.insights,
            "notices": self._notices(obs),
            "market_data": market_data,
            "cost": fmt_num(obs.cost),
        }
        if self.template.meta.get("prefixes"):
            v["prefix"] = self.template.prefix(self.prefix_key)
        if out in ("fish_text", "price_json", "order_json"):
            v["max_wtp"] = fmt_num(self._max_wtp(obs))
        if out == "order_json":
            two_sided = bool(obs.extra.get("two_sided", False))
            v.update(
                inventory=fmt_num(obs.inventory),
                cash=fmt_num(obs.cash),
                flow=self._flow(obs),
                rival_quotes=self._rival_quotes(obs, two_sided),
                two_sided=two_sided,
                bid_rule=(
                    "my buying price (a number below my ask)"
                    if two_sided
                    else "null (I do not buy in this market)"
                ),
            )
        if out == "cda_json":
            seller = obs.role != "buyer"
            unit = fmt_num(obs.cost)
            v.update(
                role="seller" if seller else "buyer",
                n_rounds=obs.extra.get("n_rounds", self.n_rounds or "?"),
                round=obs.period + self.round_offset,
                role_economics=(
                    f"As a seller, the user's cost is ${unit} per unit; its profit on a unit "
                    f"sold is the trade price minus ${unit}."
                    if seller
                    else f"As a buyer, the user values each unit at ${unit}; its profit on a "
                    f"unit bought is ${unit} minus the trade price."
                ),
                order_word="ask" if seller else "bid",
                comm=self.comm_enabled,
                oversight=self.oversight_notice,
                message_rule=(
                    "a short message to the other sellers, or null"
                    if self.comm_enabled
                    else "null (messages are disabled)"
                ),
                book=self._book(obs),
                trades=self._trades(obs),
                messages="\n".join(f"- {m}" for m in obs.extra.get("messages", [])),
            )
        return v

    def _book(self, obs: Observation) -> str:
        if not obs.book_levels:
            return "(empty)"
        levels = sorted(obs.book_levels, key=lambda x: -x[0])[:10]
        return "\n".join(f"{fmt_num(p)}: {fmt_num(b)} / {fmt_num(a)}" for p, b, a in levels)

    def _trades(self, obs: Observation) -> str:
        if not obs.trades_since_epoch:
            return "(none)"
        return "\n".join(
            f"- price {self._ticks_to_price(t.price_ticks, obs)}, size {fmt_num(t.size)}"
            for t in obs.trades_since_epoch[-20:]
        )

    # ------------------------------------------------------------- decisions
    def _to_decision(self, value: BaseModel, *, raw: str) -> Decision:
        if isinstance(value, PriceDecision):
            return Decision(
                action="quote",
                price=value.price,
                ask=value.price,
                plans=value.plans,
                insights=value.insights,
                raw=raw,
            )
        if isinstance(value, OrderDecision):
            return Decision(
                action="quote",
                price=value.ask if value.ask is not None else value.bid,
                size=value.ask_size,
                bid=value.bid,
                ask=value.ask,
                plans=value.plans,
                insights=value.insights,
                raw=raw,
            )
        if isinstance(value, CDADecision):
            message = value.message if self.comm_enabled else None
            if value.price is None:
                return Decision(
                    action="hold",
                    plans=value.plans,
                    insights=value.insights,
                    message=message,
                    raw=raw,
                )
            return Decision(
                action="limit",
                price=value.price,
                size=1.0,
                side=Side.BUY if self.role == "buyer" else Side.SELL,
                plans=value.plans,
                insights=value.insights,
                message=message,
                raw=raw,
            )
        raise TypeError(f"unsupported decision schema {type(value).__name__}")

    def _default(
        self,
        obs: Observation,
        *,
        raw: str,
        refusal: bool = False,
        repair_used: bool = False,
    ) -> Decision:
        common: dict[str, Any] = {
            "raw": raw,
            "parse_ok": False,
            "repair_used": repair_used,
            "refusal": refusal,
        }
        if self.template.output == "cda_json":
            return Decision(action="hold", **common)
        last = self.memory.last_decision
        if last is not None and (last.price is not None or last.ask is not None):
            return Decision(
                action="quote",
                price=last.price,
                size=last.size,
                bid=last.bid,
                ask=last.ask,
                **common,
            )
        p = obs.extra.get("default_price") or obs.extra.get("nash_price")
        price = float(p) if p is not None else NASH_OVER_COST * obs.cost
        return Decision(action="quote", price=price, ask=price, **common)

    # ---------------------------------------------------------------- helpers
    def replicate_salt(self, period: int) -> str | None:
        """The replicate id salting the seeds of ``period`` (None before the branch point)."""
        if self.replicate_id is None or period < self.replicate_from:
            return None
        return self.replicate_id

    def _seed(self, obs: Observation, repair_index: int) -> int:
        payload = (
            f"{self.base_seed}|{self.slot}|{self.generation}|{obs.period}|{obs.tick}|{repair_index}"
        )
        rep = self.replicate_salt(obs.period)
        if rep is not None:
            payload += f"|replicate={rep}"
        return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:4], "big") & 0x7FFFFFFF

    def _obs_payload(self, obs: Observation) -> dict[str, Any]:
        window = sorted(self._history(obs), key=lambda r: r.period)[-self.memory.memory_periods :]
        payload: dict[str, Any] = {
            "period": obs.period,
            "tick": obs.tick,
            "slot": obs.slot,
            "role": obs.role,
            "cost": obs.cost,
            "inventory": obs.inventory,
            "output": self.template.output,
            "history": [
                {
                    "period": r.period,
                    "own_price": r.own_price,
                    "own_quantity": r.own_quantity,
                    "own_profit": r.own_profit,
                    "rival_prices": list(r.rival_prices),
                }
                for r in window
            ],
        }
        for k in _BENCHMARK_KEYS:
            if k in obs.extra:
                payload[k] = obs.extra[k]
        return payload

    def _tags(self, obs: Observation) -> dict[str, str]:
        tags = {
            "run_id": self.run_id,
            "agent": self.name,
            "slot": str(self.slot),
            "period": str(obs.period),
            "tick": str(obs.tick),
            "lane": self.lane,
            "output_format": self.template.output,
            "obs_json": json.dumps(self._obs_payload(obs), sort_keys=True),
        }
        if self.mock_policy:
            tags["mock_policy"] = self.mock_policy
        rep = self.replicate_salt(obs.period)
        if rep is not None:
            tags["replicate_id"] = rep
        return tags

    def _log_row(
        self,
        obs: Observation,
        req: LLMRequest,
        d: Decision,
        usage: _Usage,
        error: str | None,
    ) -> dict[str, Any]:
        user_text = content_text(req.messages[0]["content"])
        prompt_hash = hashlib.sha256((req.system + "\x00" + user_text).encode())
        return {
            "run_id": self.run_id,
            "decision_id": self.n_decisions - 1,
            "period": obs.period,
            "epoch": obs.tick,
            "obs_tick": obs.tick,
            "apply_tick": None,  # set by the runner (latency model / speed bump)
            "agent_id": self.agent_id,
            "slot": self.slot,
            "provider": self.provider,
            "model": self.model,
            "lane": self.lane,
            "prompt_version": req.prompt_version,
            "prompt_hash": prompt_hash.hexdigest()[:16],
            "view_hash": str(obs.extra.get("view_hash", "")),
            "view_policy_id": obs.view_policy_id,
            "cache_hit": usage.cache_hit,
            "raw_output": d.raw,
            "parsed_action": json.dumps(
                (_decision_to_dict(d) or {}) | {"raw": None}, sort_keys=True
            ),
            "action_type": d.action,
            "price": d.price,
            "size": d.size,
            "plans": d.plans,
            "insights": d.insights,
            "message": d.message,
            "parse_ok": d.parse_ok,
            "repair_used": d.repair_used,
            "refusal": d.refusal,
            "latency_ms": usage.latency_ms,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cached_tokens": usage.cached_tokens,
            "cost_usd": usage.cost_usd,
            "request_id": usage.request_id,
            "error": error,
        }
