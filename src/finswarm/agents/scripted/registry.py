"""Registry of scripted policies: ``make_scripted(policy, params, slot, ...)``.

Missing price parameters default to the market benchmarks when given (cooperative target =
joint-monopoly price, punishment = one-shot Nash price), else to the Calvano (2020) values
for ``alpha = 1`` (p_N = 1.4729, p_M = 1.9250, c = 1). Unknown parameters raise ``TypeError``
so a typo in a config never silently falls back to a default.
"""

from __future__ import annotations

import ast
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from finswarm.agents.scripted.base import ScriptedAgent
from finswarm.agents.scripted.competitive import (
    BestResponder,
    CostPlus,
    LeaderFollower,
    demand_dict,
    run_seed_of,
)
from finswarm.agents.scripted.mock_llm import MockLLMAgent
from finswarm.agents.scripted.no_regret import NoRegretMM
from finswarm.agents.scripted.replay import ReplayRival, load_schedule
from finswarm.agents.scripted.sting import StingDealer
from finswarm.agents.scripted.strategies import (
    GrimTrigger,
    RandomPricer,
    StaticNash,
    TitForTat,
    Undercutter,
    ZIMarketMaker,
    demand_profit_fn,
)
from finswarm.config.schema import MarketConfig
from finswarm.sim.online_metrics import Benchmarks

SCRIPTED_POLICIES: dict[str, type[ScriptedAgent]] = {
    "tit_for_tat": TitForTat,
    "grim_trigger": GrimTrigger,
    "undercutter": Undercutter,
    "static_nash": StaticNash,
    "random_pricer": RandomPricer,
    "zi_mm": ZIMarketMaker,
    "no_regret_mm": NoRegretMM,
    "sting_dealer": StingDealer,
    "replay_rival": ReplayRival,
    "mock_llm": MockLLMAgent,
    "best_responder": BestResponder,
    "cost_plus": CostPlus,
    "leader_follower": LeaderFollower,
}
ALIASES = {
    "tft": "tit_for_tat",
    "grim": "grim_trigger",
    "nash": "static_nash",
    "random": "random_pricer",
    "zi_market_maker": "zi_mm",
    "exp3": "no_regret_mm",
    "sting": "sting_dealer",
    "replay": "replay_rival",
    "best_response": "best_responder",
    "sc1": "best_responder",
}

CALVANO_ALPHA1 = Benchmarks(p_nash=1.4729, pi_nash=22.29, p_mono=1.9250, pi_mono=33.75)


def parse_policy(spec: str) -> tuple[str, dict[str, Any]]:
    """``"undercutter(eps=0.02, start=1.9)"`` -> ``("undercutter", {"eps": 0.02, "start": 1.9})``.

    ``LLMSlotConfig`` has no params field, so slot policies carry their parameters inline.
    Values are Python literals (numbers, strings, booleans, None, lists).
    """
    spec = spec.strip()
    if "(" not in spec:
        return spec, {}
    if not spec.endswith(")"):
        raise ValueError(f"bad policy spec {spec!r}")
    name, args = spec[:-1].split("(", 1)
    call = ast.parse(f"f({args})", mode="eval").body
    assert isinstance(call, ast.Call)
    if call.args:
        raise ValueError(f"policy spec {spec!r}: use keyword arguments only")
    params = {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords if kw.arg}
    return name.strip(), params


def canonical_policy(policy: str) -> str:
    p = parse_policy(policy)[0].lower()
    return ALIASES.get(p, p)


def is_scripted_policy(policy: str | None) -> bool:
    return policy is not None and canonical_policy(policy) in SCRIPTED_POLICIES


def _defaults(
    policy: str, params: dict[str, Any], b: Benchmarks, market: MarketConfig | None
) -> dict[str, Any]:
    p = dict(params)
    cost = market.cost * market.alpha if market is not None else 1.0
    if policy == "tit_for_tat":
        p.setdefault("target", b.p_mono)
    elif policy == "grim_trigger":
        p.setdefault("target", b.p_mono)
        p.setdefault("punish_price", b.p_nash)
    elif policy == "undercutter":
        p.setdefault("start", b.p_mono)
    elif policy == "static_nash":
        p.setdefault("price", b.p_nash)
    elif policy == "random_pricer":
        p.setdefault("lo", b.p_nash)
        p.setdefault("hi", b.p_mono)
    elif policy == "zi_mm":
        p.setdefault("hi", 1.1 * b.p_mono)
        if market is not None:
            p.setdefault("two_sided", market.two_sided)
    elif policy == "no_regret_mm":
        p.setdefault("lo", cost)
        p.setdefault("hi", 1.1 * b.p_mono)
        p.setdefault("reward_scale", 2.0 * b.pi_mono)
        if p.get("mode") == "hedge" and "profit_fn" not in p:
            if market is None:
                raise ValueError("no_regret_mm hedge mode needs the market config")
            d = market.demand
            tau = market.routing.tau if market.routing.tau is not None else market.alpha * d.mu
            p["profit_fn"] = demand_profit_fn(
                alpha=market.alpha, a=d.a, a0=d.a0, tau=tau, beta=d.beta, cost=cost
            )
    elif policy == "mock_llm":
        p.setdefault("lo", b.p_nash)
        p.setdefault("hi", 1.1 * b.p_mono)
    elif policy == "best_responder":
        m = market if market is not None else MarketConfig()
        p.setdefault("demand", demand_dict(m))
        p.setdefault("n_dealers", m.n_dealers)
        p.setdefault("start", b.p_nash)
    elif policy == "cost_plus":
        p.setdefault("markup", b.p_nash - cost)
    elif policy == "leader_follower":
        p.setdefault("lo", b.p_nash)
        p.setdefault("hi", b.p_mono)
    return p


def make_scripted(
    policy: str,
    params: dict[str, Any] | None,
    slot: int,
    agent_id: int | None = None,
    *,
    name: str | None = None,
    role: str = "dealer",
    memory_periods: int = 100,
    seed: int | Sequence[int] | None = None,
    benchmarks: Benchmarks | None = None,
    market: MarketConfig | None = None,
    replay_source: str | Path | None = None,
) -> ScriptedAgent:
    """Build a scripted agent. ``replay_rival`` takes ``schedule`` or ``source`` (+ ``slot``)
    in ``params``; ``replay_source`` is the default source run directory."""
    pol = canonical_policy(policy)
    if pol not in SCRIPTED_POLICIES:
        raise KeyError(f"unknown scripted policy {policy!r}; known: {sorted(SCRIPTED_POLICIES)}")
    cls = SCRIPTED_POLICIES[pol]
    b = benchmarks or CALVANO_ALPHA1
    inline = parse_policy(policy)[1]
    kw = _defaults(pol, {**inline, **(params or {})}, b, market)
    if pol == "cost_plus":  # one common shock path per run, whatever the slot
        kw.setdefault("common_seed", run_seed_of(seed))
    if pol == "replay_rival" and "schedule" not in kw:
        source = kw.pop("source", None) or replay_source
        if source is None:
            raise ValueError("replay_rival needs params.source or a replay_source run directory")
        src_slot = int(kw.pop("slot", slot))
        kw["schedule"] = load_schedule(source, src_slot)
        kw["source"] = str(source)
        kw["source_slot"] = src_slot
    aid = slot if agent_id is None else int(agent_id)
    return cls(aid, slot, name=name, role=role, memory_periods=memory_periods, seed=seed, **kw)
