"""Scripted agents (TitForTat, GrimTrigger, Undercutter, StaticNash, RandomPricer,
ZIMarketMaker, NoRegretMM, StingDealer, ReplayRival, ForcedDeviationWrapper, MockLLMAgent) and
the registered controls BestResponder (SC1), CostPlus (SC2) and LeaderFollower (SC4)."""

from finswarm.agents.scripted.base import ScriptedAgent
from finswarm.agents.scripted.competitive import BestResponder, CostPlus, LeaderFollower
from finswarm.agents.scripted.forced import ForcedDeviationWrapper
from finswarm.agents.scripted.mock_llm import MockLLMAgent
from finswarm.agents.scripted.no_regret import NoRegretMM
from finswarm.agents.scripted.registry import (
    SCRIPTED_POLICIES,
    canonical_policy,
    is_scripted_policy,
    make_scripted,
)
from finswarm.agents.scripted.replay import ReplayRival
from finswarm.agents.scripted.sting import StingDealer
from finswarm.agents.scripted.strategies import (
    GrimTrigger,
    RandomPricer,
    StaticNash,
    TitForTat,
    Undercutter,
    ZIMarketMaker,
)

__all__ = [
    "SCRIPTED_POLICIES",
    "BestResponder",
    "CostPlus",
    "ForcedDeviationWrapper",
    "GrimTrigger",
    "LeaderFollower",
    "MockLLMAgent",
    "NoRegretMM",
    "RandomPricer",
    "ReplayRival",
    "ScriptedAgent",
    "StaticNash",
    "StingDealer",
    "TitForTat",
    "Undercutter",
    "ZIMarketMaker",
    "canonical_policy",
    "is_scripted_policy",
    "make_scripted",
]
