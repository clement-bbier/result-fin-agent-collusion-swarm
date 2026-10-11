"""Configuration schema (frozen interface, 2026-10-09). Every YAML is validated here.

``extra="forbid"`` everywhere: a typo in a config is an error, never a silent default.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from finswarm.core.types import MechanismKind
from finswarm.perturb.events import Event


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PriceGrid(_Strict):
    min: float = 1.0
    max: float = 4.0
    tick: float = 0.01


class DemandConfig(_Strict):
    """Calvano (2020) logit demand used by the closed-form dealer market and as calibration."""

    kind: Literal["closed_form_logit", "swarm"] = "closed_form_logit"
    a: float = 2.0
    a0: float = 0.0
    mu: float = 0.25
    beta: float = 100.0


class RoutingConfig(_Strict):
    tau: float | None = None  # None -> alpha * mu (exactly Calvano's demand)
    outside_utility: float = 0.0


class FringeConfig(_Strict):
    share: float = 0.0  # fraction of arriving swarm agents allowed to post public limit orders
    horizon_ticks: int = 50


class BookConfig(_Strict):
    levels: int = 4096
    cap: int = 64


class BatchConfig(_Strict):
    k: int = 1
    k_min: int | None = None
    k_max: int | None = None
    clearing: Literal["uniform", "pairwise_midpoint"] = "uniform"
    rationing: Literal["prorata", "time"] = "prorata"


class MarketConfig(_Strict):
    mechanism: MechanismKind = MechanismKind.DEALER
    n_dealers: int = 2
    cost: float = 1.0
    alpha: float = 1.0
    price_grid: PriceGrid = Field(default_factory=PriceGrid)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    demand: DemandConfig = Field(default_factory=DemandConfig)
    fringe: FringeConfig = Field(default_factory=FringeConfig)
    book: BookConfig = Field(default_factory=BookConfig)
    batch: BatchConfig = Field(default_factory=BatchConfig)
    two_sided: bool = False


class ArrivalConfig(_Strict):
    lam_mean: float = 1e-3  # mean arrival intensity per agent per tick
    lam_sd_log: float = 0.5


class SwarmConfig(_Strict):
    mode: Literal["none", "closed_form", "agents"] = "closed_form"
    n_agents: int = 10_000
    arrivals: ArrivalConfig = Field(default_factory=ArrivalConfig)
    type_weights: dict[str, float] = Field(
        default_factory=lambda: {"ZI_BUYER": 0.75, "INFORMED": 0.10, "NOISE": 0.10, "TREND": 0.05}
    )
    value_private_sd: float = 0.1
    signal_noise_sd: float = 0.05
    sentiment_rho: float = 0.99
    sentiment_sigma: float = 0.05
    sentiment_weight: float = 0.1
    attention_mean: float = 0.2


class LLMSlotConfig(_Strict):
    name: str
    role: Literal["dealer", "seller", "buyer", "trader"] = "dealer"
    model: str
    family: str  # anthropic | mistral | gemini | openrouter (llama) | mock
    prompt: str = "fish_v1"
    prefix: str = "P1"
    memory_periods: int = 100
    policy: str | None = None  # MockLLM / scripted policy name when family == mock
    params: dict[str, Any] = Field(default_factory=dict)  # scripted/mock policy parameters


class LLMParams(_Strict):
    effort: str | None = "low"
    max_tokens: int = 2048
    temperature: float | None = None  # must stay None on the anthropic lane


class LLMTierConfig(_Strict):
    lane: str = "mock"
    mode: Literal["live", "replay", "mock"] = "mock"
    slots: list[LLMSlotConfig] = Field(default_factory=list)
    params: LLMParams = Field(default_factory=LLMParams)
    allow_mixed_families: bool = False
    comm_channel: Literal["none", "sellers", "all"] = "none"
    oversight_notice: bool = False
    # JSON repairs before the hold-the-previous-quote fallback (PREREG: at most 10)
    max_repairs: int = Field(default=3, ge=0, le=10)
    # MARKET_DATA window step: 0 (or 1) = exact sliding window of the last memory_periods
    # rounds; k > 1 = once the run has more rounds than the window, drop the oldest rounds k at
    # a time (window between memory_periods - k + 1 and memory_periods rounds), which keeps the
    # cached history prefix valid k - 1 rounds out of k (registered: 25)
    memory_step: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _one_family_per_run(self) -> LLMTierConfig:
        fams = {s.family for s in self.slots}
        if len(fams) > 1 and not self.allow_mixed_families:
            raise ValueError(f"mixed families {sorted(fams)} require allow_mixed_families=true")
        return self


class ViewPolicy(_Strict):
    """Public-view transformations, applied in this fixed order: delay, aggregate, truncate,
    canonicalise, anonymise."""

    id: str = "V0"
    show_rival_quotes: bool = True
    show_trades: bool = True
    show_flow: bool = False
    history_periods: int = 100
    delay_ticks: int = 0
    aggregate_levels: int = 0
    depth_truncation: int = 0
    canonicalise: bool = False
    price_round: float | None = None
    size_round: float | None = None
    anonymise_ids: bool = False
    reshuffle_every: int = 0


class ClockConfig(_Strict):
    mode: Literal["sync", "poisson"] = "sync"
    epoch_ticks: int = 1
    staleness_window: int = 0
    speed_bump_max: int = 0
    # Per-slot constant extra latency U{0..J} ticks, drawn once from the `matching` stream
    # (order-preserving: FIFO bump D + j_s); replaces `tags.speed_bump_jitter`. L7 registers 0.
    speed_bump_jitter: int = Field(0, ge=0)


class CheckpointConfig(_Strict):
    every_periods: int = 10
    keep_last: int = 3


class LoggingConfig(_Strict):
    swarm_order_sample: float = 0.01
    snapshot_every: int = 10
    snapshot_levels: int = 10
    store_prompts: bool = True


GhostRivalPolicy = Literal["static_nash", "zi_mm", "replay_parent"]


class GhostConfig(_Strict):
    enabled: bool = False
    rival_policy: GhostRivalPolicy = "replay_parent"
    parent_run_id: str | None = None


class RunConfig(_Strict):
    name: str
    periods: int = 300
    ticks_per_period: int = 1
    seed: int = 0
    market: MarketConfig = Field(default_factory=MarketConfig)
    swarm: SwarmConfig = Field(default_factory=SwarmConfig)
    llm: LLMTierConfig = Field(default_factory=LLMTierConfig)
    view: ViewPolicy = Field(default_factory=ViewPolicy)
    clock: ClockConfig = Field(default_factory=ClockConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    events: list[Event] = Field(default_factory=list)
    ghost: GhostConfig = Field(default_factory=GhostConfig)
    parent_run_id: str | None = None
    branch_period: int | None = None
    tags: dict[str, str] = Field(default_factory=dict)


class LaneConfig(_Strict):
    provider: str
    models: list[str]
    rps: float | None = None
    rpm: float | None = None
    rpd: int | None = None
    tpm: int | None = None
    max_in_flight: int = 4
    budget_usd: float | None = None
    free: bool = False


class BranchSpec(_Strict):
    name: str
    parent: str  # run name or campaign cell
    at: str  # "period:100" or "checkpoint:collusion_established"
    events: list[Event]
    periods: int | None = None  # child horizon (total periods); None = parent's
    # K ghost sub-runs of every child (one per LLM slot, rivals per this policy), run by the
    # campaign once the child is done. Campaign-level only: the child's run id is unchanged.
    ghost: GhostRivalPolicy | None = None


class JudgeConfig(_Strict):
    """Scratchpad judge of a campaign (DETECTION_PROTOCOL D9): one model for Haiku-arm runs
    and another for Sonnet-arm runs (never the family being judged), with an optional backup."""

    model: str
    model_sonnet_arm: str
    lane: str | None = None
    lane_sonnet_arm: str | None = None
    backup_sonnet_arm: str | None = None
    rubric: str = "judge_v1"
    scratchpads_per_window: int = Field(8, ge=1)


class CampaignConfig(_Strict):
    name: str
    tier: int = 0
    prereg_id: str | None = None
    base: RunConfig
    grid: dict[str, list[Any]] = Field(default_factory=dict)
    seeds: list[int] = Field(default_factory=lambda: [0])
    lanes: dict[str, LaneConfig] = Field(default_factory=dict)
    max_concurrent_runs: int = 4
    branches: list[BranchSpec] = Field(default_factory=list)
    judge: JudgeConfig | None = None
