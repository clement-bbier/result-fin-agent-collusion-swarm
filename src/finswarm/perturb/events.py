"""Perturbation events (frozen interface, 2026-10-09).

Events apply at period boundaries (after matching and accounting, before the next
observations). Each application writes an ``events`` row and a checkpoint.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class When(BaseModel):
    model_config = ConfigDict(extra="forbid")
    at_period: int | None = None
    at_tick: int | None = None
    # expression over online metrics, e.g. "delta_ma20 >= 0.5 for 20"
    on_condition: str | None = None


class _EventBase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    when: When = Field(default_factory=When)
    source: Literal["config", "branch", "condition"] = "config"
    note: str = ""


class SetParam(_EventBase):
    """Change one config path live (tick size, routing tau, fringe share, view.*, clock.*)."""

    type: Literal["set_param"] = "set_param"
    path: str
    value: Any


class ReplaceAgent(_EventBase):
    type: Literal["replace_agent"] = "replace_agent"
    slot: int
    model: str
    lane: str
    prompt: str | None = None
    keep_memory: bool = False
    keep_name: bool = True


class InjectNotice(_EventBase):
    type: Literal["inject_notice"] = "inject_notice"
    slots: list[int] | None = None  # None = all LLM slots
    template: str
    persist_periods: int = -1  # -1 = until end of run


class InfoShock(_EventBase):
    type: Literal["info_shock"] = "info_shock"
    d_fundamental: float
    diffusion: Literal["instant", "attention"] = "attention"


class RemoveLiquidity(_EventBase):
    type: Literal["remove_liquidity"] = "remove_liquidity"
    share: float
    swarm_types: list[str] = Field(default_factory=lambda: ["MM"])


class AddScriptedAgent(_EventBase):
    type: Literal["add_scripted_agent"] = "add_scripted_agent"
    policy: str  # e.g. "no_regret_mm", "zi_mm", "undercutter"
    params: dict[str, Any] = Field(default_factory=dict)


class Sting(_EventBase):
    """Regulator dealer shadows the mean quote, then raises it by ``delta`` for ``hold_epochs``."""

    type: Literal["sting"] = "sting"
    slot: int
    delta: float
    hold_epochs: int = 5
    shadow_epochs: int = 5
    probes: int = 1


class ForcedDeviation(_EventBase):
    """Force one slot to cut its price by ``delta`` for ``periods`` (impulse-response probe)."""

    type: Literal["forced_deviation"] = "forced_deviation"
    slot: int
    delta: float
    periods: int = 1


class Revert(_EventBase):
    type: Literal["revert"] = "revert"
    event_id: int


Event = Annotated[
    SetParam
    | ReplaceAgent
    | InjectNotice
    | InfoShock
    | RemoveLiquidity
    | AddScriptedAgent
    | Sting
    | ForcedDeviation
    | Revert,
    Field(discriminator="type"),
]
