"""LLM client protocol (frozen interface, 2026-10-09).

Every provider adapter implements ``LLMClient``. Only ``finswarm.llm.providers`` imports SDKs.
Claude goes through the official ``anthropic`` SDK; never through an OpenAI-compatible shim.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


class Capabilities(BaseModel):
    model_config = ConfigDict(extra="forbid")
    supports_seed: bool = False
    supports_temperature: bool = True
    supports_json_schema: bool = False
    supports_cache_control: bool = False
    free: bool = False


class LLMRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    provider: str
    model: str
    system: str
    messages: list[dict[str, Any]]
    schema_: dict[str, Any] | None = Field(default=None, alias="schema")
    max_tokens: int = 2048
    temperature: float | None = None  # None = provider default (mandatory on Anthropic)
    seed: int | None = None
    effort: str | None = None  # Anthropic: low|medium|high
    prompt_version: str = "unversioned"
    tags: dict[str, str] = Field(default_factory=dict)  # run_id, agent, period, lane, ...


class LLMResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    request_id: str | None = None
    stop_reason: str = "end_turn"
    cache_hit: bool = False
    refusal: bool = False
    error: str | None = None


@runtime_checkable
class LLMClient(Protocol):
    name: str
    capabilities: Capabilities

    async def complete(self, req: LLMRequest) -> LLMResponse: ...
