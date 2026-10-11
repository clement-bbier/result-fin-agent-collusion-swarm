"""Model catalogue (``configs/models.yaml``), prices and cost arithmetic.

Costs are USD; prices are USD per million tokens. Token conventions used across the
package (and in :class:`finswarm.llm.client.LLMResponse`):

* ``input_tokens`` = every prompt token (uncached + cache read + cache write);
* ``cached_input_tokens`` = prompt tokens served from the provider cache (cache read).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from finswarm.config.schema import LaneConfig
from finswarm.llm.client import Capabilities, LLMRequest

_MTOK = 1_000_000.0


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Price(_Strict):
    input: float = 0.0
    output: float = 0.0
    cache_read: float = 0.0
    cache_write: float = 0.0
    long_context_tokens: int | None = None
    input_long: float | None = None
    output_long: float | None = None
    verified: bool = False

    @property
    def free(self) -> bool:
        return self.input == 0.0 and self.output == 0.0


class ProviderSpec(_Strict):
    kind: Literal["anthropic", "mistral", "openai_compat", "mock"]
    env_key: str | None = None
    base_url: str | None = None
    json_mode: bool = True
    capabilities: Capabilities = Field(default_factory=Capabilities)


class ModelSpec(_Strict):
    provider: str
    role: str = ""
    price: Price = Field(default_factory=Price)


class Defaults(_Strict):
    timeout_s: float = 120.0
    max_attempts: int = 6
    backoff_base_s: float = 1.0
    backoff_cap_s: float = 60.0
    aimd_decrease: float = 0.7
    aimd_increase_per_min: float = 0.05


class Catalog(_Strict):
    version: int = 1
    defaults: Defaults = Field(default_factory=Defaults)
    providers: dict[str, ProviderSpec] = Field(default_factory=dict)
    models: dict[str, ModelSpec] = Field(default_factory=dict)
    lanes: dict[str, LaneConfig] = Field(default_factory=dict)
    # short names accepted in run configs (``haiku``) -> verbatim catalogue ids
    aliases: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _aliases_point_to_models(self) -> Catalog:
        bad = {a: m for a, m in self.aliases.items() if m not in self.models}
        if bad:
            raise ValueError(f"aliases point to unknown models: {bad}")
        return self

    def resolve_model(self, model: str) -> str:
        """Verbatim catalogue id for ``model`` (an id or an alias); unknown names unchanged."""
        return model if model in self.models else self.aliases.get(model, model)

    def price(self, model: str) -> Price | None:
        spec = self.models.get(self.resolve_model(model))
        return spec.price if spec is not None else None

    def lane_for_model(self, model: str) -> str | None:
        model = self.resolve_model(model)
        for name, lane in self.lanes.items():
            if model in lane.models:
                return name
        return None


def repo_root() -> Path:
    """Directory holding ``configs/`` (env ``FINSWARM_ROOT`` > cwd ancestors > package)."""
    env = os.environ.get("FINSWARM_ROOT")
    if env:
        return Path(env)
    for parent in (Path.cwd(), *Path.cwd().parents):
        if (parent / "configs" / "models.yaml").is_file():
            return parent
    return Path(__file__).resolve().parents[3]


def cache_dir() -> Path:
    """Directory of the response cache and the SQLite spend ledger: ``FINSWARM_CACHE_DIR`` when
    set (every worktree and terminal of a sprint points it at the root checkout's ``cache/`` so
    one ledger enforces the 195 USD stop), else ``<repo>/cache``."""
    env = os.environ.get("FINSWARM_CACHE_DIR")
    return Path(env) if env else repo_root() / "cache"


def cost_ledger_path() -> Path:
    """The append-only JSONL mirror of paid calls read by ``scripts/sprint_status.py`` and
    ``scripts/cost_report.py``: ``FINSWARM_COST_LEDGER`` when set, else
    ``<repo>/results/cost_ledger.jsonl``."""
    env = os.environ.get("FINSWARM_COST_LEDGER")
    return Path(env) if env else repo_root() / "results" / "cost_ledger.jsonl"


def default_catalog_path() -> Path:
    return repo_root() / "configs" / "models.yaml"


@lru_cache(maxsize=8)
def _load_catalog_cached(path: str) -> Catalog:
    with open(path, encoding="utf-8") as fh:
        data: dict[str, Any] = yaml.safe_load(fh) or {}
    return Catalog.model_validate(data)


def load_catalog(path: str | Path | None = None) -> Catalog:
    p = Path(path) if path is not None else default_catalog_path()
    return _load_catalog_cached(str(p.resolve()))


def cost_usd(
    price: Price | None,
    *,
    uncached_input: int,
    output: int,
    cache_read: int = 0,
    cache_write: int = 0,
) -> float:
    """Exact cost of one call from its token breakdown."""
    if price is None:
        return 0.0
    in_rate, out_rate = price.input, price.output
    prompt_total = uncached_input + cache_read + cache_write
    if price.long_context_tokens is not None and prompt_total > price.long_context_tokens:
        in_rate = price.input_long if price.input_long is not None else in_rate
        out_rate = price.output_long if price.output_long is not None else out_rate
    scale = in_rate / price.input if price.input else 1.0
    return (
        uncached_input * in_rate
        + output * out_rate
        + cache_read * price.cache_read * scale
        + cache_write * price.cache_write * scale
    ) / _MTOK


def estimate_input_tokens(req: LLMRequest) -> int:
    """Conservative prompt-size estimate (3.5 characters per token)."""
    chars = len(req.system)
    for m in req.messages:
        content = m.get("content", "")
        if isinstance(content, str):
            chars += len(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    chars += len(str(block.get("text", "")))
    if req.schema_:
        chars += len(str(req.schema_))
    return int(chars / 3.5) + 16


def estimate_call_cost(
    model: str, in_tokens: int, out_tokens: int, *, catalog: Catalog | None = None
) -> float:
    """Projected USD of one call of ``model`` (id or alias) with the given token counts.

    Used by campaign dry runs. Free and mock models cost 0; an unknown model is an error
    (a dry run must never silently price a typo at zero).
    """
    cat = catalog if catalog is not None else load_catalog()
    resolved = cat.resolve_model(model)
    spec = cat.models.get(resolved)
    if spec is None:
        raise KeyError(
            f"model {model!r} is not in configs/models.yaml (models or aliases); "
            f"known: {sorted([*cat.models, *cat.aliases])}"
        )
    provider = cat.providers.get(spec.provider)
    if spec.price.free or (provider is not None and provider.kind == "mock"):
        return 0.0
    return cost_usd(spec.price, uncached_input=int(in_tokens), output=int(out_tokens))


def estimate_cost(price: Price | None, req: LLMRequest) -> float:
    """Upper-bound cost used to reserve budget before dispatch (output = ``max_tokens``)."""
    if price is None or price.free:
        return 0.0
    return cost_usd(price, uncached_input=estimate_input_tokens(req), output=req.max_tokens)
