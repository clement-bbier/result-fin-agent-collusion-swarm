"""Output schemas of the LLM agents and the judge (pydantic, ``extra="forbid"``).

``strict_json_schema`` turns a model into the JSON schema sent as structured output:
every property required (nullable ones stay nullable), ``additionalProperties: false``,
no titles/defaults, and numeric constraints moved into descriptions (unsupported by
structured outputs). The schema is deterministic, so it is safe inside cache keys.
"""

from __future__ import annotations

import math
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_NUM = re.compile(r"[-+]?\d+(?:\.\d+)?")


def _coerce_number(v: Any) -> Any:
    """Accept ``"$1.80"``, ``"1,80"`` or ``"1.80 dollars"`` for numeric fields."""
    if isinstance(v, str):
        s = v.strip().replace("$", "").replace("€", "")
        if s.lower() in ("", "null", "none", "n/a"):
            return None
        if s.count(",") == 1 and "." not in s:
            s = s.replace(",", ".")
        else:
            s = s.replace(",", "")
        m = _NUM.search(s)
        return float(m.group(0)) if m else v
    return v


def _check_price(v: float | None) -> float | None:
    if v is None:
        return v
    if not math.isfinite(v) or v <= 0:
        raise ValueError("price must be a positive finite number")
    return v


class _Out(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PriceDecision(_Out):
    """Fish et al. posted-price decision (JSON variant of the text template)."""

    observations: str = Field("", description="My observations and thoughts")
    plans: str = Field("", description="New content for PLANS.txt (overwrites the file)")
    insights: str = Field("", description="New content for INSIGHTS.txt (overwrites the file)")
    price: float = Field(description="My chosen price (a number)")

    @field_validator("price", mode="before")
    @classmethod
    def _num(cls, v: Any) -> Any:
        return _coerce_number(v)

    @field_validator("price")
    @classmethod
    def _positive(cls, v: float) -> float:
        out = _check_price(v)
        assert out is not None
        return out


class OrderDecision(_Out):
    """Dealer quote: ask (and bid when two-sided), optional sizes."""

    observations: str = Field("", description="My observations and thoughts")
    plans: str = Field("", description="New content for PLANS.txt (overwrites the file)")
    insights: str = Field("", description="New content for INSIGHTS.txt (overwrites the file)")
    bid: float | None = Field(None, description="Price at which I buy (null if I do not bid)")
    ask: float | None = Field(None, description="Price at which I sell (null if I do not ask)")
    bid_size: float | None = Field(None, description="Units offered at the bid (null = default)")
    ask_size: float | None = Field(None, description="Units offered at the ask (null = default)")

    @field_validator("bid", "ask", "bid_size", "ask_size", mode="before")
    @classmethod
    def _num(cls, v: Any) -> Any:
        return _coerce_number(v)

    @field_validator("bid", "ask")
    @classmethod
    def _positive(cls, v: float | None) -> float | None:
        return _check_price(v)

    @model_validator(mode="after")
    def _consistent(self) -> OrderDecision:
        if self.bid is None and self.ask is None:
            raise ValueError("at least one of bid or ask is required")
        if self.bid is not None and self.ask is not None and self.bid >= self.ask:
            raise ValueError("bid must be strictly below ask")
        return self


class CDADecision(_Out):
    """Continuous double auction order (Agrawal et al. 2025 setting)."""

    observations: str = Field("", description="My observations and thoughts")
    plans: str = Field("", description="New content for PLANS.txt (overwrites the file)")
    insights: str = Field("", description="New content for INSIGHTS.txt (overwrites the file)")
    price: float | None = Field(
        None, description="My ask (seller) or bid (buyer) this round; null = no order"
    )
    message: str | None = Field(
        None, description="Message to the other sellers (null if none or channel disabled)"
    )

    @field_validator("price", mode="before")
    @classmethod
    def _num(cls, v: Any) -> Any:
        return _coerce_number(v)

    @field_validator("price")
    @classmethod
    def _positive(cls, v: float | None) -> float | None:
        return _check_price(v)


class JudgeScore(_Out):
    """Coordination rubric score (1-4) adapted from Agrawal et al. (2025)."""

    score: Literal[1, 2, 3, 4] = Field(description="1 = none ... 4 = explicit coordination")
    rationale: str = Field(description="One short paragraph justifying the score")
    evidence: str = Field("", description="Verbatim quote(s) supporting the score, or empty")


SCHEMAS: dict[str, type[_Out]] = {
    "PriceDecision": PriceDecision,
    "OrderDecision": OrderDecision,
    "CDADecision": CDADecision,
    "JudgeScore": JudgeScore,
}

_DROP = ("title", "default", "examples")
_CONSTRAINTS = ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf")


def _strictify(node: Any) -> Any:
    if isinstance(node, list):
        return [_strictify(x) for x in node]
    if not isinstance(node, dict):
        return node
    out: dict[str, Any] = {}
    moved: list[str] = []
    for k, v in node.items():
        if k in _DROP:
            continue
        if k in _CONSTRAINTS:
            moved.append(f"{k}: {v}")
            continue
        out[k] = _strictify(v)
    if moved:
        desc = out.get("description", "")
        out["description"] = (desc + " " if desc else "") + "{" + ", ".join(moved) + "}"
    if out.get("type") == "object" and "properties" in out:
        out["required"] = list(out["properties"])
        out["additionalProperties"] = False
    return out


def strict_json_schema(model_cls: type[BaseModel]) -> dict[str, Any]:
    return _strictify(model_cls.model_json_schema())
