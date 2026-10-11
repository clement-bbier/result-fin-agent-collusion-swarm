"""Metric registry (frozen interface, 2026-10-09).

Every metric is a pure function over polars frames, registered with its unit, aggregation
level, direction and provenance (``source`` = citation key, or ``"novel"``).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

Level = str  # "run" | "agent" | "pair" | "window"


@dataclass(frozen=True)
class MetricSpec:
    name: str
    unit: str
    level: Level
    higher_is_collusive: bool | None
    source: str
    fn: Callable[..., Any]
    description: str = ""


REGISTRY: dict[str, MetricSpec] = {}


def register_metric(
    name: str,
    unit: str,
    level: Level,
    higher_is_collusive: bool | None,
    source: str = "novel",
    description: str = "",
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        if name in REGISTRY:
            raise ValueError(f"metric {name!r} already registered")
        REGISTRY[name] = MetricSpec(
            name=name,
            unit=unit,
            level=level,
            higher_is_collusive=higher_is_collusive,
            source=source,
            fn=fn,
            description=description or (fn.__doc__ or "").strip(),
        )
        return fn

    return deco


def get_metric(name: str) -> MetricSpec:
    return REGISTRY[name]
