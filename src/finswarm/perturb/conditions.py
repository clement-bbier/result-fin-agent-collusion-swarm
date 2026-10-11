"""Tiny condition language over online metrics.

Grammar (case-insensitive keywords, ``and`` binds tighter than ``or``)::

    expr   := term ("or" term)*
    term   := clause ("and" clause)*
    clause := METRIC OP NUMBER ["for" INT]
    OP     := ">=" | "<=" | ">" | "<" | "==" | "!="

``for N`` means the comparison held in each of the last ``N`` completed periods. Metrics are
the names known to ``OnlineMetrics`` (``delta``, ``delta_ma20``, ``price_mean``, ``period``...).
NaN never satisfies a comparison.
"""

from __future__ import annotations

import math
import operator
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

_OPS: dict[str, Callable[[float, float], bool]] = {
    ">=": operator.ge,
    "<=": operator.le,
    ">": operator.gt,
    "<": operator.lt,
    "==": operator.eq,
    "!=": operator.ne,
}
_CLAUSE = re.compile(
    r"^\s*(?P<metric>[A-Za-z_][A-Za-z0-9_]*)\s*(?P<op>>=|<=|==|!=|>|<)\s*"
    r"(?P<value>[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*"
    r"(?:for\s+(?P<n>\d+))?\s*$",
    re.IGNORECASE,
)


class MetricSource(Protocol):
    def value(self, name: str, offset: int = 0) -> float: ...

    def known(self, name: str) -> bool: ...

    def __len__(self) -> int: ...


@dataclass(frozen=True)
class Clause:
    metric: str
    op: str
    threshold: float
    for_n: int = 1

    def holds(self, src: MetricSource) -> bool:
        if len(src) < self.for_n:
            return False
        fn = _OPS[self.op]
        for k in range(self.for_n):
            v = src.value(self.metric, k)
            if not math.isfinite(v) or not fn(v, self.threshold):
                return False
        return True


@dataclass(frozen=True)
class Condition:
    text: str
    terms: tuple[tuple[Clause, ...], ...]  # OR of ANDs

    def holds(self, src: MetricSource) -> bool:
        return any(all(c.holds(src) for c in term) for term in self.terms)

    def metrics(self) -> set[str]:
        return {c.metric for term in self.terms for c in term}


class ConditionSyntaxError(ValueError):
    pass


def parse_condition(text: str) -> Condition:
    if not text or not text.strip():
        raise ConditionSyntaxError("empty condition")
    terms = []
    for or_part in re.split(r"\s+or\s+", text.strip(), flags=re.IGNORECASE):
        clauses = []
        for and_part in re.split(r"\s+and\s+", or_part, flags=re.IGNORECASE):
            m = _CLAUSE.match(and_part)
            if not m:
                raise ConditionSyntaxError(f"cannot parse clause {and_part!r} in {text!r}")
            n = int(m.group("n")) if m.group("n") else 1
            if n < 1:
                raise ConditionSyntaxError(f"'for' needs N >= 1 in {text!r}")
            clauses.append(Clause(m.group("metric"), m.group("op"), float(m.group("value")), n))
        terms.append(tuple(clauses))
    return Condition(text=text, terms=tuple(terms))


def validate_condition(cond: Condition, src: MetricSource) -> None:
    unknown = sorted(m for m in cond.metrics() if not src.known(m))
    if unknown:
        raise ConditionSyntaxError(f"unknown metrics {unknown} in {cond.text!r}")
