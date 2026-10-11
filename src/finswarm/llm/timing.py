"""Clocks for the LLM layer (wall time, unrelated to simulated ticks).

The scheduler, ``MockLLM`` and ``Flaky`` take a :class:`Clock` so tests run on
:class:`VirtualClock`, where ``sleep`` advances virtual time instantly and
deterministically.
"""

from __future__ import annotations

import asyncio
import time
from typing import Protocol


class Clock(Protocol):
    def now(self) -> float:
        """Monotonic seconds (rate limiting, backoff, AIMD)."""
        ...

    def wall(self) -> float:
        """Epoch seconds (UTC day rollover of daily quotas, ledger timestamps)."""
        ...

    async def sleep(self, seconds: float) -> None: ...


class SystemClock:
    def now(self) -> float:
        return time.monotonic()

    def wall(self) -> float:
        return time.time()

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(max(0.0, seconds))


class VirtualClock:
    """Deterministic clock for tests: ``sleep(s)`` advances time by ``s`` immediately.

    Concurrent sleepers each advance the clock, so virtual time runs at least as fast as
    real time would; limits enforced against this clock are therefore still checked
    exactly (a limiter can only look *more* permissive if it is wrong, never less).
    """

    def __init__(self, start: float = 0.0, wall_start: float = 1_760_000_000.0) -> None:
        self.t = float(start)
        self._wall_offset = float(wall_start) - float(start)

    def now(self) -> float:
        return self.t

    def wall(self) -> float:
        return self._wall_offset + self.t

    def advance(self, seconds: float) -> None:
        self.t += max(0.0, seconds)

    async def sleep(self, seconds: float) -> None:
        if seconds > 0:
            self.t += seconds
        await asyncio.sleep(0)
