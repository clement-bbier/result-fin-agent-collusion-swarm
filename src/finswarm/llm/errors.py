"""Exceptions shared by providers, the scheduler, the cache and agents.

Providers translate SDK/HTTP errors into these classes so the scheduler never needs to
know which SDK raised what. ``LaneExhausted`` is the signal for the runner to
checkpoint and pause.
"""

from __future__ import annotations


class LLMError(Exception):
    """Base class for every error raised by ``finswarm.llm``."""


class RetryableError(LLMError):
    """Transient failure (429, 5xx, timeout, connection). The scheduler retries it."""

    def __init__(
        self,
        message: str,
        *,
        kind: str = "server",
        status: int | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.kind = kind  # rate_limit | server | timeout | connection
        self.status = status
        self.retry_after = retry_after


class RateLimited(RetryableError):
    """HTTP 429. ``retry_after`` (seconds) is honoured by the scheduler when present."""

    def __init__(self, message: str = "rate limited", *, retry_after: float | None = None) -> None:
        super().__init__(message, kind="rate_limit", status=429, retry_after=retry_after)


class FatalLLMError(LLMError):
    """Non-retryable failure (bad request, authentication, permission, unknown model)."""

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class LaneExhausted(LLMError):
    """A lane cannot serve more requests now: budget, daily quota, global cap or outage.

    The runner must checkpoint and pause. ``resume_at`` is a wall-clock epoch (seconds)
    when the lane is expected to reopen (daily quota rollover), else ``None``.
    """

    def __init__(self, lane: str, reason: str, *, resume_at: float | None = None) -> None:
        super().__init__(f"lane {lane!r} exhausted: {reason}")
        self.lane = lane
        self.reason = reason  # budget | global_cap | rpd | retries_exhausted
        self.resume_at = resume_at


class ReplayMiss(LLMError):
    """``llm.mode == "replay"`` and the request is not in the response cache."""

    def __init__(self, key: str) -> None:
        super().__init__(f"replay miss for cache key {key[:16]}...")
        self.key = key


def parse_retry_after(value: str | None) -> float | None:
    """Parse a ``retry-after`` header (seconds; HTTP dates are ignored)."""
    if value is None:
        return None
    try:
        secs = float(value.strip())
    except (TypeError, ValueError):
        return None
    return max(0.0, secs)
