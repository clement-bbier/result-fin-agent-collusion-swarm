"""Lane scheduler: the only path from agents to providers.

``await scheduler.submit(req, lane) -> LLMResponse``

Per lane (``LaneConfig``): sliding-window limits for ``rps`` / ``rpm`` / ``tpm``, a daily
request quota ``rpd`` (UTC rollover, persisted through the ledger), ``max_in_flight``
workers, and ``budget_usd``. Requests wait in a priority queue that is FIFO within a run
and fair across runs (start-time fair queuing on ``tags["run_id"]``; ``tags["priority"]``,
lower first, overrides). Transient failures (429, 5xx, timeouts, connection errors) are
retried with full-jitter exponential backoff (cap 60 s, 6 attempts); ``retry-after`` is a
floor on the delay; each 429 multiplies the lane's rate and concurrency by 0.7 (AIMD) and
puts the lane in ``cooling`` until the delay has passed; the scale recovers additively
(+5 % of the configured rate per minute).

Budget guard before every dispatch: persisted spend (ledger) + reserved estimates of
in-flight calls + this call's estimate (output = ``max_tokens``) must stay within the
lane budget and within the global paid cap from ``configs/budget.yaml``
(``anthropic_total_usd: 200``, warnings at 100 and 150, hard stop at 195). When a lane
cannot serve, :class:`LaneExhausted` is raised so the runner can checkpoint and pause.

Cache: hits are served before any bucket (no rate, no quota, no cost); ``replay`` misses
raise :class:`ReplayMiss`; ``mock`` mode answers every request with ``MockLLM``.
"""

from __future__ import annotations

import asyncio
import calendar
import datetime as dt
import heapq
import itertools
import logging
import math
import random
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml

from finswarm.config.schema import LaneConfig
from finswarm.llm.cache import ResponseCache
from finswarm.llm.client import LLMClient, LLMRequest, LLMResponse
from finswarm.llm.errors import (
    FatalLLMError,
    LaneExhausted,
    RateLimited,
    RetryableError,
)
from finswarm.llm.ledger import Ledger, utc_day
from finswarm.llm.pricing import (
    Catalog,
    cache_dir,
    cost_ledger_path,
    estimate_cost,
    estimate_input_tokens,
    load_catalog,
    repo_root,
)
from finswarm.llm.timing import Clock, SystemClock

log = logging.getLogger(__name__)

MIN_SCALE = 0.05
AIMD_EVENT_S = 1.0  # 429s closer than this belong to the same congestion event
LONG_RETRY_AFTER_S = 300.0  # a longer retry-after means a quota window: exhaust the lane
CONCURRENCY_POLL_S = 0.05


class LaneState(StrEnum):
    OPEN = "open"
    COOLING = "cooling"
    EXHAUSTED = "exhausted"


# ----------------------------------------------------------------------------- budget
@dataclass(frozen=True)
class BudgetPolicy:
    """Global hard cap on paid spend (``configs/budget.yaml``, owned by the integrator)."""

    anthropic_total_usd: float = 200.0
    warn_usd: tuple[float, ...] = (100.0, 150.0)
    stop_usd: float = 195.0
    providers: tuple[str, ...] = ("anthropic",)
    per_run_max_usd: float | None = None  # enforced by the runner (pauses the run)

    @property
    def hard_stop(self) -> float:
        return min(self.stop_usd, self.anthropic_total_usd)


def load_budget_policy(path: str | Path | None = None) -> BudgetPolicy:
    """Read ``configs/budget.yaml`` if present; defaults otherwise.

    Recognised keys: ``anthropic_total_usd``, ``warn_usd`` (list) or ``warn_at_usd``,
    ``stop_usd`` or ``stop_at_usd``, ``providers``, ``per_run_max_usd``. Other keys are
    ignored.
    """
    p = Path(path) if path is not None else repo_root() / "configs" / "budget.yaml"
    if not p.is_file():
        return BudgetPolicy()
    with open(p, encoding="utf-8") as fh:
        data: dict[str, Any] = yaml.safe_load(fh) or {}
    default = BudgetPolicy()
    warn = data.get("warn_usd", data.get("warn_at_usd", default.warn_usd))
    if isinstance(warn, int | float):
        warn = (warn,)
    return BudgetPolicy(
        anthropic_total_usd=float(data.get("anthropic_total_usd", default.anthropic_total_usd)),
        warn_usd=tuple(float(w) for w in warn),
        stop_usd=float(data.get("stop_usd", data.get("stop_at_usd", default.stop_usd))),
        providers=tuple(data.get("providers", default.providers)),
        per_run_max_usd=(
            float(data["per_run_max_usd"]) if data.get("per_run_max_usd") is not None else None
        ),
    )


# ---------------------------------------------------------------------------- windows
class SlidingWindow:
    """At most ``limit`` units in any window of ``window`` seconds (exact sliding log).

    ``scale`` < 1 (AIMD) stretches the window, i.e. lowers the effective rate.
    """

    def __init__(self, limit: float, window: float, *, kind: str) -> None:
        self.limit = float(limit)
        self.window = float(window)
        self.kind = kind
        self._log: deque[list[float]] = deque()  # [t, amount]

    @classmethod
    def per_second(cls, rps: float) -> SlidingWindow:
        limit = max(1, math.floor(rps))
        return cls(limit, limit / rps, kind="rps")

    @classmethod
    def per_minute(cls, rpm: float) -> SlidingWindow:
        limit = max(1, math.floor(rpm))
        return cls(limit, 60.0 * limit / rpm, kind="rpm")

    def _prune(self, now: float, horizon: float) -> None:
        while self._log and self._log[0][0] <= now - horizon:
            self._log.popleft()

    def wait_time(self, now: float, amount: float = 1.0, scale: float = 1.0) -> float:
        horizon = self.window / max(scale, MIN_SCALE)
        self._prune(now, self.window / MIN_SCALE)
        inside = [e for e in self._log if e[0] > now - horizon]
        used = sum(e[1] for e in inside)
        if amount > self.limit:  # oversized request: wait for an empty window
            return 0.0 if used == 0 else inside[-1][0] + horizon - now
        if used + amount <= self.limit:
            return 0.0
        excess = used + amount - self.limit
        for t, a in inside:
            excess -= a
            if excess <= 0:
                return max(0.0, t + horizon - now)
        return horizon  # pragma: no cover - unreachable when amount <= limit

    def add(self, now: float, amount: float = 1.0) -> list[float]:
        entry = [now, float(amount)]
        self._log.append(entry)
        return entry


# ------------------------------------------------------------------------------ lanes
@dataclass(order=True)
class _Item:
    key: tuple[int, float, int]
    req: LLMRequest = field(compare=False)
    future: asyncio.Future[LLMResponse] = field(compare=False)
    est_cost: float = field(compare=False, default=0.0)
    est_tokens: int = field(compare=False, default=0)


@dataclass
class LaneStats:
    submitted: int = 0
    cache_hits: int = 0
    dispatched: int = 0
    succeeded: int = 0
    retries: int = 0
    rate_limited: int = 0
    failures: int = 0
    refusals: int = 0


class _Lane:
    def __init__(self, name: str, cfg: LaneConfig, clock: Clock) -> None:
        self.name = name
        self.cfg = cfg
        self.windows: list[SlidingWindow] = []
        if cfg.rps:
            self.windows.append(SlidingWindow.per_second(cfg.rps))
        if cfg.rpm:
            self.windows.append(SlidingWindow.per_minute(cfg.rpm))
        self.tpm = SlidingWindow(cfg.tpm, 60.0, kind="tpm") if cfg.tpm else None
        self.scale = 1.0
        self.scale_t = clock.now()
        self.last_decrease = -math.inf
        self.cooling_until = -math.inf
        self.exhausted_reason: str | None = None
        self.resume_at: float | None = None
        self.rpd_day = ""
        self.rpd_count = 0
        self.active = 0
        self.reserved = 0.0
        self.spend = 0.0
        self.stats = LaneStats()
        self.heap: list[_Item] = []
        self.run_vt: dict[str, float] = {}
        self.vt_floor = 0.0
        self.loop: asyncio.AbstractEventLoop | None = None
        self.cond: asyncio.Condition | None = None
        self.gate: asyncio.Lock | None = None
        self.workers: list[asyncio.Task[None]] = []


def _next_utc_midnight(wall: float) -> float:
    d = dt.datetime.fromtimestamp(wall, tz=dt.UTC).date() + dt.timedelta(days=1)
    return float(calendar.timegm(d.timetuple()))


class LaneScheduler:
    def __init__(
        self,
        lanes: dict[str, LaneConfig],
        clients: dict[str, LLMClient] | None = None,
        *,
        catalog: Catalog | None = None,
        cache: ResponseCache | None = None,
        ledger: Ledger | None = None,
        budget: BudgetPolicy | None = None,
        clock: Clock | None = None,
        seed: int = 0,
        max_attempts: int = 6,
        backoff_base_s: float = 1.0,
        backoff_cap_s: float = 60.0,
        timeout_s: float = 120.0,
        aimd_decrease: float = 0.7,
        aimd_increase_per_min: float = 0.05,
        spend_resync_s: float = 10.0,
        mock_client: LLMClient | None = None,
        on_alert: Callable[[str], None] | None = None,
    ) -> None:
        self.clock: Clock = clock or SystemClock()
        self._catalog = catalog
        self.cache = cache if cache is not None else ResponseCache(":memory:", mode="live")
        self.ledger = ledger if ledger is not None else Ledger(":memory:")
        self.budget = budget if budget is not None else BudgetPolicy()
        self.clients: dict[str, LLMClient] = dict(clients or {})
        self.mock_client = mock_client
        self.max_attempts = max_attempts
        self.backoff_base_s = backoff_base_s
        self.backoff_cap_s = backoff_cap_s
        self.timeout_s = timeout_s
        self.aimd_decrease = aimd_decrease
        self.aimd_increase_per_min = aimd_increase_per_min
        self.spend_resync_s = spend_resync_s
        self.on_alert = on_alert
        self.alerts: list[str] = []
        self._rng = random.Random(seed)
        self._seq = itertools.count()
        self._lanes = {name: _Lane(name, cfg, self.clock) for name, cfg in lanes.items()}
        self.ledger.lane_budgets.update({n: c.budget_usd for n, c in lanes.items()})
        self.ledger.global_cap_usd = self.budget.hard_stop
        self.ledger.global_providers = self.budget.providers
        self._global_reserved = 0.0
        self._global_spend = 0.0
        self._warned: set[float] = set()
        self._last_sync = -math.inf
        self._sync_spend(force=True)
        for lane in self._lanes.values():
            self._roll_day(lane)

    # ---------------------------------------------------------------- building
    @classmethod
    def from_catalog(
        cls,
        catalog: Catalog | None = None,
        *,
        lanes: Iterable[str] | None = None,
        mode: str = "live",
        cache_path: str | Path | None = None,
        ledger_path: str | Path | None = None,
        budget_path: str | Path | None = None,
        **kwargs: Any,
    ) -> LaneScheduler:
        """Build from ``configs/models.yaml``; cache and ledger default to ``cache_dir()``
        (``FINSWARM_CACHE_DIR`` or ``<repo>/cache/``), and every paid call is mirrored to the
        JSONL cost ledger (``cost_ledger_path()``) read by the sprint status scripts."""
        cat = catalog if catalog is not None else load_catalog()
        shared = cache_dir()
        cache_path = cache_path if cache_path is not None else shared / "llm_cache.sqlite"
        ledger_path = ledger_path if ledger_path is not None else shared / "llm_ledger.sqlite"
        jsonl_path = cost_ledger_path() if str(ledger_path) != ":memory:" else None
        names = list(lanes) if lanes is not None else list(cat.lanes)
        d = cat.defaults
        kwargs.setdefault("max_attempts", d.max_attempts)
        kwargs.setdefault("backoff_base_s", d.backoff_base_s)
        kwargs.setdefault("backoff_cap_s", d.backoff_cap_s)
        kwargs.setdefault("timeout_s", d.timeout_s)
        kwargs.setdefault("aimd_decrease", d.aimd_decrease)
        kwargs.setdefault("aimd_increase_per_min", d.aimd_increase_per_min)
        return cls(
            {n: cat.lanes[n] for n in names},
            catalog=cat,
            cache=ResponseCache(cache_path, mode=mode),  # type: ignore[arg-type]
            ledger=Ledger(ledger_path, jsonl_path=jsonl_path),
            budget=load_budget_policy(budget_path),
            **kwargs,
        )

    def _client(self, lane: _Lane) -> LLMClient:
        client = self.clients.get(lane.name)
        if client is None:
            from finswarm.llm.providers import make_client

            client = make_client(
                lane.cfg.provider, self._catalog or load_catalog(), clock=self.clock
            )
            self.clients[lane.name] = client
        return client

    def _price_of(self, model: str) -> Any:
        try:
            cat = self._catalog or load_catalog()
        except FileNotFoundError:
            return None
        return cat.price(model)

    # ------------------------------------------------------------ introspection
    def lane_state(self, lane: str) -> LaneState:
        L = self._lane(lane)
        if L.exhausted_reason is not None:
            return LaneState.EXHAUSTED
        if self.clock.now() < L.cooling_until:
            return LaneState.COOLING
        return LaneState.OPEN

    def lane_stats(self, lane: str) -> LaneStats:
        return self._lane(lane).stats

    def lane_scale(self, lane: str) -> float:
        return self._current_scale(self._lane(lane))

    def lane_spend(self, lane: str) -> float:
        return self._lane(lane).spend

    def global_spend(self) -> float:
        return self._global_spend

    def _lane(self, lane: str) -> _Lane:
        try:
            return self._lanes[lane]
        except KeyError:
            raise KeyError(f"unknown lane {lane!r}") from None

    # ----------------------------------------------------------------- public
    async def submit(self, req: LLMRequest, lane: str) -> LLMResponse:
        L = self._lane(lane)
        L.stats.submitted += 1
        if self.cache.mode == "mock":
            return await self._serve_mock(req)
        if req.provider != L.cfg.provider:
            raise ValueError(f"request provider {req.provider!r} != lane provider")
        if L.cfg.models and req.model not in L.cfg.models:
            raise ValueError(f"model {req.model!r} is not served by lane {lane!r}")

        cached = self.cache.get(req)  # raises ReplayMiss in replay mode
        if cached is not None:
            L.stats.cache_hits += 1
            self._record(L, req, cached, status="cache_hit", attempts=0)
            return cached

        est_cost = estimate_cost(self._price_of(req.model), req)
        self._check_open(L, est_cost)  # fail fast before queuing

        loop = asyncio.get_running_loop()
        self._ensure_workers(L, loop)
        assert L.cond is not None
        run = req.tags.get("run_id", "")
        try:
            priority = int(req.tags.get("priority", "0"))
        except ValueError:
            priority = 0
        vt = max(L.run_vt.get(run, 0.0), L.vt_floor) + 1.0
        L.run_vt[run] = vt
        item = _Item(
            key=(priority, vt, next(self._seq)),
            req=req,
            future=loop.create_future(),
            est_cost=est_cost,
            est_tokens=estimate_input_tokens(req) + req.max_tokens,
        )
        async with L.cond:
            heapq.heappush(L.heap, item)
            L.cond.notify()
        return await item.future

    async def aclose(self) -> None:
        for L in self._lanes.values():
            for w in L.workers:
                w.cancel()
            for w in L.workers:
                try:
                    await w
                except (asyncio.CancelledError, Exception):
                    pass
            L.workers = []
            L.loop = None

    # ---------------------------------------------------------------- workers
    def _ensure_workers(self, L: _Lane, loop: asyncio.AbstractEventLoop) -> None:
        if L.loop is not loop:
            L.loop = loop
            L.cond = asyncio.Condition()
            L.gate = asyncio.Lock()
            L.heap = []
            L.workers = []
        if not L.workers:
            L.workers = [
                loop.create_task(self._worker(L), name=f"lane-{L.name}-{i}")
                for i in range(max(1, L.cfg.max_in_flight))
            ]

    async def _worker(self, L: _Lane) -> None:
        assert L.cond is not None
        while True:
            async with L.cond:
                while not L.heap:
                    await L.cond.wait()
                item = heapq.heappop(L.heap)
                L.vt_floor = max(L.vt_floor, item.key[1])
            if item.future.done():  # cancelled by the caller
                continue
            try:
                resp = await self._run_item(L, item)
            except asyncio.CancelledError:
                if not item.future.done():
                    item.future.cancel()
                raise
            except BaseException as e:  # forwarded to the caller
                if not item.future.done():
                    item.future.set_exception(e)
            else:
                if not item.future.done():
                    item.future.set_result(resp)

    async def _run_item(self, L: _Lane, item: _Item) -> LLMResponse:
        req = item.req
        client = self._client(L)
        L.reserved += item.est_cost
        is_global = L.cfg.provider in self.budget.providers
        if is_global:
            self._global_reserved += item.est_cost
        attempts = 0
        try:
            while True:
                attempts += 1
                tpm_entry = await self._acquire(L, item)
                t0 = self.clock.now()
                try:
                    resp = await asyncio.wait_for(client.complete(req), timeout=self.timeout_s)
                except (RetryableError, TimeoutError) as e:
                    L.active -= 1
                    err = (
                        e
                        if isinstance(e, RetryableError)
                        else RetryableError("timeout", kind="timeout")
                    )
                    try:
                        delay = self._on_retryable(L, err, attempts)
                    except LaneExhausted:
                        L.stats.failures += 1
                        self._record(
                            L, req, None, status="failed", attempts=attempts, error=str(err)
                        )
                        raise
                    if attempts >= self.max_attempts:
                        L.stats.failures += 1
                        self._record(
                            L, req, None, status="failed", attempts=attempts, error=str(err)
                        )
                        raise LaneExhausted(L.name, "retries_exhausted") from e
                    L.stats.retries += 1
                    await self.clock.sleep(delay)
                    continue
                except FatalLLMError as e:
                    L.active -= 1
                    L.stats.failures += 1
                    self._record(L, req, None, status="error", attempts=attempts, error=str(e))
                    raise
                except BaseException:
                    L.active -= 1
                    raise
                L.active -= 1
                if tpm_entry is not None:
                    tpm_entry[1] = float(resp.input_tokens + resp.output_tokens)
                if resp.latency_ms <= 0:
                    resp = resp.model_copy(update={"latency_ms": (self.clock.now() - t0) * 1000.0})
                status = "refusal" if resp.refusal else ("error" if resp.error else "ok")
                L.stats.succeeded += 1
                L.stats.refusals += int(resp.refusal)
                self._account(L, resp.cost_usd)
                self._record(L, req, resp, status=status, attempts=attempts)
                self.cache.put(req, resp)
                return resp
        finally:
            L.reserved -= item.est_cost
            if is_global:
                self._global_reserved -= item.est_cost

    async def _acquire(self, L: _Lane, item: _Item) -> list[float] | None:
        """Wait until the lane may dispatch ``item``; account the dispatch. FIFO via gate."""
        assert L.gate is not None
        async with L.gate:
            while True:
                self._check_open(L, 0.0, reserved_included=True)
                now = self.clock.now()
                scale = self._current_scale(L)
                waits = [L.cooling_until - now]
                waits += [w.wait_time(now, 1.0, scale) for w in L.windows]
                if L.tpm is not None:
                    waits.append(L.tpm.wait_time(now, item.est_tokens, scale))
                limit = max(1, math.floor(L.cfg.max_in_flight * scale))
                if L.active >= limit:
                    waits.append(CONCURRENCY_POLL_S)
                wait = max(waits)
                if wait <= 1e-9:  # float slack: never spin on sub-nanosecond waits
                    break
                await self.clock.sleep(max(wait, 1e-6))
            now = self.clock.now()
            for w in L.windows:
                w.add(now, 1.0)
            entry = L.tpm.add(now, item.est_tokens) if L.tpm is not None else None
            L.rpd_count += 1
            L.active += 1
            L.stats.dispatched += 1
            return entry

    # ------------------------------------------------------------ retry / AIMD
    def _current_scale(self, L: _Lane) -> float:
        elapsed_min = max(0.0, self.clock.now() - L.scale_t) / 60.0
        return min(1.0, L.scale + self.aimd_increase_per_min * elapsed_min)

    def _backoff(self, attempt: int) -> float:
        ceiling = min(self.backoff_cap_s, self.backoff_base_s * (2 ** (attempt - 1)))
        return self._rng.uniform(0.0, ceiling)

    def _on_retryable(self, L: _Lane, err: RetryableError, attempt: int) -> float:
        now = self.clock.now()
        delay = self._backoff(attempt)
        if err.retry_after is not None:
            if err.retry_after > LONG_RETRY_AFTER_S:
                L.exhausted_reason = "quota"
                L.resume_at = self.clock.wall() + err.retry_after
                raise LaneExhausted(L.name, "quota", resume_at=L.resume_at) from err
            delay = max(delay, err.retry_after)
        if isinstance(err, RateLimited) or err.status == 429:
            L.stats.rate_limited += 1
            # one multiplicative decrease per congestion event: a burst of 429s from
            # concurrent workers within AIMD_EVENT_S counts once
            if now - L.last_decrease >= AIMD_EVENT_S:
                L.scale = max(MIN_SCALE, self._current_scale(L) * self.aimd_decrease)
                L.scale_t = now
                L.last_decrease = now
            L.cooling_until = max(L.cooling_until, now + delay)
        return delay

    # ------------------------------------------------------------------ budget
    def _roll_day(self, L: _Lane) -> None:
        day = utc_day(self.clock.wall())
        if day != L.rpd_day:
            L.rpd_day = day
            L.rpd_count = self.ledger.requests_on_day(L.name, day)
            if L.exhausted_reason in ("rpd", "quota"):
                L.exhausted_reason = None
                L.resume_at = None
        if L.exhausted_reason == "quota" and L.resume_at and self.clock.wall() >= L.resume_at:
            L.exhausted_reason = None
            L.resume_at = None

    def _sync_spend(self, force: bool = False) -> None:
        now = self.clock.now()
        if not force and now - self._last_sync < self.spend_resync_s:
            return
        self._last_sync = now
        for L in self._lanes.values():
            L.spend = self.ledger.spend(lane=L.name)
        self._global_spend = self.ledger.spend(providers=self.budget.providers)

    def _check_open(self, L: _Lane, est_cost: float, *, reserved_included: bool = False) -> None:
        self._roll_day(L)
        self._sync_spend()
        if L.exhausted_reason in ("rpd", "quota"):
            resume = L.resume_at if L.exhausted_reason == "quota" else None
            raise LaneExhausted(L.name, L.exhausted_reason, resume_at=resume)
        extra = 0.0 if reserved_included else est_cost
        if L.cfg.budget_usd is not None and L.spend + L.reserved + extra > L.cfg.budget_usd:
            L.exhausted_reason = "budget"
            raise LaneExhausted(L.name, "budget")
        if L.cfg.provider in self.budget.providers:
            total = self._global_spend + self._global_reserved + extra
            if total > self.budget.hard_stop:
                L.exhausted_reason = "global_cap"
                raise LaneExhausted(L.name, "global_cap")
        if L.exhausted_reason in ("budget", "global_cap"):  # budget raised since: reopen
            L.exhausted_reason = None
        if L.cfg.rpd is not None and L.rpd_count >= L.cfg.rpd:
            L.exhausted_reason = "rpd"
            L.resume_at = _next_utc_midnight(self.clock.wall())
            raise LaneExhausted(L.name, "rpd", resume_at=L.resume_at)

    def _account(self, L: _Lane, cost: float) -> None:
        L.spend += cost
        if L.cfg.provider in self.budget.providers:
            self._global_spend += cost
            for threshold in sorted(self.budget.warn_usd):
                if self._global_spend >= threshold and threshold not in self._warned:
                    self._warned.add(threshold)
                    msg = (
                        f"paid LLM spend ${self._global_spend:.2f} crossed ${threshold:.0f} "
                        f"(hard stop ${self.budget.hard_stop:.0f})"
                    )
                    self.alerts.append(msg)
                    log.warning(msg)
                    if self.on_alert is not None:
                        self.on_alert(msg)

    def _record(
        self,
        L: _Lane,
        req: LLMRequest,
        resp: LLMResponse | None,
        *,
        status: str,
        attempts: int,
        error: str | None = None,
    ) -> None:
        try:
            period = int(req.tags.get("period", "-1"))
        except ValueError:
            period = -1
        self.ledger.record(
            ts=self.clock.wall(),
            lane=L.name,
            provider=req.provider,
            model=req.model,
            run_id=req.tags.get("run_id", ""),
            agent=req.tags.get("agent", ""),
            period=period,
            input_tokens=resp.input_tokens if resp else 0,
            output_tokens=resp.output_tokens if resp else 0,
            cached_tokens=resp.cached_input_tokens if resp else 0,
            cost_usd=0.0 if resp is None or resp.cache_hit else resp.cost_usd,
            latency_ms=resp.latency_ms if resp else 0.0,
            status=status,
            attempts=attempts,
            request_id=resp.request_id if resp else None,
            error=error if error is not None else (resp.error if resp else None),
        )

    # -------------------------------------------------------------------- mock
    async def _serve_mock(self, req: LLMRequest) -> LLMResponse:
        client = self.mock_client
        if client is None:
            from finswarm.llm.providers.mock import MockLLM

            client = MockLLM(req.tags.get("mock_policy", "static_nash"), clock=self.clock)
        return await client.complete(req)
