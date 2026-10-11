"""SQLite (WAL) ledger of every LLM call: tokens, cost, latency, status, attempts.

The ledger is the persisted source of truth for spend. The scheduler reads it at start
(and periodically) so budgets survive restarts and are shared by processes that use the
same file. Cache hits are recorded with ``status='cache_hit'`` and zero cost.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Literal

_COLUMNS = (
    "ts",
    "day",
    "lane",
    "provider",
    "model",
    "run_id",
    "agent",
    "period",
    "input_tokens",
    "output_tokens",
    "cached_tokens",
    "cost_usd",
    "latency_ms",
    "status",
    "attempts",
    "request_id",
    "error",
)

GroupBy = Literal["lane", "day", "run_id", "model", "provider"]


def utc_day(ts: float) -> str:
    return dt.datetime.fromtimestamp(ts, tz=dt.UTC).strftime("%Y-%m-%d")


class Ledger:
    def __init__(
        self,
        path: str | Path = ":memory:",
        *,
        lane_budgets: dict[str, float | None] | None = None,
        global_cap_usd: float | None = None,
        global_providers: tuple[str, ...] = ("anthropic",),
        jsonl_path: str | Path | None = None,
    ) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        #: append-only JSONL mirror of provider calls (``results/cost_ledger.jsonl``), one object
        #: per call with ``usd``; read by ``scripts/sprint_status.py`` and ``scripts/cost_report.py``
        self.jsonl_path = Path(jsonl_path) if jsonl_path is not None else None
        self.lane_budgets = dict(lane_budgets or {})
        self.global_cap_usd = global_cap_usd
        self.global_providers = tuple(global_providers)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        if self.path != ":memory:":
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.execute(
            """CREATE TABLE IF NOT EXISTS calls (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL, day TEXT NOT NULL, lane TEXT NOT NULL,
                provider TEXT NOT NULL, model TEXT NOT NULL, run_id TEXT, agent TEXT,
                period INTEGER, input_tokens INTEGER, output_tokens INTEGER,
                cached_tokens INTEGER, cost_usd REAL, latency_ms REAL, status TEXT,
                attempts INTEGER, request_id TEXT, error TEXT
            )"""
        )
        self._db.execute("CREATE INDEX IF NOT EXISTS calls_lane_day ON calls(lane, day)")
        self._db.execute("CREATE INDEX IF NOT EXISTS calls_provider ON calls(provider)")

    # ------------------------------------------------------------------ writes
    def record(
        self,
        *,
        lane: str,
        provider: str,
        model: str,
        run_id: str = "",
        agent: str = "",
        period: int = -1,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cached_tokens: int = 0,
        cost_usd: float = 0.0,
        latency_ms: float = 0.0,
        status: str = "ok",
        attempts: int = 1,
        request_id: str | None = None,
        error: str | None = None,
        ts: float | None = None,
    ) -> None:
        t = time.time() if ts is None else ts
        row = (
            t,
            utc_day(t),
            lane,
            provider,
            model,
            run_id,
            agent,
            int(period),
            int(input_tokens),
            int(output_tokens),
            int(cached_tokens),
            float(cost_usd),
            float(latency_ms),
            status,
            int(attempts),
            request_id,
            error,
        )
        with self._lock:
            self._db.execute(
                f"INSERT INTO calls ({', '.join(_COLUMNS)}) VALUES ({', '.join('?' * len(_COLUMNS))})",
                row,
            )
            if self.jsonl_path is not None and status != "cache_hit":
                self._append_jsonl(dict(zip(_COLUMNS, row, strict=True)))

    def _append_jsonl(self, rec: dict[str, Any]) -> None:
        """One JSON line per provider call (cache hits excluded: no request, no cost)."""
        ts = float(rec.pop("ts"))
        rec.pop("day", None)
        line = {
            "ts": dt.datetime.fromtimestamp(ts, tz=dt.UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            **rec,
            "usd": float(rec["cost_usd"]),
        }
        try:
            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            with self.jsonl_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(line, separators=(",", ":")) + "\n")
        except OSError:  # the SQLite row is the source of truth; never fail a call on the mirror
            pass

    # ------------------------------------------------------------------- reads
    def spend(
        self,
        *,
        lane: str | None = None,
        provider: str | None = None,
        providers: tuple[str, ...] | None = None,
        run_id: str | None = None,
        day: str | None = None,
    ) -> float:
        where, args = self._where(lane=lane, provider=provider, run_id=run_id, day=day)
        if providers:
            where.append(f"provider IN ({', '.join('?' * len(providers))})")
            args.extend(providers)
        sql = "SELECT COALESCE(SUM(cost_usd), 0) FROM calls"
        if where:
            sql += " WHERE " + " AND ".join(where)
        with self._lock:
            return float(self._db.execute(sql, args).fetchone()[0])

    def requests_on_day(self, lane: str, day: str) -> int:
        """Provider requests sent on ``day`` by ``lane`` (attempts, cache hits excluded)."""
        with self._lock:
            row = self._db.execute(
                "SELECT COALESCE(SUM(attempts), 0) FROM calls "
                "WHERE lane = ? AND day = ? AND status != 'cache_hit'",
                (lane, day),
            ).fetchone()
        return int(row[0])

    def aggregate(self, by: GroupBy = "lane") -> list[dict[str, Any]]:
        if by not in ("lane", "day", "run_id", "model", "provider"):
            raise ValueError(f"cannot aggregate by {by!r}")
        sql = (
            f"SELECT {by}, COUNT(*), COALESCE(SUM(attempts),0), COALESCE(SUM(input_tokens),0), "
            "COALESCE(SUM(output_tokens),0), COALESCE(SUM(cached_tokens),0), "
            "COALESCE(SUM(cost_usd),0), COALESCE(AVG(latency_ms),0), "
            "SUM(CASE WHEN status = 'cache_hit' THEN 1 ELSE 0 END), "
            "SUM(CASE WHEN status NOT IN ('ok', 'cache_hit') THEN 1 ELSE 0 END) "
            f"FROM calls GROUP BY {by} ORDER BY {by}"
        )
        keys = (
            by,
            "calls",
            "attempts",
            "input_tokens",
            "output_tokens",
            "cached_tokens",
            "cost_usd",
            "mean_latency_ms",
            "cache_hits",
            "non_ok",
        )
        with self._lock:
            rows = self._db.execute(sql).fetchall()
        return [dict(zip(keys, r, strict=True)) for r in rows]

    def remaining_budget(self, lane: str | None = None) -> float:
        """USD left on ``lane`` (its ``budget_usd``) or on the global paid cap if ``None``.

        ``math.inf`` when no budget applies.
        """
        if lane is not None:
            budget = self.lane_budgets.get(lane)
            if budget is None:
                return math.inf
            return budget - self.spend(lane=lane)
        if self.global_cap_usd is None:
            return math.inf
        return self.global_cap_usd - self.spend(providers=self.global_providers)

    def __len__(self) -> int:
        with self._lock:
            return int(self._db.execute("SELECT COUNT(*) FROM calls").fetchone()[0])

    def close(self) -> None:
        with self._lock:
            self._db.close()

    @staticmethod
    def _where(**filters: Any) -> tuple[list[str], list[Any]]:
        where: list[str] = []
        args: list[Any] = []
        for col, val in filters.items():
            if val is not None:
                where.append(f"{col} = ?")
                args.append(val)
        return where, args
