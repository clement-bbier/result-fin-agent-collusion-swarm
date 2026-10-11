"""SQLite (WAL) response cache: the basis of ``llm.mode: replay`` determinism.

The key is ``sha256`` of the canonical JSON of the fields that define a request:
provider, model, prompt_version, system, messages, schema, max_tokens, temperature,
effort and seed. ``tags`` (run id, agent, period, observation JSON) are deliberately
excluded so that a branch replays its parent's calls byte for byte before the branch
period. Agents make otherwise-identical requests distinct through ``seed``.

Modes: ``live`` (read, then write on miss), ``replay`` (read only; a miss raises
:class:`ReplayMiss`), ``mock`` (cache disabled).
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Literal

from finswarm.llm.client import LLMRequest, LLMResponse
from finswarm.llm.errors import ReplayMiss

CacheMode = Literal["live", "replay", "mock"]
_KEY_VERSION = 1


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def cache_key(req: LLMRequest) -> str:
    payload = {
        "v": _KEY_VERSION,
        "provider": req.provider,
        "model": req.model,
        "prompt_version": req.prompt_version,
        "system": req.system,
        "messages": req.messages,
        "schema": req.schema_,
        "max_tokens": req.max_tokens,
        "temperature": req.temperature,
        "effort": req.effort,
        "seed": req.seed,
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


class ResponseCache:
    """Thread-safe SQLite cache of :class:`LLMResponse` objects."""

    def __init__(self, path: str | Path = ":memory:", mode: CacheMode = "live") -> None:
        if mode not in ("live", "replay", "mock"):
            raise ValueError(f"unknown cache mode {mode!r}")
        self.mode: CacheMode = mode
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        if self.path != ":memory:":
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.execute(
            """CREATE TABLE IF NOT EXISTS responses (
                key TEXT PRIMARY KEY,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                prompt_version TEXT NOT NULL,
                created_ts REAL NOT NULL,
                response_json TEXT NOT NULL
            )"""
        )
        self.hits = 0
        self.misses = 0

    @property
    def enabled(self) -> bool:
        return self.mode != "mock"

    def get(self, req: LLMRequest) -> LLMResponse | None:
        """Return the cached response (``cache_hit=True``) or ``None``.

        In ``replay`` mode a miss raises :class:`ReplayMiss`.
        """
        if not self.enabled:
            return None
        key = cache_key(req)
        with self._lock:
            row = self._db.execute(
                "SELECT response_json FROM responses WHERE key = ?", (key,)
            ).fetchone()
        if row is None:
            self.misses += 1
            if self.mode == "replay":
                raise ReplayMiss(key)
            return None
        self.hits += 1
        resp = LLMResponse.model_validate_json(row[0])
        return resp.model_copy(update={"cache_hit": True, "latency_ms": 0.0, "cost_usd": 0.0})

    def put(self, req: LLMRequest, resp: LLMResponse) -> None:
        """Store a successful (or refused) response. Errors are never cached."""
        if self.mode != "live" or (resp.error is not None and not resp.refusal):
            return
        key = cache_key(req)
        stored = resp.model_copy(update={"cache_hit": False})
        with self._lock:
            self._db.execute(
                "INSERT OR IGNORE INTO responses VALUES (?, ?, ?, ?, ?, ?)",
                (
                    key,
                    req.provider,
                    req.model,
                    req.prompt_version,
                    time.time(),
                    stored.model_dump_json(),
                ),
            )

    def __len__(self) -> int:
        with self._lock:
            return int(self._db.execute("SELECT COUNT(*) FROM responses").fetchone()[0])

    def close(self) -> None:
        with self._lock:
            self._db.close()
