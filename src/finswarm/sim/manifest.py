"""Run manifests: everything needed to identify, reproduce and audit a run.

``manifest.json`` is rewritten atomically at start, at every checkpoint and at the end.
``status`` is one of ``running | paused | failed | ok``.
"""

from __future__ import annotations

import datetime as _dt
import json
import platform
import subprocess
import sys
from functools import lru_cache
from importlib import metadata
from pathlib import Path
from typing import Any

from finswarm.core.rng import STREAMS
from finswarm.store.writer import atomic_write_text

MANIFEST_FILE = "manifest.json"
DONE_STATUSES = frozenset({"ok", "done"})
LLM_TOTAL_KEYS = (
    "calls",
    "cache_hits",
    "input_tokens",
    "output_tokens",
    "cached_tokens",
    "cost_usd",
    "parse_failures",
    "refusals",
)
_PACKAGES = ("finswarm", "numpy", "polars", "pyarrow", "pydantic", "anthropic", "mistralai")


def utc_now() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds")


def empty_llm_totals() -> dict[str, float]:
    return {k: (0.0 if k == "cost_usd" else 0) for k in LLM_TOTAL_KEYS}


@lru_cache(maxsize=1)
def git_info() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]

    def _git(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args], cwd=root, capture_output=True, text=True, timeout=10, check=True
            )
            return out.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {"commit": commit, "dirty": bool(status) if status is not None else None}


@lru_cache(maxsize=1)
def package_versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for p in _PACKAGES:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = None
    return out


def seeds_info(seed: int, branch_salt: int | None = None) -> dict[str, Any]:
    """How every named stream was derived (``SeedSequence(seed).spawn`` order).

    ``interventions_respawn = [seed, branch_salt]`` is the ``SeedSequence`` of the
    ``interventions`` stream re-spawned at the branch period; ``branch_salt`` hashes
    ``(parent_run_id, branch_period)`` only (``sim.run.branch_salt``), so every branch of the
    same trunk at the same period (lever, sham, sham2) shares it. Each branch event draws from
    a child of that sequence, ``[seed, branch_salt, ordinal, type hash]``
    (``PerturbationEngine.branch_stream_key``), not recorded here because it is a pure function
    of the branch events already in the manifest (``branch_events_hash``). LLM request seeds are
    salted separately with ``tags.replicate_id`` (``agents.llm.factory.replicate_of``).
    """
    streams = {name: {"entropy": int(seed), "spawn_key": [i]} for i, name in enumerate(STREAMS)}
    info: dict[str, Any] = {"root": int(seed), "streams": streams}
    if branch_salt is not None:
        info["interventions_respawn"] = [int(seed), int(branch_salt)]
    return info


def environment() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


def read_manifest(run_dir: str | Path) -> dict[str, Any]:
    path = Path(run_dir) / MANIFEST_FILE
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_manifest(run_dir: str | Path, manifest: dict[str, Any]) -> None:
    atomic_write_text(
        Path(run_dir) / MANIFEST_FILE,
        json.dumps(manifest, sort_keys=True, indent=1, ensure_ascii=False, default=str),
    )
