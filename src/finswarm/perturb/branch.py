"""Branch a finished (or sufficiently advanced) run at a period boundary.

``branch(run_dir, period, events, name) -> new_run_dir``:

1. read the parent manifest (resolved config, run id) and build the child config: the
   parent's config, ``events`` appended with ``source="branch"`` (an event without a trigger
   fires at ``period``), ``parent_run_id``, ``branch_period = period``, name
   ``<parent>--<name>``, ``tags.replicate_id = name`` (``<parent replicate>/<name>`` for a
   branch of a branch: LLM request seeds are salted with it from ``period`` on, so the child
   draws fresh samples instead of replaying its parent or a sibling). A ``set_param`` on path ``periods`` in ``events`` sets the child's
   horizon (``BranchSpec`` has no horizon field) instead of being applied live;
2. locate the latest parent checkpoint ``<= period``; copy the parent's Parquet parts written
   before it (``run_id`` column rewritten) and restore the state;
3. replay the parent up to ``period`` with ``llm.mode = replay`` (bit-identical with the
   LLM cache; trivially so with mock/scripted agents);
4. at boundary ``period``: apply the parent's own events first, then re-spawn the
   ``interventions`` stream with ``(seed, branch_salt)``, where ``branch_salt`` hashes
   ``(parent_run_id, branch_period)`` only (``sim.run.branch_salt``), so a lever branch and
   its sham share it; each branch event then fires on its own child stream
   ``(seed, branch_salt, ordinal among branch events of its type, type hash)``
   (``PerturbationEngine.branch_stream_key``) and the shared stream is put back afterwards:
   common random numbers between a lever and its sham, and a lever-specific event never
   shifts the draws of the events it shares with the sham;
5. continue in the parent's LLM mode to the child's horizon.

The child's manifest carries ``parent_run_id``, ``branch_period``, ``branch_events_hash``.
Before ``period`` the child's tables equal the parent's row for row (except ``run_id``).
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter

from finswarm.config.schema import RunConfig
from finswarm.perturb.events import Event, SetParam, When

_EVENT = TypeAdapter(Event)


def _as_event(e: Any) -> Any:
    return _EVENT.validate_python(e) if isinstance(e, dict) else e


def child_config(
    parent_cfg: RunConfig,
    parent_run_id: str,
    period: int,
    events: Sequence[Any],
    name: str,
    *,
    periods: int | None = None,
) -> RunConfig:
    if period < 0:
        raise ValueError("branch period must be >= 0")
    inherited = [
        e.model_copy(update={"source": "config"}) if e.source == "branch" else e
        for e in parent_cfg.events
    ]
    horizon = periods
    new = []
    for raw in events:
        e = _as_event(raw)
        if isinstance(e, SetParam) and e.path == "periods":
            horizon = int(e.value)
            continue
        w = e.when
        if w.at_period is None and w.at_tick is None and not w.on_condition:
            w = When(at_period=period)
        elif w.at_period is not None and w.at_period < period:
            raise ValueError(f"branch event at_period {w.at_period} < branch period {period}")
        new.append(e.model_copy(update={"source": "branch", "when": w}))
    data = parent_cfg.model_dump(mode="json")
    data["events"] = [e.model_dump(mode="json") for e in [*inherited, *new]]
    data["name"] = f"{parent_cfg.name}--{name}"
    tags = dict(data.get("tags") or {})
    prev = tags.get("replicate_id")
    tags["replicate_id"] = f"{prev}/{name}" if prev else name
    data["tags"] = tags
    data["parent_run_id"] = parent_run_id
    data["branch_period"] = int(period)
    if horizon is not None:
        data["periods"] = int(horizon)
    cfg = RunConfig.model_validate(data)
    if cfg.periods < period:
        raise ValueError(f"branch period {period} beyond the child horizon {cfg.periods}")
    return cfg


def _parent(run_dir: Path) -> tuple[RunConfig, dict[str, Any]]:
    from finswarm.sim.manifest import read_manifest

    m = read_manifest(run_dir)
    if not m:
        raise FileNotFoundError(f"{run_dir}: no manifest.json")
    return RunConfig.model_validate(m["resolved_config"]), m


async def branch_async(
    run_dir: str | Path,
    period: int,
    events: Sequence[Any],
    name: str,
    *,
    out: str | Path | None = None,
    periods: int | None = None,
    provider: str | None = None,
    **runner_kw: Any,
) -> Any:
    """Async variant returning the child's ``RunResult`` (skips/resumes like any run)."""
    from finswarm.sim.run import BranchOrigin, Runner

    run_dir = Path(run_dir)
    parent_cfg, m = _parent(run_dir)
    done = int(m.get("periods_completed", 0))
    if done < period:
        raise ValueError(f"parent {m['run_id']} only completed {done} periods (< {period})")
    cfg = child_config(parent_cfg, m["run_id"], period, events, name, periods=periods)
    runner = Runner(
        cfg,
        out=Path(out) if out is not None else run_dir.parent,
        provider=provider if provider is not None else m.get("provider"),
        branch_origin=BranchOrigin(run_dir),
        extra_manifest={"branch_name": name, "branch_horizon": cfg.periods},
        **runner_kw,
    )
    return await runner.run()


def branch(
    run_dir: str | Path,
    period: int,
    events: Sequence[Any],
    name: str,
    **kw: Any,
) -> Path:
    """Branch ``run_dir`` at boundary ``period`` with ``events``; return the child run dir."""
    return asyncio.run(branch_async(run_dir, period, events, name, **kw)).run_dir


def resolve_branch_period(parent_dir: str | Path, at: str) -> int | None:
    """``"period:100"`` -> 100; ``"checkpoint:NAME"`` -> period of that named checkpoint."""
    from finswarm.sim.checkpoints import find_checkpoint

    kind, _, value = at.partition(":")
    kind = kind.strip().lower()
    if kind == "period":
        return int(value)
    if kind == "checkpoint":
        c = find_checkpoint(parent_dir, name=value.strip())
        return c.period if c is not None else None
    raise ValueError(f"bad branch point {at!r} (use 'period:N' or 'checkpoint:NAME')")
