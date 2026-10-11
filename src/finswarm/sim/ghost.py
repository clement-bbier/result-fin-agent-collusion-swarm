"""Ghost markets: K isolated copies of a run where only slot ``k`` keeps its own agent.

For every LLM slot ``k`` of a finished run, ghost ``k`` re-runs the same configuration and
seed with every other initial slot ``j`` replaced by an open-loop ``ReplayRival`` of ``j``'s
logged quotes (``rival_policy="replay_parent"``) or by a scripted rival (``static_nash``,
``zi_mm``). Dependence measures computed on the ghost are the null for the "net of ghost"
corrections: what remains when rivals cannot react.

If the run is itself a branch (``parent_run_id`` + ``branch_period``) and its parent lives in
the same directory, ghost ``k`` starts from the parent's checkpoint (identical history before
the branch period) and the replacements apply at the branch period; otherwise the replacements
apply from period 0. Outputs: ``run_dir/ghost/<k>/``.

Each ghost is a replicate: ``tags.replicate_id = "ghost:<k>"`` (``"<run replicate>/ghost:<k>"``
when the run is itself a branch, e.g. ``sham/ghost:0``) salts the focal agent's request seeds
from the replacement period on. Without it the focal prompt reproduces the run's own prompt
(same checkpoint, same seed, rivals replaying the logged quotes), so every call would be a
cache hit and the ghost a byte-identical copy of the run.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from finswarm.config.schema import RunConfig
from finswarm.sim.manifest import read_manifest

GHOST_SEED_SALT = 0x6057


def ghost_replicate_id(cfg: RunConfig, k: int) -> str:
    parent = cfg.tags.get("replicate_id")
    return f"{parent}/ghost:{k}" if parent else f"ghost:{k}"


def ghost_config(cfg: RunConfig, parent_run_id: str, k: int, rival_policy: str) -> RunConfig:
    data = cfg.model_dump(mode="json")
    data["name"] = f"{cfg.name}--ghost{k}"
    data["ghost"] = {**data["ghost"], "enabled": False}
    data["tags"] = {
        **data.get("tags", {}),
        "ghost_of": parent_run_id,
        "ghost_slot": str(k),
        "ghost_rival_policy": rival_policy,
        "replicate_id": ghost_replicate_id(cfg, k),
    }
    return RunConfig.model_validate(data)


def rival_spec(cfg: RunConfig, j: int, rival_policy: str, source: Path) -> dict[str, Any]:
    sc = cfg.llm.slots[j]
    base = {
        "type": "scripted",
        "seed": [cfg.seed, GHOST_SEED_SALT, j],
        "role": sc.role,
        "memory_periods": sc.memory_periods,
        "name": sc.name,
    }
    if rival_policy == "replay_parent":
        return {**base, "policy": "replay_rival", "params": {"slot": j, "source": str(source)}}
    return {**base, "policy": rival_policy, "params": {}}


def ghost_slots(run_dir: str | Path) -> list[int]:
    """The slots that get a ghost: the run's initial LLM slots (all initial slots if the
    manifest lists none)."""
    m = read_manifest(Path(run_dir))
    cfg = RunConfig.model_validate(m["resolved_config"])
    n_initial = len(cfg.llm.slots)
    llm = [s["slot"] for s in m.get("slots", []) if s["kind"] == "LLM" and s["slot"] < n_initial]
    return llm or list(range(n_initial))


async def run_ghosts_async(
    run_dir: str | Path,
    *,
    rival_policy: str = "replay_parent",
    slots: list[int] | None = None,
    provider: str | None = None,
    **runner_kw: Any,
) -> list[Any]:
    from finswarm.sim.run import BranchOrigin, Runner

    run_dir = Path(run_dir)
    m = read_manifest(run_dir)
    if m.get("status") not in ("ok", "done"):
        raise ValueError(f"{run_dir}: ghosts need a finished run (status={m.get('status')})")
    cfg = RunConfig.model_validate(m["resolved_config"])
    n_initial = len(cfg.llm.slots)
    if slots is None:
        slots = ghost_slots(run_dir)
    runner_kw = {k: v for k, v in runner_kw.items() if v is not None}
    results = []
    for k in slots:
        gcfg = ghost_config(cfg, m["run_id"], k, rival_policy)
        overrides = {
            j: rival_spec(cfg, j, rival_policy, run_dir) for j in range(n_initial) if j != k
        }
        gdir = run_dir / "ghost" / str(k)
        parent_dir = run_dir.parent / cfg.parent_run_id if cfg.parent_run_id is not None else None
        if parent_dir is not None and cfg.branch_period is not None and parent_dir.exists():
            runner = Runner(
                gcfg,
                run_dir=gdir,
                provider=provider if provider is not None else m.get("provider"),
                branch_origin=BranchOrigin(parent_dir),
                agent_overrides=overrides,
                overrides_at=cfg.branch_period,
                replay_source=run_dir,
                extra_manifest={"ghost_of": m["run_id"], "ghost_slot": k},
                **runner_kw,
            )
        else:
            data = gcfg.model_dump(mode="json")
            data["parent_run_id"] = None
            data["branch_period"] = None
            data["events"] = [
                {**e, "source": "config"} if e.get("source") == "branch" else e
                for e in data["events"]
            ]
            runner = Runner(
                RunConfig.model_validate(data),
                run_dir=gdir,
                provider=provider if provider is not None else m.get("provider"),
                agent_overrides=overrides,
                overrides_at=0,
                replay_source=run_dir,
                extra_manifest={"ghost_of": m["run_id"], "ghost_slot": k},
                **runner_kw,
            )
        results.append(await runner.run())
    return results


def run_ghosts(run_dir: str | Path, **kw: Any) -> list[Any]:
    return asyncio.run(run_ghosts_async(run_dir, **kw))
