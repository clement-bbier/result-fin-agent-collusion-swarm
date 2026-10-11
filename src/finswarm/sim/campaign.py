"""Campaigns: grid x seeds, then branches once the parents' checkpoints exist.

* Grid keys are dotted config paths; a key ``"a+b"`` zips several paths (its values are lists
  of the same length), e.g. ``"market.alpha+market.price_grid.max": [[1, 4], [3.2, 12.8]]``.
  Cells are named ``<base.name>-<k=v_...>`` (last path components).
* Runs share one asyncio loop under a semaphore (``max_concurrent_runs`` or ``workers``).
* Resume is free: a run whose manifest says ``ok`` is skipped, any other run directory with a
  checkpoint resumes from its latest one, branch children likewise.
* ``BranchSpec.parent`` matches a run name, a cell label, ``c<index>`` or the base name, or
  the children of an earlier branch of the same campaign (their run name
  ``<parent run name>--<branch name>``, e.g. ``TR-H--ext``, or that branch's name): branches
  are processed in file order, each wave after the previous one has finished, so a lever can
  branch from the trend-rule extension child instead of the trunk (``scripts/resolve_registered.py
  --shift``). ``at`` is ``period:N`` or ``checkpoint:NAME`` (skipped for parents where it
  never fired).
* ``BranchSpec.ghost`` (a rival policy) adds K ghost sub-runs per finished child, one per
  initial LLM slot (``finswarm.sim.ghost``, written to ``<child>/ghost/<k>/``); each ghost is
  its own job under the semaphore and is skipped on resume like any run.
* Dry run: number of runs, LLM calls (``n_llm x periods x epochs`` per run; branches count
  only periods after the branch point; each ghost counts its one focal LLM slot after the
  branch point) and USD from a tokens-per-call profile.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import math
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from finswarm.config.load import load_campaign_config
from finswarm.config.load import run_id as make_run_id
from finswarm.config.schema import BranchSpec, CampaignConfig, RunConfig
from finswarm.sim.integration import slot_is_scripted
from finswarm.sim.manifest import DONE_STATUSES, read_manifest
from finswarm.store.writer import atomic_write_text

#: The registered "collusion established" rule (PREREG Gate 2,
#: ``delta_p_window(71, 100) >= 0.30``):
#: mean per-period Delta_pi over periods 71-100 (1-indexed; 0-indexed 70-99) >= 0.30, read at
#: boundary 100. As an ``on_condition`` over the online metrics: ``delta_ma30`` is the trailing
#: 30-period mean (NaN unless all 30 are finite) and ``period`` the last completed period.
ESTABLISHED_CONDITION = "period >= 99 and delta_ma30 >= 0.30"


@dataclass(frozen=True)
class TokenProfile:
    """Average tokens per LLM call used by dry-run projections (a Fish et al. prompt with a
    100-period history is about 1.5k input tokens; answers about 300 output tokens)."""

    input_tokens: int = 1500
    output_tokens: int = 300


@dataclass
class PlannedRun:
    cfg: RunConfig
    run_id: str
    cell: str
    cell_index: int
    seed: int
    n_llm: int
    calls: int
    usd: float = 0.0
    kind: str = "run"
    branch: str | None = None
    parent_run_id: str | None = None
    approx: bool = False


@dataclass
class Projection:
    runs: int
    branches: int
    calls: int
    usd: float
    rows: list[dict[str, Any]] = field(default_factory=list)
    by_lane: dict[str, float] = field(default_factory=dict)
    over_budget: list[str] = field(default_factory=list)
    pricing: str = "configs/models.yaml"
    ghosts: int = 0

    def table(self) -> str:
        lines = [f"{'kind':7} {'name':48} {'seed':>5} {'n_llm':>5} {'calls':>8} {'usd':>9}"]
        for r in self.rows:
            approx = "~" if r["approx"] else " "
            lines.append(
                f"{r['kind']:7} {r['name'][:48]:48} {r['seed']:>5} {r['n_llm']:>5} "
                f"{r['calls']:>8}{approx}{r['usd']:>9.4f}"
            )
        ghosts = f"{self.ghosts} ghosts, " if self.ghosts else ""
        lines.append(
            f"TOTAL: {self.runs} runs, {self.branches} branches, {ghosts}{self.calls} calls, "
            f"{self.usd:.4f} USD"
        )
        lines.append(f"pricing: {self.pricing}")
        for lane, usd in sorted(self.by_lane.items()):
            flag = "  OVER BUDGET" if lane in self.over_budget else ""
            lines.append(f"lane {lane}: {usd:.4f} USD{flag}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------------------------
# expansion
# ---------------------------------------------------------------------------------------------
def _set_path(data: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    cur: Any = data
    for k in parts[:-1]:
        cur = cur[int(k)] if isinstance(cur, list) else cur.setdefault(k, {})
    if isinstance(cur, list):
        cur[int(parts[-1])] = value
    else:
        cur[parts[-1]] = value


def _label(key: str, value: Any) -> str:
    paths = key.split("+")
    vals = value if len(paths) > 1 else [value]
    bits = [f"{p.split('.')[-1]}={v}" for p, v in zip(paths, vals, strict=True)]
    return re.sub(r"[^A-Za-z0-9._=-]+", "", "_".join(bits))


def expand_cells(cc: CampaignConfig) -> list[tuple[str, dict[str, Any]]]:
    """``[(cell_name, overrides)]`` for the cartesian product of the grid."""
    keys = list(cc.grid)
    if not keys:
        return [(cc.base.name, {})]
    for k in keys:
        n = len(k.split("+"))
        if n > 1 and any(not isinstance(v, list | tuple) or len(v) != n for v in cc.grid[k]):
            raise ValueError(f"grid key {k!r} zips {n} paths: each value must be a list of {n}")
    cells = []
    for combo in itertools.product(*(cc.grid[k] for k in keys)):
        overrides = dict(zip(keys, combo, strict=True))
        label = "_".join(_label(k, v) for k, v in overrides.items())
        cells.append((f"{cc.base.name}-{label}", overrides))
    return cells


def cell_config(cc: CampaignConfig, name: str, overrides: dict[str, Any], seed: int) -> RunConfig:
    data = cc.base.model_dump(mode="json")
    for key, value in overrides.items():
        paths = key.split("+")
        vals = value if len(paths) > 1 else [value]
        for p, v in zip(paths, vals, strict=True):
            _set_path(data, p, v)
    data["name"] = name
    data["seed"] = int(seed)
    tags = dict(data.get("tags") or {})
    tags["campaign"] = cc.name
    if cc.prereg_id:
        tags["prereg_id"] = cc.prereg_id
    data["tags"] = tags
    return RunConfig.model_validate(data)


def n_llm_slots(cfg: RunConfig) -> int:
    return sum(1 for s in cfg.llm.slots if not slot_is_scripted(s))


def epochs_per_period(cfg: RunConfig) -> int:
    # decisions per slot and period: exact for the sync clock; an upper bound under poisson
    # (expected tpp / epoch_ticks there)
    return max(1, math.ceil(max(1, cfg.ticks_per_period) / max(1, cfg.clock.epoch_ticks)))


def pricing_source() -> str:
    from finswarm.llm.pricing import default_catalog_path

    return f"finswarm.llm.pricing.estimate_call_cost ({default_catalog_path().name})"


def usd_per_call(cfg: RunConfig, cc: CampaignConfig | None, profile: TokenProfile) -> float:
    """Mean projected USD per LLM call over the run's LLM slots (every slot makes the same
    number of calls). Mock mode, free campaign lanes and the mock family cost 0."""
    if cfg.llm.mode == "mock":
        return 0.0
    lane = cc.lanes.get(cfg.llm.lane) if cc is not None else None
    if lane is not None and lane.free:
        return 0.0
    from finswarm.llm.pricing import estimate_call_cost

    costs = [
        0.0
        if s.family == "mock"
        else estimate_call_cost(s.model, profile.input_tokens, profile.output_tokens)
        for s in cfg.llm.slots
        if not slot_is_scripted(s)
    ]
    return sum(costs) / len(costs) if costs else 0.0


def plan_runs(
    cc: CampaignConfig, provider: str | None = None, profile: TokenProfile | None = None
) -> list[PlannedRun]:
    from finswarm.sim.run import apply_provider

    profile = profile or TokenProfile()
    out = []
    for idx, (name, overrides) in enumerate(expand_cells(cc)):
        for seed in cc.seeds:
            cfg = apply_provider(cell_config(cc, name, overrides, seed), provider)
            n = n_llm_slots(cfg)
            calls = n * cfg.periods * epochs_per_period(cfg)
            out.append(
                PlannedRun(
                    cfg=cfg,
                    run_id=make_run_id(cfg),
                    cell=name,
                    cell_index=idx,
                    seed=int(seed),
                    n_llm=n,
                    calls=calls,
                    usd=calls * usd_per_call(cfg, cc, profile),
                )
            )
    return out


def match_parents(
    spec: BranchSpec,
    runs: list[PlannedRun],
    base_name: str,
    children: list[PlannedRun] | None = None,
) -> list[PlannedRun]:
    """Parents of ``spec``: the trunk runs it names, then the children of earlier branches
    (``children``: ``cfg.name`` = ``<parent>--<branch>``, ``cell`` = the branch name)."""
    p = spec.parent
    trunks = [
        r
        for r in runs
        if p in (r.cell, r.cfg.name, f"c{r.cell_index}", r.cell.removeprefix(f"{base_name}-"))
        or (p == base_name and r.cell.startswith(base_name))
    ]
    kids = [c for c in (children or []) if p in (c.cfg.name, c.cell)]
    return trunks + kids


def child_name(parent_name: str, spec: BranchSpec) -> str:
    """Run name of ``spec``'s child of a parent named ``parent_name`` (``perturb.branch``)."""
    return f"{parent_name}--{spec.name}"


def _branch_horizon(spec: BranchSpec, parent: RunConfig) -> int:
    for e in spec.events:
        if e.type == "set_param" and e.path == "periods":
            return int(e.value)
    return parent.periods


def project(
    cc: CampaignConfig, provider: str | None = None, profile: TokenProfile | None = None
) -> Projection:
    """Dry-run projection: runs, branches, LLM calls and USD (no simulation)."""
    profile = profile or TokenProfile()
    runs = plan_runs(cc, provider, profile)
    rows = [
        {
            "kind": "run",
            "name": r.cfg.name,
            "seed": r.seed,
            "n_llm": r.n_llm,
            "calls": r.calls,
            "usd": r.usd,
            "approx": False,
        }
        for r in runs
    ]
    n_branch = n_ghost = 0
    branch_lanes: list[str] = []
    children: list[PlannedRun] = []  # planned branch children, parents of later branches
    for spec in cc.branches:
        kind, _, val = spec.at.partition(":")
        for parent in match_parents(spec, runs, cc.base.name, children):
            horizon = _branch_horizon(spec, parent.cfg)
            approx = kind != "period"
            start = int(val) if not approx else 0
            per_slot = max(0, horizon - start) * epochs_per_period(parent.cfg)
            calls = parent.n_llm * per_slot
            per_call = usd_per_call(parent.cfg, cc, profile)
            n_branch += 1
            branch_lanes.append(parent.cfg.llm.lane)
            name = child_name(parent.cfg.name, spec)
            rows.append(
                {
                    "kind": "branch",
                    "name": name,
                    "seed": parent.seed,
                    "n_llm": parent.n_llm,
                    "calls": calls,
                    "usd": calls * per_call,
                    "approx": approx,
                }
            )
            children.append(
                PlannedRun(
                    cfg=parent.cfg.model_copy(update={"name": name, "periods": horizon}),
                    run_id="",  # resolved at run time (the child's manifest)
                    cell=spec.name,
                    cell_index=-1,
                    seed=parent.seed,
                    n_llm=parent.n_llm,
                    calls=calls,
                    usd=calls * per_call,
                )
            )
            if spec.ghost is None:
                continue
            for k, sc in enumerate(parent.cfg.llm.slots):  # one ghost per initial LLM slot
                if slot_is_scripted(sc):
                    continue
                n_ghost += 1
                branch_lanes.append(parent.cfg.llm.lane)
                rows.append(
                    {
                        "kind": "ghost",
                        "name": f"{name}--ghost{k}",
                        "seed": parent.seed,
                        "n_llm": 1,
                        "calls": per_slot,
                        "usd": per_slot * per_call,
                        "approx": approx,
                    }
                )
    by_lane: dict[str, float] = {}
    lanes = [r.cfg.llm.lane for r in runs] + branch_lanes
    for row, lane in zip(rows, lanes, strict=True):
        by_lane[lane] = by_lane.get(lane, 0.0) + row["usd"]
    over = sorted(
        name
        for name, usd in by_lane.items()
        if name in cc.lanes
        and cc.lanes[name].budget_usd is not None
        and usd > float(cc.lanes[name].budget_usd or 0.0)
    )
    return Projection(
        runs=len(runs),
        branches=n_branch,
        calls=sum(r["calls"] for r in rows),
        usd=sum(r["usd"] for r in rows),
        rows=rows,
        by_lane=by_lane,
        over_budget=over,
        pricing=pricing_source(),
        ghosts=n_ghost,
    )


# ---------------------------------------------------------------------------------------------
# execution
# ---------------------------------------------------------------------------------------------
class CampaignProgress:
    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}
        self.t0 = time.perf_counter()

    def update(self, run_id: str, info: dict[str, Any]) -> None:
        self.rows.setdefault(run_id, {}).update(info)

    def render(self) -> Any:
        try:
            from rich import box
            from rich.table import Table
        except ImportError:  # pragma: no cover
            return self.text()
        t = Table(title=f"finswarm campaign  t={time.perf_counter() - self.t0:.0f}s", box=box.ASCII)
        for col in ("run", "seed", "status", "period", "delta_ma20", "calls", "usd"):
            t.add_column(col)
        for rid, r in sorted(self.rows.items()):
            d = r.get("delta_ma20")
            t.add_row(
                str(r.get("name", rid))[:40],
                str(r.get("seed", "")),
                str(r.get("status", "")),
                f"{r.get('period', -1) + 1}/{r.get('periods', '?')}",
                f"{d:.3f}" if isinstance(d, float) and math.isfinite(d) else "-",
                str(r.get("calls", 0)),
                f"{float(r.get('cost_usd', 0.0)):.4f}",
            )
        return t

    def text(self) -> str:
        return "\n".join(f"{k}: {v}" for k, v in sorted(self.rows.items()))


async def _dashboard(progress: CampaignProgress, interval: float, stop: asyncio.Event) -> None:
    try:
        from rich.console import Console

        console: Any = Console()
        show = console.print
    except ImportError:  # pragma: no cover
        show = print
    while not stop.is_set():
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
        except TimeoutError:
            show(progress.render())


async def run_campaign_async(
    cc: CampaignConfig,
    *,
    provider: str | None = None,
    out: str | Path = Path("results/campaigns"),
    workers: int | None = None,
    dashboard: bool = True,
    interval: float = 10.0,
    **runner_kw: Any,
) -> list[Any]:
    from finswarm.perturb.branch import branch_async, resolve_branch_period
    from finswarm.sim.ghost import ghost_slots, run_ghosts_async
    from finswarm.sim.run import RunResult, run_config_async

    root = Path(out) / cc.name
    root.mkdir(parents=True, exist_ok=True)
    runs = plan_runs(cc, provider)
    sem = asyncio.Semaphore(max(1, int(workers or cc.max_concurrent_runs)))
    progress = CampaignProgress()
    stop = asyncio.Event()
    dash = asyncio.create_task(_dashboard(progress, interval, stop)) if dashboard else None

    def _done(res: RunResult) -> RunResult:
        progress.update(res.run_id, {"status": res.status})
        return res

    async def one(r: PlannedRun) -> RunResult:
        async with sem:
            progress.update(r.run_id, {"name": r.cfg.name, "seed": r.seed, "status": "queued"})
            res = await run_config_async(
                r.cfg, out=root, provider=provider, progress=progress.update, **runner_kw
            )
            return _done(res)

    async def one_ghost(run_dir: Path, k: int, rival_policy: str) -> RunResult:
        async with sem:
            (res,) = await run_ghosts_async(
                run_dir,
                rival_policy=rival_policy,
                slots=[k],
                provider=provider,
                progress=progress.update,
                **runner_kw,
            )
            return _done(res)

    async def one_branch(spec: BranchSpec, parent_dir: Path, period: int) -> list[RunResult]:
        async with sem:
            res = await branch_async(
                parent_dir,
                period,
                list(spec.events),
                spec.name,
                out=root,
                provider=provider,
                progress=progress.update,
                **runner_kw,
            )
        out = [_done(res)]
        if spec.ghost is not None and res.status in DONE_STATUSES:
            # the semaphore is released first: ghosts queue like any other run
            ghosts = (one_ghost(res.run_dir, k, spec.ghost) for k in ghost_slots(res.run_dir))
            out.extend(await asyncio.gather(*ghosts))
        return out

    try:
        results = list(await asyncio.gather(*(one(r) for r in runs)))
        skipped = []
        # finished branch children: (branch name, child run name, run dir); later branches
        # may name them as parents (``TR-H--ext`` or ``ext``), so branches run in file order
        kids: list[tuple[str, str, Path]] = []
        for spec in cc.branches:
            parents: list[tuple[str, Path]] = [
                (p.run_id, root / p.run_id) for p in match_parents(spec, runs, cc.base.name)
            ]
            parents += [
                (run_name, run_dir)
                for bname, run_name, run_dir in kids
                if spec.parent in (bname, run_name)
            ]
            jobs = []
            for parent_id, pdir in parents:
                if read_manifest(pdir).get("status") not in DONE_STATUSES:
                    skipped.append(
                        {"branch": spec.name, "parent": parent_id, "why": "parent not ok"}
                    )
                    continue
                period = resolve_branch_period(pdir, spec.at)
                if period is None:
                    skipped.append(
                        {"branch": spec.name, "parent": parent_id, "why": f"no {spec.at}"}
                    )
                    continue
                jobs.append(one_branch(spec, pdir, period))
            for group in await asyncio.gather(*jobs):
                results.extend(group)
                child = group[0]  # the child itself; the rest are its ghosts
                if child.status in DONE_STATUSES:
                    kids.append((spec.name, str(child.manifest.get("name")), Path(child.run_dir)))
    finally:
        stop.set()
        if dash is not None:
            await dash
    summary = {
        "campaign": cc.name,
        "runs": [
            {
                "run_id": r.run_id,
                "status": r.status,
                "periods": r.periods_completed,
                "cost_usd": float(r.manifest.get("cost_usd", 0.0)),
                "dir": str(r.run_dir),
            }
            for r in results
        ],
        "skipped_branches": skipped,
        "cost_usd": sum(float(r.manifest.get("cost_usd", 0.0)) for r in results),
        "status": "ok" if all(r.status in DONE_STATUSES for r in results) else "incomplete",
    }
    atomic_write_text(root / "campaign.json", json.dumps(summary, indent=1, sort_keys=True))
    return results


def run_campaign(cc: CampaignConfig, **kw: Any) -> list[Any]:
    return asyncio.run(run_campaign_async(cc, **kw))


#: untracked files a registered launch may leave in the tree: run data, manifests and caches
#: that finished campaigns write (hooks keep Claude sessions from committing them)
LAUNCH_IGNORED_UNTRACKED: tuple[str, ...] = ("results/", "cache/")


def _git(root: Path, *args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, timeout=15, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout


def registered_launch_problems(root: str | Path) -> list[str]:
    """Why a registered launch must not start from the checkout ``root`` (empty = go).

    The rule (``.claude/rules/experiments.md``, lead decision of 2026-10-10): (i) the
    pre-registration is frozen (``docs/03_preregistration/FROZEN`` present); (ii) ``HEAD`` is
    pushed (contained in a remote-tracking branch, so the manifest's git sha is public);
    (iii) no tracked file outside ``results/**`` and ``cache/**`` is modified or staged.
    Files under those two trees (run data, manifests, caches) never block, tracked or not: a
    resumed run rewrites its committed manifest; any other untracked file blocks and is listed.
    """
    root = Path(root)
    problems: list[str] = []
    if not (root / "docs" / "03_preregistration" / "FROZEN").exists():
        problems.append("docs/03_preregistration/FROZEN is absent (run /freeze-prereg)")
    status = _git(root, "status", "--porcelain", "--untracked-files=all")
    if status is None:
        return [*problems, "git status failed: not a git checkout?"]
    for line in status.splitlines():
        if len(line) < 4:
            continue
        code, path = line[:2], line[3:].strip().strip('"').replace("\\", "/")
        if path.startswith(LAUNCH_IGNORED_UNTRACKED):
            # run data, manifests and caches are the runner's own output: a resumed run
            # rewrites its (possibly committed) manifest, which must not block the next launch
            continue
        if code == "??":
            problems.append(f"untracked file {path} (commit, delete or ignore it)")
        else:
            problems.append(f"modified or staged tracked file {path} (commit it first)")
    head = (_git(root, "rev-parse", "HEAD") or "").strip()
    remotes = _git(root, "branch", "-r", "--contains", "HEAD")
    if not head or remotes is None:
        problems.append("git HEAD unavailable")
    elif not remotes.strip():
        problems.append(f"HEAD {head[:12]} is not pushed to any remote (git push first)")
    return problems


def check_prereg(cc: CampaignConfig, root: str | Path | None = None) -> None:
    """Registered campaigns (``prereg_id`` set) may only launch from a frozen, pushed checkout
    whose tracked files are clean (:func:`registered_launch_problems`)."""
    if not cc.prereg_id:
        return
    where = Path(root) if root is not None else Path(__file__).resolve().parents[3]
    problems = registered_launch_problems(where)
    if problems:
        raise RuntimeError(
            f"campaign {cc.name!r} is registered ({cc.prereg_id}) and cannot launch: "
            + "; ".join(problems)
        )


def campaign_from_path(
    config: str | Path,
    provider: str | None = None,
    workers: int | None = None,
    dry_run: bool = False,
    out: str | Path = Path("results/campaigns"),
) -> Projection | list[Any]:
    cc = load_campaign_config(config)
    if dry_run:
        proj = project(cc, provider)
        print(proj.table())
        return proj
    check_prereg(cc)
    results = run_campaign(
        cc, provider=provider, out=out, workers=workers, mirror_dir=Path(out).parent / "manifests"
    )
    for r in results:
        print(f"{r.run_id}  {r.status}  periods={r.periods_completed}")
    return results


__all__ = [
    "ESTABLISHED_CONDITION",
    "CampaignProgress",
    "PlannedRun",
    "Projection",
    "TokenProfile",
    "campaign_from_path",
    "check_prereg",
    "cell_config",
    "expand_cells",
    "plan_runs",
    "project",
    "run_campaign",
    "run_campaign_async",
]
