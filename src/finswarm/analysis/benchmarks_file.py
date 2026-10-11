"""Build the pinned ``benchmarks.json`` of the analysis from the campaign configurations.

For every market listed under a pin of ``configs/analysis/registered_arms.yaml`` the trunk
configuration is loaded, the grid overrides and the branch's ``set_param market.*`` events
are applied and ``add_scripted_agent`` events are counted as entrants; the resolved market is
then solved:

* ``closed_form`` -- symmetric Nash / joint monopoly of the routed logit demand
  (``MarketModel.benchmarks``; ``tau`` and ``outside_utility`` folded in, ``beta`` from the
  config, ``n`` = incumbents + entrants);
* ``sweep`` -- markets with the public fringe on (L2): ``finswarm.market.sweep.profit_fn``
  fed to :func:`finswarm.metrics.benchmarks.estimate_by_sweeps` (the closed form no longer
  applies: prices barely move but the profit benchmarks fall sharply). ``auto`` picks the
  sweep when ``fringe.share > 0`` and ``finswarm.market.sweep`` is importable; otherwise the
  entry is written with the flow-diversion closed form and flagged ``provisional``.

The output is deterministic (no timestamps), so its SHA-256 can be pinned in the registry.
Each entry records the SHA-256 of its resolved market so stale pins are detected when a
configuration changes after pinning.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from finswarm.analysis.arms import (
    BENCH_KEYS,
    MarketSource,
    RegisteredArms,
    file_sha256,
)
from finswarm.config.load import canonical_json, load_campaign_config
from finswarm.metrics.benchmarks import MarketModel, d7_cartel, estimate_by_sweeps


def _set_dotted(data: dict[str, Any], path: str, value: Any) -> None:
    cur = data
    parts = path.split(".")
    for k in parts[:-1]:
        if not isinstance(cur.get(k), dict):
            raise KeyError(f"config path {path!r} does not exist")
        cur = cur[k]
    if parts[-1] not in cur:
        raise KeyError(f"config path {path!r} does not exist")
    cur[parts[-1]] = value


def resolve_market(src: MarketSource, root: str | Path = ".") -> dict[str, Any]:
    """Resolved run config (JSON dict), incumbents, entrants and ignored (non-demand) events."""
    cc = load_campaign_config(Path(root) / src.config)
    data = cc.base.model_dump(mode="json")
    for path, v in src.grid.items():
        _set_dotted(data, path, v)
    n_entrants, ignored = 0, []
    if src.branch is not None:
        spec = next((b for b in cc.branches if b.name == src.branch), None)
        if spec is None:
            raise KeyError(f"{src.config}: no branch named {src.branch!r}")
        for ev in spec.events:
            kind = getattr(ev, "type", "")
            if kind == "set_param" and ev.path.startswith("market."):
                _set_dotted(data, ev.path, ev.value)
            elif kind == "add_scripted_agent":
                n_entrants += 1
            elif kind == "set_param":
                if ev.path != "periods" and not ev.path.startswith("tags."):
                    ignored.append(f"set_param {ev.path}")
            else:
                ignored.append(kind)
    slots = data.get("llm", {}).get("slots") or []
    n_inc = src.n_incumbents or (len(slots) if slots else int(data["market"]["n_dealers"]))
    return {"run": data, "n_incumbents": int(n_inc), "n_entrants": n_entrants, "ignored": ignored}


def market_sha256(resolved: dict[str, Any]) -> str:
    body = {
        "market": resolved["run"]["market"],
        "n_incumbents": resolved["n_incumbents"],
        "n_entrants": resolved["n_entrants"],
    }
    return hashlib.sha256(canonical_json(body).encode()).hexdigest()


def _sweep_module() -> Any:
    try:
        from finswarm.market import sweep  # sim-core, present once the fringe lever landed
    except ImportError:
        return None
    return sweep if hasattr(sweep, "profit_fn") else None


#: SAP section 3: expected profits by Monte Carlo with common random numbers, 10 replications
SWEEP_REPLICATIONS = 10


def replicated_profit_fn(sw: Any, run: Any, seeds: list[int], periods: int) -> Any:
    """Mean of ``sw.profit_fn(run, seed=s, periods=periods)`` over ``seeds`` (each seed is a
    common random number across the price vectors of a batch)."""
    fns = [sw.profit_fn(run, periods=periods, seed=int(s)) for s in seeds]

    def fn(batch: np.ndarray) -> np.ndarray:
        return np.mean([np.asarray(f(batch), dtype=float) for f in fns], axis=0)

    return fn


def solve_market(
    resolved: dict[str, Any],
    method: str = "auto",
    sweep_kw: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Benchmarks entry (without ``source``) of a resolved market.

    ``sweep_kw``: ``replications`` (default 10 seeds, SAP), ``periods`` per evaluation
    (default 50) and ``n_starts`` (default 10, SAP) of the best-response iteration; the
    entry records the seeds, the converged Nash points of every start and their spread
    (distinct equilibria show up as a large ``nash_spread``).
    """
    from finswarm.config.schema import RunConfig

    run = RunConfig.model_validate(resolved["run"])
    model = MarketModel.from_config(run.market, resolved["n_incumbents"], resolved["n_entrants"])
    fringe = model.fringe_share > 0
    sw = _sweep_module()
    use_sweep = method == "sweep" or (method == "auto" and fringe and sw is not None)
    provisional = False
    if use_sweep:
        if sw is None:
            raise RuntimeError("method 'sweep' needs finswarm.market.sweep (sim-core)")
        kw = dict(sweep_kw or {})
        reps = int(kw.pop("replications", SWEEP_REPLICATIONS))
        periods = int(kw.pop("periods", 50))
        kw = {"n_starts": 10, "seed": 0, **kw}
        seeds = [int(run.seed) + r for r in range(reps)]
        closed = model.benchmarks()
        tick = float(run.market.price_grid.tick)
        lo = model.params.marginal_cost
        hi = max(closed.p_mono * 1.25, closed.p_nash + 4 * tick)
        grid = np.round(np.arange(lo + tick, hi + tick / 2, tick), 10)
        fn = replicated_profit_fn(sw, run, seeds, periods)
        b = estimate_by_sweeps(fn, grid, n_agents=model.n_sellers, **kw)
        bench = {k: float(getattr(b, k)) for k in BENCH_KEYS}
        meth = "sweep:market.sweep.profit_fn"
        extra = {
            "sweep": {
                **b.params,
                "replications": reps,
                "seeds": seeds,
                "periods_per_evaluation": periods,
                "closed_form_flow_diversion": {k: float(getattr(closed, k)) for k in BENCH_KEYS},
            }
        }
    else:
        b = model.benchmarks()
        bench = {k: float(getattr(b, k)) for k in BENCH_KEYS}
        meth = "closed_form:routed_logit"
        extra = {}
        if fringe:
            meth = "closed_form:flow_diversion"
            provisional = True
    return {
        "model": model.to_dict(),
        "benchmarks": bench,
        "method": meth,
        "provisional": provisional,
        "d7_cartel": float(d7_cartel(model)),
        **extra,
    }


def build_benchmark_file(
    spec: RegisteredArms,
    pin: str,
    root: str | Path = ".",
    sweep_kw: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The full ``benchmarks.json`` document of one pin."""
    markets = {}
    for name, src in sorted(spec.benchmark_files[pin].markets.items()):
        res = resolve_market(src, root)
        entry = solve_market(res, src.method, sweep_kw)
        entry["source"] = src.model_dump(mode="json")
        entry["market_sha256"] = market_sha256(res)
        entry["ignored_events"] = res["ignored"]
        markets[name] = entry
    return {
        "version": 1,
        "generated_by": "finswarm.analysis.benchmarks_file",
        "pin": pin,
        "prereg_id": spec.prereg_id,
        "markets": markets,
    }


def dumps(doc: dict[str, Any]) -> str:
    return json.dumps(doc, indent=2, sort_keys=True) + "\n"


def write_benchmark_file(
    spec: RegisteredArms,
    pin: str,
    root: str | Path = ".",
    sweep_kw: dict[str, Any] | None = None,
) -> str:
    """Write the pinned file (LF line endings) and return its SHA-256."""
    p = Path(root) / spec.benchmark_files[pin].file
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(dumps(build_benchmark_file(spec, pin, root, sweep_kw)))
    return file_sha256(p)


def stale_markets(
    spec: RegisteredArms, pin: str, doc: dict[str, Any], root: str | Path = "."
) -> list[str]:
    """Markets whose configuration changed since the file was built (or are missing)."""
    out = []
    for name, src in spec.benchmark_files[pin].markets.items():
        entry = doc.get("markets", {}).get(name)
        if entry is None or entry.get("market_sha256") != market_sha256(resolve_market(src, root)):
            out.append(name)
    return sorted(out)


def repin(arms_path: str | Path, pin: str, file: str, sha: str) -> bool:
    """Rewrite the ``sha256:`` line of ``pin`` in the registry YAML (comments kept; the hash
    is written quoted so YAML never reads an all-digit hash as a number)."""
    p = Path(arms_path)
    text = p.read_text(encoding="utf-8")
    pat = re.compile(
        rf"(^[ \t]+{re.escape(pin)}:[ \t]*\n[ \t]+file:[ \t]*{re.escape(file)}[ \t]*\n"
        rf"[ \t]+sha256:[ \t]*)(\S*)",
        re.MULTILINE,
    )
    new, n = pat.subn(lambda m: m.group(1) + f'"{sha}"', text)
    if n != 1:
        return False
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(new)
    return True
