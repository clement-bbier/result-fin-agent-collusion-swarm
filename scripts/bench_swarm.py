"""Throughput benchmark of the background swarm against fixed dealer quotes (F8 provenance).

Usage::

    uv run python scripts/bench_swarm.py --mode both --tier0
    uv run python scripts/bench_swarm.py --agents 10000 --ticks 100000 --arrivals 20 --mode dealer

Modes: ``dealer`` = mechanism A (dealer market, no fringe); ``fringe`` = the same market with
the public limit-order fringe on (``market.fringe.share`` = ``--fringe-share``, default 0.3,
mechanism B); ``both`` runs the two in turn. ``--tier0`` benches the registered Tier 0 market
instead of the historical two-dealer, 20-arrivals profile: TR-H's swarm (10,000 agents,
``lam_mean`` 1e-4, one arrival per tick on average, price tick 0.05, routing s1) with four
fixed dealer asks, so the figure is the one the campaigns actually pay for.

Per mode the market is advanced one period at a time (``--ticks-per-period``) and trades, flow
and period stats are drained every period, as the runner does; JIT compilation is excluded by
a warm-up period. Output: ``results/bench/<git sha>.json`` (``<sha>-dirty`` on a dirty tree, or
``--out``), one document per invocation with the provenance (sha, dirty flag, UTC time, Python,
numba, machine), the profile, one block per mode under ``modes`` and the flat ``ticks_per_s``
read by the hooks and the docs (the slowest mode when several ran). ``latest.json`` is
refreshed too. Exit 1 when any mode is below ``--min-ticks-per-s``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from finswarm.config.schema import RunConfig
from finswarm.core.rng import RNGRegistry
from finswarm.core.types import Order, OrderType, Side
from finswarm.market.registry import make_market
from finswarm.swarm._jit import NUMBA_ENABLED

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "bench"
TRUNK = ROOT / "configs" / "sprint" / "tier0" / "TR-H.yaml"
MODES = ("dealer", "fringe")
FRINGE_SHARE = 0.3


@dataclass(frozen=True)
class Profile:
    """The benched market: swarm size and arrival rate, price grid, fixed dealer asks."""

    name: str
    agents: int
    arrivals_per_tick: float
    ticks_per_period: int
    tick: float
    asks: tuple[float, ...]
    source: str = ""

    @property
    def lam_mean(self) -> float:
        return self.arrivals_per_tick / max(1, self.agents)


def legacy_profile(agents: int, arrivals: float, tpp: int) -> Profile:
    """The historical bench: two dealers at 1.50 / 1.70, 20 arrivals per tick, tick 0.01."""
    return Profile("legacy", int(agents), float(arrivals), int(tpp), 0.01, (1.50, 1.70))


def tier0_profile(agents: int | None = None, tpp: int | None = None) -> Profile:
    """TR-H's registered market: its swarm (arrival rate per tick kept when ``agents`` is
    overridden), price tick and four dealers quoting fixed asks around the Nash-monopoly band."""
    from finswarm.config.load import load_campaign_config

    cc = load_campaign_config(TRUNK)
    base = cc.base
    n_agents = int(base.swarm.n_agents)
    arrivals = float(base.swarm.arrivals.lam_mean) * n_agents
    return Profile(
        "tier0",
        int(agents) if agents is not None else n_agents,
        arrivals,
        int(tpp) if tpp is not None else int(base.ticks_per_period),
        float(base.market.price_grid.tick),
        tuple(1.50 + 0.10 * k for k in range(int(base.market.n_dealers))),
        source=str(TRUNK.relative_to(ROOT)).replace("\\", "/"),
    )


def build(profile: Profile, mode: str, seed: int, fringe_share: float) -> Any:
    if mode not in MODES:
        raise ValueError(f"unknown bench mode {mode!r}; expected one of {MODES}")
    cfg = RunConfig.model_validate(
        {
            "name": f"bench_{profile.name}_{mode}",
            "ticks_per_period": profile.ticks_per_period,
            "seed": int(seed),
            "market": {
                "mechanism": "dealer",
                "n_dealers": len(profile.asks),
                "price_grid": {"min": 1.0, "max": 4.0, "tick": profile.tick},
                "demand": {"kind": "swarm"},
                "fringe": {"share": fringe_share if mode == "fringe" else 0.0},
            },
            "swarm": {
                "mode": "agents",
                "n_agents": profile.agents,
                "arrivals": {"lam_mean": profile.lam_mean},
            },
        }
    )
    market: Any = make_market(cfg, RNGRegistry(int(seed)))
    for slot, price in enumerate(profile.asks):
        ticks = int(round((price - market.p_min) / market.tick_size))
        aid = market.dealer_agent_id(slot)
        market.submit(Order(slot + 1, aid, Side.SELL, OrderType.QUOTE, ticks, 1.0, -1, 0, 0))
    return market


def run_mode(
    profile: Profile, mode: str, ticks: int, seed: int, fringe_share: float = FRINGE_SHARE
) -> dict[str, Any]:
    """Bench one mode; the ``ticks_per_s`` block the document reports under ``modes``."""
    market = build(profile, mode, seed, fringe_share)
    tpp = profile.ticks_per_period
    t0 = time.perf_counter()
    market.advance(tpp)  # JIT warm-up, excluded from the timing
    warmup = time.perf_counter() - t0
    market.drain_trades_arrays()
    market.drain_flow()
    market.drain_periods()
    n_trades = n_fringe = 0
    n_periods = max(1, int(ticks) // tpp)
    t0 = time.perf_counter()
    for _ in range(n_periods):
        market.advance(tpp)
        tr = market.drain_trades_arrays()
        n_trades += int(tr["tick"].shape[0])
        if "seller_id" in tr:
            n_fringe += int((tr["seller_id"] < 0).sum())
        market.drain_flow()
        market.drain_periods()
    elapsed = time.perf_counter() - t0
    done = n_periods * tpp
    return {
        "mode": mode,
        "fringe_share": fringe_share if mode == "fringe" else 0.0,
        "ticks": done,
        "seconds": elapsed,
        "warmup_seconds": warmup,
        "ticks_per_s": done / elapsed if elapsed > 0 else float("inf"),
        "trades": n_trades,
        "trades_per_s": n_trades / elapsed if elapsed > 0 else float("inf"),
        "fringe_trades": n_fringe,
    }


def git_sha() -> tuple[str, bool | None]:
    """``(short sha, dirty)`` of the checkout holding this script (``nogit`` without git)."""

    def _git(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=10, check=True
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip()

    sha = _git("rev-parse", "--short=12", "HEAD")
    status = _git("status", "--porcelain", "--untracked-files=no")
    return (sha or "nogit", None if status is None else bool(status))


def bench(
    profile: Profile,
    modes: tuple[str, ...],
    *,
    ticks: int,
    seed: int,
    fringe_share: float,
    min_ticks_per_s: float,
) -> dict[str, Any]:
    """The provenance document: profile, one block per mode, flat ``ticks_per_s`` (slowest)."""
    sha, dirty = git_sha()
    blocks = {m: run_mode(profile, m, ticks, seed, fringe_share) for m in modes}
    slowest = min(blocks.values(), key=lambda b: b["ticks_per_s"])
    doc: dict[str, Any] = {
        "schema_version": 1,
        "sha": sha,
        "dirty": dirty,
        "generated_at": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "python": platform.python_version(),
        "numba": NUMBA_ENABLED,
        "machine": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count() or 0,
        "profile": asdict(profile) | {"lam_mean": profile.lam_mean},
        "modes": blocks,
        # flat fields kept for the hooks and docs that read the historical layout
        "agents": profile.agents,
        "arrivals_per_tick": profile.arrivals_per_tick,
        "ticks_per_period": profile.ticks_per_period,
        "mode": slowest["mode"] if len(modes) == 1 else "both",
        "ticks": slowest["ticks"],
        "seconds": slowest["seconds"],
        "warmup_seconds": slowest["warmup_seconds"],
        "ticks_per_s": slowest["ticks_per_s"],
        "trades": slowest["trades"],
        "trades_per_s": slowest["trades_per_s"],
        "threshold_ticks_per_s": float(min_ticks_per_s),
    }
    doc["pass"] = all(b["ticks_per_s"] >= min_ticks_per_s for b in blocks.values())
    return doc


def write_doc(doc: dict[str, Any], out: Path | None, out_dir: Path, latest: bool) -> Path:
    name = f"{doc['sha']}{'-dirty' if doc['dirty'] else ''}.json"
    path = out if out is not None else out_dir / name
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(doc, indent=2) + "\n"
    path.write_text(text, encoding="utf-8")
    if latest:
        (path.parent / "latest.json").write_text(text, encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--agents", type=int, default=None, help="swarm size (default: profile)")
    ap.add_argument("--ticks", type=int, default=100_000)
    ap.add_argument("--arrivals", type=float, default=20.0, help="mean arrivals per tick")
    ap.add_argument("--mode", choices=[*MODES, "both"], default="dealer")
    ap.add_argument("--tier0", action="store_true", help="bench TR-H's registered market")
    ap.add_argument("--fringe-share", type=float, default=FRINGE_SHARE)
    ap.add_argument("--ticks-per-period", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-ticks-per-s", type=float, default=10_000.0)
    ap.add_argument("--out", type=Path, default=None, help="explicit output file")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR, help="directory of <sha>.json")
    ap.add_argument("--no-latest", action="store_true", help="do not refresh latest.json")
    args = ap.parse_args(argv)
    if args.tier0:
        profile = tier0_profile(args.agents, args.ticks_per_period)
    else:
        profile = legacy_profile(
            args.agents if args.agents is not None else 10_000,
            args.arrivals,
            args.ticks_per_period if args.ticks_per_period is not None else 100,
        )
    modes = MODES if args.mode == "both" else (args.mode,)
    doc = bench(
        profile,
        modes,
        ticks=args.ticks,
        seed=args.seed,
        fringe_share=args.fringe_share,
        min_ticks_per_s=args.min_ticks_per_s,
    )
    path = write_doc(doc, args.out, args.out_dir, latest=not args.no_latest)
    for b in doc["modes"].values():
        print(
            f"{b['mode']} ({profile.name}: {profile.agents} agents, "
            f"{profile.arrivals_per_tick:g} arrivals/tick, {len(profile.asks)} dealers): "
            f"{b['ticks']} ticks -> {b['ticks_per_s']:.0f} ticks/s ({b['trades_per_s']:.0f} "
            f"trades/s, {b['fringe_trades']} at the fringe, warm-up {b['warmup_seconds']:.1f} s)"
        )
    print(
        f"sha {doc['sha']}{' (dirty)' if doc['dirty'] else ''}, numba={doc['numba']}: "
        f"flat ticks_per_s {doc['ticks_per_s']:.0f} "
        f"({'PASS' if doc['pass'] else 'FAIL'} at {doc['threshold_ticks_per_s']:.0f}) -> {path}"
    )
    return 0 if doc["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
