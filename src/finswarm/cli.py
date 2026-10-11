"""finswarm command line: run | campaign | bench | lanes."""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.command()
def run(config: Path, provider: str | None = None, out: Path = Path("results/runs")) -> None:
    """Run a single configuration (implemented in finswarm.sim.run)."""
    from finswarm.sim.run import run_from_path

    run_from_path(config, provider=provider, out=out)


@app.command()
def campaign(
    config: Path, provider: str | None = None, workers: int = 2, dry_run: bool = False
) -> None:
    """Run a campaign (config x seeds, plus branches)."""
    from finswarm.sim.campaign import campaign_from_path

    campaign_from_path(config, provider=provider, workers=workers, dry_run=dry_run)


@app.command()
def bench(agents: int = 10_000, ticks: int = 100_000, mechanism: str = "dealer") -> None:
    """Throughput benchmark for the background swarm and the chosen mechanism."""
    from finswarm.sim.bench import run_bench

    run_bench(agents=agents, ticks=ticks, mechanism=mechanism)


@app.command()
def lanes() -> None:
    """Health-check every configured LLM lane (one tiny request each)."""
    from finswarm.llm.lanes import health_check

    health_check()


if __name__ == "__main__":
    app()
