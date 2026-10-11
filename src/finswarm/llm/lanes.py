"""Lane health check: one tiny request per configured lane whose key is present.

``python -c "from finswarm.llm.lanes import health_check; health_check()"`` (or
``finswarm lanes``). Lanes without a key are reported as ``unconfigured`` and never
raise. Requests go through the lane scheduler (the only path to a provider), with an
in-memory cache; paid calls are written to the persisted ledger so budgets stay exact.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from finswarm.llm.cache import ResponseCache
from finswarm.llm.client import LLMRequest
from finswarm.llm.errors import LLMError
from finswarm.llm.keys import key_present, redact
from finswarm.llm.ledger import Ledger
from finswarm.llm.pricing import Catalog, cache_dir, load_catalog
from finswarm.llm.providers import make_client
from finswarm.llm.scheduler import LaneScheduler, load_budget_policy


def _probe_request(provider: str, kind: str, model: str) -> LLMRequest:
    anthropic = kind == "anthropic"
    return LLMRequest(
        provider=provider,
        model=model,
        system="You are a connectivity check for a research simulator.",
        messages=[{"role": "user", "content": "Reply with the single word: ok"}],
        max_tokens=2048 if anthropic else 16,
        effort="low" if anthropic else None,
        prompt_version="health_check",
        tags={"run_id": "health_check", "agent": "health_check"},
    )


async def health_check_async(
    catalog: Catalog | None = None,
    *,
    lanes: Iterable[str] | None = None,
    ledger_path: str | Path | None = None,
    out: Callable[[str], None] = print,
) -> dict[str, dict[str, Any]]:
    cat = catalog or load_catalog()
    names = list(lanes) if lanes is not None else list(cat.lanes)
    results: dict[str, dict[str, Any]] = {}
    ready: list[str] = []
    for name in names:
        lane = cat.lanes[name]
        spec = cat.providers.get(lane.provider)
        if spec is None:
            results[name] = {"status": "misconfigured", "detail": f"no provider {lane.provider!r}"}
        elif spec.kind != "mock" and not key_present(spec.env_key):
            results[name] = {"status": "unconfigured", "detail": f"{spec.env_key} not set"}
        else:
            ready.append(name)

    paid = any(not cat.lanes[n].free and cat.lanes[n].provider != "mock" for n in ready)
    if ledger_path is None and paid:  # the shared ledger (FINSWARM_CACHE_DIR), as the runs
        ledger_path = cache_dir() / "llm_ledger.sqlite"
    clients = {}
    for n in ready:
        try:
            clients[n] = make_client(
                cat.lanes[n].provider, cat, capture_limits=True, timeout_s=60.0
            )
        except Exception as e:  # never raise from a health check
            results[n] = {"status": "error", "detail": redact(f"{type(e).__name__}: {e}")[:200]}
    ready = [n for n in ready if n in clients]
    scheduler = LaneScheduler(
        {n: cat.lanes[n] for n in ready},
        clients,
        catalog=cat,
        cache=ResponseCache(":memory:", mode="live"),
        ledger=Ledger(ledger_path if ledger_path is not None else ":memory:"),
        budget=load_budget_policy(),
        max_attempts=2,
        timeout_s=60.0,
    )
    try:
        for name in ready:
            lane = cat.lanes[name]
            spec = cat.providers[lane.provider]
            model = lane.models[0]
            try:
                resp = await scheduler.submit(_probe_request(lane.provider, spec.kind, model), name)
            except LLMError as e:
                results[name] = {"status": "error", "detail": redact(str(e))[:200]}
                continue
            except Exception as e:  # never raise from a health check
                results[name] = {
                    "status": "error",
                    "detail": redact(f"{type(e).__name__}: {e}")[:200],
                }
                continue
            client = scheduler.clients.get(name)
            limits = dict(getattr(client, "last_limits", {}) or {})
            results[name] = {
                "status": "refusal" if resp.refusal else "ok",
                "latency_ms": round(resp.latency_ms, 1),
                "model": model,
                "cost_usd": resp.cost_usd,
                "limits": limits,
                "reply": resp.text.strip()[:40],
            }
    finally:
        await scheduler.aclose()

    out(f"{'lane':<18} {'provider':<11} {'status':<13} {'latency':>9}  detail")
    for name in names:
        r = results[name]
        lane = cat.lanes[name]
        latency = f"{r['latency_ms']:.0f} ms" if "latency_ms" in r else "-"
        if r["status"] in ("ok", "refusal"):
            lim = r["limits"]
            detail = ", ".join(f"{k}={v}" for k, v in sorted(lim.items())) if lim else "limits n/a"
            detail = f"{r['model']}; {detail}"
        else:
            detail = r.get("detail", "")
        out(f"{name:<18} {lane.provider:<11} {r['status']:<13} {latency:>9}  {detail}")
    missing = [n for n in names if results[n]["status"] == "unconfigured"]
    if missing:
        out(f"unconfigured lanes ({len(missing)}): {', '.join(missing)} -- add keys to .env")
    return results


def health_check(
    catalog: Catalog | None = None,
    *,
    lanes: Iterable[str] | None = None,
    ledger_path: str | Path | None = None,
    out: Callable[[str], None] = print,
) -> dict[str, dict[str, Any]]:
    """Synchronous wrapper (use :func:`health_check_async` inside a running loop)."""
    return asyncio.run(health_check_async(catalog, lanes=lanes, ledger_path=ledger_path, out=out))
