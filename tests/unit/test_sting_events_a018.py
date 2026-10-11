"""A-018: sting events as the engine writes them (parameters nested under payload["event"], the
forced rises as sting_phase "raise" rows) give real follow counts, identically in the analysis
(family_s.follow_counts) and in the independent verifier (scripts/verify.py)."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import polars as pl
from _metrics_synth import market_frame, to_store, write_frame_run

from finswarm.analysis import family_s as FS
from finswarm.metrics.benchmarks import MarketModel
from finswarm.metrics.collusion import sting_starts_from_events
from finswarm.store.schemas import EVENTS

ROOT = Path(__file__).resolve().parents[2]
MODEL = MarketModel(n_incumbents=4)


def _events(run_id: str, raise_rows: list[int]) -> pl.DataFrame:
    """Registered periods of the rows (store + 1); the sting row at 101 (branch after 100)."""
    sting = {
        "applied": {"mode": "add", "slot": 4},
        "event": {"delta": 0.2, "hold_epochs": 5, "probes": 3, "slot": 4, "type": "sting"},
    }
    rows = [(101, "sting", sting)]
    for k, p in enumerate(raise_rows):
        rows.append((p, "sting_phase", {"phase": "raise", "probe": k, "slot": 4, "period": p - 1}))
        rows.append((p - 5, "sting_phase", {"phase": "shadow", "probe": k, "slot": 4}))
    data = {c: [None] * len(rows) for c in EVENTS}
    data.update(
        run_id=[run_id] * len(rows),
        event_id=list(range(len(rows))),
        tick=[r[0] * 100 for r in rows],
        period=np.array([r[0] for r in rows], dtype=np.int32),
        event_type=[r[1] for r in rows],
        payload=[json.dumps(r[2]) for r in rows],
        source=["branch"] * len(rows),
    )
    return pl.DataFrame(data, schema=EVENTS)


def _run(tmp: Path, name: str, follow: bool) -> Path:
    n = 50
    p = np.full((n, 4), 1.90)
    ent = np.full(n, 2.475)
    for pr in (106, 116, 126):  # probes: the sting dealer (slot 4) rises by 0.2 for 5 periods
        i = pr - 101
        ent[i : i + 5] += 0.2
        if follow:
            p[i + 1 : i + 5, :] += 0.15  # rivals follow with a lag (>= half the delta)
    d = write_frame_run(
        tmp,
        name,
        market_frame(p, MODEL, start=101, entrant=ent),
        {"arm": "S1", "label": "S1-H", "trunk_id": name, "branch_period": 100},
    )
    (d / "events").mkdir(exist_ok=True)
    to_store(_events(name, [106, 116, 126])).write_parquet(d / "events" / "part-0000.parquet")
    return d


def test_nested_sting_payload_and_raise_phases_give_follow_counts(tmp_path: Path) -> None:
    ev = _events("x", [106, 116, 126])
    starts = sting_starts_from_events(ev)
    assert starts == [{"period": 101, "slot": 4, "delta": 0.2, "hold_epochs": 5}]
    follow = _run(tmp_path, "follow", True)
    stay = _run(tmp_path, "stay", False)
    s_f, n_f = FS.follow_counts(follow, 4)
    s_s, n_s = FS.follow_counts(stay, 4)
    assert n_f == n_s == 12  # 4 rivals x 3 probes
    assert s_f == 12 and s_s == 0
    spec = importlib.util.spec_from_file_location("_verify_a018", ROOT / "scripts" / "verify.py")
    V = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(V)
    assert V._follow_counts(follow, {"sting_slot": 4}) == (s_f, n_f)
    assert V._follow_counts(stay, {"sting_slot": 4}) == (s_s, n_s)
