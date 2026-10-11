"""scripts/peek.py: parquet tables, run and campaign directories, manifests, sqlite ledger/cache."""

from __future__ import annotations

import importlib.util
import json
import sqlite3
from pathlib import Path

import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[2]
# Looks like an Anthropic key to the redactor; built by concatenation so that no literal
# key-shaped string sits in the tree (scripts/secret_scan.py allowlists its sha256 for history).
FAKE_KEY = "sk-ant-" + "secret" * 4


def _load():
    spec = importlib.util.spec_from_file_location("_script_peek", ROOT / "scripts" / "peek.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


peek = _load()


def _run_dir(root: Path, run_id: str, *, status: str = "ok", cost: float = 0.0) -> Path:
    d = root / run_id
    (d / "quotes").mkdir(parents=True)
    pl.DataFrame(
        {"period": [0, 0, 1], "dealer": [0, 1, 0], "price": [1.5, 1.6, 1.55]}
    ).write_parquet(d / "quotes" / "part-00000.parquet")
    pl.DataFrame({"period": [1, 2], "dealer": [1, 0], "price": [1.65, 1.7]}).write_parquet(
        d / "quotes" / "part-00001.parquet"
    )
    (d / "events").mkdir()
    pl.DataFrame({"period": [2], "kind": ["shock"]}).write_parquet(
        d / "events" / "part-00000.parquet"
    )
    (d / "checkpoints").mkdir()
    (d / "checkpoints" / "p000000.json").write_text("{}", encoding="utf-8")
    manifest = {
        "run_id": run_id,
        "name": "smoke",
        "arm": "sham",
        "condition": "smoke",
        "status": status,
        "error": None if status == "ok" else "boom",
        "seed": 7,
        "provider": "mock",
        "llm_mode": "mock",
        "periods": 50,
        "periods_completed": 50 if status == "ok" else 12,
        "cost_usd": cost,
        "config_hash": "abc123",
        "git": {"commit": "0123456789abcdef", "dirty": False},
        "llm": {
            "calls": 3,
            "cost_usd": cost,
            "cache_hits": 1,
            "input_tokens": 30,
            "output_tokens": 9,
        },
        "online": {"delta": -0.123456, "price_mean": 1.6},
        "slots": [{"agent_id": 0, "name": "D0", "kind": "LLM", "model": "mock-v1"}],
        "resolved_config": {"long": "x" * 500},
    }
    (d / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return d


@pytest.fixture()
def campaign(tmp_path: Path) -> Path:
    root = tmp_path / "results" / "runs" / "smoke"
    _run_dir(root, "smoke__s0", cost=0.5)
    _run_dir(root, "smoke__s1", status="failed", cost=0.25)
    (root / "campaign.json").write_text(
        json.dumps({"campaign": "smoke", "status": "ok", "cost_usd": 0.75, "skipped_branches": []}),
        encoding="utf-8",
    )
    return root


@pytest.fixture()
def ledger(tmp_path: Path) -> Path:
    path = tmp_path / "llm_ledger.sqlite"
    con = sqlite3.connect(path)
    con.execute(
        "CREATE TABLE calls (id INTEGER PRIMARY KEY, ts REAL, day TEXT, lane TEXT, provider TEXT, "
        "model TEXT, run_id TEXT, agent TEXT, period INTEGER, input_tokens INTEGER, "
        "output_tokens INTEGER, cached_tokens INTEGER, cost_usd REAL, latency_ms REAL, status TEXT, "
        "attempts INTEGER, request_id TEXT, error TEXT)"
    )
    rows = [
        (
            1.0,
            "2026-10-10",
            "anthropic_haiku",
            "anthropic",
            "claude-haiku-5-5",
            "r1",
            "D0",
            0,
            2000,
            700,
            1000,
            0.0005,
            5500.0,
            "ok",
            1,
            "req1",
            None,
        ),
        (
            2.0,
            "2026-10-10",
            "anthropic_haiku",
            "anthropic",
            "claude-haiku-5-5",
            "r1",
            "D1",
            0,
            2000,
            700,
            1000,
            0.0005,
            5400.0,
            "ok",
            1,
            "req2",
            None,
        ),
        (
            3.0,
            "2026-10-10",
            "anthropic_sonnet",
            "anthropic",
            "claude-sonnet-5-5",
            "r2",
            "D0",
            0,
            1900,
            400,
            1000,
            0.0064,
            5100.0,
            "error",
            3,
            None,
            f"429 {FAKE_KEY}",
        ),
    ]
    con.executemany(
        "INSERT INTO calls (ts, day, lane, provider, model, run_id, agent, period, input_tokens, "
        "output_tokens, cached_tokens, cost_usd, latency_ms, status, attempts, request_id, error) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        rows,
    )
    con.commit()
    con.close()
    return path


def test_parquet_table_dir_schema_head_tail_cols_describe(campaign: Path, capsys) -> None:
    quotes = campaign / "smoke__s0" / "quotes"
    assert peek.main([str(quotes), "-n", "2"]) == 0
    out = capsys.readouterr().out
    assert "2 part(s), 5 rows, 3 columns" in out and "price: Float64" in out
    assert "first 2 row(s)" in out and "1.5" in out
    assert peek.main([str(quotes), "--tail", "-n", "1", "--cols", "price", "--describe"]) == 0
    out = capsys.readouterr().out
    assert "last 1 row(s)" in out and "1.7" in out and "std" in out
    assert peek.main([str(quotes / "part-00001.parquet")]) == 0
    assert "1 part(s), 2 rows" in capsys.readouterr().out


def test_run_dir_summary_and_table(campaign: Path, capsys) -> None:
    run = campaign / "smoke__s0"
    assert peek.main([str(run)]) == 0
    out = capsys.readouterr().out
    assert "run_id             smoke__s0" in out and "status             ok" in out
    assert "git                0123456789ab" in out
    assert "calls 3, cost 0.5000 USD, cache hits 1" in out
    assert "D0:LLM/mock-v1" in out
    assert "quotes" in out and "events" in out and "checkpoints: 1" in out
    assert "x" * 100 not in out  # resolved_config is not dumped
    assert peek.main([str(run), "--table", "quotes", "-n", "3"]) == 0
    assert "first 3 row(s)" in capsys.readouterr().out
    assert peek.main([str(run), "--table", "nope"]) == 1


def test_campaign_dir_lists_runs(campaign: Path, capsys) -> None:
    assert peek.main([str(campaign)]) == 0
    out = capsys.readouterr().out
    assert "2 run(s), 0.7500 USD" in out
    assert "smoke__s1" in out and "failed" in out and "12/50" in out and "boom" in out
    assert "campaign.json: status ok" in out
    empty = campaign.parent / "empty"
    empty.mkdir()
    assert peek.main([str(empty)]) == 1


def test_manifest_json_is_truncated_unless_full(campaign: Path, capsys) -> None:
    manifest = campaign / "smoke__s0" / "manifest.json"
    assert peek.main([str(manifest)]) == 0
    out = capsys.readouterr().out
    assert '"run_id": "smoke__s0"' in out and "(+400 chars)" in out
    assert peek.main([str(manifest), "--full"]) == 0
    assert "x" * 500 in capsys.readouterr().out
    jsonl = campaign / "log.jsonl"
    jsonl.write_text('{"a": 1}\n{"a": 2}\n{"a": 3}\n', encoding="utf-8")
    assert peek.main([str(jsonl), "-n", "2", "--tail"]) == 0
    out = capsys.readouterr().out
    assert "3 line(s); showing 2" in out and '{"a": 3}' in out and '{"a": 1}' not in out


def test_sqlite_ledger_summary_table_and_sql_read_only(ledger: Path, capsys, monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_KEY)
    assert peek.main([str(ledger)]) == 0
    out = capsys.readouterr().out
    assert "ledger: 3 calls, 0.0074 USD, tokens in/out/cached 5900/1800/3000" in out
    assert "anthropic_sonnet" in out and "by status" in out and "error" in out
    assert peek.main([str(ledger), "--table", "calls", "-n", "1", "--tail"]) == 0
    out = capsys.readouterr().out
    assert "claude-sonnet-5-5" in out
    assert "secretsecretsecretsecret" not in out  # key-looking strings are redacted
    assert (
        peek.main(
            [str(ledger), "--sql", "select lane, count(*) c from calls group by 1 order by c desc"]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "anthropic_haiku  2" in out.replace("   ", "  ")
    assert peek.main([str(ledger), "--sql", "delete from calls"]) == 1
    assert "sqlite error" in capsys.readouterr().out
    assert sqlite3.connect(ledger).execute("select count(*) from calls").fetchone()[0] == 3


def test_unknown_paths(tmp_path: Path, capsys) -> None:
    assert peek.main([str(tmp_path / "missing.parquet")]) == 1
    (tmp_path / "x.bin").write_bytes(b"\x00")
    assert peek.main([str(tmp_path / "x.bin")]) == 1
    assert "unsupported" in capsys.readouterr().out
