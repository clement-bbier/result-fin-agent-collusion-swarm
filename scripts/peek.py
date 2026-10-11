"""Peek at finswarm results without opening binary files in the editor: parquet tables, run and
campaign directories, manifests, the LLM cache and the cost ledger (sqlite).

    python scripts/peek.py results/runs/smoke                       # campaign: one line per run
    python scripts/peek.py results/runs/smoke/<run_id>              # run: manifest summary + tables
    python scripts/peek.py results/runs/smoke/<run_id> --table quotes -n 20 --cols period,price
    python scripts/peek.py results/runs/smoke/<run_id>/quotes --tail --describe
    python scripts/peek.py cache/llm_ledger.sqlite                  # cost by lane, model, status, day
    python scripts/peek.py cache/llm_cache.sqlite --table responses -n 5
    python scripts/peek.py cache/llm_ledger.sqlite --sql "select lane, sum(cost_usd) from calls group by 1"
    python scripts/peek.py results/manifests/<run_id>.json          # long values truncated (--full)

Read-only: sqlite files are opened in read-only mode and every printed line goes through the API
key redactor (``finswarm.llm.keys.redact``).
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

import polars as pl

try:
    from finswarm.llm.keys import redact
except ImportError:  # the script still works outside the project environment

    def redact(text: str) -> str:
        return text


MANIFEST_FILE = "manifest.json"
PART_GLOB = "part-*.parquet"
RUN_FIELDS = (
    "run_id",
    "name",
    "arm",
    "condition",
    "status",
    "error",
    "seed",
    "provider",
    "llm_mode",
    "periods_completed",
    "periods",
    "cost_usd",
    "wall_time_s",
    "config_hash",
    "started_at",
    "finished_at",
)


def out(text: str = "") -> None:
    print(redact(text))


def truncate(obj: Any, limit: int) -> Any:
    """Shorten long strings and long lists recursively so a manifest fits on a screen."""
    if isinstance(obj, str):
        return obj if len(obj) <= limit else obj[:limit] + f"... (+{len(obj) - limit} chars)"
    if isinstance(obj, dict):
        return {k: truncate(v, limit) for k, v in obj.items()}
    if isinstance(obj, list):
        head = [truncate(v, limit) for v in obj[:8]]
        return head if len(obj) <= 8 else [*head, f"... (+{len(obj) - 8} items)"]
    return obj


def table_text(headers: list[str], rows: list[list[Any]]) -> str:
    cells = [[str(c) for c in row] for row in rows]
    widths = [
        max(len(h), *(len(r[i]) for r in cells)) if cells else len(h) for i, h in enumerate(headers)
    ]
    line = "  ".join(h.ljust(w) for h, w in zip(headers, widths, strict=True))
    body = ["  ".join(c.ljust(w) for c, w in zip(row, widths, strict=True)) for row in cells]
    return "\n".join([line, *body])


# ----------------------------------------------------------------------------- parquet


def parquet_parts(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(path.glob(PART_GLOB)) or sorted(path.glob("*.parquet"))


def frame_text(df: pl.DataFrame, n: int) -> str:
    with pl.Config(
        tbl_rows=max(n, 1),
        tbl_cols=-1,
        fmt_str_lengths=60,
        tbl_width_chars=200,
        tbl_hide_dataframe_shape=True,
    ):
        return str(df)


def peek_parquet(path: Path, args: argparse.Namespace) -> int:
    parts = parquet_parts(path)
    if not parts:
        out(f"no parquet file under {path}")
        return 1
    lf = pl.scan_parquet([str(p) for p in parts])
    schema = lf.collect_schema()
    rows = lf.select(pl.len()).collect().item()
    out(f"{path}: {len(parts)} part(s), {rows} rows, {len(schema)} columns")
    for name, dtype in schema.items():
        out(f"  {name}: {dtype}")
    if args.cols:
        lf = lf.select([c.strip() for c in args.cols.split(",") if c.strip()])
    if args.describe:
        out()
        out(frame_text(lf.collect().describe(), 12))
    out()
    shown = lf.tail(args.n).collect() if args.tail else lf.head(args.n).collect()
    out(f"{'last' if args.tail else 'first'} {min(args.n, rows)} row(s):")
    out(frame_text(shown, args.n))
    return 0


# ---------------------------------------------------------------------- runs, campaigns


def read_manifest(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / MANIFEST_FILE).read_text(encoding="utf-8"))


def run_tables(run_dir: Path) -> list[tuple[str, int, int]]:
    """``(table, parts, rows)`` for every sub-directory that holds parquet parts."""
    found = []
    for child in sorted(run_dir.iterdir()):
        if not child.is_dir():
            continue
        parts = parquet_parts(child)
        if not parts:
            continue
        rows = pl.scan_parquet([str(p) for p in parts]).select(pl.len()).collect().item()
        found.append((child.name, len(parts), rows))
    return found


def peek_run(run_dir: Path, args: argparse.Namespace) -> int:
    if args.table:
        return peek_parquet(run_dir / args.table, args)
    m = read_manifest(run_dir)
    out(f"run {run_dir}")
    for key in RUN_FIELDS:
        if key in m and m[key] not in (None, "", []):
            out(f"  {key:18s} {m[key]}")
    git = m.get("git") or {}
    if git:
        out(
            f"  {'git':18s} {str(git.get('commit', ''))[:12]}{' DIRTY' if git.get('dirty') else ''}"
        )
    llm = m.get("llm") or {}
    if llm:
        out(
            f"  {'llm':18s} calls {llm.get('calls', 0)}, cost {llm.get('cost_usd', 0.0):.4f} USD, "
            f"cache hits {llm.get('cache_hits', 0)}, tokens in/out/cached "
            f"{llm.get('input_tokens', 0)}/{llm.get('output_tokens', 0)}/{llm.get('cached_tokens', 0)}"
        )
    online = m.get("online") or {}
    if online:
        out(
            f"  {'online':18s} "
            + ", ".join(
                f"{k} {v:.4g}" if isinstance(v, int | float) else f"{k} {v}"
                for k, v in online.items()
            )
        )
    slots = m.get("slots") or []
    if slots:
        kinds = ", ".join(
            f"{s.get('name', s.get('agent_id'))}:{s.get('kind', '?')}"
            + (f"/{s['model']}" if s.get("model") else "")
            for s in slots
        )
        out(f"  {'slots':18s} {kinds}")
    tables = run_tables(run_dir)
    out()
    out(
        table_text(["table", "parts", "rows"], [list(t) for t in tables]) if tables else "no tables"
    )
    checkpoints = (
        sorted((run_dir / "checkpoints").glob("*.json"))
        if (run_dir / "checkpoints").is_dir()
        else []
    )
    if checkpoints:
        out(f"\ncheckpoints: {len(checkpoints)} ({', '.join(p.stem for p in checkpoints[:8])})")
    return 0


def peek_campaign(root: Path, args: argparse.Namespace) -> int:
    run_dirs = sorted(p.parent for p in root.rglob(MANIFEST_FILE))
    if not run_dirs:
        out(f"no run (no {MANIFEST_FILE}) under {root}")
        return 1
    rows: list[list[Any]] = []
    total = 0.0
    for d in run_dirs:
        m = read_manifest(d)
        cost = float(m.get("cost_usd") or 0.0)
        total += cost
        rows.append(
            [
                d.relative_to(root).as_posix(),
                m.get("status", "?"),
                f"{m.get('periods_completed', '?')}/{m.get('periods', '?')}",
                m.get("seed", ""),
                m.get("arm", ""),
                m.get("condition", ""),
                f"{cost:.4f}",
                (m.get("error") or "")[:40],
            ]
        )
    out(f"campaign {root}: {len(run_dirs)} run(s), {total:.4f} USD")
    out(table_text(["run", "status", "periods", "seed", "arm", "condition", "usd", "error"], rows))
    summary = root / "campaign.json"
    if summary.is_file():
        c = json.loads(summary.read_text(encoding="utf-8"))
        skipped = c.get("skipped_branches") or []
        out(
            f"campaign.json: status {c.get('status')}, cost {c.get('cost_usd')}, skipped branches {len(skipped)}"
        )
    return 0


# ------------------------------------------------------------------------------ sqlite


def sql_rows(
    con: sqlite3.Connection, query: str, params: tuple = ()
) -> tuple[list[str], list[list[Any]]]:
    cur = con.execute(query, params)
    headers = [d[0] for d in cur.description] if cur.description else []
    return headers, [list(r) for r in cur.fetchall()]


def ledger_summary(con: sqlite3.Connection) -> None:
    h, r = sql_rows(
        con,
        "select count(*), coalesce(sum(cost_usd),0), coalesce(sum(input_tokens),0), "
        "coalesce(sum(output_tokens),0), coalesce(sum(cached_tokens),0) from calls",
    )
    calls, cost, tin, tout, tcached = r[0]
    out(f"ledger: {calls} calls, {cost:.4f} USD, tokens in/out/cached {tin}/{tout}/{tcached}")
    for title, group in (
        ("lane", "lane"),
        ("model", "provider, model"),
        ("status", "status"),
        ("day", "day"),
    ):
        h, r = sql_rows(
            con,
            f"select {group}, count(*) as calls, round(coalesce(sum(cost_usd),0), 4) as usd "
            f"from calls group by {group} order by usd desc, calls desc",
        )
        out(f"\nby {title}:")
        out(table_text(h, r))


def cache_summary(con: sqlite3.Connection) -> None:
    h, r = sql_rows(
        con,
        "select provider, model, prompt_version, count(*) as entries, "
        "round(sum(length(response_json))/1048576.0, 2) as mib from responses "
        "group by provider, model, prompt_version order by entries desc",
    )
    out("cache entries:")
    out(table_text(h, r))


def peek_sqlite(path: Path, args: argparse.Namespace) -> int:
    con = sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True)
    try:
        _, tables = sql_rows(con, "select name from sqlite_master where type='table' order by name")
        names = [t[0] for t in tables]
        counts = [[n, sql_rows(con, f'select count(*) from "{n}"')[1][0][0]] for n in names]
        out(f"{path}: {len(names)} table(s)")
        out(table_text(["table", "rows"], counts))
        if args.sql:
            out()
            h, r = sql_rows(con, args.sql)
            out(table_text(h, r[: args.n]) if h else "(no rows)")
            if len(r) > args.n:
                out(f"... {len(r) - args.n} more row(s); raise -n to see them")
            return 0
        if args.table:
            if args.table not in names:
                out(f"no table {args.table!r}")
                return 1
            order = " order by rowid desc" if args.tail else ""
            h, r = sql_rows(con, f'select * from "{args.table}"{order} limit ?', (args.n,))
            out()
            out(
                table_text(
                    h, [[truncate(c, 80) if isinstance(c, str) else c for c in row] for row in r]
                )
            )
            return 0
        out()
        if "calls" in names:
            ledger_summary(con)
        if "responses" in names:
            cache_summary(con)
        return 0
    except sqlite3.Error as exc:
        out(f"sqlite error: {exc}")
        return 1
    finally:
        con.close()


# -------------------------------------------------------------------------------- json


def peek_json(path: Path, args: argparse.Namespace) -> int:
    limit = 10**9 if args.full else 100
    if path.suffix == ".jsonl":
        lines = path.read_text(encoding="utf-8").splitlines()
        picked = lines[-args.n :] if args.tail else lines[: args.n]
        out(f"{path}: {len(lines)} line(s); showing {len(picked)}")
        for line in picked:
            try:
                out(json.dumps(truncate(json.loads(line), limit), ensure_ascii=False))
            except json.JSONDecodeError:
                out(line[:limit])
        return 0
    data = json.loads(path.read_text(encoding="utf-8"))
    out(json.dumps(truncate(data, limit), indent=1, ensure_ascii=False, default=str))
    return 0


# -------------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "path", help="parquet file or table dir, run dir, campaign dir, sqlite, json(l)"
    )
    ap.add_argument("--table", default="", help="table of a run dir or of a sqlite file")
    ap.add_argument("-n", type=int, default=10, help="rows to show (default 10)")
    ap.add_argument("--cols", default="", help="comma-separated columns (parquet)")
    ap.add_argument("--tail", action="store_true", help="show the last rows instead of the first")
    ap.add_argument("--describe", action="store_true", help="print polars describe() (parquet)")
    ap.add_argument("--sql", default="", help="read-only SQL to run on a sqlite file")
    ap.add_argument("--full", action="store_true", help="do not truncate long json values")
    args = ap.parse_args(argv)
    path = Path(args.path)
    if not path.exists():
        out(f"not found: {path}")
        return 1
    try:
        if path.suffix in (".sqlite", ".db", ".sqlite3"):
            return peek_sqlite(path, args)
        if path.suffix in (".json", ".jsonl"):
            return peek_json(path, args)
        if path.suffix == ".parquet":
            return peek_parquet(path, args)
        if path.is_dir():
            if (path / MANIFEST_FILE).is_file():
                return peek_run(path, args)
            if parquet_parts(path):
                return peek_parquet(path, args)
            return peek_campaign(path, args)
        out(f"unsupported file type: {path}")
        return 1
    except (OSError, ValueError, pl.exceptions.PolarsError) as exc:
        out(f"error: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
