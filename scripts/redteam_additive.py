"""Additive descriptives for the draft red-team (no registered value changes, exploratory).

(a) Sham cache reuse per lever: share of the post-branch (registered periods 101-150) incumbent
    LLM decisions (slots 0-3) of a lever run that were served from a response-cache entry first
    written by the sham branch of the same trunk. A decision counts when its ``cache_hit`` is
    true and the sham run of the same trunk logged, at the same period and slot, a live call
    (``cache_hit`` false) with the same prompt hash and the same raw output. Reported beside it:
    every post-branch cache hit, whatever its source, and the decisions whose prompt equals the
    sham's at the same period and slot (with how many also got the sham's exact output).
    With ``--ledger`` the SQLite call ledger is read (read-only) as a cross-check: post-branch
    ``cache_hit`` calls per arm and how many follow an earlier live call of the same run, agent
    and period (a replay of the run's own request, not of the sham's).
(b) L6c minus L6: the paired contrast of the two entrant arms on the registered per-run capped
    index, Delta C(L6c) - Delta C(L6) = idx(L6) - idx(L6c) per trunk (the sham cancels), BCa
    95 % with the registered bootstrap settings, over the trunks both contrasts use.

Qualified trunks only (the 11 of the registered contrasts); periods are registered (A-008)::

    uv run python scripts/redteam_additive.py --out-root results/analysis \
        --ledger cache/llm_ledger.sqlite
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from pathlib import Path
from typing import Any

import polars as pl

from finswarm.analysis import stats
from finswarm.analysis.io import STORE_PERIOD_OFFSET, read_table

ROW_ARMS = ("L1", "L2", "L3", "L4a", "L5", "L6", "L7", "L6b", "L6c", "R1")
#: every TR-H branch arm with LLM incumbents (context table; the rows use ROW_ARMS)
BRANCH_ARMS = (*ROW_ARMS, "L9", "R2", "R4", "S1", "S2", "S1c_LLM_TR", "sham2")
WINDOW = (101, 150)  # registered post-branch periods (branch at period 100)
MEMBERS = (0, 1, 2, 3)
COLS = ["period", "slot", "prompt_hash", "cache_hit", "raw_output"]


def index_runs(st: dict[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    """``{arm: {trunk: run}}`` for the trunk, the sham and the branch arms; one run per key."""
    runs: dict[str, dict[str, dict[str, Any]]] = {}
    for r in st["runs"]:
        arm = r["registered_arm"]
        if arm not in (*BRANCH_ARMS, "sham", "trunk"):
            continue
        slot = runs.setdefault(arm, {})
        t = str(r["trunk_id"])
        if t in slot:
            raise ValueError(f"two {arm} runs on trunk {t}: {slot[t]['run_id']}, {r['run_id']}")
        slot[t] = r
    return runs


def _post(path: str) -> pl.DataFrame:
    d = read_table(path, "llm_decisions")
    if d.height == 0:
        return pl.DataFrame(schema={c: pl.Utf8 for c in COLS})
    return d.select(COLS).filter(
        pl.col("period").is_between(WINDOW[0], WINDOW[1]) & pl.col("slot").is_in(list(MEMBERS))
    )


def _pool(per: dict[str, dict[str, Any]], keep: set[str] | None = None) -> dict[str, Any]:
    v = [x for t, x in per.items() if keep is None or t in keep]
    n = sum(x["n"] for x in v)
    k = {f: sum(x[f] for x in v) for f in ("cache_hits", "from_ref", "same_prompt", "same_out")}
    return {
        "n_trunks": len(v),
        "n_decisions": n,
        "n_cache_hits": k["cache_hits"],
        "n_from_sham": k["from_ref"],
        "n_same_prompt_as_sham": k["same_prompt"],
        "n_same_prompt_and_output": k["same_out"],
        "share_from_sham": k["from_ref"] / n if n else None,
        "share_cache_hits": k["cache_hits"] / n if n else None,
    }


def reuse(
    runs: dict[str, dict[str, dict[str, Any]]], arm: str, ref_arm: str = "sham"
) -> dict[str, dict[str, Any]]:
    """Per trunk: the arm's post-branch decisions against the same-trunk ``ref_arm`` run."""
    per: dict[str, dict[str, Any]] = {}
    for t, r in sorted(runs.get(arm, {}).items()):
        rr = runs.get(ref_arm, {}).get(t)
        if rr is None:
            continue
        s = _post(rr["path"]).select(
            "period",
            "slot",
            "prompt_hash",
            pl.col("cache_hit").alias("hit_ref"),
            pl.col("raw_output").alias("out_ref"),
        )
        j = _post(r["path"]).join(s, on=["period", "slot", "prompt_hash"], how="left")
        hit = pl.col("cache_hit").fill_null(False)
        same = pl.col("hit_ref").is_not_null()
        same_out = same & (pl.col("raw_output") == pl.col("out_ref"))
        from_ref = hit & same_out & ~pl.col("hit_ref").fill_null(True)
        per[t] = {
            "run_id": r["run_id"],
            "reference_run_id": rr["run_id"],
            "n": j.height,
            "cache_hits": int(j.select(hit.sum()).item()),
            "from_ref": int(j.select(from_ref.fill_null(False).sum()).item()),
            "same_prompt": int(j.select(same.sum()).item()),
            "same_out": int(j.select(same_out.fill_null(False).sum()).item()),
        }
    return per


def cache_reuse(runs: dict[str, dict[str, dict[str, Any]]], trunks: list[str]) -> dict[str, Any]:
    """Pooled over the qualified ``trunks`` (the rows) and over every trunk the arm ran on."""
    out: dict[str, Any] = {}
    for arm in BRANCH_ARMS:
        if arm not in runs:
            continue
        per = reuse(runs, arm)
        out[arm] = _pool(per, set(trunks)) | {"all_trunks": _pool(per), "per_trunk": per}
    return out


def ledger_check(
    ledger: Path, runs: dict[str, dict[str, dict[str, Any]]], trunks: list[str]
) -> dict[str, Any]:
    """Post-branch ledger calls per arm (every trunk the arm ran on). A cache hit carries the
    ``request_id`` of the live call that wrote the entry, so each hit is attributed exactly to
    its writer run: the same run, the same-trunk run of another arm, or another trunk."""
    db = sqlite3.connect(f"file:{ledger.as_posix()}?mode=ro", uri=True)
    lo, hi = WINDOW[0] - STORE_PERIOD_OFFSET, WINDOW[1] - STORE_PERIOD_OFFSET  # store periods
    arm_of = {r["run_id"]: a for a, v in runs.items() for r in v.values()}
    trunk_of = {r["run_id"]: t for v in runs.values() for t, r in v.items()}
    out: dict[str, Any] = {}
    try:
        writer: dict[str, str] = {}
        for rid, run in db.execute(
            "SELECT request_id, run_id FROM calls WHERE status = 'ok' AND request_id IS NOT NULL "
            "ORDER BY ts"
        ):
            writer.setdefault(rid, run)
        for arm in ("sham", *BRANCH_ARMS):
            ids = [r["run_id"] for r in runs.get(arm, {}).values()]
            if not ids:
                continue
            q = ",".join("?" * len(ids))
            base = f"FROM calls WHERE run_id IN ({q}) AND period BETWEEN ? AND ?"
            args = [*ids, lo, hi]
            live = db.execute(f"SELECT COUNT(*) {base} AND status = 'ok'", args).fetchone()[0]
            src: dict[str, int] = {}
            for rid, run in db.execute(
                f"SELECT request_id, run_id {base} AND status = 'cache_hit'", args
            ):
                w = writer.get(rid)
                if w is None:
                    key = "unresolved"
                elif w == run:
                    key = "same run"
                elif trunk_of.get(w) == trunk_of.get(run):
                    key = f"{arm_of.get(w, 'unlisted')} (same trunk)"
                else:
                    key = f"{arm_of.get(w, 'unlisted')} (other trunk)"
                src[key] = src.get(key, 0) + 1
            out[arm] = {
                "n_trunks": len(ids),
                "live_calls": live,
                "cache_hits": sum(src.values()),
                "hit_writers": src,
            }
    finally:
        db.close()
    return out


def l6c_minus_l6(st: dict[str, Any], runs: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    trunks = sorted(set(st["contrasts"]["L6"]["trunks"]) & set(st["contrasts"]["L6c"]["trunks"]))
    d = [float(runs["L6"][t]["delta_p"]) - float(runs["L6c"][t]["delta_p"]) for t in trunks]
    ci = stats.bca_ci(d, B=stats.DEFAULT_B, seed=stats.DEFAULT_SEED)
    return {
        "definition": "Delta C(L6c) - Delta C(L6) = capped idx(L6) - capped idx(L6c) per trunk "
        "(registered per-run delta_p, trunk benchmarks, window 111-150), BCa 95 %",
        "n": ci.n,
        "mean": ci.estimate,
        "bca_lo": ci.lo,
        "bca_hi": ci.hi,
        "B": stats.DEFAULT_B,
        "seed": stats.DEFAULT_SEED,
        "check_difference_of_registered_means": st["contrasts"]["L6c"]["delta_p"]["mean"]
        - st["contrasts"]["L6"]["delta_p"]["mean"],
        "per_trunk": dict(zip(trunks, d, strict=True)),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-root", type=Path, default=Path("results/analysis"))
    ap.add_argument("--ledger", type=Path, default=None, help="SQLite call ledger (read-only)")
    args = ap.parse_args(argv)
    st = json.loads((args.out_root / "TR-H" / "stats.json").read_text(encoding="utf-8"))
    runs = index_runs(st)
    trunks = sorted(st["inference"]["qualification"]["by_lane"]["H"]["qualified"])
    out: dict[str, Any] = {
        "status": "exploratory descriptives for the draft red-team; no registered value changes",
        "n_qualified": len(trunks),
        "cache_reuse": {
            "definition": "post-branch (periods 101-150) incumbent LLM decisions (slots 0-3) of "
            "the qualified trunks; from_sham = cache_hit true and the same-trunk sham logged a "
            "live call (cache_hit false) at the same period and slot with the same prompt hash "
            "and the same raw output; share = n_from_sham / n_decisions (pooled); all_trunks "
            "pools every trunk the arm ran on",
            "arms": cache_reuse(runs, trunks),
            "positive_control": {
                "definition": "the same rule with S2 as the reference: S1c-LLM-TR-H decisions "
                "served from the same-trunk S2 run's answers (a known replay)",
                **_pool(reuse(runs, "S1c_LLM_TR", "S2")),
            },
        },
        "l6c_minus_l6": l6c_minus_l6(st, runs),
    }
    if args.ledger is not None:
        out["cache_reuse"]["ledger"] = ledger_check(args.ledger, runs, trunks)
    path = args.out_root / "redteam_additive.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    (args.out_root / "redteam_additive.md").write_text(render(out), encoding="utf-8")
    print(f"-> {path}")
    return 0


def _f(v: Any, d: int = 3) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "-"
    return f"{v:.{d}f}" if isinstance(v, float) else str(v)


def render(o: dict[str, Any]) -> str:
    c = o["cache_reuse"]
    L = [
        "# Red-team additive descriptives (exploratory, no registered value changes)",
        "",
        "## (a) Sham cache reuse after the branch",
        "",
        c["definition"],
        "",
        "| arm | decisions | from sham (share) | any cache hit (share) | prompt = sham's | "
        "and output = sham's |",
        "|---|---|---|---|---|---|",
    ]
    for arm, a in c["arms"].items():
        for lab, r in (("qualified", a), ("all trunks", a["all_trunks"])):
            L.append(
                f"| {arm} ({lab}, {r['n_trunks']}) | {r['n_decisions']} | {r['n_from_sham']} "
                f"({_f(r['share_from_sham'], 4)}) | {r['n_cache_hits']} "
                f"({_f(r['share_cache_hits'], 4)}) | {r['n_same_prompt_as_sham']} | "
                f"{r['n_same_prompt_and_output']} |"
            )
    pc = c["positive_control"]
    L += [
        "",
        f"Positive control ({pc['definition']}): {pc['n_from_sham']} of {pc['n_decisions']} "
        f"decisions ({_f(pc['share_from_sham'], 4)}).",
    ]
    if "ledger" in c:
        L += [
            "",
            "Ledger cross-check (post-branch calls, every trunk the arm ran on; hits attributed to "
            "their writer by request_id):",
            "",
            "| arm | trunks | live calls | cache hits | writer of the hit entries |",
            "|---|---|---|---|---|",
        ]
        for arm, r in c["ledger"].items():
            w = ", ".join(f"{k}: {v}" for k, v in r["hit_writers"].items()) or "-"
            L.append(f"| {arm} | {r['n_trunks']} | {r['live_calls']} | {r['cache_hits']} | {w} |")
    e = o["l6c_minus_l6"]
    L += [
        "",
        "## (b) L6c minus L6 (entrant type, exploratory)",
        "",
        e["definition"],
        "",
        f"n {e['n']}: {_f(e['mean'])} [{_f(e['bca_lo'])}, {_f(e['bca_hi'])}]",
        "",
    ]
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
