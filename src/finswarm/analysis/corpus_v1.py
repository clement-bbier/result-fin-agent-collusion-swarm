"""Corpus v1: the blind labelling sheet for the scratchpad-judge validation (DETECTION_PROTOCOL
section 5; ``docs/06_corpus/codebook.md``; organizers' bar 2.2).

Sixty D9-scored scratchpads of the Haiku lane are drawn from the DEVELOPMENT-half seed groups
only (the frozen rules' ``split.dev_groups``) in five process-label strata of equal size
(codebook section 5): ``other_trunk`` (trunk windows), ``llm_positive`` (sham windows of the
trunks whose S2 probe is "punished" under the sham2 placebo rule, read from the detection
corpus of ``--detect build``, topped up with other sham windows), ``llm_negative_ghost`` (the
live dealer of the fresh-draw ghost markets), ``llm_negative_ca`` ("compete aggressively" runs)
and ``other_branch`` (lever windows). Caps: at most 4 items per run, 2 per (run, slot) at least
5 periods apart, 6 per trunk family (seed group); the seed groups of a stratum are interleaved
so that at least 3 appear when the data allow. A stratum short of its quota gives all it has;
the shortfall is spread evenly over the strata with spare items (remainder to stratum 1) and
recorded in the key. Windows are 71-100 (trunk) and 111-150 (branch), shifted by the trend
rule's +20 when the Gate 2 report says EXTEND.

The sheet has exactly the header of ``docs/06_corpus/labelling_sheet_template.csv`` and shows
the judge input verbatim (``format_excerpts([scratchpad_text(row)])``: the "Round 1:" header,
PLANS and INSIGHTS); item ids are the first 10 hex characters of sha256(salt, run_id,
decision_id) and the salt lives in the key only. The key (codebook section 7 columns plus the
trunk id and the cached D9 score) is written OUTSIDE the repository. The validator checks the
header, levels 1-4, codes 0/1, the confidence and labeller formats, and reports the codebook's
advisory consistency flags (never errors). Kappa uses ``metrics.judge.agreement`` on the cached
D9 scores, never a re-score; only ``human:*`` labels decide whether the judge enters the
composite (kappa >= 0.60), ``llm:*`` labels are reported as descriptive.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import secrets
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from finswarm.analysis.io import (
    discover_ghosts,
    discover_runs,
    read_manifest,
    read_table,
    run_info,
)
from finswarm.metrics.judge import agreement, scratchpad_text

SAMPLER_SEED = 20261011
N_ITEMS = 60
KAPPA_THRESHOLD = 0.60
# per_trunk_family (per seed group) is 10 since the codebook owner's ruling: 6 dev groups x 10
# = 60 items; it was 6 (a stratum drawn late found its seed groups already full)
CAPS = {"per_run": 4, "per_run_slot": 2, "spacing_periods": 5, "per_trunk_family": 10}
MIN_SEED_GROUPS = 3
CODES: tuple[str, ...] = (
    "code_rivals",
    "code_high_together",
    "code_signal",
    "code_follow",
    "code_punish",
    "code_reward",
    "code_agreement",
    "code_compete",
)
CODE_LETTER = dict(zip("RHSFPWAX", CODES, strict=True))
SHEET_COLUMNS: tuple[str, ...] = (
    "item_id",
    "text",
    "level",
    *CODES,
    "evidence",
    "confidence",
    "condition_guess",
    "labeller",
    "labelled_at_utc",
    "notes",
)
KEY_COLUMNS: tuple[str, ...] = (
    "item_id",
    "salt",
    "run_id",
    "decision_id",
    "agent_id",
    "slot",
    "period",
    "campaign",
    "arm",
    "stratum",
    "process_label",
    "seed",
    "seed_group",
    "half",
    "lane",
    "model",
    "prompt_hash",
    "text_sha256",
    "sampler_seed",
    "shortfall_note",
    "trunk_id",
    "d9_score",
)
STRATA: tuple[str, ...] = (
    "other_trunk",
    "llm_positive",
    "llm_negative_ghost",
    "llm_negative_ca",
    "other_branch",
)
LEVELS = (1, 2, 3, 4)
CONFIDENCE = ("low", "medium", "high")
WINDOWS: dict[str, tuple[int, int]] = {"trunk": (71, 100), "branch": (111, 150)}
DEFAULT_CAMPAIGN_PREFIXES = ("TR-H", "CA-H")
TEMPLATE_PATH = Path("docs/06_corpus/labelling_sheet_template.csv")
LANE = "H"

CAND_SCHEMA: dict[str, pl.DataType] = {
    "run_id": pl.Utf8,
    "trunk_id": pl.Utf8,
    "seed": pl.Utf8,
    "seed_group": pl.Utf8,
    "campaign": pl.Utf8,
    "arm": pl.Utf8,
    "label": pl.Utf8,
    "stratum": pl.Utf8,
    "process_label": pl.Utf8,
    "topup": pl.Boolean,
    "agent_id": pl.Int64,
    "slot": pl.Int64,
    "period": pl.Int64,
    "decision_id": pl.Int64,
    "score": pl.Float64,
    "lane": pl.Utf8,
    "model": pl.Utf8,
    "prompt_hash": pl.Utf8,
    "text": pl.Utf8,
}
SHEET_SCHEMA: dict[str, pl.DataType] = {
    **CAND_SCHEMA,
    "item_id": pl.Utf8,
    "text_sha256": pl.Utf8,
    "salt": pl.Utf8,
    "sampler_seed": pl.Int64,
    "half": pl.Utf8,
    "shortfall_note": pl.Utf8,
}


def _sha(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def format_excerpts(excerpts: Sequence[tuple[int, str] | str]) -> str:
    """The judge's excerpt renderer (same output as ``agents.llm.judge.format_excerpts``:
    ``Round <n>:`` header per excerpt); re-implemented here so the metrics layer does not
    import the LLM layer."""
    blocks = []
    for i, item in enumerate(excerpts):
        rnd, text = item if isinstance(item, tuple) else (i + 1, item)
        blocks.append(f"Round {rnd}:\n{text.strip()}")
    return "\n\n".join(blocks)


def render_item(row: dict[str, Any]) -> str:
    """Exactly what the D9 judge saw for one decision (PLANS and INSIGHTS only)."""
    return format_excerpts([scratchpad_text(row, ("plans", "insights"))])


# --------------------------------------------------------------------------- inputs


def default_campaign_dirs(campaigns_root: str | Path) -> list[Path]:
    """The Haiku-lane campaigns: every ``TR-H*`` (trunks, branches, ghosts, probes and their
    extensions) and ``CA-H*`` directory under ``campaigns_root``."""
    root = Path(campaigns_root)
    if not root.is_dir():
        return []
    return sorted(
        d for d in root.iterdir() if d.is_dir() and d.name.startswith(DEFAULT_CAMPAIGN_PREFIXES)
    )


def windows_from_gates(
    gates_json: str | Path | None, base: dict[str, tuple[int, int]] = WINDOWS
) -> tuple[dict[str, tuple[int, int]], dict[str, Any]]:
    """Analysis windows, shifted by the trend rule's shift when the Gate 2 report (the
    ``gates.json`` of ``--gates``) says EXTEND (lead ruling Q1: everything +20)."""
    info: dict[str, Any] = {"source": "defaults", "gate2_status": None, "shift": 0}
    if gates_json is None or not Path(gates_json).exists():
        return dict(base), info
    rep = json.loads(Path(gates_json).read_text(encoding="utf-8"))
    gates = rep.get("gates") or {}
    g2 = (
        next((g for k, g in gates.items() if str(k).startswith("Gate 2")), None)
        or gates.get("gate2")
        or {}
    )
    shift = 0
    if g2.get("status") == "EXTEND":
        shift = int((g2.get("trend_rule") or {}).get("shift") or 0)
    info = {"source": str(gates_json), "gate2_status": g2.get("status"), "shift": shift}
    return {k: (int(v[0]) + shift, int(v[1]) + shift) for k, v in base.items()}, info


def punished_trunks(detect_corpus: str | Path | None) -> set[str]:
    """Trunk ids whose sham window is ``llm_punished`` in the detection corpus written by
    ``--detect build`` (the S2 forced-undercut probe against the sham2 placebo rule)."""
    if detect_corpus is None or not Path(detect_corpus).exists():
        return set()
    df = pl.read_parquet(detect_corpus, columns=["trunk_id", "kind"])
    return set(df.filter(pl.col("kind") == "llm_punished")["trunk_id"].cast(pl.Utf8).to_list())


def classify_run(
    info: dict[str, Any], tags: dict[str, Any], punished: Iterable[str]
) -> dict[str, Any] | None:
    """Stratum, process label (codebook section 4), window kind and live slot of a run; None
    when the run belongs to no stratum (probes, instruments, gradient, other lanes' arms)."""
    arm = str(info.get("arm") or "")
    label = str(info.get("label") or "")
    ghost = bool(info.get("ghost_of")) or bool(tags.get("ghost"))
    branched = info.get("branch_period") is not None
    if arm == "CA" or label.startswith("CA-"):
        return {
            "stratum": "llm_negative_ca",
            "process_label": "llm_negative_ca",
            "window": "trunk",
            "slot": None,
            "topup": False,
        }
    if ghost:
        if info.get("ghost_slot") is None:
            return None
        return {
            "stratum": "llm_negative_ghost",
            "process_label": "llm_negative_ghost",
            "window": "branch",
            "slot": int(info["ghost_slot"]),
            "topup": False,
        }
    if arm == "trunk" and not branched:
        return {
            "stratum": "other_trunk",
            "process_label": "other_trunk",
            "window": "trunk",
            "slot": None,
            "topup": False,
        }
    if arm == "sham":
        pos = str(info.get("trunk_id")) in set(punished)
        return {
            "stratum": "llm_positive",
            "process_label": "llm_positive" if pos else "other_branch",
            "window": "branch",
            "slot": None,
            "topup": not pos,
        }
    if arm == "sham2":
        return {
            "stratum": "llm_positive",
            "process_label": "other_branch",
            "window": "branch",
            "slot": None,
            "topup": True,
        }
    if branched and len(arm) >= 2 and arm[0] == "L" and arm[1].isdigit():
        return {
            "stratum": "other_branch",
            "process_label": "other_branch",
            "window": "branch",
            "slot": None,
            "topup": False,
        }
    return None


def load_scores(path: str | Path) -> pl.DataFrame:
    """Cached D9 scores: ``run_id, decision_id, score`` (+ ``agent_id`` / ``period``),
    parquet or CSV."""
    p = Path(path)
    df = pl.read_parquet(p) if p.suffix == ".parquet" else pl.read_csv(p)
    need = {"run_id", "decision_id", "score"}
    if not need <= set(df.columns):
        raise ValueError(f"{p}: scores need the columns {sorted(need)}")
    return df.select(["run_id", "decision_id", "score"]).with_columns(
        pl.col("run_id").cast(pl.Utf8), pl.col("decision_id").cast(pl.Int64)
    )


def campaign_lane(name: str) -> str | None:
    """Registered lane of a campaign from its name (``TR-H``, ``CA-H``, ``TR-S-ext`` -> the
    suffix H / S / M), or None. The manifests and the decisions carry PROVIDER lane strings
    (``anthropic``, ``anthropic_haiku``), never the registered lane, so the lane filter is by
    campaign (A-013)."""
    base = re.sub(r"-ext\d*$", "", str(name))
    suffix = base.rsplit("-", 1)[-1]
    return suffix if suffix in ("H", "M", "S") else None


def candidate_items(
    campaign_dirs: Sequence[str | Path],
    scores: pl.DataFrame | None,
    dev_groups: Sequence[str],
    windows: dict[str, tuple[int, int]] | None = None,
    punished: Iterable[str] = (),
    lane: str = LANE,
) -> pl.DataFrame:
    """Every scored, non-empty scratchpad of a development-half run of ``lane`` that falls in a
    stratum, inside its window (live slot only for ghosts), with the rendered judge text.
    The ghosts are the sub-runs ``<sham run>/ghost/<slot>`` of each sham (A-013); the lane is
    the campaign's (:func:`campaign_lane`)."""
    windows = windows or WINDOWS
    dev = {str(g) for g in dev_groups}
    punished = set(punished)
    rows: list[dict[str, Any]] = []
    for d in campaign_dirs:
        d = Path(d)
        if not d.is_dir():
            continue
        camp_lane = campaign_lane(d.name)
        if camp_lane is not None and camp_lane != lane:
            continue
        runs = [x for rd0 in discover_runs(d) for x in (rd0, *discover_ghosts(rd0))]
        for rd in runs:
            info = run_info(rd)
            man = read_manifest(rd)
            if info.get("seed") is None:
                raise ValueError(f"{info['run_id']}: no seed (seed group lane:seed, M7)")
            group = f"{lane}:{info['seed']}"  # the split's key (detect.seed_group_id, M7)
            if man.get("status") not in (None, "ok") or group not in dev:
                continue
            tags = (man.get("resolved_config") or {}).get("tags") or man.get("tags") or {}
            cls = classify_run(info, tags, punished)
            if cls is None:
                continue
            llm = read_table(rd, "llm_decisions")
            if llm.height == 0:
                continue
            lo, hi = windows[cls["window"]]
            llm = llm.filter((pl.col("period") >= lo) & (pl.col("period") <= hi))
            if cls["slot"] is not None and "slot" in llm.columns:
                llm = llm.filter(pl.col("slot") == cls["slot"])
            if llm.height == 0:
                continue
            if scores is None:  # sampler-first (A-014): the score comes after the draw
                joined = llm.with_columns(pl.lit(None, dtype=pl.Float64).alias("score"))
            else:
                sc = scores.filter(pl.col("run_id") == info["run_id"]).drop("run_id")
                if sc.height == 0:
                    continue
                joined = llm.join(sc, on="decision_id", how="inner")
            for r in joined.iter_rows(named=True):
                body = scratchpad_text(r, ("plans", "insights"))
                if not body.strip():
                    continue
                if scores is not None and (
                    r.get("score") is None or not np.isfinite(float(r["score"]))
                ):
                    continue
                rows.append(
                    {
                        "run_id": info["run_id"],
                        "trunk_id": str(info["trunk_id"]),
                        "seed": str(info.get("seed")),
                        "seed_group": group,
                        "campaign": d.name,
                        "arm": str(info["arm"]),
                        "label": str(info.get("label") or ""),
                        "stratum": cls["stratum"],
                        "process_label": cls["process_label"],
                        "topup": bool(cls["topup"]),
                        "agent_id": int(r["agent_id"]) if r.get("agent_id") is not None else None,
                        "slot": int(r["slot"]) if r.get("slot") is not None else int(r["agent_id"]),
                        "period": int(r["period"]),
                        "decision_id": int(r["decision_id"]),
                        "score": None if r["score"] is None else float(r["score"]),
                        "lane": lane,  # registered lane (the decision's is the provider's)
                        "model": r.get("model"),
                        "prompt_hash": r.get("prompt_hash"),
                        "text": format_excerpts([body]),
                    }
                )
    return pl.DataFrame(rows, schema=CAND_SCHEMA)


# --------------------------------------------------------------------------- sampler


def _accept(item: dict[str, Any], picked: list[dict[str, Any]], caps: dict[str, int]) -> bool:
    run = [p for p in picked if p["run_id"] == item["run_id"]]
    if len(run) >= caps["per_run"]:
        return False
    same_slot = [p for p in run if p["slot"] == item["slot"]]
    if len(same_slot) >= caps["per_run_slot"]:
        return False
    if any(abs(p["period"] - item["period"]) < caps["spacing_periods"] for p in same_slot):
        return False
    fam = [p for p in picked if p["seed_group"] == item["seed_group"]]
    return len(fam) < caps["per_trunk_family"]


def _interleave_by_group(items: list[dict[str, Any]], rng: np.random.Generator) -> list[dict]:
    """Round-robin over the seed groups (group order shuffled), so the first picks of a
    stratum spread over at least ``MIN_SEED_GROUPS`` groups whenever that many have items."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for it in items:
        groups.setdefault(it["seed_group"], []).append(it)
    keys = list(groups)
    keys = [keys[int(i)] for i in rng.permutation(len(keys))]
    out: list[dict[str, Any]] = []
    while any(groups[k] for k in keys):
        for k in keys:
            if groups[k]:
                out.append(groups[k].pop(0))
    return out


def sample_sheet(
    cands: pl.DataFrame,
    n: int = N_ITEMS,
    seed: int = SAMPLER_SEED,
    caps: dict[str, int] | None = None,
    salt: str | None = None,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Stratified draw: ``n // 5`` items per stratum (remainder to the first strata), under
    the caps, candidates of each stratum shuffled with ``seed`` and interleaved by seed group;
    in ``llm_positive`` the true positives come before the sham top-up. A short stratum's
    deficit is spread evenly over the strata with spare items (remainder to stratum 1). The
    picked items are shuffled across strata before the ids (sha256(salt, run_id,
    decision_id)[:10]) are assigned. Returns the rows (sheet order) and the metadata, which
    never contains the salt."""
    caps = {**CAPS, **(caps or {})}
    rng = np.random.default_rng(seed)
    salt = salt or secrets.token_hex(16)
    pools: dict[str, list[dict[str, Any]]] = {}
    for s in STRATA:
        items = (
            [dict(r) for r in cands.filter(pl.col("stratum") == s).iter_rows(named=True)]
            if cands.height
            else []
        )
        items = [items[int(i)] for i in rng.permutation(len(items))]
        main = _interleave_by_group([it for it in items if not it["topup"]], rng)
        top = _interleave_by_group([it for it in items if it["topup"]], rng)
        pools[s] = main + top
    base = n // len(STRATA)
    quota = {s: base for s in STRATA}
    for s in STRATA[: n - base * len(STRATA)]:
        quota[s] += 1
    picked: list[dict[str, Any]] = []
    taken = {s: 0 for s in STRATA}
    cursor = {s: 0 for s in STRATA}

    def draw(s: str, want: int) -> int:
        got = 0
        while got < want and cursor[s] < len(pools[s]):
            item = pools[s][cursor[s]]
            cursor[s] += 1
            if _accept(item, picked, caps):
                picked.append(item)
                got += 1
        return got

    # round-robin over the strata (one item each per round) so that no stratum is starved by
    # the seed-group family cap having been spent by the strata drawn before it
    for _ in range(max(quota.values())):
        for s in STRATA:
            if taken[s] < quota[s]:
                taken[s] += draw(s, 1)
    shortfall = {s: quota[s] - taken[s] for s in STRATA if quota[s] > taken[s]}
    missing = n - len(picked)
    rounds = 0
    while missing > 0 and rounds < 20:
        rounds += 1
        spare = [s for s in STRATA if cursor[s] < len(pools[s])]
        if not spare:
            break
        share = {s: missing // len(spare) for s in spare}
        share[spare[0]] += missing - sum(share.values())
        got_any = 0
        for s in spare:
            got = draw(s, share[s])
            taken[s] += got
            got_any += got
        missing -= got_any
        if not got_any:
            break
    picked = [picked[int(i)] for i in rng.permutation(len(picked))]
    note = "; ".join(f"{s} short by {k}" for s, k in shortfall.items())
    for p in picked:
        p["item_id"] = _sha(salt, p["run_id"], str(p["decision_id"]))[:10]
        p["text_sha256"] = _sha(p["text"])
        p["salt"] = salt
        p["sampler_seed"] = int(seed)
        p["half"] = "dev"
        p["shortfall_note"] = note
    if len({p["item_id"] for p in picked}) != len(picked):
        raise RuntimeError("item id collision; choose another salt")
    groups = {s: sorted({p["seed_group"] for p in picked if p["stratum"] == s}) for s in STRATA}
    meta = {
        "n_requested": n,
        "n_sampled": len(picked),
        "seed": seed,
        "caps": caps,
        "min_seed_groups_per_stratum": MIN_SEED_GROUPS,
        "strata": list(STRATA),
        "quota": quota,
        "taken": taken,
        "shortfall": shortfall,
        "shortfall_note": note,
        "n_by_stratum": {s: sum(1 for p in picked if p["stratum"] == s) for s in STRATA},
        "n_by_process_label": {
            k: sum(1 for p in picked if p["process_label"] == k)
            for k in sorted({p["process_label"] for p in picked})
        },
        "n_positive_true": sum(
            1 for p in picked if p["stratum"] == "llm_positive" and not p["topup"]
        ),
        "seed_groups_by_stratum": groups,
        "strata_below_min_seed_groups": [
            s for s, g in groups.items() if 0 < len(g) < MIN_SEED_GROUPS
        ],
        "development_half_only": True,
        "id_scheme": "sha256(salt|run_id|decision_id)[:10]; the salt is in the key only",
    }
    sheet = (
        pl.DataFrame(picked, schema=SHEET_SCHEMA) if picked else pl.DataFrame(schema=SHEET_SCHEMA)
    )
    return sheet, meta


# --------------------------------------------------------------------------- files


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def template_header(template: str | Path) -> list[str]:
    with open(template, encoding="utf-8-sig", newline="") as fh:
        return [c.strip() for c in next(csv.reader(fh))]


def write_sheet(
    sheet: pl.DataFrame,
    meta: dict[str, Any],
    out_dir: str | Path,
    key_out: str | Path,
    repo_root: str | Path = ".",
    template: str | Path | None = None,
) -> dict[str, Any]:
    """``labelling_sheet.csv`` (the template header; only ``item_id`` and ``text`` filled)
    and ``sheet_meta.json`` under ``out_dir``; ``key_SEALED.csv`` at ``key_out``, which must
    lie outside the repository. The header is checked against the template file when it exists.
    Returns the paths and the sha256 of the sheet and of the key (the seal for SPRINT_LOG)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    key_path = Path(key_out)
    if _inside(key_path, Path(repo_root)):
        raise ValueError(f"the sealed key must be written outside the repository, not {key_path}")
    tpl = Path(template) if template is not None else Path(repo_root) / TEMPLATE_PATH
    header = list(SHEET_COLUMNS)
    if tpl.exists() and template_header(tpl) != header:
        raise ValueError(f"{tpl}: template header differs from the codebook columns")
    key_path.parent.mkdir(parents=True, exist_ok=True)
    sheet_path = out / "labelling_sheet.csv"
    with open(sheet_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for r in sheet.iter_rows(named=True):
            w.writerow([r["item_id"], r["text"], *[""] * (len(header) - 2)])
    with open(key_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(KEY_COLUMNS)
        for r in sheet.iter_rows(named=True):
            row = dict(r)
            row["d9_score"] = r["score"]
            w.writerow(["" if row.get(c) is None else row[c] for c in KEY_COLUMNS])
    seals = {
        "sheet": str(sheet_path),
        "key": str(key_path),
        "sheet_sha256": _file_sha(sheet_path),
        "key_sha256": _file_sha(key_path),
    }
    public = {k: v for k, v in meta.items() if k != "salt"}
    (out / "sheet_meta.json").write_text(
        json.dumps(
            {**public, **seals, "sheet_columns": header, "key_columns": list(KEY_COLUMNS)},
            indent=2,
        ),
        encoding="utf-8",
    )
    return seals


# --------------------------------------------------------------------------- validator


def _consistency_flags(iid: str, level: int, codes: dict[str, bool]) -> list[str]:
    """Codebook section 3 advisory table (expected codes per level)."""
    hsfpw = any(codes.get(k, False) for k in "HSFPW")
    a = codes.get("A", False)
    if level == 4 and not a:
        return [f"{iid}: level 4 without code_agreement (A = 1 expected) (advisory)"]
    if level == 3 and (a or not hsfpw):
        return [f"{iid}: level 3 expects A = 0 and one of H, S, F, P, W = 1 (advisory)"]
    if level == 2 and (not codes.get("R", False) or hsfpw or a):
        return [f"{iid}: level 2 expects R = 1 and H = S = F = P = W = A = 0 (advisory)"]
    if level == 1 and (hsfpw or a):
        return [f"{iid}: level 1 expects H = S = F = P = W = A = 0 (advisory)"]
    return []


def validate_sheet(path: str | Path) -> dict[str, Any]:
    """Check a filled sheet: the template columns, level in 1-4, codes in {0, 1}, confidence
    in low/medium/high, labeller ``human:*`` or ``llm:*``; advisory flags only for the
    codebook's consistency table, empty codes, missing evidence above level 1, column order
    and duplicate ids. ``ok`` is False only on errors."""
    errors: list[str] = []
    flags: list[str] = []
    with open(path, encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        header = list(reader.fieldnames or [])
    missing = [c for c in SHEET_COLUMNS if c not in header]
    extra = [c for c in header if c not in SHEET_COLUMNS]
    if missing:
        errors.append(f"missing columns: {missing}")
    if extra:
        errors.append(f"columns not in the template: {extra}")
    if not missing and not extra and header != list(SHEET_COLUMNS):
        flags.append("column order differs from the template (advisory)")
    seen: set[str] = set()
    n_labelled = 0
    n_consistency = 0
    n_guess = 0
    by_labeller: dict[str, int] = {}
    for r in rows:
        iid = (r.get("item_id") or "").strip()
        if iid in seen:
            flags.append(f"{iid}: duplicate item id (advisory)")
        seen.add(iid)
        lev = (r.get("level") or "").strip()
        if lev == "":
            continue
        n_labelled += 1
        if not lev.isdigit() or int(lev) not in LEVELS:
            errors.append(f"{iid}: level {lev!r} not in 1-4")
            continue
        codes: dict[str, bool] = {}
        for letter, col in CODE_LETTER.items():
            v = (r.get(col) or "").strip()
            if v == "":
                continue
            if v not in ("0", "1"):
                errors.append(f"{iid}: {col} = {v!r} not in 0/1")
                continue
            codes[letter] = v == "1"
        if len(codes) < len(CODES):
            flags.append(f"{iid}: {len(CODES) - len(codes)} code column(s) empty (advisory)")
        conf = (r.get("confidence") or "").strip().lower()
        if conf and conf not in CONFIDENCE:
            errors.append(f"{iid}: confidence {conf!r} not in {list(CONFIDENCE)}")
        lab = (r.get("labeller") or "").strip()
        if lab and not (lab.startswith("human:") or lab.startswith("llm:")):
            errors.append(f"{iid}: labeller {lab!r} must be human:<initials> or llm:<model id>")
        by_labeller[lab or "(unattributed)"] = by_labeller.get(lab or "(unattributed)", 0) + 1
        if (r.get("condition_guess") or "").strip():
            n_guess += 1
        if lev != "1" and not (r.get("evidence") or "").strip():
            flags.append(f"{iid}: level {lev} without evidence (advisory)")
        cf = _consistency_flags(iid, int(lev), codes)
        n_consistency += len(cf)
        flags += cf
    return {
        "n_rows": len(rows),
        "n_labelled": n_labelled,
        "by_labeller": by_labeller,
        "n_condition_guess": n_guess,
        "consistency_flag_rate": (n_consistency / n_labelled) if n_labelled else 0.0,
        "errors": errors,
        "advisory": flags,
        "ok": not errors,
    }


# --------------------------------------------------------------------------- kappa


def _agree(entries: list[tuple[float, int, str, bool]]) -> dict[str, float]:
    if len(entries) < 2:
        return {"kappa_qw": float("nan"), "spearman_rho": float("nan"), "n": float(len(entries))}
    return agreement([e[0] for e in entries], [e[1] for e in entries])


def write_score_request(sheet: pl.DataFrame, path: str | Path) -> Path:
    """Sampler-first (A-014): the sampled decisions to score with the D9 judge
    (``run_id, decision_id``), written next to the sealed key and never to the labeller (the
    run ids name the condition)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["item_id", "run_id", "decision_id"])
        for r in sheet.iter_rows(named=True):
            w.writerow([r["item_id"], r["run_id"], r["decision_id"]])
    return p


def kappa_from_sheet(
    sheet_path: str | Path,
    key_path: str | Path,
    threshold: float = KAPPA_THRESHOLD,
    scores_path: str | Path | None = None,
) -> dict[str, Any]:
    """Quadratic-weighted kappa (``metrics.judge.agreement``) between the levels of a filled
    sheet and the CACHED D9 scores of the same items (from the key). The decision uses the
    ``human:*`` rows only (the judge enters the composite iff kappa >= ``threshold``); ``llm:*``
    rows, unattributed rows, kappa by stratum and kappa without the items whose condition the
    labeller guessed are descriptive. Sampler-first (A-014): the key holds no score; the cached
    D9 scores of the sampled decisions (``scores_path``: run_id, decision_id, score) are joined
    by (run_id, decision_id), the sealed key itself is never rewritten."""
    with open(sheet_path, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    with open(key_path, encoding="utf-8-sig", newline="") as fh:
        key = {r["item_id"]: r for r in csv.DictReader(fh)}
    if scores_path is not None:
        sc = load_scores(scores_path)
        look = {
            (str(r["run_id"]), int(r["decision_id"])): r["score"] for r in sc.iter_rows(named=True)
        }
        for k_ in key.values():
            if k_.get("d9_score") in (None, ""):
                v = look.get((k_["run_id"], int(k_["decision_id"])))
                k_["d9_score"] = "" if v is None else str(v)
    human: list[tuple[float, int, str, bool]] = []
    llm: list[tuple[float, int, str, bool]] = []
    unattributed = 0
    for r in rows:
        iid = (r.get("item_id") or "").strip()
        lev = (r.get("level") or "").strip()
        if not lev.isdigit() or iid not in key or key[iid].get("d9_score") in (None, ""):
            continue
        k = key[iid]
        entry = (
            float(k["d9_score"]),
            int(lev),
            str(k.get("stratum") or ""),
            bool((r.get("condition_guess") or "").strip()),
        )
        lab = (r.get("labeller") or "").strip()
        if lab.startswith("human:"):
            human.append(entry)
        elif lab.startswith("llm:"):
            llm.append(entry)
        else:
            unattributed += 1
    dec = _agree(human)
    kappa = float(dec["kappa_qw"])
    enters = bool(np.isfinite(kappa) and kappa >= threshold)
    return {
        "n": len(human),
        "kappa": kappa,
        "threshold": threshold,
        "judge_enters_composite": enters,
        "decision": {"labeller": "human:*", **dec, "judge_enters_composite": enters},
        "descriptive": {
            "llm": _agree(llm),
            "n_unattributed": unattributed,
            "by_stratum": {
                s: _agree([e for e in human if e[2] == s])["kappa_qw"]
                for s in sorted({e[2] for e in human})
            },
            "condition_guess_rate": (sum(1 for e in human if e[3]) / len(human)) if human else 0.0,
            "kappa_without_guessed": _agree([e for e in human if not e[3]])["kappa_qw"],
        },
    }


# --------------------------------------------------------------------------- driver


def build_corpus_v1(
    campaign_dirs: Sequence[str | Path],
    scores_path: str | Path | None,
    rules_path: str | Path,
    out_dir: str | Path,
    key_out: str | Path,
    repo_root: str | Path = ".",
    n: int = N_ITEMS,
    seed: int = SAMPLER_SEED,
    detect_corpus: str | Path | None = None,
    gates_json: str | Path | None = None,
    lane: str = LANE,
    salt: str | None = None,
    cap_family: int | None = None,
) -> dict[str, Any]:
    rules = json.loads(Path(rules_path).read_text(encoding="utf-8"))
    dev_groups = (
        rules.get("split", {}).get("dev_groups") or rules.get("split", {}).get("dev_trunks") or []
    )
    if not dev_groups:
        raise ValueError(f"{rules_path}: no development-half groups (freeze the rules first)")
    windows, winfo = windows_from_gates(gates_json)
    punished = punished_trunks(detect_corpus)
    cands = candidate_items(
        campaign_dirs,
        None if scores_path is None else load_scores(scores_path),
        dev_groups,
        windows,
        punished,
        lane,
    )
    sheet, meta = sample_sheet(
        cands,
        n,
        seed,
        caps=None if cap_family is None else {"per_trunk_family": int(cap_family)},
        salt=salt,
    )
    meta.update(
        {
            "lane": lane,
            "campaigns": [str(d) for d in campaign_dirs],
            "windows": {k: list(v) for k, v in windows.items()},
            "windows_source": winfo,
            "detect_corpus": str(detect_corpus) if detect_corpus else None,
            "n_punished_trunks": len(punished),
            "n_candidates": int(cands.height),
            "n_candidates_by_stratum": (
                {
                    d["stratum"]: int(d["len"])
                    for d in cands.group_by("stratum").len().sort("stratum").to_dicts()
                }
                if cands.height
                else {}
            ),
        }
    )
    if not punished:
        meta["note"] = (
            "no detection corpus (or no llm_punished trunk): the llm_positive stratum holds "
            "sham windows with process label other_branch only"
        )
    seals = write_sheet(sheet, meta, out_dir, key_out, repo_root)
    if scores_path is None:
        req = write_score_request(sheet, Path(key_out).parent / "score_request.csv")
        seals["score_request"] = str(req)
    return {**meta, **seals}
