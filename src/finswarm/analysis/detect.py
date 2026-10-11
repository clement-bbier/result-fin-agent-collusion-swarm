"""Detection corpus from the campaign results, rule freeze and the single evaluation.

``configs/detection/corpus.yaml`` says which runs become labelled windows and how they are
labelled (by generating process only, DETECTION_PROTOCOL section 1); the registered arms give
members, windows, benchmarks and lanes. Three steps, run from ``scripts/analyze.py``::

    --detect build     # corpus of D1..D9 features -> results/analysis/detection/corpus.parquet
    --detect freeze    # split by seed group, thresholds on the development half,
                       # configs/detection/frozen_rules.json (sha256) + eval half saved apart
    --detect evaluate  # ONCE, after the frozen rules are committed: the held-out half,
                       # LLM-only decision, scripted validity, transfer, outcome strata

The frozen rules file carries ``status: frozen`` and refuses a second freeze; before the
freeze it is the registered protocol stub (``status: unfrozen``) with a valid hash.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import polars as pl
import yaml
from pydantic import BaseModel, ConfigDict, Field

from finswarm.analysis.arms import (
    DEFAULT_ARMS_PATH,
    ArmBenchmarks,
    RegisteredArms,
    file_sha256,
    load_registered_arms,
    resolve_arm_benchmarks,
)
from finswarm.analysis.campaign import collect_runs, members_of
from finswarm.analysis.io import discover_runs, read_manifest, read_table, run_info
from finswarm.metrics import detection as DT
from finswarm.metrics._frames import window
from finswarm.metrics.collusion import capped, delta_price_incumbents

DEFAULT_CORPUS_PATH = Path("configs/detection/corpus.yaml")
DEFAULT_RULES_PATH = DT.DEFAULT_RULES_PATH
DEFAULT_OUT = Path("results/analysis/detection")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CorpusSource(_Strict):
    campaign: str
    arm: str | None = None
    label: str | None = None
    kind: str
    window: Literal["trunk", "branch"] | None = None
    probe: str | None = None
    placebo: str | None = None
    ghosts: str | None = None
    lane: str | None = None


class DecisionSpec(_Strict):
    positives: list[str] = ["llm_punished"]
    negatives: list[str] = ["ghost", "instruction"]
    lane: str = "H"
    transfer_lane: str | None = "M"


class SubsetSpec(_Strict):
    positives: list[str]
    negatives: list[str]


class FeatureSpec(_Strict):
    n_surrogates: int = 50
    n_boot: int = 100
    d7_n_perm: int = 200
    n_shifts: int | None = None


class CorpusSpec(_Strict):
    version: int
    seed: int = DT.DEFAULT_SEED
    max_fpr: float = 0.05
    theta7_kinds: list[str] = list(DT.THETA7_KINDS)
    decision: DecisionSpec = Field(default_factory=DecisionSpec)
    scripted_validity: SubsetSpec = Field(
        default_factory=lambda: SubsetSpec(positives=["sc3"], negatives=["sc1", "sc2_common_shock"])
    )
    fpr_targets: dict[str, float] = Field(default_factory=dict)
    features: FeatureSpec = Field(default_factory=FeatureSpec)
    placebo_quantile: float = DT.PLACEBO_QUANTILE
    windows: dict[str, tuple[int, int]] = Field(
        default_factory=lambda: {"trunk": (71, 100), "branch": (111, 150)}
    )
    sources: list[CorpusSource]


def load_corpus_spec(path: str | Path = DEFAULT_CORPUS_PATH) -> CorpusSpec:
    return CorpusSpec.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


# --------------------------------------------------------------------------- corpus


def _frame(run_dir: str | Path, run_id: str) -> pl.DataFrame:
    ap = read_table(run_dir, "agent_period")
    if ap.height and ("run_id" not in ap.columns or ap["run_id"].null_count() == ap.height):
        ap = ap.with_columns(pl.lit(run_id).alias("run_id"))
    return ap


def _deviation_period(run_dir: str | Path, fallback: int | None) -> int | None:
    """Period of the forced deviation (``events`` table), else the branch period + 1."""
    ev = read_table(run_dir, "events")
    if ev.height and "event_type" in ev.columns:
        hit = ev.filter(pl.col("event_type").is_in(["forced_deviation", "sting"]))
        if hit.height:
            return int(hit["period"].min())
    return None if fallback is None else int(fallback) + 1


def _window_of(frame: pl.DataFrame, win: tuple[int, int]) -> pl.DataFrame:
    return window(frame, int(win[0]), int(win[1]))


class _Collector:
    """Matched runs per campaign (registered arms) and scripted controls by label."""

    def __init__(self, campaigns_root: Path, spec: RegisteredArms, root: Path) -> None:
        self.root = root
        self.croot = campaigns_root
        self.spec = spec
        self.abench = resolve_arm_benchmarks(spec, root)
        self._matched: dict[str, list[dict[str, Any]]] = {}

    def matched(self, campaign: str) -> list[dict[str, Any]]:
        if campaign not in self._matched:
            d = self.croot / campaign
            self._matched[campaign] = collect_runs(d, self.spec)[0] if d.is_dir() else []
        return self._matched[campaign]

    def arm_runs(self, campaign: str, arm: str) -> list[dict[str, Any]]:
        return [r for r in self.matched(campaign) if r["registered_arm"] == arm]

    def label_runs(self, campaign: str, label: str) -> list[dict[str, Any]]:
        d = self.croot / campaign
        if not d.is_dir():
            return []
        out = []
        for rd in discover_runs(d):
            info = run_info(rd)
            if info["label"] == label or info["arm"] == label:
                man = read_manifest(rd)
                if man.get("status") in (None, "ok"):
                    out.append(info)
        return out


def build_corpus_from_results(
    campaigns_root: str | Path,
    corpus_spec: CorpusSpec,
    arms_spec: RegisteredArms,
    root: str | Path = ".",
    progress: Any = None,
    qualified: set[str] | None = None,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Labelled feature rows for every source of the corpus spec; returns ``(corpus, meta)``.

    Labels come from the generating process only: ``llm_sham`` windows become
    ``llm_punished`` / ``llm_unpunished`` through the S2 probe of their trunk against the
    sham2 placebo distribution (``llm_unprobed`` without a probe), never through Delta
    (:func:`_sham_labelling`). The placebo distribution holds one sham2-vs-sham cut per
    QUALIFIED trunk (``qualified``: the Gate 2 list of the campaign analysis; ``None`` =
    every trunk, recorded as ``meta["qualified_filter"] = False``), each against that
    trunk's own sham at that trunk's own S2 deviation period. ``outcome_label`` is the
    separate Delta stratum of every LLM window (reported apart, D1 removed).

    Netting (D-21): every ``llm_sham`` window is netted against the fresh-draw ghost
    sub-runs of its seed group; ghost and CA (``instruction``) windows against the
    cyclic-shift permutation null (``meta["ca_netting"] = "cyclic_shift"``). The seed group
    is ``f"{lane}:{seed}"`` (:func:`finswarm.metrics.detection.seed_group_id`).

    Raises ValueError (never a silent fallback) on: probes without a placebo set, a probe
    or a placebo trunk without its own sham, a probe without a deviation period, an
    ``llm_sham`` window without ghosts of its seed group, a run without a manifest seed.
    """
    col = _Collector(Path(campaigns_root), arms_spec, Path(root))
    fs = corpus_spec.features
    specs: list[DT.WindowSpec] = []
    qual = None if qualified is None else {str(t) for t in qualified}
    meta: dict[str, Any] = {
        "sources": [],
        "placebo": {},
        "probe_flags": {},
        "probe_flags_v1": {},
        "qualified_filter": qual is not None,
        "qualified": sorted(qual) if qual is not None else None,
        "seed_group": "lane:seed",
        "sham_netting": "ghost_set",
        "ghost_netting": "cyclic_shift",
        "ca_netting": "cyclic_shift",
    }
    trunk_ab = col.abench[arms_spec.inference.qualification.arm]
    null_rows: list[tuple[str, str, np.ndarray]] = []

    for src in corpus_spec.sources:
        if src.arm is not None:
            a = arms_spec.arms[src.arm]
            ab = col.abench.get(src.arm) or trunk_ab
            win = corpus_spec.windows[src.window or a.window]
            lane = src.lane or a.lane
            runs = col.arm_runs(src.campaign, src.arm)
        else:
            if src.label is None:
                raise ValueError(f"source {src.campaign}: give arm or label")
            ab = trunk_ab
            win = corpus_spec.windows[src.window or "branch"]
            lane = src.lane or "CPU"
            runs = col.label_runs(src.campaign, src.label)
        n_before = len(specs)
        where = f"{src.campaign}:{src.arm or src.label}"
        shift_null = src.kind in DT.SHIFT_NULL_KINDS
        # ghost frames of the trunk (by seed group) netting the LLM windows; ghost and CA
        # windows are netted by the cyclic-shift permutation null instead
        ghosts_by_group: dict[str, list[tuple[int, pl.DataFrame]]] = {}
        if src.ghosts and not shift_null:
            g_lane = src.lane or arms_spec.arms[src.ghosts].lane
            for g in col.arm_runs(src.campaign, src.ghosts):
                gf = _window_of(_frame(g["path"], g["run_id"]), win)
                if gf.height:
                    live = members_of(arms_spec, src.ghosts, g)[0]
                    ghosts_by_group.setdefault(_seed_group(g_lane, g), []).append((live, gf))
            for entries in ghosts_by_group.values():
                entries.sort(key=lambda e: e[0])  # by live slot: draws in a fixed order
        # probes, their deviation periods and the placebo distribution (llm_sham kind)
        probes: dict[str, dict[str, Any]] = {}
        probe_t0: dict[str, int] = {}
        placebo_cuts: list[float] = []
        dev_slot = _probe_slot(arms_spec, src.probe)
        if src.kind == "llm_sham":
            probes, probe_t0, placebo_cuts, pmeta = _sham_labelling(
                col, src, arms_spec, runs, qual, corpus_spec.placebo_quantile
            )
            if pmeta is not None:
                meta["placebo"][f"{src.campaign}:{src.arm}"] = pmeta
        n_unnetted = 0
        for r in runs:
            if src.arm is not None:
                members = members_of(arms_spec, src.arm, r)
            else:
                ap = _frame(r["path"], r["run_id"])
                members = sorted(int(x) for x in ap["slot"].drop_nulls().unique().to_list())
            full = _frame(r["path"], r["run_id"])
            frame = _window_of(full, win)
            if frame.height == 0:
                continue
            kind = src.kind
            extra: dict[str, Any] = {}
            sg = _seed_group(lane, r)
            ghost: list[pl.DataFrame] | None = None
            if not shift_null:
                entries = ghosts_by_group.get(sg, [])
                ghost = [gf for _, gf in entries] or None
                if ghost is None and kind == "llm_sham":
                    raise ValueError(
                        f"{where}: sham window of trunk {r['trunk_id']} (run {r['run_id']}, "
                        f"seed group {sg}) has no ghost sub-run of its seed group to net it "
                        f"(ghost arm {src.ghosts!r}); an LLM sham window is never un-netted"
                    )
                n_unnetted += int(ghost is None and src.ghosts is not None)
                if kind == "llm_sham":
                    null_rows.append(
                        (
                            lane,
                            r["trunk_id"],
                            DT.ghost_shift_null_draws(
                                [gf for _, gf in entries],
                                [s for s, _ in entries],
                                members,
                                key="slot",
                                n_surrogates=fs.n_surrogates,
                                n_shifts=fs.n_shifts,
                            ),
                        )
                    )
            if kind == "llm_sham":
                pr = probes.get(r["trunk_id"])
                if pr is None:
                    kind = "llm_unprobed"
                else:
                    pframe = _frame(pr["path"], pr["run_id"])
                    t0 = probe_t0[r["trunk_id"]]
                    try:
                        kind = DT.llm_sham_kind(
                            pframe,
                            full,
                            dev_slot,
                            t0,
                            placebo_cuts=placebo_cuts,
                            placebo_quantile=corpus_spec.placebo_quantile,
                        )
                        v1 = DT.llm_sham_kind(pframe, full, dev_slot, t0)
                    except KeyError as e:
                        raise ValueError(
                            f"{where}: S2 probe {pr['run_id']} of trunk {r['trunk_id']} cannot "
                            f"be scored against its sham ({e})"
                        ) from e
                    extra["probe_run"] = pr["run_id"]
                    meta["probe_flags"][r["run_id"]] = kind
                    meta["probe_flags_v1"][r["run_id"]] = v1
            dp = (
                capped(delta_price_incumbents(frame, ab.primary, members))
                if lane != "CPU"
                else None
            )
            specs.append(
                DT.WindowSpec.by_process(
                    f"{r['run_id']}:{win[0]}-{win[1]}",
                    r["run_id"],
                    r["trunk_id"],
                    kind,
                    frame,
                    ghost=ghost,
                    permutation_null=shift_null,
                    members=tuple(members),
                    dep_members=tuple(
                        sorted(int(x) for x in frame["slot"].drop_nulls().unique().to_list())
                    )
                    if src.kind == "ghost"
                    else None,
                    d7_cartel=_d7c(ab),
                    seed_group=sg,
                    outcome_label=DT.label_from_delta(dp) if dp is not None else None,
                    lane=lane,
                    extra={**extra, "bench": ab.primary, "br": ab.window_model.best_response()},
                )
            )
        meta["sources"].append(
            {
                "campaign": src.campaign,
                "arm": src.arm,
                "label": src.label,
                "kind": src.kind,
                "n_windows": len(specs) - n_before,
                "netting": "cyclic_shift"
                if shift_null
                else ("ghost_set" if src.ghosts else "none"),
                "n_unnetted": n_unnetted,
            }
        )
    corpus = DT.build_corpus(
        specs,
        None,
        None,
        bench_of=lambda s: s.extra["bench"],
        br_of=lambda s: s.extra["br"],
        n_surrogates=fs.n_surrogates,
        n_boot=fs.n_boot,
        d7_n_perm=fs.d7_n_perm,
        n_shifts=fs.n_shifts,
        key="slot",
    )
    meta["n_windows"] = corpus.height
    meta["kinds"] = corpus.group_by("kind").len().sort("kind").to_dicts() if corpus.height else []
    meta["ghost_null"] = _ghost_null_meta(null_rows, qual, corpus_spec.decision.lane)
    meta["ghost_null_definition"] = (
        "per llm_sham window: every ghost sub-run of its seed group, replayed rivals "
        "cyclically shifted against its live dealer (features.n_shifts shifts), TE summary "
        "minus the mean TE of the ghost set (the sham's own D5 netting term); sham_pooled "
        f"pools the draws of the qualified trunks of lane {corpus_spec.decision.lane}; "
        f"q95 = {DT.GHOST_NULL_QUANTILE} quantile of the finite draws"
    )
    return corpus, meta


def _q(draws: list[float]) -> float:
    return float(np.quantile(draws, DT.GHOST_NULL_QUANTILE)) if draws else float("nan")


def _ghost_null_meta(
    rows: list[tuple[str, str, np.ndarray]], qualified: set[str] | None, lane: str
) -> dict[str, Any]:
    """``meta["ghost_null"]``: per trunk the ghost cyclic-shift null draws of its sham
    window's D5-net (``d5net_draws``, ``q95``, ``n``), ``sham_pooled`` (the draws of the
    qualified trunks of the decision ``lane`` pooled; ``mean_q95`` the same quantile of the
    null of the MEAN over those trunks, draw-index matched, when every trunk has as many
    draws) and ``sham_pooled_by_lane``. JSON-serialisable (lists of floats)."""
    per: dict[str, dict[str, Any]] = {}
    for ln, trunk, d in rows:
        arr = np.asarray(d, dtype=float)
        prev = per.get(trunk, {"d5net_draws": [], "n_nonfinite": 0})
        draws = prev["d5net_draws"] + [float(x) for x in arr[np.isfinite(arr)]]
        per[trunk] = {
            "lane": ln,
            "d5net_draws": draws,
            "q95": _q(draws),
            "n": len(draws),
            "n_nonfinite": prev["n_nonfinite"] + int((~np.isfinite(arr)).sum()),
        }

    def pool(ln: str) -> dict[str, Any]:
        ts = sorted(
            t for t, v in per.items() if v["lane"] == ln and (qualified is None or t in qualified)
        )
        draws = [x for t in ts for x in per[t]["d5net_draws"]]
        lens = {per[t]["n"] for t in ts}
        mean = (
            np.mean([per[t]["d5net_draws"] for t in ts], axis=0).tolist()
            if ts and len(lens) == 1 and lens != {0}
            else []
        )
        return {
            "lane": ln,
            "q95": _q(draws),
            "n": len(draws),
            "trunks": ts,
            "qualified_filter": qualified is not None,
            "mean_d5net_draws": mean,
            "mean_q95": _q(mean),
            "n_mean_draws": len(mean),
        }

    by_lane = {ln: pool(ln) for ln in sorted({v["lane"] for v in per.values()} | {lane})}
    return {**per, "sham_pooled": by_lane[lane], "sham_pooled_by_lane": by_lane}


def _seed_group(lane: str | None, run: dict[str, Any]) -> str:
    """``f"{lane}:{seed}"`` of a run; a run without a manifest seed raises ValueError."""
    try:
        return DT.seed_group_id(lane, run.get("seed"))
    except ValueError as e:
        raise ValueError(f"run {run['run_id']} (trunk {run['trunk_id']}): {e}") from e


def _probe_slot(arms_spec: RegisteredArms, probe: str | None) -> int:
    """Deviating slot of the registered probe arm (slot 0 without a probe spec)."""
    a = arms_spec.arms[probe] if probe else None
    return int(a.probe.slot) if a is not None and a.probe else 0


def _sham_labelling(
    col: _Collector,
    src: CorpusSource,
    arms_spec: RegisteredArms,
    runs: list[dict[str, Any]],
    qualified: set[str] | None,
    quantile: float,
) -> tuple[dict[str, dict[str, Any]], dict[str, int], list[float], dict[str, Any] | None]:
    """S2 probes, their deviation periods and the placebo distribution of an ``llm_sham``
    source (DETECTION_PROTOCOL section 1, D-21); returns ``(probes, t0, cuts, meta)``.

    Every probe is scored against the sham of ITS OWN trunk. The placebo cuts come from the
    qualified trunks only (``qualified``; ``None`` = every trunk with a sham2): the impulse
    response of the trunk's sham2 versus the trunk's own sham at the trunk's own S2
    deviation period (its probe's; else its branch period + 1, the registered S2 deviation
    period). Raises ValueError, never falls back to the v1 fixed cut or to another trunk's
    sham, when: a probe has no sham on its trunk or no deviation period; a qualified
    placebo trunk has no sham or no branch period; probes exist but no finite placebo cut.
    Without any probe every window is ``llm_unprobed`` and no placebo is needed.
    """
    where = f"{src.campaign}:{src.arm}"
    shams = {r["trunk_id"]: r for r in runs}
    probes = {r["trunk_id"]: r for r in col.arm_runs(src.campaign, src.probe)} if src.probe else {}
    if not probes:
        return {}, {}, [], None
    t0: dict[str, int] = {}
    for t, pr in sorted(probes.items()):
        if t not in shams:
            raise ValueError(
                f"{where}: trunk {t} has an S2 probe ({pr['run_id']}) but no {src.arm} run; "
                "a probe is never scored against another trunk's sham"
            )
        d = _deviation_period(pr["path"], pr.get("branch_period"))
        if d is None:
            raise ValueError(
                f"{where}: S2 probe {pr['run_id']} of trunk {t} has no deviation period "
                "(no forced_deviation event and no branch_period)"
            )
        t0[t] = int(d)
    placebos = (
        {r["trunk_id"]: r for r in col.arm_runs(src.campaign, src.placebo)} if src.placebo else {}
    )
    used = sorted(t for t in placebos if qualified is None or t in qualified)
    dev_slot = _probe_slot(arms_spec, src.probe)
    cut_by: dict[str, float] = {}
    t0_by: dict[str, int] = {}
    for t in used:
        pb, sham = placebos[t], shams.get(t)
        if sham is None:
            raise ValueError(
                f"{where}: trunk {t} has a sham2 placebo ({pb['run_id']}) but no {src.arm} run; "
                "a placebo is never cut against another trunk's sham"
            )
        if t in t0:
            tt = t0[t]
        else:
            bp = pb.get("branch_period")
            bp = sham.get("branch_period") if bp is None else bp
            if bp is None:
                raise ValueError(f"{where}: placebo trunk {t} has no branch period (S2 t0)")
            tt = int(bp) + 1
        try:
            cut_by[t] = DT.placebo_cut(
                _frame(pb["path"], pb["run_id"]), _frame(sham["path"], sham["run_id"]), dev_slot, tt
            )
        except KeyError as e:
            raise ValueError(f"{where}: placebo cut of trunk {t} failed ({e})") from e
        t0_by[t] = tt
    cuts = [cut_by[t] for t in used]
    finite = [v for v in cuts if np.isfinite(v)]
    if not finite:
        raise ValueError(
            f"{where}: {len(probes)} S2 probe(s) but an empty placebo set (placebo arm "
            f"{src.placebo!r}: {len(placebos)} sham2 run(s), {len(used)} on qualified trunks, "
            f"qualified filter {'on' if qualified is not None else 'off'}); the punished flag "
            "needs the sham2-vs-sham placebo distribution (no fallback to the v1 fixed cut)"
        )
    eligible = [t for t in shams if qualified is None or t in qualified]
    meta = {
        "n": len(finite),
        "quantile": quantile,
        "threshold": float(np.quantile(finite, quantile)),
        "cuts": cuts,
        "trunks": used,
        "cut_by_trunk": cut_by,
        "t0": t0_by,
        "probe_t0": t0,
        "qualified_filter": qualified is not None,
        "nonfinite_trunks": [t for t in used if not np.isfinite(cut_by[t])],
        "qualified_without_placebo": sorted(t for t in eligible if t not in placebos),
        "probes_on_unqualified_trunks": sorted(
            t for t in probes if qualified is not None and t not in qualified
        ),
        "note": "placebo percentile close to its maximum (n <= 12)" if len(finite) <= 12 else "",
    }
    return probes, t0, finite, meta


def _d7c(ab: ArmBenchmarks) -> float:
    from finswarm.metrics.benchmarks import d7_cartel

    try:
        return float(d7_cartel(ab.window_model))
    except Exception:  # noqa: BLE001 - a missing cartel value is reported as NaN
        return float("nan")


# --------------------------------------------------------------------------- freeze / evaluate


def protocol_stub(corpus_spec: CorpusSpec, corpus_path: str | Path = DEFAULT_CORPUS_PATH) -> dict:
    """The registered protocol as an unfrozen rules file (valid hash, no thresholds)."""
    body = {
        "version": 1,
        "status": "unfrozen",
        "protocol": corpus_spec.model_dump(mode="json", exclude={"sources"}),
        "corpus_spec_sha256": file_sha256(corpus_path) if Path(corpus_path).exists() else None,
        "note": "thresholds are selected on the development half at freeze; the hash of the "
        "frozen file is committed before the evaluation half is read",
    }
    return {**body, "sha256": DT.rules_sha256(body)}


def write_stub(
    corpus_spec: CorpusSpec,
    rules_path: str | Path = DEFAULT_RULES_PATH,
    corpus_path: str | Path = DEFAULT_CORPUS_PATH,
) -> str:
    p = Path(rules_path)
    if p.exists():
        cur = json.loads(p.read_text(encoding="utf-8"))
        if cur.get("status") == "frozen":
            raise FileExistsError(f"{p} is frozen; refusing to overwrite with a stub")
    stub = protocol_stub(corpus_spec, corpus_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(stub, indent=2, sort_keys=True), encoding="utf-8")
    return stub["sha256"]


def _sha_df(df: pl.DataFrame) -> str:
    return hashlib.sha256(df.write_csv().encode()).hexdigest()


def freeze(
    corpus: pl.DataFrame,
    corpus_spec: CorpusSpec,
    rules_path: str | Path = DEFAULT_RULES_PATH,
    out_dir: str | Path = DEFAULT_OUT,
    corpus_path: str | Path = DEFAULT_CORPUS_PATH,
) -> dict[str, Any]:
    """Split by seed group, select the thresholds on the development half and freeze them.

    Refuses when ``rules_path`` already holds frozen rules. Writes ``dev.parquet`` and
    ``eval.parquet`` under ``out_dir``; the evaluation half is NOT read here. Raises
    ValueError, before anything is written, when a window has no seed group or when a seed
    group appears in both halves (DETECTION_PROTOCOL section 7: no seed in both halves).
    """
    p = Path(rules_path)
    if p.exists():
        cur = json.loads(p.read_text(encoding="utf-8"))
        if cur.get("status") == "frozen":
            raise FileExistsError(f"{p} already holds frozen rules (sha {cur.get('sha256')})")
    dev, ev = DT.split_half(corpus, seed=corpus_spec.seed)
    dev_groups, eval_groups = _seed_groups(dev, "development"), _seed_groups(ev, "evaluation")
    both = sorted(dev_groups & eval_groups)
    if both:
        raise ValueError(
            f"seed group(s) in both halves of the split: {both[:10]}"
            + (f" (+{len(both) - 10} more)" if len(both) > 10 else "")
        )
    if p.exists():  # a prior split-only step (corpus v1 labels) must be reproduced exactly
        prior = (json.loads(p.read_text(encoding="utf-8")).get("split") or {}).get("dev_groups")
        if prior and sorted(prior) != sorted(dev_groups):
            raise ValueError("the split moved since the split-only step (labels were drawn on it)")
    rules = DT.select_rules(
        dev,
        max_fpr=corpus_spec.max_fpr,
        seed=corpus_spec.seed,
        theta7_kinds=corpus_spec.theta7_kinds,
    )
    rules["status"] = "frozen"
    rules["protocol"] = corpus_spec.model_dump(mode="json", exclude={"sources"})
    rules["corpus_spec_sha256"] = file_sha256(corpus_path) if Path(corpus_path).exists() else None
    rules["corpus_sha256"] = _sha_df(corpus)
    rules["split"]["eval_groups"] = sorted(eval_groups)
    rules["split"]["dev_groups"] = sorted(dev_groups)
    rules["split"]["n_dev"], rules["split"]["n_eval"] = int(dev.height), int(ev.height)
    sha = DT.freeze_rules(rules, p, overwrite=True)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dev.write_parquet(out / "dev.parquet")
    ev.write_parquet(out / "eval.parquet")
    (out / "freeze.json").write_text(
        json.dumps(
            {
                "rules_sha256": sha,
                "rules_path": str(p),
                "n_dev": int(dev.height),
                "n_eval": int(ev.height),
                "dev_groups": rules["split"]["dev_groups"],
                "note": "commit configs/detection/frozen_rules.json before --detect evaluate",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"rules_sha256": sha, "n_dev": int(dev.height), "n_eval": int(ev.height)}


def split_only(
    corpus: pl.DataFrame,
    corpus_spec: CorpusSpec,
    rules_path: str | Path = DEFAULT_RULES_PATH,
    corpus_path: str | Path = DEFAULT_CORPUS_PATH,
) -> dict[str, Any]:
    """Write ONLY the seed-group split (``status: "split"``, seed ``corpus_spec.seed``): the
    development / evaluation groups that corpus v1 needs before the labels, with no threshold
    selected and nothing frozen. Refuses frozen rules; the later ``freeze`` recomputes the
    same split (same seed, same corpus) and must reproduce these groups, else it raises."""
    p = Path(rules_path)
    cur: dict[str, Any] = {}
    if p.exists():
        cur = json.loads(p.read_text(encoding="utf-8"))
        if cur.get("status") == "frozen":
            raise FileExistsError(f"{p} already holds frozen rules (sha {cur.get('sha256')})")
    dev, ev = DT.split_half(corpus, seed=corpus_spec.seed)
    dev_groups, eval_groups = _seed_groups(dev, "development"), _seed_groups(ev, "evaluation")
    both = sorted(dev_groups & eval_groups)
    if both:
        raise ValueError(f"seed group(s) in both halves of the split: {both[:10]}")
    prior = (cur.get("split") or {}) if cur.get("status") == "split" else {}
    if prior.get("dev_groups") and sorted(prior["dev_groups"]) != sorted(dev_groups):
        raise ValueError("the split differs from the one already written; refusing to move it")
    rules = {
        **{k: v for k, v in cur.items() if k not in ("status", "split")},
        "status": "split",
        "protocol": corpus_spec.model_dump(mode="json", exclude={"sources"}),
        "corpus_spec_sha256": file_sha256(corpus_path) if Path(corpus_path).exists() else None,
        "corpus_sha256": _sha_df(corpus),
        "split": {
            "seed": int(corpus_spec.seed),
            "dev_groups": sorted(dev_groups),
            "eval_groups": sorted(eval_groups),
            "n_dev": int(dev.height),
            "n_eval": int(ev.height),
        },
        "note": "split only: no threshold selected, nothing frozen (corpus v1 labelling)",
    }
    p.write_text(json.dumps(rules, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return {
        "status": "split",
        "n_dev_groups": len(dev_groups),
        "n_eval_groups": len(eval_groups),
        "dev_groups": sorted(dev_groups),
        "rules_path": str(p),
    }


def _seed_groups(half: pl.DataFrame, name: str) -> set[str]:
    """Seed groups of one half of the split; a null seed group raises ValueError."""
    if "seed_group" not in half.columns or half["seed_group"].null_count():
        raise ValueError(f"{name} half: every window needs a seed_group (no trunk fallback)")
    return {str(x) for x in half["seed_group"].to_list()}


def evaluate_frozen(
    rules_path: str | Path = DEFAULT_RULES_PATH,
    out_dir: str | Path = DEFAULT_OUT,
    n_boot: int = 2000,
) -> dict[str, Any]:
    """The single evaluation on the held-out half saved at freeze (``eval.parquet``)."""
    rules = DT.load_rules(rules_path)
    if rules.get("status") != "frozen":
        raise ValueError(f"{rules_path}: rules are not frozen (status {rules.get('status')!r})")
    out = Path(out_dir)
    ev = pl.read_parquet(out / "eval.parquet")
    dev = pl.read_parquet(out / "dev.parquet") if (out / "dev.parquet").exists() else ev.clear()
    seed = int(rules.get("seed", DT.DEFAULT_SEED))
    proto = rules.get("protocol", {})
    dec = proto.get("decision", {})
    report = {
        "rules_sha256": rules["sha256"],
        "n_eval": int(ev.height),
        "report": DT.evaluate(ev, rules, n_boot=n_boot, seed=seed),
        "decision": DT.decision_reports(
            ev, rules, n_boot=n_boot, seed=seed, transfer_lane=dec.get("transfer_lane", "M")
        ),
        "outcome_strata": DT.outcome_report(
            dev, ev, float(rules.get("max_fpr", 0.05)), n_boot, seed
        ),
        "fpr_targets": proto.get("fpr_targets", {}),
    }
    report["decision_rules"] = _h9_rules(report, proto)
    (out / "report.json").write_text(json.dumps(_clean(report), indent=2), encoding="utf-8")
    (out / "report.md").write_text(report_md(_clean(report)), encoding="utf-8")
    return report


def _h9_rules(report: dict[str, Any], proto: dict[str, Any]) -> dict[str, Any]:
    """PREREG H9 criteria on the LLM-only decision report (AUC >= 0.85, FPR targets)."""
    llm = report["decision"].get("llm_only", {})
    out: dict[str, Any] = {"status": llm.get("status")}
    if llm.get("status") != "ok":
        return out
    comp = llm["composite"]
    out["auc"] = comp["auc"]
    out["auc_ge_0.85"] = bool(np.isfinite(comp["auc"]) and comp["auc"] >= 0.85)
    targets = proto.get("fpr_targets", {})
    by = {**comp.get("fpr_by_type", {})}
    sv = report["decision"].get("scripted_validity", {})
    if sv.get("status") == "ok":
        by.update({k: v for k, v in sv["composite"]["fpr_by_type"].items() if k not in by})
    out["fpr_targets"] = {
        k: {"value": by.get(k), "max": v, "ok": by.get(k) is not None and by[k] <= v}
        for k, v in targets.items()
    }
    feat = llm.get("feature_auc", {})
    if "D9" in feat and np.isfinite(feat.get("D9", float("nan"))):
        best_action = max(v for k, v in feat.items() if k != "D9" and np.isfinite(v))
        out["actions_beat_judge_by_0.10"] = bool(best_action - feat["D9"] >= 0.10)
    tr = report["decision"].get("transfer", {})
    if tr.get("status") == "ok":
        out["transfer_auc"] = tr["composite"]["auc"]
        out["transfer_ge_0.75"] = bool(tr["composite"]["auc"] >= 0.75)
    return out


def _clean(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, list | tuple):
        return [_clean(v) for v in o]
    if isinstance(o, np.generic):
        o = o.item()
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def report_md(report: dict[str, Any]) -> str:
    from finswarm.analysis import tables

    lines = [
        "# Detection: single evaluation on the held-out half",
        "",
        f"Frozen rules sha256 `{report['rules_sha256']}`; {report['n_eval']} windows.",
        "",
        "## H9 decision set (LLM-only: S2-punished sham windows vs ghosts and CA)",
        "",
    ]
    llm = report["decision"].get("llm_only", {})
    if llm.get("status") == "ok":
        lines += [tables.detector_table(llm), ""]
    else:
        lines += [f"Status: {llm.get('status')}.", ""]
    lines += [
        "### Decision rules",
        "",
        "```",
        json.dumps(report["decision_rules"], indent=2),
        "```",
        "",
    ]
    for key, title in (
        ("scripted_validity", "Scripted ROC (implementation validity, reported apart)"),
        ("transfer", "Transfer to the Mistral lane (fixed rules)"),
    ):
        sub = report["decision"].get(key)
        if sub:
            lines += [f"## {title}", ""]
            lines += (
                [tables.detector_table(sub), ""]
                if sub.get("status") == "ok"
                else [
                    f"Status: {sub.get('status')}.",
                    "",
                ]
            )
    lines += ["## Full held-out corpus", "", tables.detector_table(report["report"]), ""]
    oc = report.get("outcome_strata", {})
    lines += [
        "## Outcome strata (reported apart, D1 removed)",
        "",
        f"Status: {oc.get('status')}"
        + (f"; AUC {tables.fmt(oc['composite']['auc'], 3)}" if oc.get("status") == "ok" else "")
        + ".",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- FPR of D5-net


def fpr_null(
    campaigns_root: str | Path,
    corpus_spec: CorpusSpec,
    arms_spec: RegisteredArms,
    root: str | Path = ".",
    level: float = 0.05,
    n_surrogates: int = 50,
    n_shifts: int | None = None,
    campaign: str | None = None,
    ghost_arm: str = "ghost",
) -> dict[str, Any]:
    """EXPLORATORY (A-019): false-positive rate of D5-net on the ghost windows (negatives by
    construction: replayed rivals, no feedback from the live dealer) under two nulls, each a
    one-sided test at ``level`` (a window is flagged when its statistic exceeds the
    ``1 - level`` quantile of the null draws):

    * ``plain_shift``: the window's own transfer entropy against the draws of the plain
      cyclic-shift null of that window (rival series shifted against the live dealer);
    * ``ghost_null``: the window netted by the OTHER ghosts of its trunk (D5-net =
      TE(window) - mean TE of the other ghosts) against :func:`DT.ghost_shift_null_draws` of
      those other ghosts, i.e. the null the family-S H6 guard uses for the sham.

    Both rates, their counts and the per-window rows are returned; nothing is frozen and no
    decision depends on it."""
    from finswarm.metrics._frames import as_matrix
    from finswarm.metrics.dependence import transfer_entropy

    col = _Collector(Path(campaigns_root), arms_spec, Path(root))
    src = next(
        (s for s in corpus_spec.sources if s.kind == "ghost" and (campaign in (None, s.campaign))),
        None,
    )
    if src is None:
        raise ValueError("fpr_null: the corpus spec has no ghost source")
    runs = col.arm_runs(src.campaign, src.arm or ghost_arm)
    win = corpus_spec.windows[src.window or arms_spec.arms[src.arm or ghost_arm].window]
    by_trunk: dict[str, list[dict[str, Any]]] = {}
    for r in runs:
        by_trunk.setdefault(r["trunk_id"], []).append(r)
    rows: list[dict[str, Any]] = []
    for trunk, grs in sorted(by_trunk.items()):
        frames = [
            (_window_of(_frame(g["path"], g["run_id"]), win), int(g["ghost_slot"])) for g in grs
        ]
        frames = [(f, s) for f, s in frames if f.height]
        for i, (f, live) in enumerate(frames):
            slots = sorted(int(x) for x in f["slot"].drop_nulls().unique().to_list())
            others = [fs for j, (fs, _) in enumerate(frames) if j != i]
            o_live = [s for j, (_, s) in enumerate(frames) if j != i]
            dq = as_matrix(f.filter(pl.col("slot").is_in(slots)), key="slot", diff=True)
            te = transfer_entropy(dq, n_surrogates=n_surrogates, seed=0).summary
            plain = DT.shift_null_draws(
                f,
                key="slot",
                members=[live],
                dep_members=slots,
                live=live,
                n_surrogates=n_surrogates,
                n_shifts=n_shifts,
            )
            row: dict[str, Any] = {
                "run_id": grs[i]["run_id"],
                "trunk_id": trunk,
                "live_slot": live,
                "te": float(te),
                "plain_q": float(np.quantile(plain, 1 - level)) if plain.size else None,
                "plain_flag": bool(plain.size and te > np.quantile(plain, 1 - level)),
            }
            if others:
                from finswarm.metrics.detection import _ghost_te_mean  # noqa: PLC0415

                net = te - _ghost_te_mean(others, slots, "slot", n_surrogates, 0)
                gdraws = DT.ghost_shift_null_draws(
                    others, o_live, slots, key="slot", n_surrogates=n_surrogates, n_shifts=n_shifts
                )
                row["d5net"] = float(net)
                row["ghost_q"] = float(np.quantile(gdraws, 1 - level)) if gdraws.size else None
                row["ghost_flag"] = bool(gdraws.size and net > np.quantile(gdraws, 1 - level))
            rows.append(row)

    def rate(flag: str) -> dict[str, Any]:
        v = [r[flag] for r in rows if flag in r]
        return {"n": len(v), "flagged": int(sum(v)), "fpr": (sum(v) / len(v)) if v else None}

    return {
        "status": "exploratory (A-019), nothing frozen",
        "level": level,
        "window": list(win),
        "campaign": src.campaign,
        "n_windows": len(rows),
        "n_shifts": n_shifts,
        "plain_shift": rate("plain_flag"),
        "ghost_null": rate("ghost_flag"),
        "rows": rows,
    }


# --------------------------------------------------------------------------- driver


def run_detect(
    step: str,
    campaigns_root: str | Path = "results/campaigns",
    corpus_path: str | Path = DEFAULT_CORPUS_PATH,
    rules_path: str | Path = DEFAULT_RULES_PATH,
    arms_path: str | Path = DEFAULT_ARMS_PATH,
    root: str | Path = ".",
    out_dir: str | Path = DEFAULT_OUT,
    n_boot: int = 2000,
    qualified: set[str] | None = None,
    fpr_shifts: int | None = None,
) -> dict[str, Any]:
    """``build`` (corpus.parquet), ``freeze`` (rules + halves) or ``evaluate`` (report).

    ``qualified`` (the Gate 2 qualified trunks of the campaign analysis) restricts the
    placebo distribution of the corpus build; ``None`` uses every trunk and records
    ``qualified_filter: false`` in ``corpus_meta.json``."""
    cs = load_corpus_spec(corpus_path)
    out = Path(out_dir)
    if step == "stub":
        return {"rules_sha256": write_stub(cs, rules_path, corpus_path)}
    if step == "evaluate":
        return evaluate_frozen(rules_path, out, n_boot)
    arms = load_registered_arms(arms_path)
    if step == "fpr-null":
        res = fpr_null(campaigns_root, cs, arms, root, n_shifts=fpr_shifts)
        out.mkdir(parents=True, exist_ok=True)
        (out / "fpr_null.json").write_text(json.dumps(_clean(res), indent=2), encoding="utf-8")
        return {k: v for k, v in res.items() if k != "rows"} | {"path": str(out / "fpr_null.json")}
    corpus_file = out / "corpus.parquet"
    if step == "build" or not corpus_file.exists():
        corpus, meta = build_corpus_from_results(
            campaigns_root, cs, arms, root, qualified=qualified
        )
        out.mkdir(parents=True, exist_ok=True)
        corpus.write_parquet(corpus_file)
        (out / "corpus_meta.json").write_text(json.dumps(_clean(meta), indent=2), encoding="utf-8")
        if step == "build":
            return {"n_windows": corpus.height, "kinds": meta["kinds"], "path": str(corpus_file)}
    corpus = pl.read_parquet(corpus_file)
    if step == "split":
        return split_only(corpus, cs, rules_path, corpus_path)
    if step == "freeze":
        return freeze(corpus, cs, rules_path, out, corpus_path)
    raise ValueError(
        f"unknown detect step {step!r} (build | split | freeze | evaluate | stub | fpr-null)"
    )
