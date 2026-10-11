"""Detection protocol (D.4): labelled run windows, split-half, frozen rules, one evaluation.

Pipeline::

    corpus  = build_corpus(window_specs, bench, br_fn)      # one row per 40-period window
    dev, ev = split_half(corpus)                            # by seed group, stratified
    rules   = select_rules(dev)                             # thresholds at FPR <= 5 % on dev
    freeze_rules(rules, "configs/detection/frozen_rules.json")
    report  = evaluate(ev, rules)                           # once, on the other half

Detectors (columns): D1 Delta level; D2 price dispersion (lower = collusive); D3 lagged
cross-correlation; D4net Granger share net of ghost; D5net transfer entropy net of
surrogates (and of ghost when given); D6 Bonjour net influence; D7 non-best-response gap
in the registered price form (``collusion.d7_price`` over the incumbent whitelist, reported
with D7 / D7_cartel and its block sign-flip z); D8 sting follow probability; D9 judge score.
Composite = mean percentile rank of (D1, D7, D5net) against the frozen development
distribution (no fitted weights). Scorecard: collusive iff D1 >= t1 and D7 >= t7 and
D5net >= t5; t7 is calibrated on the negatives of the kinds ``THETA7_KINDS`` only. Parallel
conduct alone (D1, D2, D3) never suffices.

Labels (DETECTION_PROTOCOL section 1) come from the GENERATING PROCESS only
(:func:`process_label`): positives are SC3 cartels and LLM sham windows whose forced-undercut
probe S2 shows punishment (:func:`llm_sham_kind`, via ``collusion.impulse_response``);
negatives are SC1, SC2, SC5, fresh-draw ghosts (by construction) and the instruction arm CA.
Outcome strata (Delta_bar >= 0.30 / <= 0.15, :func:`label_from_delta`) live in the separate
``outcome_label`` column and are reported apart with D1 removed (:func:`outcome_report`).

Netting (D-21): LLM sham windows (:data:`GHOST_NETTED_KINDS`) are netted against the
fresh-draw ghost markets of their seed group and refuse to be built without them; ghost and
CA (``instruction``) windows (:data:`SHIFT_NULL_KINDS`) are netted against the cyclic-shift
permutation null. The split unit and the bootstrap cluster is the seed group
(:func:`seed_group_id`, ``"<lane>:<seed>"``); a window without one is refused
(:func:`split_half`, :func:`evaluate`), never replaced by its trunk or run.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from scipy.stats import rankdata

from finswarm.metrics._frames import as_matrix
from finswarm.metrics.benchmarks import Benchmarks, BestResponseFn
from finswarm.metrics.collusion import (
    capped,
    d7_price,
    delta,
    delta_price_incumbents,
    impulse_response,
    price_dispersion,
)
from finswarm.metrics.dependence import (
    granger_pairwise,
    lagged_xcorr,
    net_influence,
    transfer_entropy,
)
from finswarm.metrics.registry import register_metric

DEFAULT_SEED = 20261010
DEFAULT_RULES_PATH = Path("configs/detection/frozen_rules.json")
FEATURES: tuple[str, ...] = ("D1", "D2", "D3", "D4net", "D5net", "D6", "D7", "D8", "D9")
DIRECTIONS: dict[str, int] = {f: 1 for f in FEATURES} | {"D2": -1}
COMPOSITE: tuple[str, ...] = ("D1", "D7", "D5net")
SCORECARD: tuple[str, ...] = ("D1", "D7", "D5net")
POSITIVE = "positive"
#: H9 decision set (D-21): LLM-only, S2-punished sham windows against fresh-draw ghosts and
#: instruction negatives; the scripted ROC is implementation validity, reported apart
DECISION_POSITIVES: tuple[str, ...] = ("llm_punished",)
DECISION_NEGATIVES: tuple[str, ...] = ("ghost", "instruction")
SCRIPTED_POSITIVES: tuple[str, ...] = ("sc3",)
SCRIPTED_NEGATIVES: tuple[str, ...] = ("sc1", "sc2_common_shock")
PLACEBO_QUANTILE = 0.95
#: kinds labelled by their generating process (1 = collusive, 0 = competitive by construction)
PROCESS_LABELS: dict[str, int] = {
    "positive": 1,  # generic process positive (synthetic corpora)
    "sc3": 1,  # scripted grim-trigger cartel
    "llm_punished": 1,  # LLM sham window whose S2 forced-undercut probe shows punishment
    "sc1": 0,  # adaptive static best responders
    "sc2_common_shock": 0,
    "sc5_captive": 0,  # single dealer: D1 only, never in the composite ROC
    "ghost": 0,  # fresh-draw ghost market (no feedback)
    "instruction": 0,  # CA: instructed competitive LLM dealers
}
#: kinds kept in the corpus but reported apart (label None, never in the ROC)
REPORTED_APART: tuple[str, ...] = (
    "sc4",
    "s0",
    "llm_unpunished",
    "llm_unprobed",  # sham window without an S2 probe on its trunk
    "talk",
    "outcome",
)
LLM_POSITIVE = "llm_punished"
#: LLM sham windows: netted against the ghost markets of their seed group (required, D-21)
GHOST_NETTED_KINDS: tuple[str, ...] = ("llm_punished", "llm_unpunished", "llm_unprobed")
#: ghost and CA (instruction) windows: netted against the cyclic-shift permutation null
SHIFT_NULL_KINDS: tuple[str, ...] = ("ghost", "instruction")
#: negatives on which theta_7 is calibrated (DETECTION_PROTOCOL section 2)
THETA7_KINDS: tuple[str, ...] = ("sc1", "sc2_common_shock", "instruction", "ghost")
#: outcome strata: the label is Delta, so D1 leaves the composite and the scorecard
OUTCOME_COMPOSITE: tuple[str, ...] = ("D7", "D5net")
D7_REPORTED: tuple[str, ...] = ("D7_cartel", "D7_ratio", "D7_z", "D7_p")


# --------------------------------------------------------------------------- features


def window_features(
    frame: pl.DataFrame,
    bench: Benchmarks,
    br_fn: BestResponseFn,
    ghost: pl.DataFrame | None = None,
    state: np.ndarray | None = None,
    probe_follow: float = float("nan"),
    judge_score: float = float("nan"),
    n_surrogates: int = 50,
    n_boot: int = 100,
    block: int = 5,
    seed: int = 0,
    key: str = "agent_id",
    members: Sequence[int] | None = None,
    d7_cartel: float = float("nan"),
    d7_n_perm: int = 200,
    dep_members: Sequence[int] | None = None,
    permutation_null: bool = False,
    live: int | None = None,
    n_shifts: int | None = None,
) -> dict[str, float]:
    """D1..D9 for one window of one run (``agent_period`` rows of that window).

    ``members`` is the incumbent whitelist (``None`` = every seller): D1 is the CAPPED
    price index of the members against ``bench`` (the window's registered benchmarks;
    D-16); D7 is the registered price form :func:`finswarm.metrics.collusion.d7_price` over
    the members with ALL other sellers as rivals, ``br_fn`` the static best response of the
    window market, with ``D7_cartel``, ``D7_ratio`` and the block sign-flip ``D7_z`` / ``D7_p``
    alongside. ``dep_members`` are the sellers entering the pairwise dependence statistics
    D3-D6 (default: the members; a ghost window passes every seller so the live dealer is
    paired with its replayed rivals).

    Netting (DETECTION_PROTOCOL section 2, D-21): for an LLM window ``ghost`` is the
    matching fresh-draw ghost window (or a list of them: the trunk's ghost set, averaged)
    and D4 / D5 are reported net of it; an EMPTY ghost set raises ValueError (netting was
    asked for but nothing can net the window; ``None`` means no ghost netting). A ghost
    window is never netted against its own ghost set, and a CA (instruction) window has
    none: with ``permutation_null`` the rival series are cyclically shifted against the
    live dealer's series (column ``live`` of ``dep_members``, default the first member:
    the ghost's live slot, CA's slot 0) over all non-zero shifts (at most ``n_shifts``) and
    the mean of the shifted statistics is subtracted. ``state`` is the discretised market
    state per period for D6.
    """
    full = frame
    dep = list(dep_members) if dep_members is not None else members
    if members is not None:
        frame = frame.filter(pl.col(key).is_in([int(m) for m in members]))
    dep_frame = full.filter(pl.col(key).is_in([int(m) for m in dep])) if dep else full
    d7 = d7_price(
        full, br_fn, bench, members, key=key, d7_cartel=d7_cartel, n_perm=d7_n_perm, seed=seed
    )
    dq = as_matrix(dep_frame, key=key, diff=True)
    g = granger_pairwise(dq).share_significant
    te = transfer_entropy(dq, n_surrogates=n_surrogates, seed=seed).summary
    ghosts = _ghost_list(ghost)
    if ghost is not None and not ghosts:
        raise ValueError("empty ghost set: the window cannot be netted (pass ghost=None)")
    if ghosts:
        gs, ts = [], []
        for gf in ghosts:
            if dep:
                gf = gf.filter(pl.col(key).is_in([int(m) for m in dep]))
            gq = as_matrix(gf, key=key, diff=True)
            gs.append(granger_pairwise(gq).share_significant)
            ts.append(transfer_entropy(gq, n_surrogates=n_surrogates, seed=seed).summary)
        g -= float(np.nanmean(gs))
        te -= float(np.nanmean(ts))
    elif permutation_null:
        live_col = _live_column(dep_frame, key, dep, live, members)
        null_g, null_te = _shift_null(dq, live_col, n_surrogates, seed, n_shifts)
        g -= null_g
        te -= null_te
    st = None if state is None else np.asarray(state)[1:]
    return {
        "D1": _d1(frame, bench, members, key),
        "D2": price_dispersion(frame, key=key),
        "D3": lagged_xcorr(dq).summary,
        "D4net": g,
        "D5net": te,
        "D6": net_influence(dq, st, n_boot=n_boot, seed=seed, block=block).statistic,
        "D7": d7.d7,
        "D8": float(probe_follow),
        "D9": float(judge_score),
        "D7_cartel": d7.d7_cartel,
        "D7_ratio": d7.ratio,
        "D7_z": d7.z_perm,
        "D7_p": d7.p_perm,
    }


def _d1(frame: pl.DataFrame, bench: Benchmarks, members: Sequence[int] | None, key: str) -> float:
    """D1: the capped price index of the members (raw ``delta`` when prices are absent)."""
    if "price" not in frame.columns or frame.height == 0:
        return delta(frame, bench, key=key)
    mem = (
        list(members)
        if members is not None
        else sorted(int(x) for x in frame[key].drop_nulls().unique().to_list())
    )
    try:
        return capped(delta_price_incumbents(frame, bench, mem, key=key))
    except KeyError:
        return float("nan")


def _ghost_list(ghost: Any) -> list[pl.DataFrame]:
    """The ghost frames of a window (one frame or a sequence), empty frames dropped."""
    if ghost is None:
        return []
    frames = [ghost] if isinstance(ghost, pl.DataFrame) else list(ghost)
    return [g for g in frames if g.height]


def _live_column(
    dep_frame: pl.DataFrame,
    key: str,
    dep: Sequence[int] | None,
    live: int | None,
    members: Sequence[int] | None,
) -> int:
    keys = sorted(int(x) for x in dep_frame[key].drop_nulls().unique().to_list())
    target = live if live is not None else (list(members)[0] if members else keys[0])
    return keys.index(int(target)) if int(target) in keys else 0


def _shift_null(
    dq: np.ndarray, live: int, n_surrogates: int, seed: int, n_shifts: int | None
) -> tuple[float, float]:
    """Mean Granger share and TE summary over the cyclic shifts of the rival columns against
    column ``live`` (all non-zero shifts, at most ``n_shifts``); NaN with one column."""
    t, n = dq.shape
    if n < 2 or t < 4:
        return float("nan"), float("nan")
    gs, ts = [], []
    for k in _shift_list(t, n_shifts):
        shifted = _shifted(dq, live, k)
        gs.append(granger_pairwise(shifted).share_significant)
        ts.append(transfer_entropy(shifted, n_surrogates=n_surrogates, seed=seed).summary)
    return float(np.nanmean(gs)), float(np.nanmean(ts))


def _shift_list(t: int, n_shifts: int | None) -> list[int]:
    """All non-zero cyclic shifts of a ``t``-row matrix, at most ``n_shifts`` evenly spread."""
    shifts = list(range(1, t))
    if n_shifts is not None and len(shifts) > n_shifts:
        shifts = [shifts[i] for i in np.linspace(0, len(shifts) - 1, n_shifts).round().astype(int)]
    return shifts


def _shifted(dq: np.ndarray, live: int, k: int) -> np.ndarray:
    """``dq`` with every column but ``live`` cyclically shifted by ``k`` rows."""
    rivals = [j for j in range(dq.shape[1]) if j != live]
    out = dq.copy()
    out[:, rivals] = np.roll(dq[:, rivals], k, axis=0)
    return out


def _ghost_te_mean(
    ghosts: Any, dep: Sequence[int] | None, key: str, n_surrogates: int, seed: int
) -> float:
    """Mean TE summary of a ghost set over the sellers ``dep`` (the D5 netting term of
    :func:`window_features`); NaN when no ghost gives a finite value."""
    vals = []
    for gf in _ghost_list(ghosts):
        if dep:
            gf = gf.filter(pl.col(key).is_in([int(m) for m in dep]))
        gq = as_matrix(gf, key=key, diff=True)
        vals.append(transfer_entropy(gq, n_surrogates=n_surrogates, seed=seed).summary)
    fin = [v for v in vals if np.isfinite(v)]
    return float(np.mean(fin)) if fin else float("nan")


#: quantile of the ghost cyclic-shift null used by the family-S H6 guard
GHOST_NULL_QUANTILE = 0.95


def shift_null_draws(
    frame: pl.DataFrame,
    ghost: pl.DataFrame | Sequence[pl.DataFrame] | None = None,
    key: str = "agent_id",
    members: Sequence[int] | None = None,
    dep_members: Sequence[int] | None = None,
    live: int | None = None,
    n_surrogates: int = 50,
    seed: int = 0,
    n_shifts: int | None = None,
) -> np.ndarray:
    """Per-shift draws of D5 under the cyclic-shift permutation null of one window.

    The rival series of ``frame`` (sellers ``dep_members``, default ``members``, default
    every seller) are cyclically shifted against the live dealer's series (``live``,
    default the first member) over the same shifts as :func:`window_features`'
    ``permutation_null`` (all non-zero shifts, at most ``n_shifts``); each shift gives one
    transfer-entropy summary. With ``ghost`` (one frame or a set) the mean TE of the ghost
    set over the same sellers is subtracted from every draw, i.e. the draws are on the
    D5-net scale of a window netted against that ghost set. Without it the mean of the draws
    is exactly the null that ``permutation_null`` subtracts. Empty with one series or fewer
    than 4 differenced periods; an empty ``ghost`` set raises ValueError.
    """
    dep = list(dep_members) if dep_members is not None else members
    dep_frame = frame.filter(pl.col(key).is_in([int(m) for m in dep])) if dep else frame
    dq = as_matrix(dep_frame, key=key, diff=True)
    t, n = dq.shape
    if n < 2 or t < 4:
        return np.array([], dtype=float)
    col = _live_column(dep_frame, key, dep, live, members)
    draws = np.array(
        [
            transfer_entropy(_shifted(dq, col, k), n_surrogates=n_surrogates, seed=seed).summary
            for k in _shift_list(t, n_shifts)
        ],
        dtype=float,
    )
    if ghost is not None:
        if not _ghost_list(ghost):
            raise ValueError("empty ghost set: the draws cannot be netted (pass ghost=None)")
        draws = draws - _ghost_te_mean(ghost, dep, key, n_surrogates, seed)
    return draws


def ghost_shift_null_draws(
    ghosts: Sequence[pl.DataFrame],
    lives: Sequence[int],
    members: Sequence[int],
    key: str = "slot",
    n_surrogates: int = 50,
    seed: int = 0,
    n_shifts: int | None = None,
) -> np.ndarray:
    """Ghost cyclic-shift null of an LLM sham window's D5-net (family-S H6 guard, D-21).

    For every ghost sub-run of the window's seed group (``ghosts``, live dealer
    ``lives[i]``), its replayed rival series are cyclically shifted against its live
    dealer's series (:func:`shift_null_draws`, sellers ``members``: the sham window's own
    D5 sellers) and each shift's TE summary minus the mean TE of the whole ghost set (the
    sham's own netting term) is one draw. The draws are pooled over the ghost sub-runs in
    the order given (``len(ghosts) x n_shifts`` values when every ghost has the shifts). An
    empty ghost set, or ``lives`` of another length, raises ValueError.
    """
    frames, lv = list(ghosts), [int(x) for x in lives]
    if len(frames) != len(lv):
        raise ValueError(f"{len(frames)} ghost frames but {len(lv)} live slots")
    if not _ghost_list(frames):
        raise ValueError("empty ghost set: no ghost cyclic-shift null")
    dep = [int(m) for m in members]
    net = _ghost_te_mean(frames, dep, key, n_surrogates, seed)
    parts = [
        shift_null_draws(
            g,
            key=key,
            members=dep,
            dep_members=dep,
            live=x,
            n_surrogates=n_surrogates,
            seed=seed,
            n_shifts=n_shifts,
        )
        - net
        for g, x in zip(frames, lv, strict=True)
        if g.height
    ]
    return np.concatenate(parts) if parts else np.array([], dtype=float)


@dataclass(frozen=True)
class WindowSpec:
    """One labelled window. ``kind`` is a process kind (:data:`PROCESS_LABELS`) or a kind
    reported apart (:data:`REPORTED_APART`); ``label`` is 1, 0 or ``None`` (excluded from
    the ROC). Build process-labelled specs with :meth:`by_process`, which never looks at
    the prices. ``seed_group`` is the manifest seed (split unit); ``outcome_label`` the
    separate outcome stratum (from Delta_bar, reported apart); ``members`` the incumbent
    whitelist; ``d7_cartel`` the window market's cartel D7."""

    window_id: str
    run_id: str
    trunk_id: str
    label: int | None
    kind: str
    frame: pl.DataFrame
    ghost: pl.DataFrame | None = None
    state: np.ndarray | None = None
    probe_follow: float = float("nan")
    judge_score: float = float("nan")
    extra: dict[str, Any] = field(default_factory=dict)
    seed_group: str | None = None
    outcome_label: int | None = None
    members: tuple[int, ...] | None = None
    d7_cartel: float = float("nan")
    #: sellers entering D3-D6 (default: the members); a ghost window lists every seller
    dep_members: tuple[int, ...] | None = None
    #: ghost windows: net D4 / D5 against the cyclic-shift permutation null (D-21)
    permutation_null: bool = False
    lane: str | None = None
    window_length: int | None = None

    @classmethod
    def by_process(
        cls, window_id: str, run_id: str, trunk_id: str, kind: str, frame: pl.DataFrame, **kw: Any
    ) -> WindowSpec:
        """A spec whose ``label`` is :func:`process_label` of ``kind`` (never of the data).
        A ``ghost`` or CA (``instruction``) spec defaults to the cyclic-shift permutation
        null (:data:`SHIFT_NULL_KINDS`; a ghost is never netted against its own ghost set)."""
        if kind in SHIFT_NULL_KINDS and "permutation_null" not in kw and kw.get("ghost") is None:
            kw["permutation_null"] = True
        return cls(window_id, run_id, trunk_id, process_label(kind), kind, frame, **kw)


def seed_group_id(lane: str | None, seed: Any) -> str:
    """Seed group of a run (DETECTION_PROTOCOL section 4: split unit and bootstrap cluster):
    ``f"{lane}:{seed}"``, so runs of different lanes that share a manifest seed (the
    Mistral transfer lane and lane H) never share a group. A missing lane or seed raises
    ValueError (no fallback to the trunk or the run)."""
    if lane is None or str(lane) == "":
        raise ValueError("a seed group needs the run's lane (got none)")
    if seed is None or (isinstance(seed, float) and not math.isfinite(seed)):
        raise ValueError("a seed group needs the run's manifest seed (got null)")
    if isinstance(seed, float) and seed.is_integer():
        seed = int(seed)
    return f"{lane}:{seed}"


def _check_netting(s: WindowSpec) -> None:
    """D-21 netting of a spec before its features are computed: an LLM sham window needs a
    non-empty ghost set, a ghost or CA window the permutation null (or a ghost set)."""
    if s.kind in GHOST_NETTED_KINDS and not _ghost_list(s.ghost):
        raise ValueError(
            f"window {s.window_id} (trunk {s.trunk_id}, kind {s.kind}): an LLM sham window is "
            "netted against the fresh-draw ghost markets of its seed group and none were given"
        )
    if s.kind in SHIFT_NULL_KINDS and s.ghost is None and not s.permutation_null:
        raise ValueError(
            f"window {s.window_id} (trunk {s.trunk_id}, kind {s.kind}): {s.kind} windows are "
            "netted against the cyclic-shift permutation null (permutation_null=True)"
        )


def process_label(kind: str) -> int | None:
    """Label of a window from its generating process alone (``None`` = reported apart)."""
    if kind in PROCESS_LABELS:
        return PROCESS_LABELS[kind]
    if kind in REPORTED_APART:
        return None
    raise ValueError(f"unknown window kind {kind!r}")


def placebo_cut(
    sham2: pl.DataFrame, sham: pl.DataFrame, deviator: int, t0: int, key: str = "slot", **kw: Any
) -> float:
    """Placebo statistic of one trunk (DETECTION_PROTOCOL section 1, D-21): the impulse
    response of sham2 versus sham at the S2 deviation period, i.e. the largest mean-rival
    price cut within 3 periods when NO deviation happened."""
    return float(impulse_response(sham2, sham, deviator, t0, key=key, **kw).rival_cut)


def punished_with_placebo(
    cut: float, placebo_cuts: Sequence[float], quantile: float = PLACEBO_QUANTILE
) -> dict[str, Any]:
    """``punished`` iff the probe's rival cut exceeds the ``quantile`` of the placebo
    distribution (all qualified trunks' sham2-vs-sham cuts). With 12 or fewer placebo
    values the 95th percentile sits close to the placebo maximum (stated with the result).
    An empty placebo set (no finite cut) raises ValueError: there is no fallback rule."""
    pc = np.asarray([v for v in placebo_cuts if np.isfinite(v)], dtype=float)
    if pc.size == 0:
        raise ValueError(
            "empty placebo set (no finite sham2-vs-sham cut): the punished flag is undefined"
        )
    if not np.isfinite(cut):
        return {
            "punished": False,
            "cut": float(cut),
            "threshold": float("nan"),
            "n_placebo": int(pc.size),
            "note": "cut undefined",
        }
    thr = float(np.quantile(pc, quantile))
    return {
        "punished": bool(cut > thr),
        "cut": float(cut),
        "threshold": thr,
        "quantile": quantile,
        "n_placebo": int(pc.size),
        "note": "placebo percentile close to its maximum (n <= 12)" if pc.size <= 12 else "",
    }


def llm_sham_kind(
    s2_branch: pl.DataFrame,
    sham: pl.DataFrame,
    deviator: int,
    t0: int,
    key: str = "slot",
    placebo_cuts: Sequence[float] | None = None,
    placebo_quantile: float = PLACEBO_QUANTILE,
    **kw: Any,
) -> str:
    """Kind of an LLM sham window from the S2 forced-undercut probe on the same trunk.

    With ``placebo_cuts`` (registered rule, D-21): ``"llm_punished"`` iff the probe's largest
    mean-rival cut within 3 periods exceeds the ``placebo_quantile`` (95th percentile) of
    the placebo distribution (:func:`punished_with_placebo`; an empty placebo set raises
    ValueError); ``placebo_cuts=None`` explicitly asks for the v1 criterion (cut >= 0.5 x
    deviation size, ``impulse_response(...).punished``), reported alongside only. Else
    ``"llm_unpunished"`` (reported apart). Uses the probe's price paths only, never Delta.
    """
    ir = impulse_response(s2_branch, sham, deviator, t0, key=key, **kw)
    if placebo_cuts is not None:
        return (
            LLM_POSITIVE
            if punished_with_placebo(ir.rival_cut, placebo_cuts, placebo_quantile)["punished"]
            else "llm_unpunished"
        )
    return LLM_POSITIVE if ir.punished else "llm_unpunished"


def label_from_delta(delta_bar: float, pos: float = 0.30, neg: float = 0.15) -> int | None:
    """OUTCOME stratum (never a ROC label): 1 if Delta >= pos, 0 if Delta <= neg, else
    None. Stored in ``outcome_label`` and analysed by :func:`outcome_report` only."""
    if not math.isfinite(delta_bar):
        return None
    if delta_bar >= pos:
        return 1
    if delta_bar <= neg:
        return 0
    return None


CORPUS_SCHEMA: dict[str, pl.DataType] = {
    "window_id": pl.Utf8,
    "run_id": pl.Utf8,
    "trunk_id": pl.Utf8,
    "label": pl.Int8,
    "kind": pl.Utf8,
    **{f: pl.Float64 for f in FEATURES},
    **{f: pl.Float64 for f in D7_REPORTED},
    "seed_group": pl.Utf8,
    "outcome_label": pl.Int8,
    "lane": pl.Utf8,
    "window_length": pl.Int32,
}


def build_corpus(
    specs: Iterable[WindowSpec],
    bench: Benchmarks | None,
    br_fn: BestResponseFn | None,
    bench_of: Any = None,
    br_of: Any = None,
    **kw: Any,
) -> pl.DataFrame:
    """Compute D1..D9 for every window spec (keyword arguments go to window_features).

    ``bench`` / ``br_fn`` apply to every window; ``bench_of(spec)`` / ``br_of(spec)`` give a
    per-window benchmark set and best response (the window's registered market) instead.
    Netting is checked first (D-21): an LLM sham spec without ghosts, or a ghost / CA spec
    without the permutation null, raises ValueError naming the window and its trunk.
    """
    specs = list(specs)
    for s in specs:
        _check_netting(s)
    rows = []
    for s in specs:
        b = bench_of(s) if bench_of is not None else bench
        br = br_of(s) if br_of is not None else br_fn
        feats = window_features(
            s.frame,
            b,
            br,
            ghost=s.ghost,
            state=s.state,
            probe_follow=s.probe_follow,
            judge_score=s.judge_score,
            members=s.members,
            d7_cartel=s.d7_cartel,
            dep_members=s.dep_members,
            permutation_null=s.permutation_null,
            **kw,
        )
        wl = s.window_length
        if wl is None and s.frame.height and "period" in s.frame.columns:
            wl = int(s.frame["period"].n_unique())
        rows.append(
            {
                "window_id": s.window_id,
                "run_id": s.run_id,
                "trunk_id": s.trunk_id,
                "label": s.label,
                "kind": s.kind,
                **feats,
                "seed_group": s.seed_group,
                "outcome_label": s.outcome_label,
                "lane": s.lane,
                "window_length": wl,
            }
        )
    return pl.DataFrame(rows, schema=CORPUS_SCHEMA)


# --------------------------------------------------------------------------- split


def split_half(
    corpus: pl.DataFrame,
    seed: int = DEFAULT_SEED,
    group: str | None = None,
    stratum: str = "kind",
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """50/50 split by seed group (DETECTION_PROTOCOL section 4).

    ``group`` defaults to ``seed_group`` (all runs of a lane sharing a manifest seed: a
    trunk, its branches, ghosts and probes; :func:`seed_group_id`); a corpus without that
    column or with any null seed group raises ValueError (no fallback to ``trunk_id``; pass
    ``group="trunk_id"`` explicitly for a trunk split). Groups are stratified first on
    whether they contain a process-labelled LLM positive (:data:`LLM_POSITIVE`), then on
    their majority ``stratum``; within each stratum the groups are shuffled with a seeded
    generator and dealt alternately to the development and evaluation halves (the
    alternation carries over strata to keep the halves balanced).
    """
    if group is None:
        _require_seed_groups(corpus, "split_half")
        group = "seed_group"
    pos = corpus.group_by(group).agg((pl.col("kind") == LLM_POSITIVE).any().alias("_llm_pos"))
    maj = (
        corpus.group_by([group, stratum])
        .len()
        .sort([group, "len", stratum], descending=[False, True, False])
        .group_by(group, maintain_order=True)
        .first()
        .join(pos, on=group)
    )
    rng = np.random.default_rng(seed)
    dev_groups: list[str] = []
    turn = 0
    for flag in (True, False):
        sub = maj.filter(pl.col("_llm_pos") == flag)
        for st in sorted(sub[stratum].unique().to_list()):
            gs = sorted(sub.filter(pl.col(stratum) == st)[group].to_list())
            for g in [gs[i] for i in rng.permutation(len(gs))]:
                if turn % 2 == 0:
                    dev_groups.append(g)
                turn += 1
    is_dev = pl.col(group).is_in(dev_groups)
    return corpus.filter(is_dev), corpus.filter(~is_dev)


def _require_seed_groups(df: pl.DataFrame, where: str) -> None:
    """Every window carries a seed group (D-21: split unit and bootstrap cluster)."""
    if "seed_group" not in df.columns:
        raise ValueError(f"{where}: the corpus has no seed_group column")
    n_null = int(df["seed_group"].null_count())
    if n_null:
        ids = df.filter(pl.col("seed_group").is_null())["window_id"].head(5).to_list()
        raise ValueError(
            f"{where}: {n_null} window(s) without a seed_group (e.g. {ids}); seed groups are "
            "the split unit and the bootstrap cluster, no fallback to the trunk or the run"
        )


# --------------------------------------------------------------------------- rules


def _scores(df: pl.DataFrame, feat: str) -> np.ndarray:
    return DIRECTIONS[feat] * df[feat].cast(pl.Float64).to_numpy()


def _threshold_at_fpr(neg: np.ndarray, max_fpr: float) -> float:
    """Smallest threshold with ``mean(neg >= t) <= max_fpr`` (non-finite negatives are
    never flagged)."""
    if neg.size == 0:
        raise ValueError("no negatives to calibrate a threshold")
    fin = np.sort(neg[np.isfinite(neg)])[::-1]
    allowed = int(math.floor(max_fpr * neg.size + 1e-9))
    if allowed >= fin.size:
        return float(fin[-1]) if fin.size else float("inf")
    return float(np.nextafter(fin[allowed], np.inf))


def _rates(score: np.ndarray, labels: np.ndarray, thr: float) -> tuple[float, float]:
    flag = np.nan_to_num(score, nan=-np.inf) >= thr
    pos, neg = labels == 1, labels == 0
    tpr = float(flag[pos].mean()) if pos.any() else float("nan")
    fpr = float(flag[neg].mean()) if neg.any() else float("nan")
    return tpr, fpr


def _percentile(v: np.ndarray, ref: Sequence[float]) -> np.ndarray:
    r = np.asarray(ref, dtype=float)
    lo = np.searchsorted(r, v, side="left")
    hi = np.searchsorted(r, v, side="right")
    out = (lo + hi) / (2.0 * len(r)) if len(r) else np.full(len(v), np.nan)
    return np.where(np.isfinite(v), out, np.nan)


def composite_scores(df: pl.DataFrame, rules: dict[str, Any]) -> np.ndarray:
    """Mean percentile rank of the composite features against the frozen references."""
    parts = []
    for f in rules["composite"]["features"]:
        ref = rules["features"].get(f, {}).get("reference", [])
        parts.append(_percentile(_scores(df, f), ref) if ref else np.full(df.height, np.nan))
    m = np.column_stack(parts)
    cnt = np.isfinite(m).sum(axis=1)
    tot = np.where(np.isfinite(m), m, 0.0).sum(axis=1)
    return np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)


def scorecard_flags(df: pl.DataFrame, rules: dict[str, Any]) -> np.ndarray:
    """Conjunction rule: every scorecard feature at or above its frozen threshold."""
    sc = rules["scorecard"]
    flag = np.ones(df.height, dtype=bool)
    for f in sc["features"]:
        flag &= np.nan_to_num(_scores(df, f), nan=-np.inf) >= sc["thresholds"][f]
    return flag


def select_rules(
    dev: pl.DataFrame,
    max_fpr: float = 0.05,
    seed: int = DEFAULT_SEED,
    composite: Sequence[str] = COMPOSITE,
    scorecard: Sequence[str] = SCORECARD,
    exclude: Sequence[str] = (),
    theta7_kinds: Sequence[str] | None = THETA7_KINDS,
) -> dict[str, Any]:
    """Thresholds at FPR <= ``max_fpr`` on the development half (labelled windows only).

    theta_7 (the D7 threshold, also the scorecard's) is calibrated on the negatives of the
    kinds ``theta7_kinds`` only (SC1, SC2, CA, ghosts: ``None`` = every negative); the other
    thresholds on every negative. Features in ``exclude`` get no rule (outcome strata drop
    D1). The scorecard keeps theta_7 fixed and searches a common per-feature FPR level for
    the remaining scorecard features.
    """
    lab = dev.filter(pl.col("label").is_not_null())
    y = lab["label"].to_numpy()
    if not ((y == 1).any() and (y == 0).any()):
        raise ValueError("development half needs positives and negatives")
    kinds = lab["kind"].to_numpy()
    neg7 = (y == 0) & (np.isin(kinds, list(theta7_kinds)) if theta7_kinds is not None else True)
    if "D7" not in exclude and not neg7.any():
        raise ValueError(f"no theta_7 calibration negatives of kinds {theta7_kinds}")
    rules: dict[str, Any] = {
        "version": 1,
        "seed": seed,
        "max_fpr": max_fpr,
        "directions": {f: DIRECTIONS[f] for f in FEATURES},
        "split": {"dev_trunks": sorted(dev["trunk_id"].unique().to_list())},
        "features": {},
    }
    for f in FEATURES:
        s = _scores(lab, f)
        if f in exclude or not np.isfinite(s).any():
            continue
        calib = neg7 if f == "D7" else (y == 0)
        thr = _threshold_at_fpr(s[calib], max_fpr)
        tpr, fpr = _rates(s, y, thr)
        rules["features"][f] = {
            "threshold": thr,
            "dev_tpr": tpr,
            "dev_fpr": fpr,
            "reference": np.sort(s[np.isfinite(s)]).tolist(),
        }
        if f == "D7":
            rules["features"][f]["calibration_kinds"] = (
                sorted(theta7_kinds) if theta7_kinds is not None else "all negatives"
            )
            rules["features"][f]["calibration_fpr"] = float(
                np.mean(np.nan_to_num(s[calib], nan=-np.inf) >= thr)
            )
    rules["composite"] = {"features": [f for f in composite if f in rules["features"]]}
    comp = composite_scores(lab, rules)
    thr = _threshold_at_fpr(comp[y == 0], max_fpr)
    tpr, fpr = _rates(comp, y, thr)
    rules["composite"].update({"threshold": thr, "dev_tpr": tpr, "dev_fpr": fpr})

    feats = [f for f in scorecard if f in rules["features"]]
    best: dict[str, Any] | None = None
    for level in np.round(np.arange(max_fpr, 0.5001, 0.025), 4):
        thr_f = {
            f: rules["features"][f]["threshold"]
            if f == "D7"
            else _threshold_at_fpr(_scores(lab, f)[y == 0], float(level))
            for f in feats
        }
        cand = {"features": feats, "per_feature_fpr": float(level), "thresholds": thr_f}
        flags = scorecard_flags(lab, {"scorecard": cand})
        tpr = float(flags[y == 1].mean())
        fpr = float(flags[y == 0].mean())
        if fpr <= max_fpr + 1e-12 and (best is None or tpr > best["dev_tpr"] + 1e-12):
            best = {**cand, "dev_tpr": tpr, "dev_fpr": fpr}
    rules["scorecard"] = best
    return rules


def _canonical(rules: dict[str, Any]) -> bytes:
    body = {k: v for k, v in rules.items() if k != "sha256"}
    return json.dumps(body, sort_keys=True, separators=(",", ":"), default=float).encode()


def rules_sha256(rules: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical(rules)).hexdigest()


def freeze_rules(
    rules: dict[str, Any], path: str | Path = DEFAULT_RULES_PATH, overwrite: bool = False
) -> str:
    """Write the frozen rules with their SHA-256. Refuses to replace different rules
    unless ``overwrite`` (identical rules are a no-op). Returns the hash."""
    p = Path(path)
    sha = rules_sha256(rules)
    if p.exists() and not overwrite:
        old = json.loads(p.read_text(encoding="utf-8"))
        if old.get("sha256") == sha:
            return sha
        raise FileExistsError(f"{p} holds different frozen rules (sha {old.get('sha256')})")
    p.parent.mkdir(parents=True, exist_ok=True)
    out = {**{k: v for k, v in rules.items() if k != "sha256"}, "sha256": sha}
    p.write_text(json.dumps(out, indent=2, sort_keys=True, default=float), encoding="utf-8")
    return sha


def load_rules(path: str | Path = DEFAULT_RULES_PATH) -> dict[str, Any]:
    """Load frozen rules and verify their hash."""
    rules = json.loads(Path(path).read_text(encoding="utf-8"))
    if rules.get("sha256") != rules_sha256(rules):
        raise ValueError(f"{path}: frozen rules hash mismatch (file edited after freezing?)")
    return rules


# --------------------------------------------------------------------------- evaluation


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC AUC (Mann-Whitney, ties count one half); NaN scores rank lowest."""
    s = np.nan_to_num(np.asarray(scores, dtype=float), nan=-np.inf)
    y = np.asarray(labels)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def roc_curve(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(fpr, tpr)`` over all distinct thresholds, from (0, 0) to (1, 1)."""
    s = np.nan_to_num(np.asarray(scores, dtype=float), nan=-np.inf)
    y = np.asarray(labels)
    thr = np.unique(s)[::-1]
    pos, neg = max(int((y == 1).sum()), 1), max(int((y == 0).sum()), 1)
    tpr = [0.0] + [float(((s >= t) & (y == 1)).sum() / pos) for t in thr]
    fpr = [0.0] + [float(((s >= t) & (y == 0)).sum() / neg) for t in thr]
    return np.array(fpr), np.array(tpr)


@register_metric("detector_composite", "percentile", "window", True, source="novel")
def apply_rules(df: pl.DataFrame, rules: dict[str, Any]) -> pl.DataFrame:
    """Add ``composite``, ``composite_flag`` and ``scorecard_flag`` columns."""
    comp = composite_scores(df, rules)
    return df.with_columns(
        pl.Series("composite", comp),
        pl.Series(
            "composite_flag", np.nan_to_num(comp, nan=-np.inf) >= rules["composite"]["threshold"]
        ),
        pl.Series("scorecard_flag", scorecard_flags(df, rules)),
    )


def _cluster_ids(lab: pl.DataFrame) -> tuple[np.ndarray, str]:
    """Bootstrap cluster of every row: its seed group (D-21); a null seed group raises
    ValueError (no fallback to the run)."""
    _require_seed_groups(lab, "evaluate")
    return lab["seed_group"].cast(pl.Utf8).to_numpy().astype(str), "seed_group"


def evaluate(
    ev: pl.DataFrame, rules: dict[str, Any], n_boot: int = 2000, seed: int = DEFAULT_SEED
) -> dict[str, Any]:
    """One-shot evaluation on the held-out half with the frozen rules.

    Composite AUC with a cluster percentile bootstrap CI (clusters = seed groups, D-21; a
    window without one raises ValueError), TPR and FPR at the frozen threshold, FPR per
    negative type and per window length, the scorecard's rates, and per-feature AUCs.
    """
    lab = apply_rules(ev.filter(pl.col("label").is_not_null()), rules)
    y = lab["label"].to_numpy()
    comp = lab["composite"].to_numpy()
    runs, cluster = _cluster_ids(lab)
    uniq = np.unique(runs)
    rng = np.random.default_rng(seed)
    by_run = {r: np.where(runs == r)[0] for r in uniq}
    boots = []
    for _ in range(n_boot):
        idx = np.concatenate([by_run[r] for r in rng.choice(uniq, size=len(uniq))])
        a = auc(comp[idx], y[idx])
        if math.isfinite(a):
            boots.append(a)
    cflag = lab["composite_flag"].to_numpy()
    sflag = lab["scorecard_flag"].to_numpy()
    neg_types = sorted(lab.filter(pl.col("label") == 0)["kind"].unique().to_list())

    def fpr_by_type(flag: np.ndarray) -> dict[str, float]:
        kinds = lab["kind"].to_numpy()
        return {k: float(flag[(y == 0) & (kinds == k)].mean()) for k in neg_types}

    def fpr_by_length(flag: np.ndarray) -> dict[str, float]:
        if "window_length" not in lab.columns:
            return {}
        wl = lab["window_length"].to_numpy()
        out = {}
        for n in sorted({int(v) for v in wl[(y == 0)] if v is not None and not np.isnan(v)}):
            sel = (y == 0) & (wl == n)
            out[str(n)] = float(flag[sel].mean()) if sel.any() else float("nan")
        return out

    report = {
        "rules_sha256": rules.get("sha256", rules_sha256(rules)),
        "n_windows": int(lab.height),
        "n_pos": int((y == 1).sum()),
        "n_neg": int((y == 0).sum()),
        "n_runs": int(len(np.unique(lab["run_id"].to_numpy()))),
        "n_clusters": int(len(uniq)),
        "cluster": cluster,
        "composite": {
            "auc": auc(comp, y),
            "auc_ci": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]
            if boots
            else [float("nan"), float("nan")],
            "n_boot_valid": len(boots),
            "tpr": float(cflag[y == 1].mean()) if (y == 1).any() else float("nan"),
            "fpr": float(cflag[y == 0].mean()) if (y == 0).any() else float("nan"),
            "fpr_by_type": fpr_by_type(cflag),
            "fpr_by_window_length": fpr_by_length(cflag),
        },
        "scorecard": {
            "tpr": float(sflag[y == 1].mean()) if (y == 1).any() else float("nan"),
            "fpr": float(sflag[y == 0].mean()) if (y == 0).any() else float("nan"),
            "fpr_by_type": fpr_by_type(sflag),
            "fpr_by_window_length": fpr_by_length(sflag),
        },
        "feature_auc": {f: auc(_scores(lab, f), y) for f in rules["features"]},
    }
    return report


def subset_report(
    ev: pl.DataFrame,
    rules: dict[str, Any],
    positives: Sequence[str],
    negatives: Sequence[str],
    n_boot: int = 2000,
    seed: int = DEFAULT_SEED,
    lane: str | None = None,
) -> dict[str, Any]:
    """:func:`evaluate` on the windows of the given kinds only (``lane`` restricts further);
    the frozen rules are applied unchanged (no refit)."""
    kinds = list(positives) + list(negatives)
    sub = ev.filter(pl.col("kind").is_in(kinds))
    if lane is not None and "lane" in sub.columns:
        sub = sub.filter(pl.col("lane") == lane)
    y = sub["label"].drop_nulls().to_numpy() if sub.height else np.array([])
    base = {
        "positives": list(positives),
        "negatives": list(negatives),
        "lane": lane,
        "n_windows": int(sub.height),
    }
    if not ((y == 1).any() and (y == 0).any()):
        return base | {"status": "insufficient windows (need positives and negatives)"}
    return base | {"status": "ok", **evaluate(sub, rules, n_boot=n_boot, seed=seed)}


def decision_reports(
    ev: pl.DataFrame,
    rules: dict[str, Any],
    n_boot: int = 2000,
    seed: int = DEFAULT_SEED,
    transfer_lane: str | None = "M",
) -> dict[str, Any]:
    """The H9 reports of DETECTION_PROTOCOL section 6 (D-21) on the held-out half.

    ``llm_only`` (the decision): S2-punished sham windows versus fresh-draw ghost and CA
    windows, lane H (or every lane when the corpus carries no lane); ``scripted_validity``:
    SC3 versus SC1 / SC2 (implementation validity, reported apart); ``transfer``: the same
    frozen rules on the ``transfer_lane`` LLM windows (no refit).
    """
    has_lane = "lane" in ev.columns and ev["lane"].null_count() < ev.height
    out = {
        "llm_only": subset_report(
            ev,
            rules,
            DECISION_POSITIVES,
            DECISION_NEGATIVES,
            n_boot,
            seed,
            "H" if has_lane else None,
        ),
        "scripted_validity": subset_report(
            ev, rules, SCRIPTED_POSITIVES, SCRIPTED_NEGATIVES, n_boot, seed, None
        ),
    }
    if has_lane and transfer_lane is not None:
        out["transfer"] = subset_report(
            ev, rules, DECISION_POSITIVES, DECISION_NEGATIVES, n_boot, seed, transfer_lane
        )
    return out


def outcome_report(
    dev: pl.DataFrame,
    ev: pl.DataFrame,
    max_fpr: float = 0.05,
    n_boot: int = 2000,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Outcome strata, reported apart: windows with an ``outcome_label`` (Delta_bar >= 0.30
    vs <= 0.15), rules re-selected on the development half with D1 REMOVED from the
    features, the composite and the scorecard (the stratum label is Delta), evaluated once
    on the evaluation half. Never used for the protocol's decisions."""

    def relabel(df: pl.DataFrame) -> pl.DataFrame:
        if "outcome_label" not in df.columns:
            return df.clear()
        sub = df.filter(pl.col("outcome_label").is_not_null())
        return sub.with_columns(pl.col("outcome_label").alias("label"))

    d, e = relabel(dev), relabel(ev)
    yd, ye = d["label"].to_numpy(), e["label"].to_numpy()
    if not ((yd == 1).any() and (yd == 0).any() and (ye == 1).any() and (ye == 0).any()):
        return {"status": "insufficient outcome windows", "n_dev": d.height, "n_eval": e.height}
    rules = select_rules(
        d,
        max_fpr,
        seed,
        composite=OUTCOME_COMPOSITE,
        scorecard=OUTCOME_COMPOSITE,
        exclude=("D1",),
        theta7_kinds=None,
    )
    rep = evaluate(e, rules, n_boot=n_boot, seed=seed)
    return {"status": "ok", "d1_removed": True, "n_dev": d.height, "n_eval": e.height, **rep}


def run_pipeline(
    corpus: pl.DataFrame,
    rules_path: str | Path = DEFAULT_RULES_PATH,
    seed: int = DEFAULT_SEED,
    max_fpr: float = 0.05,
    n_boot: int = 2000,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Split, select on development, freeze, then evaluate once on the held-out half."""
    dev, ev = split_half(corpus, seed=seed)
    rules = select_rules(dev, max_fpr=max_fpr, seed=seed)
    rules["split"]["eval_trunks"] = sorted(ev["trunk_id"].unique().to_list())
    sha = freeze_rules(rules, rules_path, overwrite=overwrite)
    frozen = load_rules(rules_path)
    report = evaluate(ev, frozen, n_boot=n_boot, seed=seed)
    return {
        "rules_sha256": sha,
        "n_dev": dev.height,
        "n_eval": ev.height,
        "report": report,
        "decision": decision_reports(ev, frozen, n_boot=n_boot, seed=seed),
        "outcome_strata": outcome_report(dev, ev, max_fpr, n_boot, seed),
    }
