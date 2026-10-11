"""Registered arms of the analysis: roles, incumbent slot whitelists, pinned benchmarks.

``configs/analysis/registered_arms.yaml`` is the single source of truth the campaign
analysis reads (frozen with the pre-registration). Only runs that match a registered arm are
analysed; every arm declares the incumbent LLM dealer slots it averages over (entrants added
by ``add_scripted_agent``, sting dealers, replay rivals and the fringe are never members) and
pins its benchmarks to a ``benchmarks.json`` file by path and SHA-256.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from finswarm.metrics.benchmarks import Benchmarks, MarketModel

DEFAULT_ARMS_PATH = Path("configs/analysis/registered_arms.yaml")
#: primary_lever = the H1-R family (lane H); control = trunks, shams, ghosts, scripted and
#: instruction negatives; probe = stings and forced deviations; secondary = family S, H1-M and
#: H-S contrasts; instrument = Gate 1 / 1b / AC-H runs analysed by the gates with their own
#: closed-form benchmarks; exploratory = reported, never confirmatory
Role = Literal["primary_lever", "control", "probe", "secondary", "instrument", "exploratory"]
Lane = Literal["H", "M", "S", "L", "CPU"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MarketSource(_Strict):
    """Where a window market comes from: a campaign config, optionally one of its branches
    (its ``set_param market.*`` events and ``add_scripted_agent`` entrants are applied) and
    grid overrides (``{"market.routing.tau": 0.125}``)."""

    config: str
    branch: str | None = None
    grid: dict[str, Any] = Field(default_factory=dict)
    n_incumbents: int | None = None
    method: Literal["auto", "closed_form", "sweep"] = "auto"


class BenchmarkFile(_Strict):
    file: str
    sha256: str | None = None
    markets: dict[str, MarketSource]


class BenchRef(_Strict):
    """``pin``: key of ``benchmark_files``; ``primary``: market of the pre-lever (trunk)
    benchmarks used by the primary estimand; ``window``: market of the arm's own window
    (per-window secondary estimand and mechanical decomposition)."""

    pin: str
    primary: str
    window: str


class MatchSpec(_Strict):
    """A run belongs to the arm when its manifest ``arm`` (and ``label``, when given) match,
    it is (``ghost: true``) or is not (default) a ghost sub-run, and every dotted ``config``
    path of its resolved config equals the given value."""

    arm: str
    label: str | None = None
    ghost: bool = False
    config: dict[str, Any] = Field(default_factory=dict)


class ProbeSpec(_Strict):
    kind: Literal["forced_undercut", "sting"]
    slot: int


class LeverSpec(_Strict):
    """The registered lever: which campaign config and branch implement it and the exact
    parameter values its events must carry (dotted ``set_param`` paths, or
    ``add_scripted_agent`` -> policy name). ``scripts/check_prereg.py`` verifies them."""

    config: str
    branch: str
    params: dict[str, Any]


class ArmSpec(_Strict):
    """One registered arm. ``window`` names an entry of ``windows``; ``benchmarks`` pins the
    markets (required unless ``benchmarks_mode = run_closed_form``: Gate 1 / 1b / AC-H
    instrument runs are scored against the closed form of their own resolved market, alpha
    included). ``fresh_draw_required`` enforces the registered cache-miss rule (ghosts and
    sham2: any post-branch cache hit excludes the run). ``config`` is the campaign file as
    the engine names it (a missing file is reported at freeze, not fatal)."""

    role: Role
    match: MatchSpec
    members: list[int] | Literal["ghost_slot"]
    window: str
    benchmarks: BenchRef | None = None
    benchmarks_mode: Literal["pinned", "run_closed_form"] = "pinned"
    lane: Lane = "H"
    paired_with: str | None = None
    hypothesis: str | None = None
    mqc_label: str | None = None
    probe: ProbeSpec | None = None
    lever: LeverSpec | None = None
    fresh_draw_required: bool = False
    config: str | None = None
    description: str = ""


class LaneSpec(_Strict):
    """Per-lane roles: the trunk arm whose runs qualify at Gate 2, its sham (zero point of the
    lane's contrasts), the lane's pre-specified lever and the hypothesis it carries (H1 on
    H, H1-M on M, H-S on S)."""

    trunk: str
    sham: str
    primary_lever: str | None = None
    hypothesis: str | None = None
    model: str = ""
    description: str = ""


class Gate2Spec(_Strict):
    """Gate 2 outcome table (PREREG section 1, D-25) as data the code reads.

    ``n >= branch_all_min``: branch every qualified trunk; ``reduced_power_min <= n <
    branch_all_min``: branch, reduced power declared, split-sample estimate off, fewer H8 /
    H9 positives; ``n < reduced_power_min``: the fallback (F1 then F2), H1 is not tested on
    these trunks. The split estimate also needs ``split_estimate_min_per_half`` trunks per
    seed-parity half (PREREG section 2).
    """

    branch_all_min: int = 8
    reduced_power_min: int = 5
    fallback: str = "F1 (s0.5, relaunch 12 trunks) then F2 (2 dealers)"
    split_estimate_min_qualified: int = 8
    split_estimate_min_per_half: int = 3

    def outcome(self, n_qualified: int) -> dict[str, Any]:
        if n_qualified >= self.branch_all_min:
            return {
                "n_qualified": int(n_qualified),
                "status": "PASS",
                "decision": "branch every qualified trunk",
                "h1_tested": True,
                "reduced_power": False,
                "split_estimate": "on",
            }
        if n_qualified >= self.reduced_power_min:
            return {
                "n_qualified": int(n_qualified),
                "status": "PASS",
                "decision": "branch every qualified trunk; reduced power declared; "
                "split-sample best-lever estimate off; S1/S2 on all qualified trunks, so H8 "
                "has fewer collusive markets and H9 at most that many LLM positives (D-25)",
                "h1_tested": True,
                "reduced_power": True,
                "split_estimate": "off",
            }
        return {
            "n_qualified": int(n_qualified),
            "status": "FAIL",
            "decision": self.fallback,
            "h1_tested": False,
            "reduced_power": True,
            "split_estimate": "off",
        }


class TrendRuleSpec(_Strict):
    """Trend rule (PREREG section 1, D-22): evaluated on every trunk before qualification;
    an OLS slope of the per-period cross-trunk mean ``Delta_p`` over ``window`` above
    ``slope_max`` per period shifts every window by ``shift`` periods (T0 100 -> 120)."""

    window: tuple[int, int] = (81, 100)
    slope_max: float = 0.005
    shift: int = 20


class H1Spec(_Strict):
    """Primary H1 (D-3'): one pre-specified lever, one-sided paired test, no multiplicity.
    Supported iff ``p < alpha`` and ``MQC <= mqc_max_pp``; practically significant iff also
    the one-sided 95 % lower bound exceeds ``practical_lower_bound_gt``."""

    mqc_max_pp: float = 5.0
    practical_lower_bound_gt: float = 0.10


class QualificationSpec(_Strict):
    """Gate 2 ("collusion established", D-16): a trunk is qualified when the ``arm`` run's
    ``measure`` over its window is at least ``threshold`` AND, when a ``co_measure`` is
    given, that measure is at least ``co_threshold`` (registered: ``delta_p >= 0.30`` and
    ``delta_pi >= 0.30`` on periods 71-100). H1, H1-R, H7, H8 and H10 use qualified trunks
    only; the same rule classifies the sting markets."""

    arm: str = "trunk"
    measure: str = "delta_p"
    threshold: float = 0.30
    co_measure: str | None = "delta_pi"
    co_threshold: float | None = 0.30

    def measures(self) -> dict[str, float]:
        """``{measure: threshold}`` of every condition (one or two)."""
        out = {self.measure: self.threshold}
        if self.co_measure is not None:
            out[self.co_measure] = float(
                self.co_threshold if self.co_threshold is not None else self.threshold
            )
        return out

    def passes(self, row: dict[str, Any]) -> bool:
        """True when every condition holds on the run row (NaN or missing = fails)."""
        for m, thr in self.measures().items():
            v = row.get(m)
            if v is None or not math.isfinite(float(v)) or float(v) < thr:
                return False
        return True

    def describe(self, window: tuple[int, int] | None = None) -> str:
        w = f"({window[0]}-{window[1]})" if window else ""
        return " AND ".join(f"{m}{w} >= {thr:g}" for m, thr in self.measures().items())


class InferenceSpec(_Strict):
    sidedness: Literal["greater"] = "greater"
    alpha: float = 0.05
    h1: H1Spec = Field(default_factory=H1Spec)
    qualification: QualificationSpec = Field(default_factory=QualificationSpec)
    split_selection: Literal["odd"] = "odd"
    split_estimation: Literal["even"] = "even"
    tost_margin: float = 0.20
    gate2: Gate2Spec = Field(default_factory=Gate2Spec)
    trend_rule: TrendRuleSpec = Field(default_factory=TrendRuleSpec)
    B: int = 10_000
    seed: int = 20261010


class MQCSpec(_Strict):
    reference_arm: str = "sham"
    max_pp: float = 5.0
    n_seeds: int = 20
    window: Literal["branch", "trunk"] = "branch"


class RegisteredArms(_Strict):
    version: int
    prereg_id: str
    windows: dict[str, tuple[int, int]]
    reference_arm: str = "sham"
    primary_lever: str
    lever_family: list[str]
    lanes: dict[str, LaneSpec] = Field(default_factory=dict)
    inference: InferenceSpec = Field(default_factory=InferenceSpec)
    max_fail_rate: float = 0.10
    mqc: MQCSpec = Field(default_factory=MQCSpec)
    benchmark_files: dict[str, BenchmarkFile]
    arms: dict[str, ArmSpec]

    @model_validator(mode="after")
    def _consistent(self) -> RegisteredArms:
        if len(self.lever_family) != 8 or len(set(self.lever_family)) != 8:
            raise ValueError("lever_family must list the 8 registered levers exactly once")
        if self.primary_lever not in self.lever_family:
            raise ValueError("primary_lever must be one of lever_family")
        q = self.inference.qualification
        if q.arm not in self.arms:
            raise ValueError(f"qualification arm {q.arm!r} is not registered")
        for k in ("branch", "trunk"):
            if k not in self.windows:
                raise ValueError(f"windows.{k} missing")
        if not self.lanes:  # single-lane registry: lane H from the top-level fields
            self.lanes = {
                "H": LaneSpec(
                    trunk=q.arm,
                    sham=self.reference_arm,
                    primary_lever=self.primary_lever,
                    hypothesis="H1",
                )
            }
        for lane, ls in self.lanes.items():
            for role, name in (("trunk", ls.trunk), ("sham", ls.sham)):
                if name not in self.arms:
                    raise ValueError(f"lane {lane!r}: {role} arm {name!r} is not registered")
            if ls.primary_lever is not None and ls.primary_lever not in self.arms:
                raise ValueError(f"lane {lane!r}: lever {ls.primary_lever!r} is not registered")
        for lev in self.lever_family:
            arm = self.arms.get(lev)
            if arm is None or arm.role != "primary_lever":
                raise ValueError(f"lever {lev!r} must be a registered primary_lever arm")
        for name, a in self.arms.items():
            if a.role == "primary_lever" and name not in self.lever_family:
                raise ValueError(f"primary_lever {name!r} is not in lever_family")
            if a.window not in self.windows:
                raise ValueError(f"arm {name!r}: unknown window {a.window!r}")
            if a.benchmarks is None:
                if a.benchmarks_mode != "run_closed_form":
                    raise ValueError(f"arm {name!r}: pinned benchmarks required")
            else:
                pin = self.benchmark_files.get(a.benchmarks.pin)
                if pin is None:
                    raise ValueError(f"arm {name!r}: unknown benchmark pin {a.benchmarks.pin!r}")
                for mk in (a.benchmarks.primary, a.benchmarks.window):
                    if mk not in pin.markets:
                        raise ValueError(
                            f"arm {name!r}: market {mk!r} not in pin {a.benchmarks.pin}"
                        )
            if a.paired_with is not None and a.paired_with not in self.arms:
                raise ValueError(f"arm {name!r}: paired_with {a.paired_with!r} not registered")
            if isinstance(a.members, list) and not a.members:
                raise ValueError(f"arm {name!r}: empty member whitelist")
        if self.reference_arm not in self.arms:
            raise ValueError("reference_arm is not a registered arm")
        return self

    def window_of(self, arm: str) -> tuple[int, int]:
        return tuple(self.windows[self.arms[arm].window])  # type: ignore[return-value]


def load_registered_arms(path: str | Path = DEFAULT_ARMS_PATH) -> RegisteredArms:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return RegisteredArms.model_validate(data)


# --------------------------------------------------------------------------- lever check

_IGNORED_PATHS = ("periods",)


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b or a == b
    if isinstance(a, int | float) and isinstance(b, int | float):
        return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-12)
    return a == b


def check_lever_params(
    repo_root: str | Path, registered: dict[str, Any] | str | Path | RegisteredArms
) -> list[str]:
    """Problems between registered lever values and the campaign configs' branch events.

    Stable entry point for ``scripts/check_prereg.py`` (infra): ``repo_root`` is the
    repository root the lever configs are resolved against; ``registered`` is the parsed
    ``registered_arms.yaml`` (dict or model) or its path (relative paths are resolved
    against ``repo_root``). Returns the mismatch messages (empty when consistent).

    For every arm with a ``lever`` block, the branch's ``set_param`` events (except
    ``periods`` and ``tags.*``) and ``add_scripted_agent`` policies must equal the registered
    ``params`` exactly: a missing, different or unregistered extra change is reported.
    An empty list means consistent.
    """
    from finswarm.config.load import load_campaign_config

    root = Path(repo_root)
    if isinstance(registered, RegisteredArms):
        spec = registered
    elif isinstance(registered, dict):
        spec = RegisteredArms.model_validate(registered)
    else:
        rp = Path(registered)
        spec = load_registered_arms(rp if rp.is_absolute() else root / rp)
    problems = []
    for name, a in spec.arms.items():
        lv = a.lever
        if lv is None:
            continue
        path = root / lv.config
        if not path.exists():
            problems.append(f"{name}: config {lv.config} missing")
            continue
        try:
            cc = load_campaign_config(path)
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            problems.append(f"{name}: {lv.config} does not validate ({exc})")
            continue
        br = next((b for b in cc.branches if b.name == lv.branch), None)
        if br is None:
            problems.append(f"{name}: {lv.config} has no branch {lv.branch!r}")
            continue
        seen: dict[str, Any] = {}
        for ev in br.events:
            kind = getattr(ev, "type", "")
            if kind == "set_param":
                if ev.path in _IGNORED_PATHS or ev.path.startswith("tags."):
                    continue
                seen[ev.path] = ev.value
            elif kind == "add_scripted_agent":
                seen["add_scripted_agent"] = ev.policy
        for k, v in lv.params.items():
            if k not in seen:
                problems.append(f"{name}: {lv.config}#{lv.branch} lacks {k} = {v!r}")
            elif not _same(seen[k], v):
                problems.append(
                    f"{name}: {lv.config}#{lv.branch} sets {k} = {seen[k]!r}, registered {v!r}"
                )
        for k in sorted(set(seen) - set(lv.params)):
            problems.append(
                f"{name}: {lv.config}#{lv.branch} changes {k} = {seen[k]!r}, not registered"
            )
    return problems


# --------------------------------------------------------------------------- pinning


class PinError(RuntimeError):
    """A pinned benchmarks file is missing, unpinned or does not match its SHA-256."""


def file_sha256(path: str | Path) -> str:
    """SHA-256 of a text file with line endings normalised to LF (Windows-safe)."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def load_pinned(spec: RegisteredArms, pin: str, root: str | Path = ".") -> dict[str, Any]:
    """Read a pinned benchmarks file after checking its SHA-256 against the registry."""
    bf = spec.benchmark_files[pin]
    p = Path(root) / bf.file
    if not p.exists():
        raise PinError(f"benchmarks pin {pin!r}: {bf.file} missing (run analyze --make-benchmarks)")
    got = file_sha256(p)
    if not bf.sha256:
        raise PinError(f"benchmarks pin {pin!r}: no sha256 recorded (file hash {got})")
    if got != bf.sha256.lower():
        raise PinError(
            f"benchmarks pin {pin!r}: {bf.file} sha256 {got} != pinned {bf.sha256} "
            "(file edited or regenerated without re-pinning)"
        )
    return json.loads(p.read_text(encoding="utf-8"))


class ArmBenchmarks:
    """Benchmarks of one arm resolved from its pinned file."""

    def __init__(self, data: dict[str, Any], ref: BenchRef, pin_sha: str, path: str) -> None:
        self.primary = benchmarks_from_entry(data["markets"][ref.primary])
        self.window = benchmarks_from_entry(data["markets"][ref.window])
        self.primary_model = MarketModel.from_dict(data["markets"][ref.primary]["model"])
        self.window_model = MarketModel.from_dict(data["markets"][ref.window]["model"])
        self.ref = ref
        self.pin_sha = pin_sha
        self.path = path

    @classmethod
    def from_model(cls, model: MarketModel, label: str = "run_closed_form") -> ArmBenchmarks:
        """Benchmarks of an instrument run solved from its own resolved market (closed form,
        alpha included): primary = window = the run's market (Gate 1 / 1b / AC-H)."""
        obj = cls.__new__(cls)
        b = model.benchmarks()
        obj.primary = obj.window = b
        obj.primary_model = obj.window_model = model
        obj.ref = BenchRef(pin=label, primary=label, window=label)
        obj.pin_sha = ""
        obj.path = label
        return obj

    def to_dict(self) -> dict[str, Any]:
        return {
            "file": self.path,
            "sha256": self.pin_sha,
            "primary_market": self.ref.primary,
            "window_market": self.ref.window,
            "primary": {k: getattr(self.primary, k) for k in BENCH_KEYS},
            "window": {k: getattr(self.window, k) for k in BENCH_KEYS},
            "window_method": self.window.method,
        }


BENCH_KEYS = ("p_nash", "p_mono", "pi_nash", "pi_mono")


def benchmarks_from_entry(entry: dict[str, Any]) -> Benchmarks:
    b = entry["benchmarks"]
    vals = {k: float(b[k]) for k in BENCH_KEYS}
    if not all(math.isfinite(v) for v in vals.values()):
        raise PinError("benchmarks entry with non-finite values")
    return Benchmarks(
        **vals, method=str(entry.get("method", "closed_form")), params={"market": entry["model"]}
    )


def resolve_arm_benchmarks(
    spec: RegisteredArms, root: str | Path = "."
) -> dict[str, ArmBenchmarks]:
    """Per-arm benchmarks from the pinned files (each file read and verified once)."""
    cache: dict[str, dict[str, Any]] = {}
    out = {}
    for name, a in spec.arms.items():
        if a.benchmarks is None:  # instrument arms: solved per run from the resolved market
            continue
        pin = a.benchmarks.pin
        if pin not in cache:
            cache[pin] = load_pinned(spec, pin, root)
        bf = spec.benchmark_files[pin]
        out[name] = ArmBenchmarks(cache[pin], a.benchmarks, str(bf.sha256), bf.file)
    return out
