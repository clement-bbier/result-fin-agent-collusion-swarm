"""Resolve a registered template config into a runnable config (D-26).

Some campaign files are templates whose content is fixed by a registered RULE applied after a
gate, never by hand: the seeds of the sting probes (``S1-H``, ``S2-H``: the first 6 qualified
TR-H trunks in seed order; ``S1c-LLM-TR-H``: the non-qualified trunks, at most 3) and the lever
branches of the LT campaigns (``LT-M``: L2 plus the best eligible Tier 0 lever other than L2;
``LT-S``: L2 plus the two best). This script applies the rule and writes
``<template stem>.resolved.yaml`` next to the template, with a provenance header (template
SHA-256, arguments, UTC time) and the resolution line that ``lead`` copies into
``docs/03_preregistration/AMENDMENTS.md``::

    uv run python scripts/resolve_registered.py configs/sprint/tier1/S1-H.yaml --qualified 0 2 3 5 7 8 11
    uv run python scripts/resolve_registered.py configs/sprint/tier1/S1c-LLM-TR-H.yaml --qualified 0 2 3 5 7 8 11
    uv run python scripts/resolve_registered.py configs/sprint/tier1/LT-M.yaml --lever L3
    uv run python scripts/resolve_registered.py configs/sprint/tier1/LT-S.yaml --lever L3 --lever L4a

The resolved config is validated with the schema before it is written. Lever events are copied
from the registered Tier 0 lever file (``configs/sprint/tier0/<L>-H.yaml``), so a resolved LT
branch is the H lever verbatim apart from ``tags.arm`` (= the lever id) and ``tags.label``.
Exit codes: 0 written, 2 nothing to run (no non-qualified trunk), 1 error.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
TIER0 = ROOT / "configs" / "sprint" / "tier0"
TRUNK_H = TIER0 / "TR-H.yaml"
#: Tier 0 levers a rule may select (L2 is the fixed first LT lever, never the rule's choice).
ELIGIBLE_LEVERS = ("L1", "L3", "L4a", "L6")
#: branch events the template keeps for itself when a lever's events are copied in
TEMPLATE_OWNED = ("periods", "tags.arm", "tags.label")


@dataclass(frozen=True)
class SeedRule:
    first_n: int
    qualified: bool  # True: the first qualified seeds; False: the first non-qualified ones
    universe: Path = TRUNK_H  # trunk seeds the complement is taken from


@dataclass(frozen=True)
class LeverRule:
    branches: tuple[str, ...]  # branch names filled by the rule, in order of the --lever values


RULES: dict[str, SeedRule | LeverRule] = {
    "S1-H": SeedRule(first_n=6, qualified=True),
    "S2-H": SeedRule(first_n=6, qualified=True),
    "S1c-LLM-TR-H": SeedRule(first_n=3, qualified=False),
    # LT-M carries L2 only since A-005 (Llama second family, lane budget 10 USD): no rule lever
    "LT-S": LeverRule(branches=("LT_2", "LT_3")),
}


class NothingToRun(Exception):
    """The rule selects no run (e.g. every trunk qualified): no file is written."""


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: not a mapping")
    return data


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_seeds(rule: SeedRule, qualified: list[int]) -> list[int]:
    q = sorted(set(qualified))
    if rule.qualified:
        seeds = q[: rule.first_n]
    else:
        universe = [int(s) for s in _load_yaml(rule.universe)["seeds"]]
        seeds = [s for s in universe if s not in set(q)][: rule.first_n]
    if not seeds:
        raise NothingToRun("the rule selects no trunk")
    return seeds


def lever_events(lever: str) -> list[dict[str, Any]]:
    """The registered events of Tier 0 lever ``lever`` minus the template-owned set_params."""
    if lever not in ELIGIBLE_LEVERS:
        raise ValueError(f"lever {lever!r} is not eligible; expected one of {ELIGIBLE_LEVERS}")
    src = TIER0 / f"{lever}-H.yaml"
    branches = [b for b in _load_yaml(src).get("branches", []) if b.get("name") == lever]
    if len(branches) != 1:
        raise ValueError(f"{src}: expected one branch named {lever!r}")
    return [
        e
        for e in branches[0]["events"]
        if not (e.get("type") == "set_param" and e.get("path") in TEMPLATE_OWNED)
    ]


def resolve_levers(data: dict[str, Any], rule: LeverRule, levers: list[str]) -> list[str]:
    if len(levers) != len(rule.branches):
        raise ValueError(f"expected {len(rule.branches)} --lever value(s) for {rule.branches}")
    if len(set(levers)) != len(levers) or "L2" in levers:
        raise ValueError("levers must be distinct and other than L2 (L2 is the fixed first lever)")
    by_name = {b["name"]: b for b in data.get("branches", [])}
    done = []
    for name, lever in zip(rule.branches, levers, strict=True):
        if name not in by_name:
            raise ValueError(f"template has no branch {name!r}")
        branch = by_name[name]
        owned = [
            e
            for e in branch["events"]
            if e.get("type") == "set_param" and e.get("path") in TEMPLATE_OWNED
        ]
        for e in owned:
            if e["path"] == "tags.arm":
                e["value"] = lever
        branch["events"] = [*owned, *lever_events(lever)]
        done.append(f"{name} = {lever}")
    return done


def resolve(
    template: Path, *, qualified: list[int] | None, levers: list[str] | None
) -> tuple[dict[str, Any], str]:
    """Apply the template's rule; return the resolved mapping and the resolution text."""
    rule = RULES.get(template.stem)
    if rule is None:
        raise ValueError(f"{template.name} is not a registered template; known: {sorted(RULES)}")
    data = _load_yaml(template)
    if isinstance(rule, SeedRule):
        if qualified is None:
            raise ValueError(f"{template.name} needs --qualified <seeds>")
        if levers:
            raise ValueError(f"{template.name} takes no --lever")
        seeds = resolve_seeds(rule, qualified)
        data["seeds"] = seeds
        what = "qualified" if rule.qualified else "non-qualified"
        text = f"seeds {seeds} ({what}, first {rule.first_n} in seed order; qualified {sorted(set(qualified))})"
    else:
        if not levers:
            raise ValueError(f"{template.name} needs --lever <L> (x{len(rule.branches)})")
        if qualified is not None:
            raise ValueError(f"{template.name} takes no --qualified (its seeds are its trunks')")
        text = "lever " + ", ".join(resolve_levers(data, rule, levers))
    return data, text


def validate(data: dict[str, Any]) -> str:
    """Schema-validate the resolved mapping; return its config hash."""
    from finswarm.config import load

    cfg = load.CampaignConfig.model_validate(data)
    return load.config_hash(cfg)


def render(data: dict[str, Any], header: list[str]) -> str:
    body = yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=120)
    return "".join(f"# {line}\n" for line in header) + body


# ------------------------------------------------------------------------- trend-rule shift
EXT_FILE = TIER0 / "TR-H-ext.yaml"
SHIFT_TAG = "shift"
T0_TAG = "t0"
#: integer ``tags.*`` keys that reference a period and move with the windows
WINDOW_TAG_KEYS = ("t0", "window_start", "window_end", "window")


def _ext_branch(ext_path: Path, base_name: str) -> dict[str, Any]:
    ext = _load_yaml(ext_path)
    if ext.get("name") != base_name or ext["base"]["name"] != base_name:
        raise ValueError(f"{ext_path.name} extends {ext.get('name')!r}, not {base_name!r}")
    branches = [b for b in ext.get("branches", []) if b.get("name") == "ext"]
    if len(branches) != 1:
        raise ValueError(f"{ext_path.name}: expected one branch named 'ext'")
    return branches[0]


def _has_llm_slot(data: dict[str, Any]) -> bool:
    return any(s.get("model") != "scripted" for s in data["base"].get("llm", {}).get("slots", []))


def make_ext_branch(data: dict[str, Any], n: int) -> dict[str, Any]:
    """The generic trend-rule ``ext`` branch of a 100-period LLM campaign: one child per run at
    ``period:<periods>`` running ``n`` more periods, ``tags.arm = <arm>_ext``,
    ``tags.label = <label>-ext``, a marker whose note names the checkpoint
    ``collusion_established_<periods + n>`` (the hand-written TR-H-ext.yaml is this branch)."""
    base = data["base"]
    if not _has_llm_slot(data):
        raise ValueError("the trend rule applies to LLM campaigns only (no LLM slot here)")
    t0 = int(base["periods"])
    if t0 != 100:
        raise ValueError(f"the trend rule extends 100-period campaigns, this one has {t0}")
    tags = base.get("tags") or {}
    arm, label = tags.get("arm", base["name"]), tags.get("label", base["name"])
    return {
        "name": "ext",
        "parent": base["name"],
        "at": f"period:{t0}",
        "events": [
            {"type": "set_param", "path": "periods", "value": t0 + n},
            {"type": "set_param", "path": "tags.arm", "value": f"{arm}_ext"},
            {"type": "set_param", "path": "tags.label", "value": f"{label}-ext"},
            {
                "type": "set_param",
                "path": "tags.milestone",
                "value": f"collusion_established_{t0 + n}",
                "when": {"at_period": t0 + n},
                "note": f"collusion_established_{t0 + n}",
            },
        ],
    }


def ext_campaign(data: dict[str, Any], n: int) -> dict[str, Any]:
    """``<campaign>-ext.yaml``: the campaign (grid kept, branches dropped) plus its ext branch."""
    out = {k: v for k, v in data.items() if k != "branches"}
    out["branches"] = [make_ext_branch(data, n)]
    return out


def find_ext_branch(
    template: Path, data: dict[str, Any], n: int, explicit: Path | None
) -> tuple[dict[str, Any], str]:
    """The ext branch for ``data``: ``--ext`` when given, else ``<base>-ext.yaml`` next to the
    template or anywhere under configs/sprint, else the generic branch (``make_ext_branch``)."""
    base_name = data["base"]["name"]
    if explicit is not None:
        return _ext_branch(explicit, base_name), str(explicit)
    candidates = [template.parent / f"{base_name}-ext.yaml"]
    candidates += sorted((ROOT / "configs" / "sprint").rglob(f"{base_name}-ext.yaml"))
    for c in candidates:
        if c.is_file():
            return _ext_branch(c, base_name), str(c.relative_to(ROOT)).replace("\\", "/")
    return make_ext_branch(data, n), "generic rule (no <campaign>-ext.yaml found)"


def _shift_event(e: dict[str, Any], n: int, tpp: int) -> dict[str, Any]:
    e = dict(e)
    if e.get("type") == "set_param":
        if e.get("path") == "periods":
            e["value"] = int(e["value"]) + n
        elif str(e.get("path", "")).startswith("tags."):  # tags are strings in the schema
            key = e["path"].split(".", 1)[1]
            v = e.get("value")
            if key in WINDOW_TAG_KEYS and (
                isinstance(v, int) or (isinstance(v, str) and v.isdigit())
            ):
                e["value"] = str(int(v) + n)
    when = e.get("when")
    if isinstance(when, dict):
        when = dict(when)
        if when.get("at_period") is not None:
            when["at_period"] = int(when["at_period"]) + n
        if when.get("at_tick") is not None:
            when["at_tick"] = int(when["at_tick"]) + n * tpp
        e["when"] = when
    return e


def shift_config(
    data: dict[str, Any],
    n: int,
    ext_path: Path | None = None,
    template: Path | None = None,
) -> tuple[dict[str, Any], str]:
    """The trend-rule variant of a campaign: every branch hangs off the ``ext`` child
    (``parent: <base>--ext``) at ``period:<T0 + n>``, its horizon, ``when`` triggers and
    window tags move by ``n`` periods, and ``tags.shift = n`` / ``tags.t0`` record the shift.
    A from-scratch campaign (no branches: AC-H, GR-H, CA-H, PREV-H) gets its ext branch only."""
    if n <= 0:
        raise ValueError("--shift must be a positive number of periods")
    base_name = data["base"]["name"]
    tpp = int(data["base"].get("ticks_per_period", 1))
    ext, ext_src = find_ext_branch(template or ROOT, data, n, ext_path)
    if not data.get("branches"):
        if not _has_llm_slot(data):
            raise ValueError("the trend rule applies to LLM campaigns only (no LLM slot here)")
        out = dict(data)
        out["branches"] = [ext]
        return out, f"shift +{n}: from-scratch campaign, ext branch only ({ext_src})"
    ext_at = int(str(ext["at"]).partition(":")[2])
    ext_horizon = next(
        int(e["value"])
        for e in ext["events"]
        if e.get("type") == "set_param" and e.get("path") == "periods"
    )
    parent = f"{base_name}--{ext['name']}"
    shifted = []
    for b in data["branches"]:
        kind, _, val = str(b["at"]).partition(":")
        if kind != "period" or int(val) != ext_at:
            raise ValueError(f"branch {b['name']!r}: at {b['at']!r} is not period:{ext_at}")
        if b.get("parent") != base_name:
            raise ValueError(f"branch {b['name']!r}: parent {b.get('parent')!r} is not the trunk")
        nb = dict(b)
        nb["parent"] = parent
        nb["at"] = f"period:{ext_horizon}"
        nb["events"] = [_shift_event(e, n, tpp) for e in b["events"]] + [
            {"type": "set_param", "path": f"tags.{SHIFT_TAG}", "value": str(int(n))},
            {"type": "set_param", "path": f"tags.{T0_TAG}", "value": str(int(ext_horizon))},
        ]
        shifted.append(nb)
    out = dict(data)
    out["branches"] = [ext, *shifted]
    names = ", ".join(b["name"] for b in shifted)
    text = (
        f"shift +{n}: branches {names} hang off {parent} at period:{ext_horizon} "
        f"(horizons, triggers and window tags +{n}; tags.{SHIFT_TAG} = {n}, tags.{T0_TAG} = {ext_horizon}; "
        f"ext from {ext_src})"
    )
    return out, text


def _write(
    template: Path,
    data: dict[str, Any],
    text: str,
    suffix: str,
    args: Any,
    argv: list[str] | None,
    out_name: str | None = None,
) -> int:
    chash = validate(data)
    now = dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    out_dir = args.out_dir if args.out_dir is not None else template.parent
    out = out_dir / (out_name or f"{template.stem}.{suffix}.yaml")
    sha = _sha256(template)
    label = suffix.upper()
    line = f"{label} {template.name} -> {out.name}: {text}; template sha256 {sha[:12]}; config_hash {chash}; {now}"
    header = [
        f"{template.stem} {label} from {template.name} by scripts/resolve_registered.py on {now}.",
        f"Template sha256 {sha}; arguments: {' '.join(sys.argv[1:] if argv is None else argv)}.",
        f"Resolution: {text}.",
        "Generated file: do not edit; rerun the script. The template's comments document the rule.",
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(data, header), encoding="utf-8", newline="\n")
    print(line)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("template", type=Path, nargs="+", help="template(s); several only with --shift")
    ap.add_argument("--qualified", type=int, nargs="*", help="seeds of the qualified trunks")
    ap.add_argument("--lever", action="append", help="selected lever id (repeat for LT-S)")
    ap.add_argument(
        "--shift",
        type=int,
        default=None,
        help="trend rule: write <stem>.shifted.yaml hanging every branch off the ext child",
    )
    ap.add_argument(
        "--ext",
        type=Path,
        default=None,
        help="the extension config (--shift); default: <campaign>-ext.yaml, else the generic rule",
    )
    ap.add_argument(
        "--make-ext",
        type=int,
        default=None,
        metavar="N",
        help="write <campaign>-ext.yaml (one ext branch per run, +N periods) next to the template",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        help="directory of the resolved file (default: next to the template)",
    )
    args = ap.parse_args(argv)
    templates: list[Path] = args.template
    for t in templates:
        if not t.is_file():
            print(f"MISSING {t}", file=sys.stderr)
            return 1
    if args.shift is not None or args.make_ext is not None:
        if args.qualified is not None or args.lever:
            print("ERROR --shift / --make-ext take no --qualified / --lever", file=sys.stderr)
            return 1
        if args.shift is not None and args.make_ext is not None:
            print("ERROR --shift and --make-ext are exclusive", file=sys.stderr)
            return 1
        rc = 0
        for t in templates:
            try:
                raw = _load_yaml(t)
                if args.make_ext is not None:
                    data = ext_campaign(raw, args.make_ext)
                    base = raw["base"]["name"]
                    text = (
                        f"trend rule: one ext child per {base} run at period:{raw['base']['periods']} "
                        f"running to {int(raw['base']['periods']) + args.make_ext}"
                    )
                    _write(t, data, text, "ext", args, argv, out_name=f"{base}-ext.yaml")
                else:
                    data, text = shift_config(raw, args.shift, args.ext, template=t)
                    _write(t, data, text, "shifted", args, argv)
            except Exception as exc:
                print(f"ERROR {t.name}: {type(exc).__name__}: {exc}", file=sys.stderr)
                rc = 1
        return rc
    if len(templates) != 1:
        print("ERROR one template at a time without --shift", file=sys.stderr)
        return 1
    template = templates[0]
    try:
        data, text = resolve(template, qualified=args.qualified, levers=args.lever)
        return _write(template, data, text, "resolved", args, argv)
    except NothingToRun as exc:
        print(f"NOTHING-TO-RUN {template.name}: {exc}; no file written")
        return 2
    except Exception as exc:
        print(f"ERROR {template.name}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
