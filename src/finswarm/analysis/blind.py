"""Blind mode for the pre-registered frontier (PREREG section 5, SAP section 11).

Lever arms are replaced by letters before any number is read. The key (letter -> arm, the
real ids of the blinded runs and the JSON paths that were rewritten) is written to a separate
file outside the blinded output directory and only its SHA-256 is recorded next to the
blinded numbers, so the blinded ``stats.json`` can be committed before unblinding and the
mapping checked afterwards. Control arms (the sham arm, sham2, the trunk) keep their names:
they are the zero point of every contrast and carry no lever identity.

What is blinded in an ``analyze_campaign`` result: every dictionary key and every string
value equal to a lever name (``arms``, ``contrasts``, ``h1``, ``h1r``, ``inference``,
``mqc``, power tables), the ``run_id``, ``label`` and ``path`` of every run of a blinded arm
(they would otherwise name the lever) and the per-arm pinned benchmarks (their window market
and method would single out a lever). Letters come from a seeded permutation of the sorted
lever names; an existing key is reused (new arms get the next unused letters) so re-running
the analysis keeps the letters stable. :func:`check_no_leak` refuses to write a blinded file
in which a lever name still appears as a token.
"""

from __future__ import annotations

import hashlib
import json
import re
import string
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_CONTROLS: tuple[str, ...] = ("sham", "sham2", "trunk")
DEFAULT_SEED = 20261010
KEY_DIRNAME = "_blind_keys"
BLINDED_ROLES: tuple[str, ...] = ("primary_lever", "secondary", "exploratory")
RUN_LISTS: tuple[str, ...] = ("runs", "ignored_runs")
BANNER = (
    "**Blind analysis.** Lever arms are shown as letters; the key is outside this folder "
    "(sha256 `{sha}`). Commit this file before unblinding (PREREG section 5)."
)

JsonPath = list[str | int]


class BlindLeak(RuntimeError):
    """A lever name survived blinding."""


# --------------------------------------------------------------------------- key


def letter_key(
    arms: Iterable[str],
    controls: Iterable[str] = DEFAULT_CONTROLS,
    seed: int = DEFAULT_SEED,
    existing: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Mapping ``letter -> arm`` for every non-control arm.

    Arms already present in ``existing`` keep their letter; the others are dealt the unused
    letters in the order of a permutation seeded with ``seed`` over the sorted arm names.
    """
    ctrl = set(controls)
    levers = sorted({a for a in arms if a not in ctrl})
    mapping: dict[str, str] = {}
    if existing:
        for letter, arm in existing.items():
            if arm in levers:
                mapping[letter] = arm
    todo = [a for a in levers if a not in mapping.values()]
    free = [c for c in string.ascii_uppercase if c not in mapping]
    if len(todo) > len(free):
        raise ValueError(f"blind: {len(todo)} lever arms but only {len(free)} letters left")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(todo))
    for pos, arm in zip(order, todo, strict=True):
        mapping[free[int(pos)]] = arm
    return dict(sorted(mapping.items()))


def key_sha256(
    mapping: Mapping[str, str],
    runs: Mapping[str, Any] | None = None,
    paths: Mapping[str, Any] | None = None,
) -> str:
    body = {
        "mapping": dict(sorted(mapping.items())),
        "runs": dict(sorted((runs or {}).items())),
        "paths": paths or {},
    }
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def key_path(out_root: str | Path, campaign: str) -> Path:
    return Path(out_root) / KEY_DIRNAME / f"{campaign}.key.json"


def write_key(
    path: str | Path,
    campaign: str,
    mapping: Mapping[str, str],
    seed: int,
    controls: Sequence[str],
    runs: Mapping[str, Any] | None = None,
    paths: Mapping[str, Any] | None = None,
) -> str:
    """Write the key file (letter -> arm, blinded run ids -> real ids, rewritten JSON paths)
    with its hash; returns the hash."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    runs = dict(sorted((runs or {}).items()))
    paths = dict(paths or {})
    sha = key_sha256(mapping, runs, paths)
    body = {
        "campaign": campaign,
        "seed": int(seed),
        "controls": sorted(controls),
        "mapping": dict(sorted(mapping.items())),
        "runs": runs,
        "paths": paths,
        "sha256": sha,
        "_warning": "key of a blind analysis; do not read before the blind commit",
    }
    p.write_text(json.dumps(body, indent=2, sort_keys=True), encoding="utf-8")
    return sha


def load_key(path: str | Path) -> dict[str, Any]:
    """Load a key file and verify its hash."""
    key = json.loads(Path(path).read_text(encoding="utf-8"))
    sha = key_sha256(key.get("mapping", {}), key.get("runs", {}), key.get("paths", {}))
    if key.get("sha256") != sha:
        raise ValueError(f"{path}: blind key hash mismatch (file edited?)")
    return key


# --------------------------------------------------------------------------- arms and runs


def lever_arms(result: Mapping[str, Any], controls: Iterable[str]) -> set[str]:
    """Arms to blind: the H1-R family, every paired (contrasted) arm, every run of a lever,
    secondary or exploratory role and every MQC lever, minus the controls."""
    levers: set[str] = set(result.get("h1r", {}).get("family", []))
    levers |= set(result.get("contrasts", {}))
    for r in result.get("runs", []):
        if r.get("role") in BLINDED_ROLES:
            levers.add(str(r.get("registered_arm", r.get("arm"))))
    mqc = result.get("mqc") or {}
    if isinstance(mqc, Mapping):
        levers |= set(mqc.get("levers", mqc))
    return {a for a in levers if a not in set(controls)}


def blind_run(
    r: Mapping[str, Any], inv: Mapping[str, str], runs: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """A run (or exclusion) row with its arm as a letter; a lever run also gets an opaque
    id, the letter as label and no path (the real values are recorded in ``runs``)."""
    arm = r.get("registered_arm", r.get("arm"))
    out = dict(r)
    for k in ("arm", "registered_arm"):
        if isinstance(out.get(k), str):
            out[k] = inv.get(out[k], out[k])
    if not (isinstance(arm, str) and arm in inv):
        return out
    rid = str(r.get("run_id"))
    bid = "blind:" + hashlib.sha256(rid.encode()).hexdigest()[:12]
    runs[bid] = {k: r.get(k) for k in ("run_id", "label", "path") if k in r}
    out["run_id"] = bid
    if "label" in out:
        out["label"] = inv[arm]
    if "path" in out:
        out["path"] = None
    return out


def unblind_run(
    r: Mapping[str, Any], mapping: Mapping[str, str], runs: Mapping[str, Any]
) -> dict[str, Any]:
    out = dict(r)
    for k in ("arm", "registered_arm"):
        if isinstance(out.get(k), str):
            out[k] = mapping.get(out[k], out[k])
    real = runs.get(str(r.get("run_id")))
    if real:
        out.update({k: v for k, v in real.items() if k in out})
    return out


# --------------------------------------------------------------------------- tree rewriting


def _blind_tree(obj: Any, inv: Mapping[str, str], path: JsonPath, touched: dict[str, list]) -> Any:
    """Rename every dict key and replace every string value equal to a lever name; record the
    paths of the rewritten dicts (``keys``) and values (``values``)."""
    if isinstance(obj, Mapping):
        out = {}
        renamed = False
        for k, v in obj.items():
            nk = k
            if isinstance(k, str) and k in inv:
                nk, renamed = inv[k], True
            out[nk] = _blind_tree(v, inv, [*path, nk], touched)
        if renamed:
            touched["keys"].append(list(path))
        return out
    if isinstance(obj, list):
        return [_blind_tree(v, inv, [*path, i], touched) for i, v in enumerate(obj)]
    if isinstance(obj, str) and obj in inv:
        touched["values"].append(list(path))
        return inv[obj]
    return obj


def _get(obj: Any, path: JsonPath) -> Any:
    for k in path:
        obj = obj[k]
    return obj


def _unblind_tree(obj: Any, mapping: Mapping[str, str], touched: Mapping[str, list]) -> Any:
    out = json.loads(json.dumps(obj))  # deep copy of a JSON document
    for p in touched.get("values", []):
        try:
            parent, last = _get(out, p[:-1]), p[-1]
            if isinstance(parent[last], str):
                parent[last] = mapping.get(parent[last], parent[last])
        except (KeyError, IndexError, TypeError):
            continue
    for p in sorted(touched.get("keys", []), key=len, reverse=True):
        try:
            d = _get(out, p)
        except (KeyError, IndexError, TypeError):
            continue
        if isinstance(d, dict):
            items = list(d.items())
            d.clear()
            for k, v in items:
                d[mapping.get(k, k) if isinstance(k, str) else k] = v
    return out


def blind_result(
    result: Mapping[str, Any],
    exclusions: Sequence[Mapping[str, Any]],
    mapping: Mapping[str, str],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], dict[str, list]]:
    """``(blinded result, blinded exclusions, blinded run ids -> real ids, rewritten paths)``.

    Besides the renaming, the per-arm ``benchmarks`` (window market and method identify a
    lever) are replaced by a count, ``provisional_benchmarks`` / ``stale_benchmarks`` by their
    counts, each contrast's ``window_market_method`` is dropped and the MQC campaign labels
    are nulled.
    """
    inv = {arm: letter for letter, arm in mapping.items()}
    runs: dict[str, dict[str, Any]] = {}
    body = {k: v for k, v in result.items() if k not in RUN_LISTS}
    bench = body.get("benchmarks")
    if isinstance(bench, Mapping):
        body["benchmarks"] = {"blinded": True, "n_arms": len(bench)}
    for k in ("provisional_benchmarks", "stale_benchmarks"):
        if isinstance(body.get(k), list):
            body[k] = {"blinded": True, "n": len(body[k])}
    body["contrasts"] = {
        a: {kk: vv for kk, vv in c.items() if kk != "window_market_method"}
        for a, c in body.get("contrasts", {}).items()
    }
    mqc = body.get("mqc")
    if isinstance(mqc, Mapping) and isinstance(mqc.get("levers"), Mapping):
        body["mqc"] = {
            **mqc,
            "levers": {
                lev: ({**v, "label": None} if isinstance(v, Mapping) and "label" in v else v)
                for lev, v in mqc["levers"].items()
            },
        }
    touched: dict[str, list] = {"keys": [], "values": []}
    out = _blind_tree(body, inv, [], touched)
    for k in RUN_LISTS:
        if k in result:
            out[k] = [blind_run(r, inv, runs) for r in result[k]]
    exc = [blind_run(e, inv, runs) for e in exclusions]
    return out, exc, runs, touched


def check_no_leak(text: str, levers: Iterable[str]) -> None:
    """Raise :class:`BlindLeak` if a lever name appears as a token in ``text`` (letters,
    digits and underscores are token characters; ``L2-H`` therefore leaks ``L2``)."""
    for arm in levers:
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(arm)}(?![A-Za-z0-9_])", text):
            raise BlindLeak(f"blind: lever name {arm!r} survived blinding")


# --------------------------------------------------------------------------- output


def _jsonable(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, list | tuple):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.generic):
        o = o.item()
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def summary_md(result: Mapping[str, Any], exclusions: Sequence[Mapping[str, Any]]) -> str:
    """The campaign ``summary.md`` with the blind banner on top."""
    from finswarm.analysis.campaign import _summary_md

    md = _summary_md(dict(result), list(exclusions))
    sha = result.get("blind", {}).get("key_sha256", "?")
    head, _, rest = md.partition("\n")
    return f"{head} (BLIND)\n\n{BANNER.format(sha=sha)}\n{rest}"


def write_blind_outputs(
    out_dir: str | Path,
    result: Mapping[str, Any],
    exclusions: Sequence[Mapping[str, Any]],
    levers: Iterable[str],
) -> None:
    """Write ``stats.json``, ``exclusions.json`` and ``summary.md`` after the leak check."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    res = _jsonable(result)
    exc = _jsonable(list(exclusions))
    stats = json.dumps(res, indent=2)
    excl = json.dumps(exc, indent=2)
    md = summary_md(res, exc)
    check_no_leak(stats + "\n" + excl + "\n" + md, levers)
    (out / "stats.json").write_text(stats, encoding="utf-8")
    (out / "exclusions.json").write_text(excl, encoding="utf-8")
    (out / "summary.md").write_text(md, encoding="utf-8")


# --------------------------------------------------------------------------- pipeline


def analyze_blind(
    campaign_dir: str | Path,
    out_root: str | Path = "results/analysis",
    key_root: str | Path | None = None,
    controls: Sequence[str] = DEFAULT_CONTROLS,
    blind_seed: int = DEFAULT_SEED,
    **kw: Any,
) -> dict[str, Any]:
    """Run ``analyze_campaign`` and write only the blinded outputs.

    The open results are produced in a temporary directory that is deleted afterwards. The key
    goes to ``<key_root or out_root>/_blind_keys/<campaign>.key.json``. Keyword arguments are
    passed to :func:`finswarm.analysis.campaign.analyze_campaign`.
    """
    from finswarm.analysis.campaign import analyze_campaign

    cdir = Path(campaign_dir)
    with tempfile.TemporaryDirectory() as tmp:
        res = analyze_campaign(cdir, out_root=tmp, **kw)
        exclusions = json.loads(
            (Path(tmp) / cdir.name / "exclusions.json").read_text(encoding="utf-8")
        )
    ctrl = tuple(dict.fromkeys(controls))
    levers = lever_arms(res, ctrl) | ({e["arm"] for e in exclusions if e.get("arm")} - set(ctrl))
    kpath = key_path(key_root or out_root, cdir.name)
    existing = load_key(kpath)["mapping"] if kpath.exists() else None
    mapping = letter_key(levers, ctrl, blind_seed, existing)
    blinded, exc_b, run_map, touched = blind_result(_jsonable(res), exclusions, mapping)
    sha = write_key(kpath, cdir.name, mapping, blind_seed, ctrl, run_map, touched)
    blinded["blind"] = {
        "key_sha256": sha,
        "key_path": str(kpath),
        "controls": sorted(ctrl),
        "letters": sorted(mapping),
        "seed": int(blind_seed),
        "hidden": ["lever arm names", "lever run ids, labels and paths", "per-arm benchmarks"],
    }
    write_blind_outputs(Path(out_root) / cdir.name, blinded, exc_b, mapping.values())
    return blinded


def unblind_stats(
    stats_path: str | Path, key: str | Path | Mapping[str, Any], out_path: str | Path | None = None
) -> dict[str, Any]:
    """Map the letters of a blinded ``stats.json`` back to arm names.

    Writes ``<stem>.unblinded.json`` next to the blinded file (or ``out_path``) and returns
    the open result. ``key`` is the key file path or its parsed content; its hash must be
    the one recorded in the blinded stats. The per-arm benchmarks dropped at blinding are not
    restored (re-run the open analysis for them).
    """
    sp = Path(stats_path)
    blinded = json.loads(sp.read_text(encoding="utf-8"))
    kd = dict(key) if isinstance(key, Mapping) else load_key(key)
    if "mapping" not in kd:
        kd = {"mapping": kd}
    mapping, runs, paths = kd["mapping"], kd.get("runs", {}), kd.get("paths", {})
    recorded = blinded.get("blind", {}).get("key_sha256")
    if recorded is not None and recorded != key_sha256(mapping, runs, paths):
        raise ValueError("unblind: key does not match the hash recorded in the blinded stats")
    exc_path = sp.with_name("exclusions.json")
    exclusions = json.loads(exc_path.read_text(encoding="utf-8")) if exc_path.exists() else []
    body = {k: v for k, v in blinded.items() if k not in RUN_LISTS}
    opened = _unblind_tree(body, mapping, paths)
    for k in RUN_LISTS:
        if k in blinded:
            opened[k] = [unblind_run(r, mapping, runs) for r in blinded[k]]
    opened["blind"] = {**blinded.get("blind", {}), "unblinded": True, "mapping": dict(mapping)}
    opened["exclusions"] = [unblind_run(e, mapping, runs) for e in exclusions]
    dest = Path(out_path) if out_path else sp.with_name(f"{sp.stem}.unblinded.json")
    dest.write_text(json.dumps(_jsonable(opened), indent=2), encoding="utf-8")
    return opened
