"""Atomic, generic run-state checkpoints.

A checkpoint is a directory. ``save_checkpoint(path, state)`` writes one file per top-level
key of ``state``:

* a value that contains numpy arrays (possibly in nested dicts) -> ``<key>.npz``; nested keys
  are flattened with ``/`` and every non-array leaf (int, float, str, None, list, ...) goes
  into a JSON sidecar stored inside the same npz under ``__json__``;
* any other JSON-serialisable value -> ``<key>.json``.

The standard layout used by the runner is ``meta.json, rng.json, market.npz, swarm.npz,
accounts.npz, agents.json, pending_orders.json``; ``load_checkpoint`` returns the same dict
(arrays bit-identical, dtypes preserved). Files are written into a temporary sibling
directory, hashed into ``_manifest.json`` and renamed into place, so a crash never leaves a
half-written checkpoint under the final name. ``keep_last`` prunes older siblings.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import uuid
from pathlib import Path
from typing import Any

import numpy as np

FORMAT_VERSION = 1
MANIFEST = "_manifest.json"
_JSON_KEY = "__json__"
_SEP = "/"
_NAME_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]*$")


def checkpoint_path(root: str | Path, period: int) -> Path:
    """Canonical checkpoint directory name; sorts chronologically."""
    return Path(root) / f"period_{int(period):07d}"


def _contains_array(v: Any) -> bool:
    if isinstance(v, np.ndarray):
        return True
    if isinstance(v, dict):
        return any(_contains_array(x) for x in v.values())
    return False


def _plain(v: Any) -> Any:
    """Convert numpy scalars inside JSON-able structures to Python scalars."""
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, list | tuple):
        return [_plain(x) for x in v]
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        raise TypeError("numpy arrays are only supported inside dicts, not lists")
    return v


def _flatten(prefix: str, v: dict[str, Any], arrays: dict[str, np.ndarray], other: dict) -> None:
    for k, x in v.items():
        k = str(k)
        if _SEP in k or k == _JSON_KEY:
            raise ValueError(f"invalid checkpoint key {k!r}")
        key = f"{prefix}{_SEP}{k}" if prefix else k
        if isinstance(x, np.ndarray):
            if x.dtype == object:
                raise TypeError(f"object arrays are not checkpointable ({key})")
            arrays[key] = x
        elif isinstance(x, dict) and _contains_array(x):
            other[key] = {"__dict__": True}
            _flatten(key, x, arrays, other)
        else:
            other[key] = _plain(x)


def _unflatten(arrays: dict[str, np.ndarray], other: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}

    def put(key: str, value: Any) -> None:
        parts = key.split(_SEP)
        node = out
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value

    for key in sorted(other, key=lambda s: s.count(_SEP)):
        v = other[key]
        put(key, {} if isinstance(v, dict) and v.get("__dict__") is True and len(v) == 1 else v)
    for key, arr in arrays.items():
        put(key, arr)
    return out


def _write_json(path: Path, obj: Any) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(_plain(obj), fh, ensure_ascii=False, indent=1, allow_nan=True)
        fh.flush()
        os.fsync(fh.fileno())


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_checkpoint(
    path: str | Path, state: dict[str, Any], *, keep_last: int | None = None
) -> Path:
    """Write ``state`` atomically to directory ``path`` (replacing it if present)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    tmp.mkdir()
    try:
        files: dict[str, str] = {}
        for name, value in state.items():
            if not _NAME_RE.match(name) or name == MANIFEST.removesuffix(".json"):
                raise ValueError(f"invalid checkpoint part name {name!r}")
            if isinstance(value, np.ndarray):
                value = {"array": value}
                wrapped = True
            else:
                wrapped = False
            if isinstance(value, dict) and _contains_array(value):
                arrays: dict[str, np.ndarray] = {}
                other: dict[str, Any] = {}
                _flatten("", value, arrays, other)
                if wrapped:
                    other["__wrapped_array__"] = True
                payload = dict(arrays)
                payload[_JSON_KEY] = np.array(json.dumps(other, allow_nan=True))
                fname = f"{name}.npz"
                with open(tmp / fname, "wb") as fh:
                    np.savez(fh, allow_pickle=False, **payload)
                    fh.flush()
                    os.fsync(fh.fileno())
            else:
                fname = f"{name}.json"
                _write_json(tmp / fname, value)
            files[fname] = _sha256(tmp / fname)
        _write_json(tmp / MANIFEST, {"format": FORMAT_VERSION, "files": files})
        if path.exists():
            trash = path.parent / f".{path.name}.old-{uuid.uuid4().hex[:8]}"
            os.replace(path, trash)
            os.replace(tmp, path)
            shutil.rmtree(trash, ignore_errors=True)
        else:
            os.replace(tmp, path)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    if keep_last is not None:
        prune_checkpoints(path.parent, keep_last)
    return path


def load_checkpoint(path: str | Path, *, verify: bool = True) -> dict[str, Any]:
    """Load a checkpoint directory written by ``save_checkpoint``."""
    path = Path(path)
    manifest = json.loads((path / MANIFEST).read_text(encoding="utf-8"))
    if manifest.get("format") != FORMAT_VERSION:
        raise ValueError(f"unsupported checkpoint format {manifest.get('format')}")
    out: dict[str, Any] = {}
    for fname, digest in sorted(manifest["files"].items()):
        fpath = path / fname
        if verify and _sha256(fpath) != digest:
            raise ValueError(f"checkpoint file {fpath} is corrupt (sha256 mismatch)")
        name, ext = fname.rsplit(".", 1)
        if ext == "json":
            out[name] = json.loads(fpath.read_text(encoding="utf-8"))
        else:
            with np.load(fpath, allow_pickle=False) as z:
                arrays = {k: z[k] for k in z.files if k != _JSON_KEY}
                other = json.loads(str(z[_JSON_KEY][()]))
            wrapped = other.pop("__wrapped_array__", False)
            value = _unflatten(arrays, other)
            out[name] = value["array"] if wrapped else value
    return out


def list_checkpoints(root: str | Path) -> list[Path]:
    """Complete checkpoint directories under ``root``, oldest first (by name)."""
    root = Path(root)
    if not root.exists():
        return []
    return sorted(
        p
        for p in root.iterdir()
        if p.is_dir() and not p.name.startswith(".") and (p / MANIFEST).exists()
    )


def latest_checkpoint(root: str | Path) -> Path | None:
    cps = list_checkpoints(root)
    return cps[-1] if cps else None


def prune_checkpoints(root: str | Path, keep_last: int) -> list[Path]:
    """Delete all but the ``keep_last`` newest checkpoints and stale temporary dirs."""
    if keep_last < 1:
        raise ValueError("keep_last must be >= 1")
    root = Path(root)
    removed: list[Path] = []
    cps = list_checkpoints(root)
    for p in cps[: max(0, len(cps) - keep_last)]:
        shutil.rmtree(p, ignore_errors=True)
        removed.append(p)
    for p in root.iterdir():
        if p.is_dir() and p.name.startswith(".") and (".tmp-" in p.name or ".old-" in p.name):
            if f"-{os.getpid()}-" in p.name and ".tmp-" in p.name:
                continue  # a concurrent save in this process
            shutil.rmtree(p, ignore_errors=True)
    return removed
