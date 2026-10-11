"""Checkpoint files: ``run_dir/checkpoints/pNNNNNN.ckpt`` (pickle) + ``pNNNNNN.json`` (meta).

A checkpoint at boundary ``B`` holds the full runner state after the events of ``B`` were
applied and before period ``B`` starts. Both files are written atomically; the meta file is
written last and is the commit marker (a ``.ckpt`` without meta is ignored).
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from finswarm.store.writer import atomic_write_bytes, atomic_write_text

CKPT_DIR = "checkpoints"
PROTECTED = {"initial", "final", "branch_origin"}


@dataclass
class CheckpointInfo:
    period: int
    names: list[str]
    path: Path
    meta: dict[str, Any]

    @property
    def meta_path(self) -> Path:
        return self.path.with_suffix(".json")


def _stem(period: int) -> str:
    return f"p{int(period):06d}"


def list_checkpoints(run_dir: str | Path) -> list[CheckpointInfo]:
    cdir = Path(run_dir) / CKPT_DIR
    if not cdir.exists():
        return []
    out = []
    for meta_path in sorted(cdir.glob("p*.json")):
        ckpt = meta_path.with_suffix(".ckpt")
        if not ckpt.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        out.append(CheckpointInfo(int(meta["period"]), list(meta.get("names", [])), ckpt, meta))
    return sorted(out, key=lambda c: c.period)


def write_checkpoint(
    run_dir: str | Path,
    period: int,
    payload: dict[str, Any],
    names: list[str],
    meta: dict[str, Any],
) -> CheckpointInfo:
    cdir = Path(run_dir) / CKPT_DIR
    cdir.mkdir(parents=True, exist_ok=True)
    path = cdir / f"{_stem(period)}.ckpt"
    meta_path = path.with_suffix(".json")
    if meta_path.exists():
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        names = list(dict.fromkeys([*old.get("names", []), *names]))
    full_meta = {**meta, "period": int(period), "names": names}
    atomic_write_bytes(path, pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
    atomic_write_text(meta_path, json.dumps(full_meta, sort_keys=True, indent=1))
    return CheckpointInfo(int(period), names, path, full_meta)


def load_payload(info: CheckpointInfo) -> dict[str, Any]:
    with open(info.path, "rb") as fh:
        return pickle.load(fh)  # noqa: S301 - local, self-written checkpoints only


def find_checkpoint(
    run_dir: str | Path, *, at_or_before: int | None = None, name: str | None = None
) -> CheckpointInfo | None:
    ckpts = list_checkpoints(run_dir)
    if name is not None:
        ckpts = [c for c in ckpts if name in c.names]
    if at_or_before is not None:
        ckpts = [c for c in ckpts if c.period <= at_or_before]
    return ckpts[-1] if ckpts else None


def prune_checkpoints(run_dir: str | Path, keep_last: int) -> None:
    """Delete old checkpoints that carry no name other than ``periodic``."""
    if keep_last <= 0:
        return
    plain = [c for c in list_checkpoints(run_dir) if set(c.names) <= {"periodic"}]
    for c in plain[:-keep_last]:
        c.meta_path.unlink(missing_ok=True)
        c.path.unlink(missing_ok=True)
