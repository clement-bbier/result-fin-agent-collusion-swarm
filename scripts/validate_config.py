"""Validate run or campaign YAML configs with the pydantic schema (finswarm.config.load)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def validate(path: Path) -> str | None:
    """Return None if valid, else an error message."""
    from finswarm.config import load

    try:
        data = load.load_yaml(path)
        if "base" in data:
            cfg = load.CampaignConfig.model_validate(data)
        else:
            cfg = load.RunConfig.model_validate(data)
        load.config_hash(cfg)
    except Exception as exc:
        return f"{type(exc).__name__}: {str(exc)[:500]}"
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="*", help="YAML files or directories (default: configs/)")
    args = ap.parse_args(argv)
    files: list[Path] = []
    if not args.paths and not Path("configs").exists():
        print("no configs/ directory yet; nothing to validate")
        return 0
    for p in map(Path, args.paths or ["configs"]):
        if p.is_dir():
            files += sorted(p.rglob("*.yaml")) + sorted(p.rglob("*.yml"))
        elif p.exists():
            files.append(p)
        else:
            print(f"MISSING {p}")
            return 1
    bad = 0
    for f in files:
        if f.name == "models.yaml":  # lane registry, not a run config
            continue
        err = validate(f)
        print(f"{'OK  ' if err is None else 'FAIL'} {f}" + (f"\n     {err}" if err else ""))
        bad += err is not None
    print(f"{len(files)} file(s) checked, {bad} invalid")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
