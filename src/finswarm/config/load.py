"""Load, validate and hash configurations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from finswarm.config.schema import CampaignConfig, RunConfig


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    return data


def load_run_config(path: str | Path) -> RunConfig:
    return RunConfig.model_validate(load_yaml(path))


def load_campaign_config(path: str | Path) -> CampaignConfig:
    return CampaignConfig.model_validate(load_yaml(path))


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def config_hash(cfg: RunConfig | CampaignConfig) -> str:
    """Stable sha256 of the resolved config (first 12 hex chars used in run ids)."""
    return hashlib.sha256(canonical_json(cfg.model_dump(mode="json")).encode()).hexdigest()


def run_id(cfg: RunConfig, seed: int | None = None) -> str:
    """Deterministic run id: name, seed, config hash, parent and branch info."""
    s = cfg.seed if seed is None else seed
    payload = canonical_json(
        {
            "config_hash": config_hash(cfg),
            "seed": s,
            "parent_run_id": cfg.parent_run_id,
            "branch_period": cfg.branch_period,
        }
    )
    digest = hashlib.sha256(payload.encode()).hexdigest()[:12]
    return f"{cfg.name}__s{s}__{digest}"
