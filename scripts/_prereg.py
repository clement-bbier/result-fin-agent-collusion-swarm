"""Shared helpers for the pre-registration freeze (PREREG.md section 6, decision D-14).

The hashed file set (two stages), LF-normalised hashing, ``sha256sum``-style lines and the
``[TBD-FREEZE`` placeholder scan live here so that ``freeze.py`` and ``check_prereg.py`` agree
byte for byte. Imported by those two scripts only (``sys.path`` insert of the ``scripts`` dir).

Stages (lead decision of 2026-10-10): stage 1 freezes the documents, the configs and the prompt
templates so that trunks and gates can launch; stage 2 (``freeze.py --amend --stage 2``, before
Gate 2 is read and before any lever contrast is computed) adds ``src/finswarm/analysis/**`` and
``src/finswarm/metrics/**`` once the registered estimators have landed. ``FROZEN`` records the
current stage; ``check_prereg.py`` judges the set of that stage only.
"""

from __future__ import annotations

import hashlib
import importlib
from pathlib import Path

PREREG_DIR = Path("docs") / "03_preregistration"
SHA_FILE = PREREG_DIR / "PREREG.sha256"
FROZEN_FILE = PREREG_DIR / "FROZEN"
AMENDMENTS_FILE = PREREG_DIR / "AMENDMENTS.md"

# Documents of the hashed set (companion records such as AMENDMENTS.md are not hashed);
# power_table.md is written by scripts/power.py and hashed when present.
DOCS = ("PREREG.md", "SAP.md", "DETECTION_PROTOCOL.md", "campaign_table.md", "power_table.md")

# Files that must exist before a freeze (PREREG.md section 6; lead decision of 2026-10-10: the
# pinned benchmarks are one JSON per campaign under configs/analysis/benchmarks/, TR-H first).
REQUIRED = tuple(f"{PREREG_DIR.as_posix()}/{d}" for d in DOCS[:4]) + (
    "configs/analysis/registered_arms.yaml",
    "configs/analysis/benchmarks/TR-H.json",
)

# Glob groups of the hashed set, relative to the repo root, by stage.
STAGE1_GLOBS = (
    "configs/sprint/**/*.yaml",
    "configs/sprint/**/*.yml",
    "configs/controls/**/*.yaml",
    "configs/controls/**/*.yml",
    "configs/analysis/**/*.yaml",
    "configs/analysis/**/*.yml",
    "configs/analysis/**/*.json",
    "src/finswarm/agents/llm/prompts/**/*",
)
STAGE2_GLOBS = (
    "src/finswarm/analysis/**/*",
    "src/finswarm/metrics/**/*",
)
STAGES = (1, 2)
GROUP_PREFIXES = (
    "docs/03_preregistration",
    "configs/sprint",
    "configs/controls",
    "configs/analysis",
    "src/finswarm/agents/llm/prompts",
    "src/finswarm/analysis",
    "src/finswarm/metrics",
)
EXCLUDED_PARTS = frozenset({"__pycache__"})
EXCLUDED_SUFFIXES = frozenset({".pyc", ".pyo"})
TBD = "[TBD-FREEZE"
# The literal token used by PREREG.md to describe the convention itself is not a placeholder.
TBD_DOC_TOKEN = "[TBD-FREEZE: ...]"


def rel(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def hashed_set(root: Path, stage: int = 2) -> list[str]:
    """Sorted repo-relative POSIX paths of every file of the hashed set of ``stage`` that exists."""
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")
    found: set[str] = set()
    for doc in DOCS:
        p = root / PREREG_DIR / doc
        if p.is_file():
            found.add(rel(root, p))
    patterns = STAGE1_GLOBS + (STAGE2_GLOBS if stage >= 2 else ())
    for pattern in patterns:
        for p in root.glob(pattern):
            if not p.is_file():
                continue
            if EXCLUDED_PARTS & set(p.parts) or p.suffix in EXCLUDED_SUFFIXES:
                continue
            found.add(rel(root, p))
    return sorted(found)


def group_counts(files: list[str]) -> dict[str, int]:
    counts = dict.fromkeys(GROUP_PREFIXES, 0)
    for f in files:
        for prefix in GROUP_PREFIXES:
            if f.startswith(prefix + "/"):
                counts[prefix] += 1
                break
    return counts


def sha256_lf(path: Path) -> str:
    """SHA-256 of the file with CRLF normalised to LF (the committed content under ``eol=lf``)."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def sha_lines(root: Path, files: list[str]) -> list[str]:
    """``sha256sum``-format lines (``<hex>  <path>``), one per file, sorted by path."""
    return [f"{sha256_lf(root / f)}  {f}" for f in sorted(files)]


def sha_text(lines: list[str]) -> str:
    return "".join(line + "\n" for line in lines)


def digest_of_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_sha_file(text: str) -> dict[str, str]:
    """Parse ``<hex>  <path>`` lines into ``{path: hex}``; blank lines and ``#`` comments skip."""
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        digest, sep, path = line.partition("  ")
        if not sep or len(digest) != 64 or not path.strip():
            raise ValueError(f"malformed PREREG.sha256 line: {raw!r}")
        out[path.strip().lstrip("*")] = digest.lower()
    return out


def missing_required(root: Path) -> list[str]:
    return [f for f in REQUIRED if not (root / f).is_file()]


def find_tbd(root: Path, files: list[str]) -> list[tuple[str, int, str]]:
    """``(path, line number, excerpt)`` for every ``[TBD-FREEZE`` marker in the given files."""
    hits: list[tuple[str, int, str]] = []
    for f in files:
        try:
            text = (root / f).read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for number, line in enumerate(text.splitlines(), 1):
            stripped = line.replace(TBD_DOC_TOKEN, "")
            if TBD in stripped:
                start = stripped.index(TBD)
                hits.append((f, number, stripped[start : start + 90].strip()))
    return hits


def lever_param_problems(root: Path) -> list[str] | None:
    """Registered lever parameters vs the configs, through
    ``finswarm.analysis.arms.check_lever_params(repo_root, registered) -> list[str]``.

    Returns ``None`` while that module is not importable (it lands with the analysis branch), so
    the scripts keep working before the merge; a missing ``registered_arms.yaml`` is reported by
    :func:`missing_required`, not here.
    """
    try:
        arms = importlib.import_module("finswarm.analysis.arms")
    except ImportError:
        return None
    registered = root / "configs" / "analysis" / "registered_arms.yaml"
    if not registered.is_file():
        return []
    return [str(problem) for problem in arms.check_lever_params(root, registered)]


def frozen_digests(text: str) -> list[str]:
    """Every ``sha256 <hex>`` recorded in FROZEN, in order; the last one is the current digest."""
    out: list[str] = []
    for line in text.splitlines():
        parts = line.split()
        for i, token in enumerate(parts[:-1]):
            if token == "sha256" and len(parts[i + 1]) == 64:
                out.append(parts[i + 1].lower())
    return out


def frozen_stage(text: str) -> int:
    """The last ``stage N`` recorded in FROZEN (1 when none is recorded)."""
    stage = 1
    for line in text.splitlines():
        parts = line.split()
        for i, token in enumerate(parts[:-1]):
            if token == "stage" and parts[i + 1].isdigit():
                stage = int(parts[i + 1])
    return stage


def current_stage(root: Path) -> int:
    """Stage recorded in FROZEN, or 1 (the stage a freeze would create) while not frozen."""
    frozen = root / FROZEN_FILE
    return frozen_stage(frozen.read_text(encoding="utf-8")) if frozen.is_file() else 1
