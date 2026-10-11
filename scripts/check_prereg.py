"""Check the pre-registration (PREREG.md section 6, D-14): hashed file set vs PREREG.sha256,
FROZEN marker (with its stage) and [TBD-FREEZE] placeholders.

The set judged is the one of the stage recorded in FROZEN (stage 1 while not frozen): at stage 1
the analysis and metrics code is not in the set yet and is neither hashed nor flagged; at stage 2
everything is. Exit 0 when consistent, 1 when a check fails. Default mode (used by
/prereg-check) fails while a placeholder remains or a registered lever parameter disagrees with
its config (finswarm.analysis.arms.check_lever_params) and, once frozen, on any hashed file
that was modified, deleted or added without an amendment. --strict also fails while not frozen.
--ci fails only on a frozen set (a DRAFT passes). --list prints the hashed file set
(--stage 1|2 to pick the stage) and exits.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _prereg import (  # noqa: E402
    FROZEN_FILE,
    PREREG_DIR,
    SHA_FILE,
    STAGES,
    current_stage,
    find_tbd,
    frozen_digests,
    hashed_set,
    lever_param_problems,
    missing_required,
    parse_sha_file,
    sha256_lf,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", default=".")
    ap.add_argument("--strict", action="store_true", help="fail unless frozen and consistent")
    ap.add_argument("--ci", action="store_true", help="fail only on a frozen-but-modified set")
    ap.add_argument("--list", action="store_true", help="print the hashed file set and exit")
    ap.add_argument(
        "--stage", type=int, choices=list(STAGES), default=None, help="stage for --list"
    )
    args = ap.parse_args(argv)
    root = Path(args.root).resolve()
    stage = args.stage if args.stage is not None else current_stage(root)
    files = hashed_set(root, stage)
    if args.list:
        print("\n".join(files))
        return 0
    if not (root / PREREG_DIR / "PREREG.md").is_file():
        print("PREREG.md missing")
        return 0 if args.ci else 1

    tbd = find_tbd(root, files)
    missing = missing_required(root)
    lever = lever_param_problems(root)
    for path, number, excerpt in tbd:
        print(f"TBD      {path}:{number}: {excerpt}")
    for path in missing:
        print(f"MISSING  {path} (required in the hashed set before freeze)")
    if lever is None:
        print("NOTE     lever parameters not checked (finswarm.analysis.arms not available yet)")
    for problem in lever or []:
        print(f"LEVER    {problem}")

    frozen, shaf = root / FROZEN_FILE, root / SHA_FILE
    if not frozen.exists():
        print(
            f"DRAFT: not frozen; {len(files)} files in the stage-1 set, "
            f"{len(tbd)} placeholder(s), {len(missing)} required file(s) missing, "
            f"{len(lever or [])} lever parameter problem(s)"
        )
        if args.ci:
            return 0
        if args.strict:
            return 1
        return 1 if tbd or lever else 0

    if not shaf.exists():
        print("FAIL: FROZEN present but PREREG.sha256 missing")
        return 1
    try:
        recorded = parse_sha_file(shaf.read_text(encoding="utf-8"))
    except ValueError as exc:
        print(f"FAIL: {exc}")
        return 1

    problems: list[str] = []
    for path, digest in recorded.items():
        p = root / path
        if not p.is_file():
            problems.append(f"deleted after freeze: {path}")
        elif sha256_lf(p) != digest:
            problems.append(f"modified after freeze: {path}")
    for path in files:
        if path not in recorded:
            problems.append(f"outside the hashed set (added after freeze, no amendment): {path}")
    if tbd:
        problems.append(f"{len(tbd)} [TBD-FREEZE] placeholder(s) in a frozen set")
    problems.extend(f"lever parameter: {problem}" for problem in lever or [])
    current = sha256_lf(shaf)
    marks = frozen_digests(frozen.read_text(encoding="utf-8"))
    if not marks:
        problems.append("FROZEN records no sha256 of PREREG.sha256")
    elif marks[-1] != current:
        problems.append(
            "PREREG.sha256 changed without a FROZEN amendment line "
            f"(recorded {marks[-1][:12]}, current {current[:12]}); use scripts/freeze.py --amend"
        )
    if problems:
        print(f"FAIL (stage {stage}): " + "\n      ".join(problems))
        return 1
    amended = f", {len(marks) - 1} amendment(s)" if len(marks) > 1 else ""
    pending = " (stage 2 pending: analysis and metrics code not hashed yet)" if stage == 1 else ""
    print(
        f"FROZEN stage {stage} and consistent: {len(recorded)} files, "
        f"PREREG.sha256 sha256 {current}{amended}{pending}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
