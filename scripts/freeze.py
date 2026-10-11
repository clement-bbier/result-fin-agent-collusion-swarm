"""Freeze the pre-registration (IRREVERSIBLE): hash the D-14 file set, write PREREG.sha256 and
FROZEN, record the digest in the README, snapshot pilots, commit, tag prereg-freeze.

Two stages (lead decision of 2026-10-10). A freeze is stage 1: the four documents (plus
power_table.md), every config under configs/sprint|controls|analysis and the prompt templates.
Stage 2 adds src/finswarm/analysis/** and src/finswarm/metrics/** before Gate 2 is read:
    python scripts/freeze.py --amend --amendment A-001 --stage 2 --confirm AMEND
and is accepted once; later documented changes are plain amendments at the current stage:
    python scripts/freeze.py --amend --amendment A-002 --confirm AMEND
Every amendment needs its "### A-NNN" entry in AMENDMENTS.md and appends a line to FROZEN.

Never pushes; the /freeze-prereg skill pushes after a literal human confirmation. A real freeze
requires --confirm FREEZE. --dry-run prints the checklist and exits 1 while anything blocks.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _prereg import (  # noqa: E402
    AMENDMENTS_FILE,
    FROZEN_FILE,
    PREREG_DIR,
    SHA_FILE,
    current_stage,
    digest_of_text,
    find_tbd,
    group_counts,
    hashed_set,
    lever_param_problems,
    missing_required,
    sha_lines,
    sha_text,
)

TAG = "prereg-freeze"
README_PREFIX = "Pre-registration freeze:"


def git(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)


def readme_line(digest: str, n_files: int, stamp: str, stage: int, amendment: str | None) -> str:
    line = (
        f"{README_PREFIX} stage {stage}, `PREREG.sha256` sha256 `{digest}` ({n_files} files), "
        f"tag `{TAG}`, {stamp}."
    )
    if amendment:
        line += f" Amended {amendment}."
    return line


def update_readme(root: Path, line: str) -> bool:
    """Replace the existing freeze line, or insert one after the ``Status:`` paragraph."""
    readme = root / "README.md"
    if not readme.is_file():
        return False
    lines = readme.read_text(encoding="utf-8").splitlines()
    existing = next((i for i, text in enumerate(lines) if text.startswith(README_PREFIX)), None)
    if existing is not None:
        lines[existing] = line
    else:
        status = next((i for i, text in enumerate(lines) if text.startswith("Status:")), None)
        if status is None:
            lines += ["", line]
        else:
            end = status
            while end + 1 < len(lines) and lines[end + 1].strip():
                end += 1
            lines[end + 1 : end + 1] = ["", line]
    readme.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", default=".")
    ap.add_argument("--dry-run", action="store_true", help="print the checklist, write nothing")
    ap.add_argument("--confirm", default="", help="FREEZE for a freeze, AMEND for an amendment")
    ap.add_argument("--amend", action="store_true", help="re-hash after a registered amendment")
    ap.add_argument("--amendment", default="", help="AMENDMENTS.md entry id (A-NNN) for --amend")
    ap.add_argument(
        "--stage",
        type=int,
        choices=[2],
        default=None,
        help="with --amend: move to stage 2 (analysis and metrics code enter the set); once",
    )
    args = ap.parse_args(argv)
    root = Path(args.root).resolve()
    prereg = root / PREREG_DIR / "PREREG.md"
    shaf, frozen = root / SHA_FILE, root / FROZEN_FILE
    mode = "amend" if args.amend else "freeze"
    if not prereg.exists():
        print("PREREG.md missing")
        return 1

    blockers: list[str] = []
    recorded_stage = current_stage(root)
    if mode == "freeze":
        stage = 1
        if frozen.exists():
            blockers.append("already frozen (FROZEN exists); a registered change uses --amend")
        if git(root, "rev-parse", "-q", "--verify", f"refs/tags/{TAG}").returncode == 0:
            blockers.append(f"tag {TAG} already exists")
        if args.stage is not None:
            blockers.append("--stage 2 requires --amend (a freeze is always stage 1)")
    else:
        stage = args.stage if args.stage is not None else recorded_stage
        if not frozen.exists():
            blockers.append("not frozen yet (FROZEN missing); run a freeze first")
        elif args.stage is not None and recorded_stage >= args.stage:
            blockers.append(
                f"stage {args.stage} already recorded in FROZEN; later changes are plain amendments"
            )
        if not re.fullmatch(r"A-\d{3}", args.amendment):
            blockers.append("--amendment A-NNN is required with --amend")
        else:
            amendments = root / AMENDMENTS_FILE
            text = amendments.read_text(encoding="utf-8") if amendments.is_file() else ""
            if f"### {args.amendment}" not in text:
                blockers.append(f"AMENDMENTS.md has no entry '### {args.amendment}'")
    files = hashed_set(root, stage)
    for path in missing_required(root):
        blockers.append(f"required file missing: {path}")
    for path, number, excerpt in find_tbd(root, files):
        blockers.append(f"placeholder {path}:{number}: {excerpt}")
    lever = lever_param_problems(root)
    blockers.extend(f"lever parameter: {problem}" for problem in lever or [])
    # run data, manifests and caches are the runner's own output (a resumed run rewrites its
    # committed manifest) and are never part of the hashed set: they do not block a freeze
    dirty = [
        line
        for line in git(root, "status", "--porcelain", "--untracked-files=no").stdout.splitlines()
        if line.strip() and not line[3:].strip().strip('"').startswith(("results/", "cache/"))
    ]
    if dirty:
        blockers.append(f"working tree has uncommitted tracked changes ({len(dirty)} file(s))")
    untracked = [
        line
        for line in git(root, "status", "--porcelain", "--untracked-files=all").stdout.splitlines()
        if line.startswith("??")
    ]

    branch = git(root, "branch", "--show-current").stdout.strip()
    head = git(root, "rev-parse", "--short", "HEAD").stdout.strip()
    lines = sha_lines(root, files)
    text = sha_text(lines)
    digest = digest_of_text(text)
    pilots = root / "results" / "pilots"
    groups = ", ".join(f"{k} {v}" for k, v in group_counts(files).items() if v or stage == 2)
    print(f"mode             : {mode}")
    print(
        f"stage            : {stage}"
        + (f" (FROZEN records {recorded_stage})" if frozen.exists() else "")
    )
    print(f"hashed file set  : {len(files)} files ({groups})")
    print(f"PREREG.sha256    : {digest}  (sha256 of the file to be written)")
    on_main = " WARNING: on the protected branch" if branch in ("main", "master") else ""
    print(f"branch / commit  : {branch} @ {head}{on_main}")
    print(f"pilots to freeze : {pilots if pilots.exists() else 'none'}")
    print(f"tag              : {TAG if mode == 'freeze' else 'none (amendment)'}")
    if untracked:
        print(
            f"note             : {len(untracked)} untracked file(s) are neither hashed nor committed"
        )
    for blocker in blockers:
        print(f"BLOCKED          : {blocker}")
    if args.dry_run:
        print("dry run: nothing written" + ("; fix the blockers above" if blockers else ""))
        return 1 if blockers else 0
    if blockers:
        print("refusing: fix the blockers above")
        return 1
    word = "AMEND" if mode == "amend" else "FREEZE"
    if args.confirm != word:
        print(f"refusing: pass --confirm {word} (after explicit human confirmation)")
        return 2

    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    shaf.write_text(text, encoding="utf-8", newline="\n")
    if mode == "freeze":
        frozen.write_text(
            f"frozen {stamp}\nsha256 {digest}\nfiles {len(files)}\ncommit {head}\nstage 1\n",
            encoding="utf-8",
            newline="\n",
        )
    else:
        with frozen.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(
                f"amended {stamp} sha256 {digest} files {len(files)} "
                f"amendment {args.amendment} commit {head} stage {stage}\n"
            )
    paths = [SHA_FILE.as_posix(), FROZEN_FILE.as_posix()]
    amendment = args.amendment if mode == "amend" else None
    if update_readme(root, readme_line(digest, len(files), stamp, stage, amendment)):
        paths.append("README.md")
    if mode == "freeze" and pilots.exists():
        shutil.copytree(pilots, root / "results" / "frozen" / "pilots", dirs_exist_ok=True)
        paths.append("results/frozen")
    if mode == "freeze":
        message = f"chore(prereg): freeze pre-registration stage 1 {digest[:12]}"
    elif args.stage is not None:
        message = (
            f"chore(prereg): freeze stage {stage}, analysis and metrics code hashed "
            f"({args.amendment}) {digest[:12]}"
        )
    else:
        message = f"chore(prereg): amend pre-registration hash ({args.amendment}) {digest[:12]}"
    commands = [["add", "-f", *paths], ["commit", "-m", message]]
    if mode == "freeze":
        commands.append(
            ["tag", "-a", TAG, "-m", f"pre-registration frozen (stage 1) {stamp} sha256 {digest}"]
        )
    for cmd in commands:
        cp = git(root, *cmd)
        if cp.returncode != 0:
            print(f"git {' '.join(cmd[:2])} failed:\n{cp.stderr}")
            return 1
    done = "frozen (stage 1)" if mode == "freeze" else f"amended (stage {stage})"
    print(f"{done}. Push with: git push origin HEAD --follow-tags")
    return 0


if __name__ == "__main__":
    sys.exit(main())
