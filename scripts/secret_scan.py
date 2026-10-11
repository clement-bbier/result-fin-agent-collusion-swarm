"""Scan for probable secrets: sk-ant- keys, AIza keys, private keys, *_API_KEY=<value>.

Default scans tracked files; --staged scans the index diff; --all-history scans `git log -p`.
Exit 1 when something is found (values are redacted in the output).
"""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
from pathlib import Path

PATTERNS = {
    "anthropic-key": re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}"),
    "google-key": re.compile(r"AIza[0-9A-Za-z_\-]{30,}"),
    "private-key": re.compile(r"-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----"),
    "api-key-assignment": re.compile(
        r"\b[A-Z][A-Z0-9_]*API_KEY\s*[=:]\s*['\"]?(?![\s'\"]|your|xxx|changeme|<)[A-Za-z0-9_\-\.]{16,}"
    ),
}
SKIP_SUFFIX = {".png", ".jpg", ".pdf", ".parquet", ".sqlite", ".lock", ".gz", ".zip"}
# Known test fixtures that merely look like keys, identified by the SHA-256 of the exact match so
# that this file never contains them: the fake Anthropic key of tests/unit/test_peek.py (committed
# literally at d016a9d, now built by concatenation; history is never rewritten).
ALLOWLIST_SHA256 = frozenset({"8e2a423c26ff96c562bd7051147ae416d8f9760392d8c8b6056e2ff900845181"})


def allowlisted(match: str) -> bool:
    return hashlib.sha256(match.encode("utf-8")).hexdigest() in ALLOWLIST_SHA256


def scan_text(text: str, where: str) -> list[str]:
    hits = []
    for i, line in enumerate(text.splitlines(), 1):
        for name, rx in PATTERNS.items():
            m = rx.search(line)
            if m and not allowlisted(m.group(0)):
                hits.append(f"{where}:{i}: {name}")
    return hits


def git(*args: str) -> str:
    cp = subprocess.run(
        ["git", *args], capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    return cp.stdout


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="*", help="explicit files to scan")
    ap.add_argument("--staged", action="store_true")
    ap.add_argument("--all-history", action="store_true")
    args = ap.parse_args(argv)
    hits: list[str] = []
    if args.all_history:
        hits += scan_text(git("log", "-p", "--all", "--no-color"), "history")
    elif args.staged:
        for line in git("diff", "--cached", "-U0", "--no-color").splitlines():
            if line.startswith("+") and not line.startswith("+++"):
                hits += scan_text(line[1:], "staged")
    else:
        files = args.paths or git("ls-files").splitlines()
        for f in files:
            p = Path(f)
            if p.suffix in SKIP_SUFFIX or not p.is_file():
                continue
            try:
                hits += scan_text(p.read_text(encoding="utf-8", errors="ignore"), f)
            except OSError:
                continue
    for h in hits[:50]:
        print(h)
    print(f"{len(hits)} probable secret(s) found" if hits else "clean")
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())
