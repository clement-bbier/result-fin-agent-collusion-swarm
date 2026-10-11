"""scripts/secret_scan.py: key-looking strings are reported, known test fixtures are allowlisted."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load():
    spec = importlib.util.spec_from_file_location(
        "_script_secret_scan", ROOT / "scripts" / "secret_scan.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


scan = _load()
# The fixture of tests/unit/test_peek.py, built by concatenation so that no literal key-looking
# string sits in the tree; its sha256 is the allowlist entry.
FIXTURE = "sk-ant-" + "secret" * 4


def test_fixture_is_allowlisted_and_real_looking_keys_are_not(tmp_path: Path, capsys) -> None:
    assert scan.allowlisted(FIXTURE)
    assert scan.scan_text(f"error = '429 {FIXTURE}'\n", "x") == []
    other = "sk-ant-" + "a1B2c3D4e5F6g7H8i9J0" + "k"
    assert scan.scan_text(f"key = {other}\n", "x") == ["x:1: anthropic-key"]
    assert scan.scan_text("AIza" + "Q" * 35 + "\n", "y") == ["y:1: google-key"]
    assert scan.scan_text("ANTHROPIC_API_KEY=\n", "z") == []  # empty assignment is fine

    f = tmp_path / "leak.txt"
    f.write_text(f"token: {other}\nfixture: {FIXTURE}\n", encoding="utf-8")
    assert scan.main([str(f)]) == 1
    out = capsys.readouterr().out
    assert "leak.txt:1: anthropic-key" in out and "leak.txt:2" not in out
    assert other not in out  # values are never printed
    f.write_text(f"fixture: {FIXTURE}\n", encoding="utf-8")
    assert scan.main([str(f)]) == 0
