"""Freeze protocol scripts (PREREG.md section 6, D-14): two-stage hashed file set, placeholders,
freeze, check, stage-2 amendment and later amendments, exercised on a throwaway git repository."""

from __future__ import annotations

import hashlib
import importlib.util
import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
PREREG = "docs/03_preregistration"


def _load(name: str):
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))  # the scripts import their private helper by name
    spec = importlib.util.spec_from_file_location(f"_script_{name}", SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


prereg = _load("_prereg")
freeze = _load("freeze")
check = _load("check_prereg")


def _fake_arms(monkeypatch: pytest.MonkeyPatch, problems: list[str] | None) -> None:
    """Stand-in for ``finswarm.analysis.arms`` (``None`` = module not importable yet)."""
    if problems is None:
        monkeypatch.setitem(sys.modules, "finswarm.analysis.arms", None)
        return
    module = types.ModuleType("finswarm.analysis.arms")
    module.check_lever_params = lambda root, registered: list(problems)
    monkeypatch.setitem(sys.modules, "finswarm.analysis.arms", module)


@pytest.fixture(autouse=True)
def lever_params_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_arms(monkeypatch, [])


def _git(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True)


def _write(root: Path, rel: str, text: str) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8", newline="\n")
    return p


STAGE1 = sorted(
    [
        f"{PREREG}/PREREG.md",
        f"{PREREG}/SAP.md",
        f"{PREREG}/DETECTION_PROTOCOL.md",
        f"{PREREG}/campaign_table.md",
        f"{PREREG}/power_table.md",
        "configs/sprint/tier0/TR-H.yaml",
        "configs/controls/SC-1.yaml",
        "configs/analysis/registered_arms.yaml",
        "configs/analysis/benchmarks/TR-H.json",
        "src/finswarm/agents/llm/prompts/dealer_v1.md",
    ]
)
STAGE2 = sorted([*STAGE1, "src/finswarm/analysis/stats.py", "src/finswarm/metrics/collusion.py"])
ANALYSIS = "src/finswarm/analysis/stats.py"
FROZEN = f"{PREREG}/FROZEN"
SHA = f"{PREREG}/PREREG.sha256"


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    for doc in prereg.DOCS:
        _write(root, f"{PREREG}/{doc}", f"# {doc}\nfrozen prose\n")
    _write(root, f"{PREREG}/AMENDMENTS.md", "# Amendments\n\n## Entries\n\n_None yet._\n")
    _write(root, "configs/sprint/tier0/TR-H.yaml", "name: TR-H\n")
    _write(root, "configs/controls/SC-1.yaml", "name: SC-1\n")
    _write(root, "configs/analysis/registered_arms.yaml", "arms: [TR-H]\n")
    _write(root, "configs/analysis/benchmarks/TR-H.json", '{"d7_cartel": 0.58}\n')
    _write(root, "configs/smoke.yaml", "name: smoke\n")  # not in the set
    _write(root, ANALYSIS, "X = 1\n")
    _write(root, "src/finswarm/analysis/__pycache__/stats.cpython-312.pyc", "not hashed\n")
    _write(root, "src/finswarm/metrics/collusion.py", "Y = 2\n")
    _write(root, "src/finswarm/agents/llm/prompts/dealer_v1.md", "You are a dealer.\n")
    _write(root, "src/finswarm/agents/llm/templates.py", "T = 3\n")  # not in the set
    _write(
        root,
        "README.md",
        "# repo\n\nStatus: scaffolding. See docs.\nSecond line.\n\n## Quickstart\n\nrun it\n",
    )
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "tests@example.com")
    _git(root, "config", "user.name", "tests")
    _git(root, "config", "core.autocrlf", "false")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "init")
    return root


def _amendment(repo: Path, entry: str) -> None:
    path = repo / PREREG / "AMENDMENTS.md"
    path.write_text(
        path.read_text(encoding="utf-8") + f"\n### {entry}\n- What changed: test\n",
        encoding="utf-8",
        newline="\n",
    )
    _git(repo, "commit", "-q", "-am", f"amendment {entry}")


def _freeze(repo: Path) -> None:
    assert freeze.main(["--root", str(repo), "--confirm", "FREEZE"]) == 0


def test_hashed_set_by_stage(repo: Path) -> None:
    assert prereg.hashed_set(repo, 1) == STAGE1
    assert prereg.hashed_set(repo, 2) == STAGE2
    assert prereg.hashed_set(repo) == STAGE2  # the full set by default
    with pytest.raises(ValueError):
        prereg.hashed_set(repo, 3)
    lines = prereg.sha_lines(repo, STAGE2)
    assert [line.split("  ", 1)[1] for line in lines] == STAGE2
    assert all(len(line.split("  ", 1)[0]) == 64 for line in lines)
    assert prereg.parse_sha_file(prereg.sha_text(lines)) == {
        f: prereg.sha256_lf(repo / f) for f in STAGE2
    }
    counts = prereg.group_counts(STAGE2)
    assert counts["docs/03_preregistration"] == 5 and counts["src/finswarm/analysis"] == 1
    assert prereg.current_stage(repo) == 1  # not frozen: a freeze would create stage 1


def test_hash_normalises_crlf(tmp_path: Path) -> None:
    lf, crlf = tmp_path / "lf.md", tmp_path / "crlf.md"
    lf.write_bytes(b"x\ny\n")
    crlf.write_bytes(b"x\r\ny\r\n")
    assert prereg.sha256_lf(lf) == prereg.sha256_lf(crlf) == hashlib.sha256(b"x\ny\n").hexdigest()


def test_frozen_stage_parsing() -> None:
    assert prereg.frozen_stage("frozen t\nsha256 x\n") == 1
    assert prereg.frozen_stage("frozen t\nstage 1\namended t sha256 y stage 2\n") == 2
    assert prereg.frozen_stage("stage 2\namended t sha256 y files 3 amendment A-002 stage 2") == 2


def test_check_draft_fails_while_a_placeholder_remains(repo: Path, capsys) -> None:
    _write(
        repo,
        f"{PREREG}/PREREG.md",
        "# PREREG\nEvery `[TBD-FREEZE: ...]` marker must be replaced before the freeze.\n"
        "power = [TBD-FREEZE: power from power.py]\n",
    )
    assert check.main(["--root", str(repo)]) == 1
    out = capsys.readouterr().out
    assert f"TBD      {PREREG}/PREREG.md:3: [TBD-FREEZE: power from power.py]" in out
    assert f"{PREREG}/PREREG.md:2:" not in out  # the rule's own example token is not a placeholder
    assert "DRAFT: not frozen; 10 files in the stage-1 set, 1 placeholder(s)" in out
    assert check.main(["--root", str(repo), "--ci"]) == 0  # a draft never breaks CI
    assert check.main(["--root", str(repo), "--strict"]) == 1
    _write(repo, f"{PREREG}/PREREG.md", "# PREREG\npower = 0.81\n")
    assert check.main(["--root", str(repo)]) == 0
    assert check.main(["--root", str(repo), "--strict"]) == 1  # still not frozen


def test_check_lists_the_hashed_set_by_stage(repo: Path, capsys) -> None:
    assert check.main(["--root", str(repo), "--list"]) == 0
    assert capsys.readouterr().out.split() == STAGE1  # not frozen: the stage a freeze creates
    assert check.main(["--root", str(repo), "--list", "--stage", "2"]) == 0
    assert capsys.readouterr().out.split() == STAGE2


def test_freeze_dry_run_reports_blockers_and_refuses(repo: Path, capsys) -> None:
    (repo / "configs/analysis/benchmarks/TR-H.json").unlink()
    _write(repo, f"{PREREG}/SAP.md", "# SAP\n[TBD-FREEZE: D7_cartel]\n")  # also dirties the tree
    assert freeze.main(["--root", str(repo), "--dry-run"]) == 1
    out = capsys.readouterr().out
    assert "stage            : 1" in out
    assert "BLOCKED          : required file missing: configs/analysis/benchmarks/TR-H.json" in out
    assert f"BLOCKED          : placeholder {PREREG}/SAP.md:2: [TBD-FREEZE: D7_cartel]" in out
    assert "uncommitted tracked changes" in out
    assert freeze.main(["--root", str(repo), "--confirm", "FREEZE"]) == 1
    assert not (repo / FROZEN).exists()
    # --stage belongs to --amend
    assert freeze.main(["--root", str(repo), "--dry-run", "--stage", "2"]) == 1
    assert "--stage 2 requires --amend" in capsys.readouterr().out


def test_stage_1_freeze_ignores_analysis_code_then_checks_the_rest(repo: Path, capsys) -> None:
    assert freeze.main(["--root", str(repo), "--dry-run"]) == 0
    assert freeze.main(["--root", str(repo)]) == 2  # no confirmation word
    assert not (repo / FROZEN).exists()
    _freeze(repo)
    shaf, frozen = repo / SHA, repo / FROZEN
    assert list(prereg.parse_sha_file(shaf.read_text(encoding="utf-8"))) == STAGE1
    digest = prereg.sha256_lf(shaf)
    frozen_text = frozen.read_text(encoding="utf-8")
    assert prereg.frozen_digests(frozen_text) == [digest]
    assert "files 10" in frozen_text and frozen_text.rstrip().endswith("stage 1")
    assert prereg.current_stage(repo) == 1
    readme = (repo / "README.md").read_text(encoding="utf-8")
    assert (
        f"Pre-registration freeze: stage 1, `PREREG.sha256` sha256 `{digest}` (10 files)" in readme
    )
    assert (
        readme.index("Second line.")
        < readme.index("Pre-registration freeze")
        < readme.index("## Quickstart")
    )
    assert _git(repo, "tag", "-l", "prereg-freeze").stdout.strip() == "prereg-freeze"
    assert _git(repo, "status", "--porcelain").stdout == ""  # everything committed
    capsys.readouterr()
    assert check.main(["--root", str(repo), "--strict"]) == 0
    out = capsys.readouterr().out
    assert "FROZEN stage 1 and consistent: 10 files" in out and "stage 2 pending" in out
    assert freeze.main(["--root", str(repo), "--confirm", "FREEZE"]) == 1  # never twice

    # analysis and metrics code is not in the stage-1 set: changes and additions pass
    _write(repo, ANALYSIS, "X = 2  # estimator landed after stage 1\n")
    _write(repo, "src/finswarm/metrics/new_estimator.py", "M = 6\n")
    assert check.main(["--root", str(repo), "--strict"]) == 0
    assert check.main(["--root", str(repo), "--ci"]) == 0

    # everything else is judged
    _write(repo, "configs/sprint/tier0/TR-H.yaml", "name: TR-H\nseed: 7\n")
    capsys.readouterr()
    assert check.main(["--root", str(repo), "--ci"]) == 1
    assert "modified after freeze: configs/sprint/tier0/TR-H.yaml" in capsys.readouterr().out
    _git(repo, "restore", "configs/sprint/tier0/TR-H.yaml")
    _write(repo, "configs/sprint/tier1/TR-M.yaml", "name: TR-M\n")
    capsys.readouterr()
    assert check.main(["--root", str(repo)]) == 1
    assert "outside the hashed set" in capsys.readouterr().out
    (repo / "configs/sprint/tier1/TR-M.yaml").unlink()
    (repo / "src/finswarm/agents/llm/prompts/dealer_v1.md").unlink()
    capsys.readouterr()
    assert check.main(["--root", str(repo)]) == 1
    assert "deleted after freeze: src/finswarm/agents/llm/prompts/dealer_v1.md" in (
        capsys.readouterr().out
    )


def test_stage_2_amendment_hashes_analysis_code_once(repo: Path, capsys) -> None:
    _freeze(repo)
    _write(repo, ANALYSIS, "X = 2  # registered estimator\n")
    _git(repo, "commit", "-q", "-am", "estimators land")
    stage2 = ["--root", str(repo), "--amend", "--amendment", "A-001", "--stage", "2"]
    assert freeze.main([*stage2, "--confirm", "AMEND"]) == 1
    assert "AMENDMENTS.md has no entry '### A-001'" in capsys.readouterr().out
    _amendment(repo, "A-001")
    assert freeze.main([*stage2, "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "stage            : 2 (FROZEN records 1)" in out and "12 files" in out
    assert freeze.main([*stage2, "--confirm", "AMEND"]) == 0
    shaf, frozen = repo / SHA, repo / FROZEN
    assert list(prereg.parse_sha_file(shaf.read_text(encoding="utf-8"))) == STAGE2
    frozen_text = frozen.read_text(encoding="utf-8")
    assert prereg.current_stage(repo) == 2 and "amendment A-001" in frozen_text
    marks = prereg.frozen_digests(frozen_text)
    assert len(marks) == 2 and marks[-1] == prereg.sha256_lf(shaf)
    assert "stage 2," in (repo / "README.md").read_text(encoding="utf-8")
    assert _git(repo, "status", "--porcelain").stdout == ""
    assert _git(repo, "tag", "-l").stdout.split() == ["prereg-freeze"]  # amendments add no tag
    capsys.readouterr()
    assert check.main(["--root", str(repo), "--strict"]) == 0
    assert "FROZEN stage 2 and consistent: 12 files" in capsys.readouterr().out

    # from now on the analysis code is judged
    _write(repo, ANALYSIS, "X = 3\n")
    capsys.readouterr()
    assert check.main(["--root", str(repo), "--ci"]) == 1
    assert f"modified after freeze: {ANALYSIS}" in capsys.readouterr().out

    # stage 2 is accepted once; a documented change is a plain amendment at stage 2
    _git(repo, "commit", "-q", "-am", "fix an estimator")
    _amendment(repo, "A-002")
    again = ["--root", str(repo), "--amend", "--amendment", "A-002", "--stage", "2"]
    assert freeze.main([*again, "--confirm", "AMEND"]) == 1
    assert "stage 2 already recorded" in capsys.readouterr().out
    plain = ["--root", str(repo), "--amend", "--amendment", "A-002", "--confirm", "AMEND"]
    assert freeze.main(plain) == 0
    assert prereg.current_stage(repo) == 2
    assert len(prereg.frozen_digests(frozen.read_text(encoding="utf-8"))) == 3
    capsys.readouterr()
    assert check.main(["--root", str(repo), "--strict"]) == 0
    assert "2 amendment(s)" in capsys.readouterr().out
    with pytest.raises(SystemExit):  # only stage 2 exists as a target
        freeze.main(["--root", str(repo), "--amend", "--amendment", "A-003", "--stage", "1"])


def test_lever_parameter_check_contract(repo: Path, capsys, monkeypatch) -> None:
    problem = "L1-H: tau 0.0625 differs from the registered 0.125"
    _fake_arms(monkeypatch, [problem])
    assert check.main(["--root", str(repo)]) == 1
    out = capsys.readouterr().out
    assert f"LEVER    {problem}" in out and "1 lever parameter problem(s)" in out
    assert check.main(["--root", str(repo), "--ci"]) == 0  # not frozen: CI stays green
    assert check.main(["--root", str(repo), "--strict"]) == 1
    capsys.readouterr()
    assert freeze.main(["--root", str(repo), "--dry-run"]) == 1
    assert f"BLOCKED          : lever parameter: {problem}" in capsys.readouterr().out

    _fake_arms(monkeypatch, [])
    _freeze(repo)
    _fake_arms(monkeypatch, [problem])
    capsys.readouterr()
    assert check.main(["--root", str(repo), "--ci"]) == 1  # frozen: CI fails on it
    assert f"lever parameter: {problem}" in capsys.readouterr().out
    assert check.main(["--root", str(repo)]) == 1

    _fake_arms(monkeypatch, None)  # module not merged yet: the hashed set is still verified
    capsys.readouterr()
    assert check.main(["--root", str(repo), "--strict"]) == 0
    assert "NOTE     lever parameters not checked" in capsys.readouterr().out
