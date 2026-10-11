"""Registered-launch pre-check (lead decision, 2026-10-10): FROZEN present, HEAD pushed, no
modified or staged tracked file; untracked files under results/** and cache/** are ignored,
any other untracked file blocks and is listed. Exercised on a throwaway git repository with a
bare remote."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from fakes import base_config

from finswarm.config.schema import CampaignConfig
from finswarm.sim.campaign import check_prereg, registered_launch_problems


def _git(root: Path, *args: str) -> str:
    cp = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True)
    return cp.stdout.strip()


def _write(root: Path, rel: str, text: str = "x\n") -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A checkout with one pushed commit, FROZEN present, and a bare remote ``origin``."""
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.email", "t@example.com")
    _git(root, "config", "user.name", "t")
    _write(root, "README.md", "hi\n")
    _write(root, "docs/03_preregistration/FROZEN", "frozen\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "init")
    remote = tmp_path / "origin.git"
    _git(root, "init", "-q", "--bare", str(remote))
    _git(root, "remote", "add", "origin", str(remote))
    _git(root, "push", "-q", "-u", "origin", "main")
    return root


def registered() -> CampaignConfig:
    base = base_config().model_dump(mode="json")
    return CampaignConfig.model_validate(
        {"name": "camp", "base": base, "seeds": [0], "prereg_id": "PREREG-v1"}
    )


def test_clean_pushed_frozen_checkout_may_launch(repo: Path) -> None:
    assert registered_launch_problems(repo) == []
    check_prereg(registered(), root=repo)
    unregistered = registered().model_copy(update={"prereg_id": None})
    check_prereg(unregistered, root=repo / "nowhere")  # never guarded


def test_untracked_run_data_is_ignored_but_other_untracked_files_block(repo: Path) -> None:
    _write(repo, "results/manifests/TR-H__s0__abc.json", "{}\n")
    _write(repo, "results/campaigns/TR-H/x/part-0.parquet", "")
    _write(repo, "cache/llm_cache.sqlite", "")
    assert registered_launch_problems(repo) == []
    _write(repo, "notes.txt")
    _write(repo, "configs/sprint/tier0/NEW.yaml")
    problems = registered_launch_problems(repo)
    assert [p for p in problems if p.startswith("untracked file")] == [
        "untracked file configs/sprint/tier0/NEW.yaml (commit, delete or ignore it)",
        "untracked file notes.txt (commit, delete or ignore it)",
    ]
    with pytest.raises(RuntimeError, match="notes.txt"):
        check_prereg(registered(), root=repo)


def test_modified_or_staged_tracked_files_block(repo: Path) -> None:
    _write(repo, "README.md", "changed\n")
    (problem,) = registered_launch_problems(repo)
    assert problem.startswith("modified or staged tracked file README.md")
    _git(repo, "add", "README.md")
    (problem,) = registered_launch_problems(repo)
    assert "README.md" in problem
    _git(repo, "commit", "-q", "-m", "edit")  # committed but not pushed yet
    (problem,) = registered_launch_problems(repo)
    assert "not pushed" in problem
    _git(repo, "push", "-q", "origin", "main")
    assert registered_launch_problems(repo) == []


def test_missing_frozen_marker_blocks(repo: Path) -> None:
    _git(repo, "rm", "-q", "docs/03_preregistration/FROZEN")
    _git(repo, "commit", "-q", "-m", "unfreeze")
    _git(repo, "push", "-q", "origin", "main")
    (problem,) = registered_launch_problems(repo)
    assert "FROZEN is absent" in problem
    with pytest.raises(RuntimeError, match="FROZEN"):
        check_prereg(registered(), root=repo)


def test_outside_a_git_checkout_everything_blocks(tmp_path: Path) -> None:
    problems = registered_launch_problems(tmp_path / "plain")
    assert any("FROZEN" in p for p in problems) and any("git" in p for p in problems)


def test_modified_tracked_manifest_under_results_does_not_block(repo: Path) -> None:
    """A resumed run rewrites its committed manifest; run data never blocks a launch."""
    man = repo / "results" / "manifests"
    man.mkdir(parents=True)
    (man / "TR-H--S2__s0__abc.json").write_text('{"status": "paused"}', encoding="utf-8")
    _git(repo, "add", "results/manifests/TR-H--S2__s0__abc.json")
    _git(repo, "commit", "-q", "-m", "manifests")
    _git(repo, "push", "-q", "origin", "HEAD")
    (man / "TR-H--S2__s0__abc.json").write_text('{"status": "ok"}', encoding="utf-8")
    assert registered_launch_problems(repo) == []
