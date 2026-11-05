"""Offline tests for ``scripts/release_check.py`` on synthetic repositories."""
import importlib.util
import subprocess
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "release_check.py"
if not _SCRIPT.exists():  # the wheel smoke harness runs a copy of tests/ outside the repository
    pytest.skip("scripts/release_check.py is repo tooling, not part of the wheel", allow_module_level=True)
_SPEC = importlib.util.spec_from_file_location("release_check", _SCRIPT)
release_check = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(release_check)

SETUP = """RUNTIME_REQUIRES = [
    "pandas>=2.3.3",
]
DEV_REQUIRES = [
    "ruff>=0.17.3",
]
setup(name='x', version='1.4.0')
"""
PRECOMMIT = """repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.17.3
  - repo: local
    hooks:
      - id: pylint
        additional_dependencies:
          - pandas>=2.3.3
"""
CHANGELOG = """# Changelog

## [Unreleased]

{unreleased}
## [v1.4.0] — 2026-11-05

- something
"""
CITATION = "version: {version}\ndate-released: {date}\n"
README = "  version   = {{{version}}},\n"


def _git(root, *args):
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


def _repo(tmp_path, monkeypatch, *, requirements="pandas>=2.3.3\nruff>=0.17.3\n", unreleased="",
          cff_version="1.4.0", date="2026-11-05", readme_version="1.4.0"):
    (tmp_path / "setup.py").write_text(SETUP)
    (tmp_path / "requirements.txt").write_text(requirements)
    (tmp_path / ".pre-commit-config.yaml").write_text(PRECOMMIT)
    (tmp_path / "CHANGELOG.md").write_text(CHANGELOG.format(unreleased=unreleased))
    (tmp_path / "CITATION.cff").write_text(CITATION.format(version=cff_version, date=date))
    (tmp_path / "README.md").write_text(README.format(version=readme_version))
    _git(tmp_path, "init", "-q")
    monkeypatch.setattr(release_check, "ROOT", tmp_path)
    monkeypatch.setattr(release_check, "PROBLEMS", [])
    return tmp_path


@pytest.mark.core
def test_consistent_release_metadata_passes(tmp_path, monkeypatch):
    _repo(tmp_path, monkeypatch)
    release_check.check_versions("HEAD")
    release_check.check_deps("HEAD")
    assert not release_check.PROBLEMS


@pytest.mark.core
def test_version_date_and_unreleased_drift_is_reported(tmp_path, monkeypatch):
    _repo(tmp_path, monkeypatch, unreleased="### Added\n\n- late feature\n", cff_version="1.3.2",
          date="2026-11-03", readme_version="1.3.9")
    release_check.check_versions("HEAD")
    text = "\n".join(release_check.PROBLEMS)
    assert "CITATION.cff says 1.3.2" in text and "README citation says 1.3.9" in text
    assert "date 2026-11-05 != CITATION.cff date-released 2026-11-03" in text
    assert "[Unreleased] is not empty" in text


def _tagged_repo(tmp_path, monkeypatch, changed_file):
    root = _repo(tmp_path, monkeypatch)
    _git(root, "add", "-A")
    _git(root, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init")
    _git(root, "tag", "v1.4.0")
    (root / changed_file).parent.mkdir(exist_ok=True)
    (root / changed_file).write_text("changed\n")
    _git(root, "add", "-A")
    _git(root, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "after the tag")


@pytest.mark.core
def test_existing_tag_with_package_changes_means_the_version_was_not_bumped(tmp_path, monkeypatch):
    _tagged_repo(tmp_path, monkeypatch, "fractal/new_module.py")
    release_check.check_versions("HEAD")
    assert release_check.PROBLEMS == [
        "[versions] tag v1.4.0 already exists and the package changed since: bump the version"
    ]


@pytest.mark.core
def test_docs_only_follow_up_to_a_tagged_release_passes(tmp_path, monkeypatch):
    _tagged_repo(tmp_path, monkeypatch, "docs/guide.md")
    release_check.check_versions("HEAD")
    assert not release_check.PROBLEMS


@pytest.mark.core
def test_dependency_floor_drift_is_reported(tmp_path, monkeypatch):
    _repo(tmp_path, monkeypatch, requirements="pandas>=2.2.0\nruff>=0.17.3\n")
    release_check.check_deps("HEAD")
    assert release_check.PROBLEMS == ["[deps] pandas: setup.py=pandas>=2.3.3 requirements.txt=pandas>=2.2.0"]


@pytest.mark.core
def test_example_imports_that_no_longer_resolve_are_reported(tmp_path, monkeypatch):
    root = _repo(tmp_path, monkeypatch)
    (root / "examples").mkdir()
    (root / "examples" / "ok.py").write_text("from fractal.loaders import PriceHistory\n")
    (root / "examples" / "broken.py").write_text("from fractal.loaders import NoSuchLoader\n")
    _git(root, "add", "examples")
    release_check.check_examples("HEAD")
    assert release_check.PROBLEMS == ["[examples] examples/broken.py:1: fractal.loaders has no NoSuchLoader"]


@pytest.mark.core
def test_the_real_repository_passes_the_offline_structure_checks():
    """docs pages, ``__all__`` entries and dependency floors hold on every commit, not only at release time."""
    problems_before = len(release_check.PROBLEMS)
    for check in (release_check.check_deps, release_check.check_docs, release_check.check_exports):
        check("HEAD")
    assert release_check.PROBLEMS[problems_before:] == []
