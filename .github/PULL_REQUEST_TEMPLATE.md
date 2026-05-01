<!--
Thanks for the PR! Keep the title short and descriptive (under 70 chars).
Detailed context goes in the body below.
-->

## Summary

<!-- One or two sentences: what changed and why. Link issues with `Closes #N`. -->

## What changed

<!--
Tick the level(s) this PR touches and add a one-line note on each.
Levels mirror the repo's mental model — same set on bug_report.md
and feature_request.md, so reviewers can filter consistently.
-->

- [ ] **entity** — `fractal/core/entities/` (Aave, Hyperliquid, Uniswap, stETH, GMX, simple/*):
- [ ] **strategy** — `fractal/strategies/` (BasisTrading, HyperliquidBasis, TauReset):
- [ ] **loaders** — `fractal/loaders/` (Binance, Hyperliquid, Aave, GMX, TheGraph, sims):
- [ ] **core** — `fractal/core/base/`, `fractal/core/pipeline.py` (Action / Observation / BaseStrategy / MLflow pipeline):
- [ ] **infra** — `setup.py`, `Makefile`, `.github/`, `scripts/`, `docs/`, pre-commit, pyproject:
- [ ] **tests** — `tests/` (core / loaders / mlflow_tests):

## Test plan

<!-- Commands you ran locally + their outcome. Mark each with ✓ when green. -->

- [ ] `pytest -m core`
- [ ] `pytest -m slow` (if real-data / CSV-replay touched)
- [ ] `pytest -m integration` (if loaders / external APIs touched)
- [ ] `flake8 fractal/ tests/` and `pylint fractal/`
- [ ] `cd docs && make html` (if docstrings / Sphinx config touched)
- [ ] `bash tests/mlflow_tests/scripts/e2e.sh` (if pipeline / examples touched)

## Checklist

- [ ] PR targets `dev`
- [ ] Appropriate label applied
- [ ] CHANGELOG.md updated (under the relevant `Unreleased` heading)
- [ ] Lock-in test added for any closed bug
- [ ] Reviewer requested
