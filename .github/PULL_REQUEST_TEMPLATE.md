<!--
Thanks for the PR! Keep the title short and descriptive (under 70 chars).
Detailed context goes in the body below.
-->

## Summary

<!-- One or two sentences: what changed and why. Link issues with `Closes #N`. -->

## What changed

<!-- Bullet list, one line per item. Group by area if helpful. -->

- [ ] entities:
- [ ] strategies:
- [ ] loaders:
- [ ] pipeline / MLflow:
- [ ] tests / docs:

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
