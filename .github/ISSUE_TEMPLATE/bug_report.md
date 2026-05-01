---
name: Bug report
about: Something works incorrectly or crashes — not how to use the library.
title: ''
labels: bug
assignees: ''
---

<!--
Thanks for the report! Keep the title short and descriptive (under 70 chars).
Detail goes in the body below. The clearer the repro, the faster the fix.
-->

## Summary

<!-- One or two sentences: what's broken and where. -->

## Repro

<!--
A minimal, runnable Python snippet that triggers the bug. Trim away
everything not directly required to reproduce. If it depends on a CSV /
real-data fixture, paste a short slice instead of attaching the file.
-->

```python
```

## Expected vs actual

- **Expected:**
- **Actual:**

<!-- If a stack trace is involved, paste it inside a fenced block. -->

```
```

## Affected level

<!--
Same 6-level taxonomy used on the PR template. Tick all that apply.
-->

- [ ] **entity** — `fractal/core/entities/` (Aave, Hyperliquid, Uniswap, stETH, GMX, simple/*)
- [ ] **strategy** — `fractal/strategies/` (BasisTrading, HyperliquidBasis, TauReset)
- [ ] **loaders** — `fractal/loaders/` (Binance, Hyperliquid, Aave, GMX, TheGraph, sims)
- [ ] **core** — `fractal/core/base/`, `fractal/core/pipeline.py` (Action / Observation / BaseStrategy / MLflow pipeline)
- [ ] **infra** — `setup.py`, `Makefile`, `.github/`, `scripts/`, `docs/`, pre-commit, pyproject
- [ ] **tests** — `tests/` (core / loaders / mlflow_tests)

## Environment

- `fractal-defi` version: <!-- e.g. 1.3.0; check with `pip show fractal-defi` -->
- Python version: <!-- `python --version` -->
- OS: <!-- macOS 14, Ubuntu 22.04, etc. -->
- Install method: <!-- pip from PyPI / pip from source / editable -->

## Notes

<!--
Anything else worth knowing: when you first saw it, last known good
version, which test/script surfaced it, related issues.
-->
