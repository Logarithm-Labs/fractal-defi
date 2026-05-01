---
name: Feature request
about: Propose a new entity, strategy, loader, or framework capability.
title: ''
labels: enhancement
assignees: ''
---

<!--
Thanks for the proposal! Lead with the use case, not the implementation.
Keep the title short and descriptive (under 70 chars).
-->

## Use case

<!--
What are you trying to do? Describe the research / strategy / data
problem in plain terms before discussing solutions. A concrete scenario
("I want to backtest X across Y under Z conditions") is much more
useful than "we should add ABC".
-->

## Proposed API or behavior

<!--
Sketch the surface — class names, method signatures, expected inputs
and outputs. Pseudocode is fine; doesn't need to compile.
-->

```python
```

## Alternatives considered

<!-- What did you try first? Why didn't an existing primitive cover it? -->

## Affected level

<!--
Same 6-level taxonomy used on the PR template. Tick all that apply.
-->

- [ ] **entity** — `fractal/core/entities/` (new protocol, new action, new state field)
- [ ] **strategy** — `fractal/strategies/` (new strategy, new hyperparam shape)
- [ ] **loaders** — `fractal/loaders/` (new venue / endpoint / cache layer / simulator)
- [ ] **core** — `fractal/core/base/`, `fractal/core/pipeline.py` (new contract surface, MLflow pipeline change)
- [ ] **infra** — `setup.py`, `Makefile`, `.github/`, `scripts/`, `docs/`, pre-commit, pyproject
- [ ] **tests** — `tests/` (new harness, new fixture, new test layer)

## Backwards compatibility

<!--
Would this change existing public types, method signatures, or behavior?
If yes, what migration path do users have? If you'd hide it behind a
new flag/class, say so.
-->

- [ ] Pure addition — no existing API affected
- [ ] Additive but touches existing types (new optional param, new method)
- [ ] Breaking — requires migration

## Notes

<!--
Links to papers, prior art (other DeFi libraries), related issues, or
prototypes you've already drafted. Skip if none.
-->
