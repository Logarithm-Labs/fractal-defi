# Contributing to Fractal

Thanks for your interest in contributing. This document covers what to
read first, how to set up your environment, and what kind of tests we
expect for what kind of change. If anything is unclear, open an issue
or draft PR — we'd rather over-discuss than get stuck.

## Read first

In order, before opening a PR:

1. **[`README.md`](README.md)** — what Fractal is, what it covers,
   how to run the simplest example.
2. **[`ARCHITECTURE.md`](ARCHITECTURE.md)** — entity-as-state-machine
   semantics, the `InternalState` / `GlobalState` split,
   delegate-resolved actions, notional conventions, pipeline
   internals. The "why" behind the code.
3. **[`CHANGELOG.md`](CHANGELOG.md)** — recent breaking changes and
   areas under active work; helps you avoid landing on a moving target.
4. **Sphinx API reference** — full docstring-driven docs:
   ```bash
   cd docs && make html
   open build/html/index.html
   ```
5. **Examples** — pick the one closest to what you're adding and
   read its `backtest.py` end-to-end before touching the framework:
   - `examples/quick_start.py` — minimal lending example
   - `examples/holder/` — toy spot strategy with a custom entity
   - `examples/basis/` — multi-entity strategy (perp + spot)
   - `examples/tau_reset/` — active LP rebalancing
   - `examples/agentic_trader/` — LLM-driven decisions
6. **Tests for the area you're touching** — best documentation of
   the contract, especially for entities (`tests/core/test_*.py`,
   `tests/core/invariant_testing/*.py`). Search for `lock-in` to find
   regression-pinning tests and read the surrounding ones.

## Local setup

```bash
git clone https://github.com/Logarithm-Labs/fractal-defi.git
cd fractal-defi
python3 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"

pre-commit install
```

`pre-commit install` registers hooks under `.git/hooks/pre-commit`.
After that every `git commit` runs flake8, pylint and the offline
`pytest -m core` suite (~10 seconds total). Skip with
`git commit --no-verify` only when you genuinely need to (CI will run
the same hooks anyway).

Verify the stack is healthy:

```bash
pytest                 # core suite, ~1100 tests, ~10s
pre-commit run --all-files
cd docs && make html   # 0 warnings expected
```

## Branching and PR flow

- **Branch off `dev`**, not `main`. `main` is reserved for releases.
- **Naming.** `feat/`, `fix/`, `refactor/`, `docs/`, `test/` prefixes
  followed by a short slug (`feat/aave-flash-loan`,
  `fix/hyperliquid-pagination`, etc.).
- **Commit messages.** Imperative, present tense, ≤ 72 chars subject:
  `Add cumulative LTV check to AaveEntity` not
  `Added cumulative LTV checks`.
- **Open a PR against `dev`** using the
  [`PULL_REQUEST_TEMPLATE.md`](.github/PULL_REQUEST_TEMPLATE.md). Fill
  the Test plan checklist honestly — uncheck the boxes that don't
  apply, don't pre-tick what you didn't run.
- **Update `CHANGELOG.md`** under `[v<next>] — Unreleased` with a
  one-line bullet describing the user-visible change.
- **Lock-in test for every closed bug.** Required, not optional —
  search any `tests/core/test_*.py` for the pattern.

## Code style

- **Linters at 10/10.** `flake8` and `pylint` both run on every
  commit. Repo configs: `.flake8` (line length 120) and `.pylintrc`
  (test-friendly disables in `tests/.pylintrc`).
- **Type hints.** Required on public methods. Use PEP 604 unions
  (`int | None`) — Python 3.10+ minimum is enforced in `setup.py`.
- **Docstrings.** RST-style (Sphinx renders them). For dataclasses,
  use `Attributes:` sections — `napoleon_use_ivar = True` makes them
  render as inline `:ivar:` blocks without conflicting with autodoc.
- **No `print()` in framework code.** Use `self._debug(...)` inside
  strategies (gated by `debug=True`); `loguru` directly elsewhere.
- **Imports.** Standard library, third-party, local — three blocks,
  alphabetical within each.
- **Comments.** Default to writing none. Only comment the *why* when
  it's non-obvious — a hidden constraint, a subtle invariant, a
  workaround for an upstream bug. Don't paraphrase the code.

## Testing pyramid

Fractal organizes tests in five layers, applied selectively based on
what you're changing. Mark each test with the appropriate
`@pytest.mark.<layer>` so it lands in the right CI stage.

| Layer | Marker | Purpose | Example file |
|---|---|---|---|
| **L1 — Unit** | `core` | Single method on a single entity, deterministic, no I/O | `tests/core/test_aave.py` |
| **L2 — Synthetic e2e** | `core` | Full strategy run on hand-rolled observations, deterministic | `tests/core/e2e/test_e2e_lending_synthetic.py` |
| **L3 — Invariants** | `core` | Post-condition properties on any successful action (LTV ≤ max, leverage bounds, range invariance, ...) | `tests/core/invariant_testing/test_lending_invariants.py` |
| **L4 — Randomized property** | `core` | L3 invariants under N random seeds | `tests/core/invariant_testing/test_basis_invariants.py` |
| **L5 — Real-data smoke** | `slow` | Replay CSV fixtures from `examples/` to catch behaviour drift on real markets | `tests/core/test_hyperliquid_basis_real_data.py` |
| **Integration** | `integration` | Live API calls (Binance, Hyperliquid, Aave, TheGraph) | `tests/loaders/test_aave_loader.py` |

Default `pytest` runs L1–L4 only. `slow` and `integration` are opt-in
via `-m slow` / `-m integration`. The end-to-end Docker MLflow harness
(`tests/mlflow_tests/`) is invoked separately as a shell script — not
collected by pytest.

### What to write for what change

#### New **entity** (e.g. a new lending protocol, a new perp venue)

| Test layer | Required? | What to cover |
|---|---|---|
| L1 unit | **yes** | Each `action_*` method: happy path, validation errors, accrual via `update_state`, edge cases (zero amounts, missing prices). |
| L3 invariants | **yes** | Post-condition properties: balance non-negative after deposit, LTV ≤ max after borrow, leverage ≤ max after open, all `update_state` rates respected, etc. |
| Parity vs sibling | **yes** if applicable | If your entity shares a paradigm (e.g. a new lending entity vs `SimpleLendingEntity`), add a parity test feeding both the same observations and asserting identical or analogous outputs. |
| L4 randomized | recommended | If your entity has many states (multiple positions, multiple price ranges), randomize over seeds. |
| L5 real-data | optional | Only if the protocol has cleanly downloadable CSV fixtures and the behaviour matters end-to-end. |

#### New **strategy**

| Test layer | Required? | What to cover |
|---|---|---|
| L1 unit | **yes** | `set_up`, `predict` branches (entry, exit, idle), specific `predict` outputs given specific entity states. |
| L2 synthetic e2e | **yes** | Run on hand-rolled observations covering each `predict` branch at least once. Assert `result.get_default_metrics()` is finite and `net_balance` is finite throughout. |
| L3 invariants | **yes** | Strategy-level invariants: e.g. basis hedge `|hedge.size| ≈ spot.amount`, LP range formula `1.0001^(TAU·tick_spacing)`, no money printing on rebalance. |
| L4 randomized | recommended | Run 50–100 randomized synthetic markets and assert invariants hold. |
| L5 real-data | **yes if** the strategy is going into `fractal/strategies/` (not just an example) | Replay one or two CSV fixtures from `examples/`, assert finiteness, equity within a reasonable band. |

#### New **loader**

| Test layer | Required? | What to cover |
|---|---|---|
| Offline `transform` unit | **yes** | Construct loader, set `_data` to a hand-rolled DataFrame, call `transform()`, assert columns and types match the typed return struct. Cover empty/single-row cases. |
| Cache key determinism | **yes** | Two loaders with identical inputs share `_cache_key()`; changing any input changes it. |
| Integration smoke | recommended | One `read(with_run=True)` call against the real endpoint with a small recent window. Mark `@pytest.mark.integration`. |
| Sign / unit conventions | **yes** | Lock-in `borrowing_rate > 0` ⇒ debt grows downstream — see `tests/loaders/test_aave_loader_offline.py`. |

#### New **pipeline / MLflow** feature

| Test layer | Required? | What to cover |
|---|---|---|
| Mock-based unit | **yes** | Stub `mlflow.*` calls via `monkeypatch.setattr` and verify the right sequence of calls is made. See `tests/core/test_pipeline.py`. |
| End-to-end Docker | recommended | Add a script under `tests/mlflow_tests/scripts/` and verify it via the existing `verify_artifacts.py`. |

#### **Bug fix** (any layer)

- **Always add a lock-in / regression test** that fails on the
  pre-fix code and passes after the fix. Use the same naming pattern
  as existing lock-ins. Cite the original symptom in the docstring.

#### **Refactor only** (no behaviour change)

- Existing tests must continue to pass without modification.
- If they don't, you're refactoring behaviour, not just shape — split
  into a "pure refactor" PR and a separate "behaviour change" PR with
  proper test coverage.

### Test naming and structure

```python
@pytest.mark.core
def test_<thing>_<expected_behaviour>_<condition>():
    """One-line lock-in or invariant statement.

    Optional: cite the original bug or rationale.
    """
    # Arrange
    e = SomeEntity(...)
    e.update_state(...)
    # Act
    e.action_borrow(100)
    # Assert
    assert e.ltv == pytest.approx(0.5)
```

Avoid:

- Multiple unrelated assertions in one test (split them).
- Hidden setup in module-scope variables that other tests mutate.
- Network calls in a `core`-marked test.

## Documentation

- **Docstrings.** Every public class and method that lands in
  `fractal/` needs at least a one-line docstring. Constructor /
  config classes deserve an `Attributes:` section explaining each
  field's role and unit.
- **`ARCHITECTURE.md`** for cross-cutting design decisions that
  affect multiple modules (a new entity paradigm, a change to the
  framework loop, a new state convention).
- **`CHANGELOG.md`** for every user-visible change. One line per
  bullet under `[v<next>] — Unreleased`. Group by area.
- **Sphinx build clean.** Run `cd docs && make html` and confirm
  zero warnings before opening the PR.

## Pre-commit hooks — what they run

| Hook | What it checks | Auto-fix? |
|---|---|---|
| `trailing-whitespace`, `end-of-file-fixer` | Whitespace hygiene | yes |
| `check-yaml`, `check-toml`, `check-merge-conflict` | Config + merge-conflict markers | no |
| `check-added-large-files` | Files > 2 MB | no (warns) |
| `debug-statements` | Stray `pdb.set_trace()` | no |
| `flake8` | PEP 8 + complexity (`fractal/`, `tests/`) | no |
| `pylint (fractal/)` | Lint at 10/10 | no |
| `pylint (tests/)` | Test-friendly subset (`tests/.pylintrc`) | no |
| `pytest -m core` | 1100+ unit/invariant tests, ~10s | no |

If a hook fails, **fix the underlying issue** rather than passing
`--no-verify`. The same checks run in CI; bypassing locally just
defers the failure.

## Continuous integration

`.github/workflows/ci.yml` runs on every PR and push to `main` or
`dev`:

| Job | When | What |
|---|---|---|
| `lint` | always | `pre-commit run --all-files` (isort + flake8 + pylint + file-shape) |
| `core-tests` | always | `pytest -m core` on Python 3.10, 3.11, 3.12, 3.13 |
| `docs` | always | `sphinx-build -W` (warnings = errors) |
| `slow-tests` | push to `main`/`dev` + weekly + manual | `pytest -m slow` (CSV-replay) |
| `integration-tests` | weekly + manual | `pytest -m integration` (live APIs) |
| `smoke` | push + weekly + PRs labelled `release-prep` | build wheel, install in throwaway venv, run imports + tests against it |
| `e2e-mlflow` | weekly + manual | `bash tests/mlflow_tests/scripts/e2e.sh` |

PR feedback stays fast (lint + core + docs ≈ 2 min). Heavier suites
that talk to the network or spin up Docker run on schedule + on demand
to avoid blocking PRs on flaky upstreams.

To trigger a manual run, go to **Actions → CI → Run workflow** and
pick the branch. To run only one job from a PR, push a draft commit
and watch the matrix.

If you need a `THE_GRAPH_API_KEY` for the integration / slow-tests
jobs, set it as a repository secret (Settings → Secrets and variables
→ Actions). Most loaders work without it — only the TheGraph
Uniswap/Lido subset requires the key.

## Examples

Adding an example under `examples/<name>/`:

- **One backtest entry-point** at `examples/<name>/backtest.py` —
  fully self-contained, runs as `PYTHONPATH=. python
  examples/<name>/backtest.py`.
- **One grid entry-point** at `examples/<name>/grid.py` if your
  example benefits from MLflow grid search. Read `MLFLOW_URI` from env
  with a clear error if unset.
- **No API keys hard-coded.** Use `os.getenv(...)` and skip cleanly
  if missing.
- **Update the README table** — add a row to the Examples section.
- **Smoke-test it locally** with the simplest possible parameters
  before opening the PR.

## Releasing (maintainers)

```bash
# 1. Bump setup.py version to <next>
# 2. Move CHANGELOG.md "[v<next>] — Unreleased" to "[v<next>] — YYYY-MM-DD"
# 3. Commit on dev: chore: bump version to <next>
# 4. Open PR dev → main, squash-merge after review

git checkout main && git pull
git tag v<next>
git push origin v<next>
python -m build
twine upload dist/*
```

## Getting unstuck

- Search closed PRs for similar patterns.
- Open a draft PR early — even rough work gets faster feedback than a
  finished PR sitting on questions.
- Open an issue if the framework genuinely doesn't support what you
  need; we'd rather know than have you build around a missing primitive.

Thanks again. Looking forward to your PR.
