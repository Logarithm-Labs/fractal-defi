# Changelog

All notable changes to **fractal-defi** are documented here. The format
is loosely based on [Keep a Changelog](https://keepachangelog.com/),
with one-line bullets per change.

## [v1.3.1] — Unreleased

Coordinated dependency-floor bump + Codex audit follow-ups + Uniswap
V2 LP fee-modelling extensions. Existing strategies behave identically
(default config preserves prior semantics); real-data V2 backtests get
a small-magnitude correction to the loader/entity tvl contract. The
supported install matrix narrows — flag this if you depend on the
older floors.

### Added

- **`fractal.__version__`** — top-level version attribute, read via
  `importlib.metadata` so there's a single source of truth (`setup.py`).
  Matches the numpy/pandas/mlflow convention; useful for sanity-checks
  like `print(f"fractal-defi {fractal.__version__}")` in notebooks.
- **`UniswapV2LPConfig.fees_compounding_model`** — opt-in `"cash"`
  (default, backward-compatible) / `"compound"` flag controlling how
  per-bar pool fees flow through the position.
  - `"cash"` — fees flow to `InternalState.cash`; position token
    amounts purely reflect price-divergence; `impermanent_loss` is
    the textbook hodl-vs-LP gap (useful for IL modelling in
    isolation).
  - `"compound"` — fees implicitly reinvested into the position by
    growing `token0/token1_amount` (split 50/50 by value at current
    price); `liquidity` (LP-token count) stays constant — mirrors
    on-chain V2 where reserves grow but LP supply does not. `balance`
    matches the `"cash"` total; `impermanent_loss` becomes a
    fee-adjusted gap (pool yield offsets price-divergence cost).
- **`UniswapV2LPInternalState.compounded_token0_amount` /
  `compounded_token1_amount`** — cumulative buffer fields used by
  the new compound mode. Stay 0 in `"cash"` mode; reset on
  `action_close_position`.

### Dependencies (user-visible)

- **Runtime floors raised.** `setup.py::install_requires` now requires:
  - `numpy>=2.2.6` (was `>=1.26.0`) — **major bump**, drops NumPy 1.x
  - `mlflow>=3.11.1` (was `>=2.14.1`) — **major bump**, drops MLflow 2.x
  - `pandas>=2.3.3` (was `>=2.2.2`)
  - `scikit-learn>=1.7.2` (was `>=1.5.0`)
  - `loguru>=0.7.3` (was `>=0.7.2`), `requests>=2.33.1` (was `>=2.32.3`)
- **Migration:** consumers pinned to `numpy<2` or `mlflow<3` cannot
  upgrade — stay on `fractal-defi==1.3.0`. The new floors were chosen
  after running the full 1108-test core suite plus the closed-form
  `quick_start.py` smoke (mismatch ≈ 7e-11) on the bumped matrix.
  v1.3.0's documented `numpy>=1.26.0` floor served the Python 3.13
  wheel-availability problem of mid-2025; that's no longer a
  constraint with NumPy 2.x.
- **Dev tooling floors raised** to match what CI installs:
  `pytest>=9.0.3`, `pylint>=4.0.5`, `flake8>=7.3.0`, `isort>=8.0.1`,
  `pre-commit>=4.6.0`, `sphinx>=8.1.3`, `sphinx-rtd-theme>=3.1.0`.
- **`.pre-commit-config.yaml` synchronized** with the new floors so
  local hooks and CI run the same toolchain (was a source of pylint
  drift between `git commit` and the CI lint job).
- **Release tooling added to `[dev]` extra**: `build>=1.2.0`,
  `twine>=5.0.0`. `make build` / `make release` now work after a
  plain `pip install -e ".[dev]"` (closes Codex M3).
- `requirements.txt` re-synced as a faithful mirror of `setup.py`
  (closes Codex L2).

### Fixes

- **Aave `_request` type annotation.** `AaveV3RatesLoader._request`
  was declared `-> List[Dict[str, Any]]` but actually returned the
  GraphQL `data` object. Now correctly typed `-> Dict[str, Any]`
  with guards that raise `RuntimeError` on malformed payload (non-
  object `payload` or non-object `data`) instead of silent NoneType
  drift downstream (closes Codex L1).
- **V2 loader / entity `tvl` contract.**
  `EthereumUniswapV2PoolDataLoader.transform()` now produces
  **pre-fee tvl** (`tvl = reserveUSD - fees`). `UniswapV2LPEntity`
  expects pre-fee tvl per its loader contract so that
  `share * tvl + share * fees` reconstructs the post-fee position
  correctly. Previously the loader passed `reserveUSD` through as-is
  (= post-fee end-of-bar reserves), which combined with the entity's
  `cash += calculate_fees()` step double-counted each bar's fees on
  real-data backtests. Magnitude is small (1 bar's fees ≈
  `volume × fee_tier`, on the order of 0.001-0.005% of TVL per hour
  on USDC/WETH 30bps), but cumulative — equity curves shift by that
  amount, no longer drift over long windows.

### Not included (deliberately deferred)

- Docker base-image bump from `python:3.11-slim` → `python:3.14-slim`
  in `tests/mlflow_tests/Dockerfile`: rejected. `mlflow==2.14.1` (the
  pin in that Dockerfile) has no Python 3.14-compatible NumPy wheel,
  build fails. Will land paired with an `mlflow` bump in the
  Dockerfile.
- `examples/agentic_trader/requirements.txt` bumps (`dash 4.1.0`,
  `openai-agents 0.14.8`): land via Dependabot in their own PRs —
  example-only, doesn't gate this release.

## [v1.3.0] — 2026-05-01

First fully reviewed release. Refactors loaders/entities/strategies to a
single contract, fixes a long list of correctness bugs, ships an
end-to-end MLflow harness, and brings linters and Sphinx docs to clean
green.

### Loaders

- **Aave migrated to V3 GraphQL** (`api.v3.aave.com/graphql`); the
  legacy V2 REST endpoint was retired upstream. `AaveV2EthereumLoader`
  preserved as a deprecated alias.
- **Aave borrow rate sign fixed.** `AaveV3RatesLoader` no longer flips
  `borrowing_rate` to negative; positive ⇒ debt grows, matching every
  other entity and the universal sign convention.
- **`Loader.__init__` keyword-only fix.** Six subclasses (Aave, GMX,
  Binance prices/funding, Hyperliquid, MonteCarlo) passed `loader_type`
  positionally — Python silently routed it into `*args` and the value
  defaulted to CSV. Now passed by name.
- **Monte-Carlo simulator replaced** the arithmetic-returns walk with
  proper log-normal GBM in `MonteCarloPriceLoader`. Trajectories are
  strictly positive and reproducible by seed.
- **MonteCarlo persistence.** `MonteCarloPriceLoader` now correctly
  uses `LoaderType.PICKLE` (it dumps a list of DataFrames); was
  silently falling back to CSV which raised on save.
- **Hyperliquid candle pagination.** `HyperliquidPerpsPricesLoader`
  no longer aborts on the first empty leading window; advances past
  Hyperliquid's undocumented history horizon and picks up data when
  it appears.
- **EVM-address validation** on pool / token / asset addresses across
  Aave, GMX, Uniswap V2/V3 pool, Uniswap V3 spot loaders. Invalid
  addresses are rejected at construction.
- **Loader contract unified** under a common `extract → transform →
  load → read` lifecycle with deterministic on-disk caching.
- **Binance / UniswapV3 bugfixes** — interval handling and pool
  decimals.

### Entities

- **Aave + Spot refactored** with looping-flow examples; strategy
  `total_balance` respects unit conventions across entities.
- **Aave LTV is now cumulative.** `AaveEntity.action_borrow` checks
  the LTV against `borrowed + amount`, not just the new amount —
  closes a stacking exploit where multiple sub-limit borrows could
  exceed `max_ltv`.
- **Lending price guards.** `SimpleLendingEntity` borrow / withdraw
  reject zero-priced collateral or debt with a domain exception
  instead of raising `ZeroDivisionError`.
- **Uniswap V2 + V3 refactor** — notional bug fixes, V2 LP-token
  minting math reworked, end-to-end tests added.
- **`UniswapV3LPEntity.update_state` guards.** Rejects `price <= 0`,
  `liquidity < 0`, `fees < 0`, `tvl < 0` when a position is open —
  was silently producing nonsense token amounts.
- **GMX legacy model removed.** The on-chain V1 endpoint was retired
  upstream; the loader is preserved for self-hosted subgraphs.
- **Hyperliquid formulas refactored** (perps + maintenance margin
  closed-form).
- **Perp leverage at-open enforcement.** `SimplePerpEntity` and
  `HyperliquidEntity` validate post-trade margin and leverage on
  risk-increasing trades; rejected trades roll back atomically.
  Risk-reducing trades (close / partial close) are still always
  permitted.

### Strategies

- **Basis trading + Tau-reset** ship with full test suites spanning
  Layer 1 (unit) → Layer 4 (randomized property) + real-data smoke
  (Layer 5).
- **Reversed notional strategy** added.
- **`HyperliquidBasis.EXECUTION_COST` semantics changed** to total
  round-trip basis spread, split equally across HEDGE and SPOT legs
  in `set_up`. Previously charged on each leg independently (double
  the configured value).
- **Atomic action rejection** at the entity level (per-action validate
  then mutate). Step-level execution stays non-atomic by design — a
  failing action aborts the run rather than rolling back prior actions.

### Strategy / framework core

- **`get_metrics` guards** against degenerate inputs (empty / single
  timestamp / zero initial balance / zero notional) — returns zero
  metrics instead of `inf`/`nan`/exception.
- **`set_params(dict)` coercion.** When a strategy declares
  `BaseStrategy[Params]`, dict-shaped grid cells are splatted into the
  declared dataclass — defaults flow, unknown keys raise.
- **Storage write moved after observation validation.** Rejected
  observations no longer pollute `ObservationsStorage`.
- **Observation equality includes timestamp.** Set / dict no longer
  collapse same-state snapshots taken at different times.
- **`get_all_available_entities()`** returns a `MappingProxyType`
  read-only view; external callers can no longer mutate the registry.
- **`UniswapV3Loader` lifecycle methods raise `NotImplementedError`**
  instead of silently `pass` when subclasses forget to override.

### Pipeline / MLflow

- **Lazy MLflow connect.** Network access deferred to first `run` /
  `grid_step`. Construction is offline.
- **AWS env-var preservation.** `MLflowConfig` does not overwrite host
  credentials with empty strings.
- **`ExperimentConfig.step_size`** exposes the sliding-window stride
  that was previously hardcoded inside `Launcher.run_scenario`.
- **End-to-end MLflow harness** added under `tests/mlflow_tests/` —
  Docker compose stack, 4 example pipelines, artifact verifier.

### Logging

- **Per-instance loguru handler** with run-id filter. `DefaultLogger`
  no longer wipes other sinks via global `logger.remove()`.
- **Default stderr sink stripped once per process** so `debug=True`
  doesn't fan out to the console; strategies log to file only.
- **Drop `PYTHONPATH` as a filesystem root.** Loader cache uses
  `DATA_PATH` or `cwd`; strategy runs use `FRACTAL_RUNS_PATH` or `cwd`.

### Packaging

- **Python 3.10–3.13** supported (PEP 604 unions are used in runtime
  code; every runtime dep ships 3.13 wheels).
- **`numpy>=1.26.0` floor.** v1.1.0 had `numpy<2,>=1.16.0` which
  forced pip onto numpy 1.26.4 — that release has no Python 3.13
  wheel, so installs without a C compiler failed at build time. The
  upper bound is gone and the floor moved to 1.26.0 so pip resolves
  cleanly to numpy 2.x on Python 3.13.
- **Runtime / dev split via `extras_require`.** `pytest`, `pylint`,
  `flake8`, `isort`, `pre-commit`, `sphinx` moved out of
  `install_requires` into the `dev` extra.
- **License metadata corrected** — README badge now matches BSD-3-Clause.

### Tests, examples and docs

- **1100+ unit / invariant / real-data / integration tests.** Default
  `pytest` runs the offline `core` layer; `slow` and `integration`
  layers are opt-in via `-m`.
- **Linters at 10/10** (pylint + flake8) across `fractal/` and
  `tests/`.
- **Test structure** — `invariant_testing/`, `e2e/`, `slow` marker
  conventions.
- **Examples renamed** for consistency: `examples/managed_basis_strategy/`
  → `examples/basis/`, `examples/tau_strategy/` → `examples/tau_reset/`,
  `examples/agent_trader/` → `examples/agentic_trader/`. Pipeline
  scripts uniformly named `grid.py`, single backtests `backtest.py`.
- **`quick_start.py` rewritten** as a passive Aave-style lending
  position with hourly compounding — minimal DeFi example, no network,
  result matches the closed-form compound-interest formula bit-for-bit.
- **Sphinx documentation regenerated** and builds cleanly with no
  warnings (`cd docs && make html`).
- **`ARCHITECTURE.md`** added documenting entity-as-state-machine
  semantics, IS/GS split, delegate-resolved actions, notional
  conventions and pipeline internals.
- **Pre-commit hooks** wired up — `isort`, `flake8`, `pylint` (fractal +
  tests) plus standard file-shape checks; `pytest -m core` lives on the
  manual stage. Install once with `pre-commit install` after
  `pip install -e ".[dev]"`.
- **GitHub Actions CI** under `.github/workflows/ci.yml`. PR + push
  to `main`/`dev` runs lint and core tests on Python 3.10–3.13, plus
  a Sphinx build with warnings-as-errors. Slow real-data tests run
  on push-to-default-branch and weekly; integration (live API) and
  e2e MLflow Docker harness run on manual dispatch and weekly cron.

## [v1.2.1] — 2025

Last release before the v1.3.0 review pass. See git history for
incremental changes.
