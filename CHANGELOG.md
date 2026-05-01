# Changelog

All notable changes to **fractal-defi** are documented here. The format
is loosely based on [Keep a Changelog](https://keepachangelog.com/),
with one-line bullets per change.

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

- **Python ≥ 3.10** (PEP 604 unions are now used in runtime code).
- **Runtime / dev split via `extras_require`.** `pytest`, `pylint`,
  `flake8`, `pre-commit`, `sphinx` moved out of `install_requires`
  into the `dev` extra.
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
- **Pre-commit hooks** wired up — `flake8`, `pylint` (fractal + tests),
  `pytest -m core` plus standard file-shape checks. Install once with
  `pre-commit install` after `pip install -e ".[dev]"`.

## [v1.2.1] — 2025

Last release before the v1.3.0 review pass. See git history for
incremental changes.
