# Fractal

[![PyPI version](https://badge.fury.io/py/fractal-defi.svg)](https://badge.fury.io/py/fractal-defi)
[![Python Versions](https://img.shields.io/pypi/pyversions/fractal-defi.svg)](https://pypi.org/project/fractal-defi/)
[![License: BSD 3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Downloads](https://pepy.tech/badge/fractal-defi)](https://pepy.tech/project/fractal-defi)
[![Downloads](https://pepy.tech/badge/fractal-defi/month)](https://pepy.tech/project/fractal-defi)

**Fractal** — open-source Python research library for DeFi strategies.
Compose protocol-agnostic entities (lending, perps, DEX and LP) into
typed strategies; backtest, simulate, track experiments.

## Why Fractal

Most DeFi backtesters are product-shaped: pick a protocol, run a
strategy, get a P&L curve. Fractal is library-shaped — small primitives
with a big composition surface. You write a strategy once against the
generic `BasePerpEntity` / `BaseLendingEntity` / `BasePoolEntity` /
`BaseSpotEntity` contracts and swap concrete implementations
(Hyperliquid, Aave, Uniswap V3, GMX, your own) without touching the
strategy code.

## Features

- **Protocol-agnostic entities.** Concrete implementations for Aave V3
  lending, Hyperliquid perps, Uniswap V2/V3 LP and spot, Lido stETH,
  plus generic `Simple*` building blocks. Each entity is a typed state
  machine with `update_state` for accruals and protocol-correct
  `action_*` methods.
- **Composable strategies.** Register any number of entities under
  named slots, return `ActionToTake` from a single `predict()` hook,
  get a typed `StrategyResult` with metrics and a flat DataFrame.
- **Live + synthetic data.** Loaders for Binance public REST,
  Hyperliquid info API, Aave V3 GraphQL, GMX, TheGraph (Uniswap V2/V3,
  Lido), plus log-normal GBM and bootstrap simulators for offline
  stress tests.
- **Experiment tracking.** `DefaultPipeline` runs your strategy across
  a parameter grid, logs metrics + artifacts per run via MLflow, and
  supports sliding-window scenarios for stability analysis.
- **Type-safe and dev-friendly.** `BaseStrategy[Params]` generic +
  dataclass coercion, atomic per-action validation, delegate-resolved
  action arguments, deterministic replay.
- **Battle-tested.** 1100+ unit / invariant / real-data / integration
  tests, plus a Docker-based end-to-end MLflow harness in
  `tests/mlflow_tests/`.

## Structure

```
fractal/
├── core/
│   ├── base/           # Entity / Strategy / Observation contracts
│   ├── entities/       # Aave, Hyperliquid, Uniswap V2/V3, stETH, simple/*
│   └── pipeline.py     # MLflow grid-search pipelines
├── loaders/            # Binance / Hyperliquid / Aave / GMX / TheGraph / sims
└── strategies/         # BasisTrading, HyperliquidBasis, TauReset
examples/               # quick_start, basis, tau_reset, holder, agentic_trader
docs/                   # Sphinx site (make html)
tests/
├── core/               # offline unit + invariant + e2e synthetic
├── loaders/            # real-API loader tests
└── mlflow_tests/       # Docker MLflow + end-to-end pipeline scripts
```

## Install

```bash
pip install fractal-defi
```

Or from source:

```bash
git clone https://github.com/Logarithm-Labs/fractal-defi.git
cd fractal-defi
pip install -e ".[dev]"
```

Requires Python 3.10–3.12.

## Quick start

The smallest end-to-end DeFi example — a passive lender accruing 5% APY for
a year — is in `examples/quick_start.py`:

```bash
PYTHONPATH=. python examples/quick_start.py
```

Output:

```
simulating 8761 hourly observations (1 year @ 5% APY)
metrics: StrategyMetrics(accumulated_return=0.05127, apy=0.05127, sharpe=0.0, max_drawdown=0.0)
final balance: 10,512.7095
closed-form  : 10,512.7095
mismatch     : 7.6e-11
```

The 80-line script demonstrates the full lifecycle: typed `BaseStrategyParams`,
`BaseStrategy[Params]` generic, `register_entity`, `predict`,
`strategy.run(observations)`, metrics extraction, CSV dump.

## Examples

| Path | What it shows |
|---|---|
| [`examples/quick_start.py`](examples/quick_start.py) | Hello-Fractal: passive lending position with hourly compounding |
| [`examples/holder/`](examples/holder/) | Toy spot HODL with simple buy / sell triggers |
| [`examples/basis/`](examples/basis/) | Hyperliquid basis trade — perp short hedged against spot long |
| [`examples/tau_reset/`](examples/tau_reset/) | Active Uniswap V3 LP with τ-reset rebalancing |
| [`examples/agentic_trader/`](examples/agentic_trader/) | LLM-driven trading agent over historical klines |

Run any example from the repo root:

```bash
PYTHONPATH=. python examples/<name>/backtest.py     # single backtest
PYTHONPATH=. python examples/<name>/grid.py         # MLflow grid search
```

The grid-search variants need a running MLflow server. The simplest local
setup ships with the repo:

```bash
bash tests/mlflow_tests/scripts/start_mlflow.sh   # docker compose up MLflow on :5500
export MLFLOW_URI=http://localhost:5500
PYTHONPATH=. python examples/basis/grid.py
```

## Tests

```bash
pytest                 # default = offline core suite (~1100 tests, ~10s)
pytest -m slow         # add real-data CSV replays (~50 tests, ~50s)
pytest -m integration  # live API tests; requires THE_GRAPH_API_KEY for some
```

End-to-end MLflow harness (Docker-based):

```bash
bash tests/mlflow_tests/scripts/e2e.sh             # full pipeline cycle
```

## Documentation

- **Sphinx API reference.** Build locally with `cd docs && make html`,
  then open `docs/build/html/index.html` (or
  `python3 -m http.server -d docs/build/html 8000` and browse
  http://localhost:8000).
- **Architecture & design notes.** See [`ARCHITECTURE.md`](ARCHITECTURE.md)
  for entity-as-state-machine semantics, the InternalState / GlobalState
  split, delegate-resolved actions, notional accounting and pipeline
  internals.
- **Changelog.** See [`CHANGELOG.md`](CHANGELOG.md).
- **Contributing.** PRs are welcome — see
  [`CONTRIBUTING.md`](CONTRIBUTING.md) for setup, branching, the
  testing pyramid (what kind of tests are required for what kind of
  change), and the pre-commit pipeline. Run `pre-commit install`
  once after cloning so hooks fire on every commit.

## License

BSD 3-Clause. See [`LICENSE`](LICENSE).

Built by [Logarithm Labs](https://github.com/Logarithm-Labs).
