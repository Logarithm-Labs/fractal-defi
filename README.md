# Fractal

[![CI](https://github.com/Logarithm-Labs/fractal-defi/actions/workflows/ci.yml/badge.svg)](https://github.com/Logarithm-Labs/fractal-defi/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/Logarithm-Labs/fractal-defi/branch/main/graph/badge.svg)](https://codecov.io/gh/Logarithm-Labs/fractal-defi)
[![PyPI version](https://img.shields.io/pypi/v/fractal-defi.svg)](https://pypi.org/project/fractal-defi/)
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

Or from source (latest unreleased ``dev`` branch):

```bash
git clone https://github.com/Logarithm-Labs/fractal-defi.git
cd fractal-defi
pip install .
```

Requires Python 3.10–3.13.

Contributors and anyone running the test suite want the editable
install with the dev extras (pulls in pytest, pylint, flake8, isort,
pre-commit, sphinx). See [`CONTRIBUTING.md`](CONTRIBUTING.md):

```bash
pip install -e ".[dev]"
pre-commit install
```

## Quick start

A passive Aave-style lender accruing 5% APY for a year, no network or
MLflow needed. After installing, write the file below as
`quick_start.py` and run it:

```python
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import List

from fractal.core.base import (Action, ActionToTake, BaseStrategy,
                               BaseStrategyParams, NamedEntity, Observation)
from fractal.core.entities import SimpleLendingEntity, SimpleLendingGlobalState


@dataclass
class LendingParams(BaseStrategyParams):
    INITIAL_BALANCE: float = 10_000.0
    LENDING_APY: float = 0.05


class PassiveLender(BaseStrategy[LendingParams]):
    def set_up(self) -> None:
        self.register_entity(NamedEntity("LENDING", SimpleLendingEntity()))
        self._funded = False

    def predict(self) -> List[ActionToTake]:
        if self._funded:
            return []
        self._funded = True
        return [ActionToTake("LENDING", Action(
            "deposit", {"amount_in_notional": self._params.INITIAL_BALANCE},
        ))]


hours = 365 * 24
start = datetime(2024, 1, 1, tzinfo=UTC)
observations = [
    Observation(timestamp=start + timedelta(hours=i), states={
        "LENDING": SimpleLendingGlobalState(
            collateral_price=1.0, debt_price=1.0,
            lending_rate=0.05 / hours, borrowing_rate=0.0,
        ),
    })
    for i in range(hours + 1)
]

result = PassiveLender(params=LendingParams()).run(observations)
print(result.get_default_metrics())
```

```
StrategyMetrics(accumulated_return=0.05127, apy=0.05127, sharpe=0.0, max_drawdown=0.0)
```

The same script lives at [`examples/quick_start/quick_start.py`](examples/quick_start/quick_start.py)
in the repo, alongside [`quick_start.ipynb`](examples/quick_start/quick_start.ipynb)
— a notebook that walks the same example then layers on a Uniswap V2
LP entity-as-model demo (with an IL chart) and a real-data ETH/USDC V2
hold-and-fees backtest. The repo also ships heavier examples covering
basis trading, LP rebalancing, agentic trading and a toy hodler — see
below.

## Examples

| Path | What it shows |
|---|---|
| [`examples/quick_start/`](examples/quick_start/) | Hello-Fractal + Uniswap V2 IL demo + real-data V2 backtest (script + notebook) |
| [`examples/holder/`](examples/holder/) | Toy spot HODL with simple buy / sell triggers |
| [`examples/basis/`](examples/basis/) | Hyperliquid basis trade — perp short hedged against spot long |
| [`examples/tau_reset/`](examples/tau_reset/) | Active Uniswap V3 LP with τ-reset rebalancing |
| [`examples/agentic_trader/`](examples/agentic_trader/) | LLM-driven trading agent over historical klines |
| [`examples/ml_funding_rate_forecasting/`](examples/ml_funding_rate_forecasting/) | ML pipeline: forecasting Binance funding rates with feature engineering + CatBoost |

After cloning the repo and installing the package, run an example
directly:

```bash
python examples/quick_start/quick_start.py
python examples/basis/backtest.py
python examples/tau_reset/backtest.py
```

The grid-search variants (`examples/<name>/grid.py`) log results to
MLflow. The repo ships a self-contained Docker MLflow stack — bring it
up, point your shell at it, run the grid:

```bash
bash tests/mlflow_tests/scripts/start_mlflow.sh   # MLflow on :5500
export MLFLOW_URI=http://localhost:5500
python examples/basis/grid.py
```

## Documentation

- [`ARCHITECTURE.md`](ARCHITECTURE.md) — entity-as-state-machine
  semantics, the `InternalState` / `GlobalState` split,
  delegate-resolved actions, notional accounting, pipeline internals.
- [`CHANGELOG.md`](CHANGELOG.md) — release notes.
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — setup, branching, the
  testing pyramid, the pre-commit / CI pipeline. Required reading
  before opening a PR.
- Sphinx API reference can be built locally from the repo
  (`make docs`); a hosted version is on the project's docs site.

## License

BSD 3-Clause. See [`LICENSE`](LICENSE).

Built by [Logarithm Labs](https://github.com/Logarithm-Labs).
