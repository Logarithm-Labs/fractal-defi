# Architecture

This document captures the design decisions behind Fractal — the things
that aren't obvious from skimming the code, that you need to know to
extend the framework correctly, and that we keep coming back to in code
review. README is for "how do I use it"; this is for "why is it shaped
this way".

## Contents

- [Entities are state machines](#entities-are-state-machines)
- [GlobalState vs InternalState](#globalstate-vs-internalstate)
- [Action methods, delegates and per-step atomicity](#action-methods-delegates-and-per-step-atomicity)
- [Strategies as orchestrators](#strategies-as-orchestrators)
- [Notional accounting and unit conventions](#notional-accounting-and-unit-conventions)
- [Loaders, caching and the data pipeline](#loaders-caching-and-the-data-pipeline)
- [MLflow pipelines and the experiment grid](#mlflow-pipelines-and-the-experiment-grid)
- [Testing layers](#testing-layers)
- [Logging and observability](#logging-and-observability)

## Entities are state machines

An entity is a deterministic state machine modelling one on-chain protocol
position (an Aave loan, a Hyperliquid perp position, a Uniswap V3 LP, a
spot account, …). Formally, every entity is a transition function:

```
T : (IS, GS, A) → IS'
```

Where:

- **`IS`** — the user's `InternalState`: collateral, debt, position size,
  LP token amounts, accumulated cash. Whatever fully describes the
  position inside the protocol.
- **`GS`** — the `GlobalState`: market context the protocol exposes —
  prices, funding rate, lending/borrowing APYs, pool TVL/fees/liquidity.
  Read-only from the entity's perspective; produced upstream by the
  data pipeline and applied via `update_state(GS)`.
- **`A`** — an `Action` instance: a method name (`deposit`, `borrow`,
  `open_position`, …) plus a positional/keyword arg payload.
- **`IS'`** — the new internal state after applying both market evolution
  and the action.

The `update_state(state)` hook is the analog of "time passes" — it ingests
a new `GS` and applies any per-step accruals (interest, funding, fees,
liquidation checks). Action methods are analogs of "user does something" —
they mutate `IS` while reading the current `GS`. The framework drives
both each step:

```
for observation in observations:                # observation = (timestamp, {entity_name: GS})
    for entity_name, gs in observation.states.items():
        entity.update_state(gs)                 # market evolution: IS, GS → IS', new GS
    actions = strategy.predict()                # current entity states → list of (entity, A)
    for action in actions:
        entity.execute(action)                  # IS', GS, A → IS''
    snapshot every entity's IS', GS for the result
```

This makes a backtest a deterministic replay of `(IS₀, GS₀) → … → (ISₙ, GSₙ)`
fully described by `(initial state, observation sequence, strategy)`.

### Implementing a new entity

Subclass `BaseEntity[GS, IS]` (Generic over the two state types):

```python
class MyEntity(BaseEntity[MyGlobalState, MyInternalState]):
    def _initialize_states(self):
        self._global_state = MyGlobalState()
        self._internal_state = MyInternalState()

    def update_state(self, state: MyGlobalState) -> None:
        # apply state, accrue interest/fees, check liquidation, …
        self._global_state = state
        self._internal_state.collateral *= 1 + state.lending_rate

    def action_deposit(self, amount_in_notional: float) -> None:
        # validate, mutate IS
        ...

    @property
    def balance(self) -> float:
        return self._internal_state.collateral * self._global_state.collateral_price
```

Methods whose name starts with `action_` are auto-discovered by
`BaseEntity.execute` — strategies dispatch to them via
`Action(name, args)` without the entity needing to know what dispatch
table to register.

## GlobalState vs InternalState

Splitting state into two types isn't decoration — it gates several
properties:

1. **Determinism.** `update_state` is the *only* place a `GlobalState`
   enters the entity. Re-running with the same observation sequence
   produces the same `InternalState` trajectory bit-for-bit.

2. **No silent dependence on environment.** Action methods can read
   `self._global_state` (current market context) but never silently
   *fetch* one. If you need an external value to decide an action, the
   strategy must put it on the next observation.

3. **Snapshot fidelity.** `StrategyResult` stores `(IS, GS)` per step.
   Every column in `result.to_dataframe()` traces back to either a user
   decision (in `IS`) or the upstream feed (in `GS`); there's no hidden
   third source.

4. **Re-export safety.** Both state types are dataclasses. Copy-on-step
   gives you a stable history without aliasing — see
   `tests/core/test_step_immutability.py`.

`GlobalState` instances live for the duration of one observation; they
are replaced wholesale on each `update_state`. `InternalState` is owned
by the entity for the lifetime of the run.

### GlobalState construction is observation-driven

Strategies don't construct `GlobalState`. Loaders do, via the
`Observation` builder:

```python
Observation(timestamp=ts, states={
    "SPOT":  UniswapV3SpotGlobalState(price=p),
    "HEDGE": HyperliquidGlobalState(mark_price=p, funding_rate=r),
})
```

The `Observation` validates that every named entity is registered and
that no required state is missing (under `STRICT_OBSERVATIONS = True`).

## Action methods, delegates and per-step atomicity

Each `action_*` method is a pure validator-mutator:

```python
def action_borrow(self, amount_in_product: float) -> None:
    if amount_in_product < 0:
        raise EntityException(...)
    if self._global_state.debt_price <= 0:
        raise EntityException(...)
    new_debt = self._internal_state.borrowed + amount_in_product
    new_ltv = new_debt * self._global_state.debt_price / collateral_value
    if new_ltv > self.max_ltv:
        raise EntityException(...)
    self._internal_state.borrowed = new_debt
```

Two semantics worth knowing:

### Atomic per-action

Validation happens *before* mutation. If a check fails, the action raises
and `IS` is untouched. Risk-increasing perp trades that would breach
`max_leverage` snapshot-and-rollback inside the action so a partial
mutation can't leak.

### Step-level non-atomic

The framework does *not* wrap a list of actions returned by `predict()`
in a transaction. If action 2 raises after action 1 succeeds, action 1
stays applied and the run aborts. This is intentional: backtest runs
fail loudly when a strategy emits an invalid action sequence; we don't
silently mask logic bugs with rollback. Strategies must keep their own
sequences consistent — typically by emitting the dependent actions as a
single bundle in `predict`, not by computing one then re-deriving the
other.

### Delegate-resolved arguments

Action arg values can be **callables** that the framework resolves at
execute time:

```python
ActionToTake(entity_name="HEDGE", action=Action(
    "deposit", {"amount_in_notional": lambda s: s.get_entity("SPOT").internal_state.cash}
))
```

The lambda receives the strategy and returns a float. This is critical
for action sequences that depend on state mutated by prior actions in
the same step:

```python
return [
    ActionToTake("SPOT",  Action("sell",     {"amount_in_product": x})),
    ActionToTake("HEDGE", Action("deposit",  {"amount_in_notional": _spot_cash_now})),
    ActionToTake("SPOT",  Action("withdraw", {"amount_in_notional": _spot_cash_now})),
]
```

`_spot_cash_now` is a delegate that resolves to `SPOT.cash` *at execute
time*, after the `sell` ran but reused by both the `deposit` and the
following `withdraw`. Without delegates the strategy author would have
to know exactly how `SPOT.sell` computes its cash credit (factoring in
fees, slippage, …) — instead it just asks the entity afterwards.

Order matters: the `deposit` must come *before* the `withdraw` in this
sequence, otherwise the withdraw zeroes `SPOT.cash` before the deposit
delegate evaluates. Action lists are executed top-to-bottom.

## Strategies as orchestrators

`BaseStrategy` is a thin coordinator:

- `set_up()` registers entities under unique names. Called once from
  `__init__`.
- `predict()` is called once per observation. Returns a (possibly empty)
  list of `ActionToTake`.
- `step(observation)` and `run(observations)` drive the framework loop.

Hyperparameters live on a `BaseStrategyParams` subclass — typically a
dataclass — referenced through the `BaseStrategy[Params]` generic:

```python
@dataclass
class MyParams(BaseStrategyParams):
    INITIAL_BALANCE: float = 1_000_000
    TARGET_LEVERAGE: float = 3.0

class MyStrategy(BaseStrategy[MyParams]):
    ...
```

The generic argument is the source of truth for `PARAMS_CLS`, which is
what `set_params` uses to coerce dict-shaped grid cells:

```python
strategy = MyStrategy(params={"INITIAL_BALANCE": 500_000, "TARGET_LEVERAGE": 2.0})
# → PARAMS_CLS(**dict) — defaults flow through, unknown keys raise TypeError
```

If your strategy needs extra fields beyond what the parent's PARAMS_CLS
declares, override `PARAMS_CLS` explicitly on the subclass — this is
how `HyperliquidBasis` adds `EXECUTION_COST` to the basis-trading params:

```python
class HyperliquidBasis(BasisTradingStrategy):
    PARAMS_CLS = HyperliquidBasisParams   # subclass with extra fields
```

## Notional accounting and unit conventions

Entities are **unit-agnostic**: `GlobalState.price` / `mark_price` /
`collateral_price` are whatever scale the strategy supplies, and
`entity.balance` returns the same scale. This gives flexibility (run
USD-, ETH- or BTC-denominated strategies with the same code) and shifts
responsibility to the strategy author to keep units **consistent across
all entities** in a single run.

### Sign conventions (uniform across all entities and loaders)

| Quantity | Positive ⇒ |
|---|---|
| `lending_rate` | collateral grows per step |
| `borrowing_rate` | debt grows per step |
| `funding_rate` | longs pay shorts |
| `trading_fee` | execution cost on traded notional |

Loaders in `fractal.loaders` follow these uniformly. The Aave V3 loader
historically flipped `borrowing_rate` sign — that was a bug, fixed in
v1.4.0; see [`CHANGELOG.md`](CHANGELOG.md).

### Pricing recipes

**USD-denominated (default).** Pass every entity prices in `<asset>/USD`:

```python
aave.update_state(AaveGlobalState(collateral_price=1.0, debt_price=3000.0))   # USDC=$1, ETH=$3000
spot.update_state(UniswapV3SpotGlobalState(price=3000.0))                      # ETH/USD
hl.update_state(HyperliquidGlobalState(mark_price=3000.0))                     # ETH/USD
strategy.total_balance                                                          # USD
```

**ETH-denominated (opt-in).** Same rule, different unit:

```python
aave.update_state(AaveGlobalState(collateral_price=0.000333, debt_price=1.0))  # USDC≈0.000333 ETH
strategy.total_balance                                                          # ETH
```

### Direction flag for lending entities

`AaveEntity` and `SimpleLendingEntity` accept `collateral_is_volatile`:

- `False` (default) — stable collateral, volatile debt (synthetic short).
- `True` — volatile collateral, stable debt (leveraged-long setup).

The flag is informational; the math is symmetric. It only affects how
strategies read the position when computing things like
`liquidation_price`. See `tests/core/e2e/test_e2e_leveraged_long.py`.

### Common pitfalls

- **Inverted price.** Passing `price = 1/3000` instead of `3000` is a
  silent corruption of every downstream balance. Always re-derive the
  convention from the data source (subgraph token0Price vs token1Price,
  oracle quote currency, …).
- **Mixed units in a single strategy.** Math still runs but
  `total_balance` becomes meaningless. Verify by computing a known
  scenario and checking `entity.balance` in the unit you expect.

## Loaders, caching and the data pipeline

Each loader implements the `Loader` ABC with a three-step lifecycle:

```
extract  → fetch raw data from the source (HTTP, GraphQL, file, simulation)
transform → coerce into a typed pandas-backed structure (PriceHistory, …)
load      → write a deterministic-named cache file under <DATA_PATH>/fractal_data/<class>/<key>.<ext>
```

`run()` is `extract → transform → load`; `read(with_run=False)` reads
the cache. Cache keys are derived from all parameters that affect the
output (ticker + window + interval + …) so two instances with identical
inputs share the same on-disk file.

### Loader return types (`fractal.loaders.structs`)

- `PriceHistory` — single-column `price`, `DatetimeIndex`
- `FundingHistory` / `RateHistory` — single-column `rate`
- `LendingHistory` — `lending_rate` + `borrowing_rate`
- `PoolHistory` — `tvl` / `volume` / `fees` / `liquidity` (+ optional `price`)
- `KlinesHistory` — OHLCV
- `TrajectoryBundle` — `List[PriceHistory]` for Monte-Carlo simulators

All are subclasses of `pd.DataFrame`, so callers can `.join` and
`.loc[start:end]` them directly. The constructor enforces a UTC
`DatetimeIndex` named `time` for join-compatibility across loaders.

### Cache root

`DATA_PATH` env var first; falls back to `cwd`. Loader cache is *opt-in*
— a fresh `read(with_run=True)` always re-fetches and overwrites. To
persist results across runs, drop `with_run=True` after the first call.

`PYTHONPATH` is **not** consulted as a filesystem root (it's a
colon-separated import list, not a directory).

## MLflow pipelines and the experiment grid

`DefaultPipeline` wraps a strategy + observation set + parameter grid
and writes everything to one MLflow experiment, one MLflow run per
grid cell:

```python
mlflow_config = MLflowConfig(mlflow_uri="http://localhost:5500", experiment_name="my_exp")
exp_config = ExperimentConfig(
    strategy_type=MyStrategy,
    backtest_observations=observations,
    params_grid=ParameterGrid({"TAU": [10, 20, 30]}),
    window_size=24*7,            # sliding-window scenarios in addition to backtest
    debug=True,                  # log per-strategy debug log file as artifact
)
pipeline = DefaultPipeline(mlflow_config, exp_config)
pipeline.run()
```

Each MLflow run gets:

| Per run |
|---|
| **params** — all fields of the grid cell (via `_params_to_dict` coercion) |
| **metrics** — `accumulated_return`, `apy`, `sharpe`, `max_drawdown` |
| **artifact `strategy_backtest_data.csv`** — the full per-step DataFrame |
| **artifact `window_trajectories_metrics.csv`** (if `window_size`) — per-window metrics |
| **secondary metrics** (if `window_size`) — mean / q05 / q95 / cvar05 across windows |
| **artifact `logs/`** (if `debug=True`) — the strategy's loguru log directory |

### Lazy connect

The pipeline does **not** open the MLflow connection during `__init__`.
Network access is deferred to the first `run()` or `grid_step(params)`
call (`_ensure_connected`). This makes the pipeline trivially testable
and importable.

### AWS credentials

`MLflowConfig.aws_access_key_id` / `aws_secret_access_key` are only
injected into the environment when explicitly provided and non-empty.
Empty strings or `None` leave the host's existing credentials (or AWS
profile chain) intact. Local Docker MLflow with `--serve-artifacts`
doesn't need AWS at all.

### Running a real MLflow server locally

`tests/mlflow_tests/docker-compose.yml` ships a self-contained Docker
service — sqlite backend, filesystem artifact store, proxy artifact
serving (so client uploads via HTTP, no host-side mount required):

```bash
bash tests/mlflow_tests/scripts/start_mlflow.sh   # MLflow at :5500
export MLFLOW_URI=http://localhost:5500
PYTHONPATH=. python examples/basis/grid.py
bash tests/mlflow_tests/scripts/stop_mlflow.sh    # tear down
```

The same harness drives a 4-pipeline end-to-end test
(`bash tests/mlflow_tests/scripts/e2e.sh`) that verifies every layer:
strategy run, MLflow logging, artifact upload, pipeline orchestration.

## Testing layers

Tests are categorized by pytest markers and live in three trees:

| Marker | Path | Scope |
|---|---|---|
| `core` | `tests/core/`, `tests/loaders/test_*_offline.py`, `tests/loaders/test_simulations_loaders.py` | Pure-Python unit, invariant, and synthetic e2e — no network, no API keys. |
| `slow` | `tests/core/e2e/test_*_real_data.py`, `tests/core/test_*_real_data.py`, `tests/loaders/test_uniswap_v3_pool_loader.py::test_uniswap_v3_data_types` | Real-data CSV replays from `examples/`. No network for most; some need `THE_GRAPH_API_KEY`. |
| `integration` | `tests/loaders/test_*_loader.py` (Aave, Binance, Hyperliquid, GMX, Uniswap V2/V3, Lido) | Live API tests. Some need `THE_GRAPH_API_KEY`. |

Default `pytest` runs only the `core` layer (set in `pytest.ini::addopts`).
The end-to-end Docker MLflow harness is invoked separately as a shell
script and is not collected by pytest.

### Property and invariant tests

`tests/core/invariant_testing/` contains randomized property tests over
randomized seeds (basis ratio, equity conservation, leverage bounds,
range invariance, …). Seeds are stable so failures are deterministic.

### Lock-in tests

Each closed bug gets a regression test that pins the exact behaviour.
These live alongside the unit tests of the relevant module — search for
"lock-in" in docstrings.

## Logging and observability

`DefaultLogger` writes a per-run debug log via loguru when
`Strategy(debug=True)` is set. Two design choices:

- **Per-instance handler with a filter.** Each strategy instance gets a
  unique loguru handler tagged with its run id; the strategy logger
  binds that id and the handler filters by it. Parallel debug runs do
  not contaminate each other.
- **Default stderr sink stripped once per process.** Loguru installs a
  default stderr sink at handler id 0 on import. We strip it the first
  time `DefaultLogger` is instantiated so `debug=True` doesn't fan out
  to the console — strategies log to file only. User-added sinks
  (handler id ≥ 1) are left untouched.

`MLflowConfig.debug=True` flag propagates through `ExperimentConfig.debug`,
which causes the pipeline to upload the strategy's `logs/` directory to
the MLflow run as an artifact alongside the backtest CSV. Useful for
explaining single-run outliers in a grid.

Run output directory: `FRACTAL_RUNS_PATH` env var first, falls back to
`cwd`. Pattern: `runs/<StrategyClass>/<uuid>/{logs,datasets}`.
