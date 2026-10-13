# Morpho Blue supply strategies

A backtesting example for supplying loan tokens to isolated Morpho Blue markets.
The library provides two strategies:

- `MorphoSupplyRotationStrategy`: one primary position, ranked on a lagged
  trailing signal; liquidity-blocked balances may remain in other markets.
- `MorphoSupplyAllocationStrategy`: multiple positions using equal-average-rate
  allocation or a marginal-income grid objective with switching costs.

## Run the example

From the repository root, using the project environment:

```bash
# Download a candidate registry; inspect it before running a backtest.
venv/bin/python examples/morpho_supply_rotation/discover.py --chains 1,8453

# Optional live formula consistency checks.
venv/bin/python examples/morpho_supply_rotation/validate_rq1.py 100

# Only after reviewing registry/<chain_id>.csv:
venv/bin/python examples/morpho_supply_rotation/backtest.py --chains 1 --days 30 --top 8
```

These commands access the live Morpho API. They are not part of the offline
test suite. Registries and generated `results/` stay gitignored.

## Use the allocator

The same reviewed listings and observation stream can be passed to the allocator:

```python
from fractal.core.entities.simple.spot import SimpleSpotExchange
from fractal.strategies import MorphoSupplyAllocationParams, MorphoSupplyAllocationStrategy

strategy = MorphoSupplyAllocationStrategy(
    markets=listings,
    cash_entity=SimpleSpotExchange(trading_fee=0.0),
    params=MorphoSupplyAllocationParams(
        ALLOCATION_MODE="marginal",
        TARGET_APY=0.0,
        SIGNAL_LAG_BARS=1,
        HOLD_HORIZON_BARS=24,
    ),
)
strategy.get_entity("CASH").action_deposit(10_000.0)
result = strategy.run(stream)
```

`listings` and `stream` come from the registry/panel helpers used by
`backtest.py`. Values above illustrate the API, not recommended allocations.
Configure gas, proportional costs and the holding horizon for the experiment;
default zero costs do not represent free execution.

## Interpretation and limitations

- Evidence is **synthetic** for offline tests and **provider-backed** for these
  live-data examples. A successful run is a smoke test, not evidence of returns.
- Timestamps are UTC. Signals lag by observed bars; missing bars do not advance
  a market's buffer and its previous state can remain eligible. Stale liquidity
  is not an executable quote.
- Universe selection uses today's listing/liquidity data. Applying that universe
  retrospectively introduces survivorship/selection bias. Current market fees
  are applied across the history because this loader has no historical fee series.
- Hourly data misses intra-hour liquidity shortages and rate changes.
- Amounts are USD notional at a stablecoin price of one. Bridging costs, bridge
  latency, depegs, bad debt, oracle failures and reward-token price risk are omitted.
- Own-position impact is a counterfactual model. At zero absorption the position
  adds its own withdrawable liquidity, so zero blocked exits follows by construction.
  Controller feedback is optional; borrower response is not estimated.
- Rewards are optional, diluted by added supply, and credited as yield rather than
  separate claimable tokens. Missing reward observations contribute zero.
- Average mode equalizes average rates, not marginal income. Marginal mode uses
  a finite capital grid and a fixed-horizon objective; it is not a continuous
  global optimum. `MAX_POSITIONS` applies only to average mode.
- The printed static reference is an unweighted mean of market APYs, not a
  compounded, executable portfolio or a vault benchmark.

## Files

| File | Purpose |
| --- | --- |
| `discover.py` | Candidate address-pinned registries for human review |
| `validate_rq1.py` | Live supply-rate, IRM and liquidity consistency checks |
| `helpers.py` | Registry, UTC history panels and observations |
| `backtest.py` | Single-position rotation, metrics and deferral reporting |
| `registry/`, `results/` | Local-only artifacts |
