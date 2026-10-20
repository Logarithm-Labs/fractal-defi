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
venv/bin/python examples/morpho_supply/discover.py --chains 1,8453

# Optional live formula consistency checks.
venv/bin/python examples/morpho_supply/validate_formulas.py 100

# Only after reviewing registry/<chain_id>.csv:
venv/bin/python examples/morpho_supply/backtest.py --chains 1 --days 30 --top 8
venv/bin/python examples/morpho_supply/backtest.py --strategy allocation --mode marginal \
    --chains 1,8453 --days 30 --top 20 --target-apy 0 --hold-horizon-bars 24

# Replication of Baude, Danos & El Khalloufi (arXiv:2608.24206).
venv/bin/python examples/morpho_supply/baude_replication.py paper
venv/bin/python examples/morpho_supply/baude_replication.py ladder --days 90
```

These commands access the live Morpho API. They are not part of the offline
test suite. Registries and generated `results/` stay gitignored.

## Strategies in `backtest.py`

`--strategy rotation` (default) runs the single-position rotation strategy.
`--strategy allocation` runs the allocator on the same listings and observation
stream:

- `--mode average` equalizes the whole-position rate across funded markets
  above `--target-apy`; `--min-rebalance-fraction` skips small resizes.
- `--mode marginal` maximizes income net of costs over `--hold-horizon-bars`.

The allocator's report prints the last solution: position count, cash share,
gated legs, the largest targets and `lambda_apy`. That value is the common
whole-position APY in average mode and the lowest last-chunk marginal APY among
funded pools in marginal mode. Default flag values illustrate the interface, not
recommended allocations.

Costs: Morpho supply and withdraw carry no protocol fee or slippage, so the
default cost is gas only. Each leg pays a fixed USD amount from an illustrative
per-chain table in `helpers.py`; set your own figures for an experiment.
`--gas-usd-per-leg` replaces the table with one figure for every chain.
`--cost-bps` adds a proportional cost per leg, which fits swaps (for example
USDC to USDT) or bridges. This example models neither, so the default is 0.

## Baude et al. replication

`baude_replication.py` reruns the backtest of Baude, Danos & El Khalloufi,
"Capital allocation on decentralized lending platforms"
([arXiv:2608.24206](https://arxiv.org/abs/2608.24206)), with this allocator.
Both experiments write a CSV to `results/`.

- `paper` uses the paper's window (2026-01-01 to 2026-04-01) and Ethereum USDC
  market sets. Its protocol is the paper's: daily re-solve on one-day averages,
  no gas, borrows held fixed, $100k and $100M budgets, and the whole budget in
  USDC/cbBTC as the benchmark. It also runs our hourly protocol with gas, and
  prints the paper's published value next to each paper-protocol row. No
  registry is needed.
- `ladder` uses a recent window on the reviewed registry. It relaxes one choice
  at a time: hourly re-solve, all Ethereum USDC markets with PT collateral
  allowed, USDT, all chains, then per-chain gas. Use `--end YYYY-MM-DD` for a
  fixed window.

The paper lists its markets by Morpho-API ids that the current API no longer
serves. Each is mapped to the largest Ethereum USDC market with that collateral
on 2026-01-01; the weETH and syrupUSDC mappings are uncertain because both
markets were below $1M then. The paper protocol ignores gas and holds borrows
fixed, so it is an in-sample replication of a published backtest, not evidence
of returns.

## Interpretation and limitations

- Evidence is **synthetic** for offline tests and **provider-backed** for these
  live-data examples. A successful run is a smoke test, not evidence of returns.
- Timestamps are UTC. Signals lag by observed bars. A market missing the current
  bar is excluded from that bar's eligible universe: strategies require
  current-bar freshness, and holdings whose feed goes stale are held, not
  traded on old state. Stale liquidity is not an executable quote.
- Universe selection uses today's listing/liquidity data. Applying that universe
  retrospectively introduces survivorship/selection bias.
- Market fees enter only through the observed supply APY, which is already net
  of the fee; the own-impact model scales that APY by a curve ratio in which
  the fee cancels, so no fee history is needed.
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
| `validate_formulas.py` | Live supply-rate, IRM and liquidity consistency checks |
| `helpers.py` | Registry, UTC history panels and observations |
| `backtest.py` | Rotation or allocation backtest, metrics and strategy report |
| `baude_replication.py` | Replication of Baude et al. (2026): paper window and design ladder |
| `registry/`, `results/` | Local-only artifacts |
