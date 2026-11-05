# DEX LP Position Replication Backtest

Replicate real on-chain Uniswap V3 LP positions in Fractal 1:1 and
validate the backtested fees against the chain's own accounting.

Each position from `positions.json` is re-opened in a backtest with its
exact mint token amounts, tick range and mint price, then driven over
native hourly pool snapshots from TheGraph. The accrued fees are
compared per token leg with the on-chain ground truth — the un-floored
`feeGrowthInside` accrual read straight from the pool contracts.

---

## Files

| File | Purpose |
|---|---|
| `positions.json` | Registry of real positions: ticks, mint amounts and block, exact mint price, `protocol_fee`. |
| `helpers.py` | Registry loading, on-chain ground truth (web3.py), observation building, mint-anchor correction. |
| `backtest.py` | The replication run (`fee_model="fee_growth"`). Saves per-bar trajectories + `results/summary.csv`. |
| `methods_grid_backtests.py` | Reproducible comparison of three fee-accounting methods over the same positions. |

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env       # fill in THE_GRAPH_API_KEY (+ BASE_RPC_URL for the grid)
set -a && source .env && set +a
```

## Run

```bash
# replication with exact per-leg fee accounting
python backtest.py

# three-method comparison grid (needs eth_getLogs on the RPC)
python methods_grid_backtests.py
```

## Outputs (`results/`)

* `trajectory_<position_id>.csv` — per-bar entity states and balances
  for every replicated position.
* `summary.csv` — one row per position:
  `replicated_position_id, pair, pool_address, fee_tier,
  fees_accrued_{usd,token0,token1}`, the same `_onchain`, and the
  `usd/token0/token1_ratio` accuracy columns.
* `methods_grid_results.csv` — per position × method: accuracy ratios
  vs the on-chain accrual (USD and per token leg) and timings (pure
  backtest compute and full wall time incl. data loading; the hourly
  data is pulled once per position and shared by both Fractal models).

Ratio semantics: a ratio is NaN only when the on-chain reference is
zero/missing; a backtest that accrued exactly nothing against positive
truth reads as `0.0`. Positions with no hourly bars after the mint yet
(fresh mints in quiet pools) are skipped with a warning.

## Fee-accounting methods

| Method | Data | Idea |
|---|---|---|
| `fee_growth` | ~130 hourly subgraph rows/pool | Per-bar deltas of the pool's cumulative `feeGrowthGlobal` counters × position `L`. The chain performs the per-swap liquidity weighting itself and nets out the protocol fee, so the backtest reads the pool's own ledger — per-leg ratios land at ~1.000 of the on-chain accrual. |
| `aggregate` | same hourly rows | Classic estimate: `feesUSD × L_pos/(L_pool+L_pos)` per bar × `(1 − protocol_fee)`. Fallback for synthetic/counterfactual data; end-of-bar liquidity snapshots overstate thin bursty pools. |
| `per_event` | every Swap log since mint (up to 10^5+/pool) | Event-replay reference: per-swap fee share with the event's own liquidity (`fee = gross_input × tier/1e6` — the event amounts include the fee). Highest data cost; still approximate for swaps that cross ticks. |

## Notes

* USD == USDC; other tokens are priced from the ticks of USDC-quoted
  reference pools — no external price feeds.
* Ground truth needs only head-state RPC reads (`positions()`,
  `slot0()`, `ticks()`, `feeGrowthGlobal*()`): the mint-side checkpoint
  is stored in the position manager. An archive-capable node makes the
  first-bar mint correction exact (one `eth_call` at the mint block);
  on a plain node it degrades to time-proration — accurate on dense
  pools, approximate on sparse bursty ones.
* The feeGrowth window is also anchored at the head: subgraph hourly
  rows snapshot the counters before the triggering swap's own fee, so
  the newest row lags the chain by up to one swap — a synthetic final
  bar carries the not-yet-indexed growth (`append_head_bar`).
* On Base, V3 pools run with the protocol-fee switch ON
  (`slot0.feeProtocol`): the protocol takes 1/4 of swap fees on the
  100/300/500 tiers and 1/6 on 3000/10000 — `positions.json` carries
  the measured `protocol_fee` per pool. `feeGrowth` counters are
  already net of it.
