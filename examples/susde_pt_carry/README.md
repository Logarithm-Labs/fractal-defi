# sUSDe PT Carry: Leveraged Loans With and Without a Boros Overlay

The strategy of PR #83 on real data: the leveraged PT-sUSDe loan on
Morpho, alone and with a long Boros yield unit as an overlay on the PT's
mark-to-market, on eight instruments.

* **Loop** (`RATE_HEDGE="none"`): buy PT-sUSDe, post it on Morpho, borrow
  the stablecoin, buy more PT; hold to expiry. Fixed PT yield against a
  floating borrow rate.
* **Loop + Boros** (`"boros"`): the same loop plus long yield units on the
  Boros ETHUSDT (or BTCUSDT) market. When the market's implied APY rises
  the PT is marked down; the unit's mark-to-maturity gains, so it offsets
  part of that move. It is sized as a hedge ratio on the PT value
  (`HEDGE_BETA` = 0.10, the pooled regression estimate below), capped by
  the margin parked in the leg (10 % of the capital), re-synced after every
  loop action, and held to its own maturity after the PT is redeemed
  (`BOROS_EXIT_POLICY="settle"`).

All data is keyless: Pendle, Morpho, Boros, Binance. Boros markets list
roughly five weeks before their maturity (the MAR2026 units from
2025-11-27), so on most full-life markets the leg opens part-way through
the hold.

## Files

| File | Purpose |
|---|---|
| `markets.json` | Eight markets: PT-sUSDe / PT-USDe maturities SEP2025, NOV2025 (ETH and BTC units), FEB2026 (ETH and BTC units), MAY2026 on their Morpho markets, daily to redemption; PT-sUSDe-26NOV2026 and PT-sUSDS-26NOV2026 live on hourly bars. The Boros market is the ETH/BTC-USDT yield unit whose maturity is closest after the PT's (MAY2026 uses the MAR2026 unit, which matures first). |
| `helpers.py` | Frame builder (Pendle + Morpho from `examples/pendle_pt_backtests`, plus Binance price/funding and the Boros mark, extended to the unit's maturity) and per-variant observations. |
| `run.py` | Both variants on every market → `results/validation.csv` and the showcase trajectories `results/susde_27nov2025_usds_{none,boros}.csv`. |
| `grid.py` | Target LTV × hedge kind × `HEDGE_BETA` × margin share × exit policy through `DefaultPipeline`; one MLflow run per cell. |
| `analysis.ipynb` | All markets side by side; the showcase's equity curves, PnL decomposition, APY / drawdown, costs and the Boros leg (coverage, settlements, mark-to-maturity). |

```bash
python run.py            # or: python run.py susde_26nov2026
MLFLOW_URI=http://localhost:5000 python grid.py susde_27nov2025_usds
python -m nbconvert --to notebook --execute --inplace analysis.ipynb
```

## Results (2026-09-21, `INITIAL_BALANCE` 100k, target LTV 0.80, `HEDGE_BETA` 0.10, margin share 10 %)

| Market | Variant | Leverage | PT APY | Borrow | Funding (ann.) | Boros mark | Realised APY | Max DD | Coverage | Leg PnL |
|---|---|---|---|---|---|---|---|---|---|---|
| sUSDe-25SEP2025 / DAI, daily | loop | 4.33× | 8.1 % | 7.9 % | 5.8 % | 6.7 % | **+9.8 %** | -3.4 % | — | — |
|  | + Boros | 3.90× |  | | |  | +8.4 % | -3.1 % | 0.05 | -145 |
| sUSDe-27NOV2025 / USDS, daily | loop | 4.33× | 9.4 % | 2.8 % | 3.9 % | 5.3 % | **+52.5 %** | -0.7 % | — | — |
|  | + Boros | 3.90× |  | | |  | +45.8 % | -0.6 % | 0.04 | -16 |
| USDe-27NOV2025 / USDS, daily | loop | 4.33× | 9.3 % | 2.8 % | 4.9 % | 5.6 % | **+33.6 %** | -0.5 % | — | — |
|  | + Boros | 3.90× |  | | |  | +29.5 % | -0.4 % | 0.04 | +4 |
| sUSDe-5FEB2026 / USDC, daily | loop | 4.33× | 6.2 % | 4.2 % | 1.3 % | 3.7 % | **+18.6 %** | -0.1 % | — | — |
|  | + Boros | 3.90× |  | | |  | +8.6 % | -0.1 % | 0.10 | -285 |
| USDe-5FEB2026 / USDC, daily | loop | 4.33× | 5.3 % | 0.4 % | 2.3 % | 4.0 % | **+23.0 %** | -0.5 % | — | — |
|  | + Boros | 3.90× |  | | |  | +10.1 % | -0.5 % | 0.10 | -464 |
| sUSDe-7MAY2026 / PYUSD, daily | loop | 4.33× | 3.7 % | 2.9 % | -1.3 % | 0.7 % | **+6.5 %** | -0.2 % | — | — |
|  | + Boros | 3.90× |  | | |  | +5.6 % | -0.2 % | 0.03 | -14 |
| sUSDe-26NOV2026 / USDC, hourly | loop | 4.33× | 4.1 % | 3.3 % | 5.6 % | 5.6 % | **-0.8 %** | -0.6 % | — | — |
|  | + Boros | 3.90× |  | | |  | +3.3 % | -0.6 % | 0.10 | +350 |
| sUSDS-26NOV2026 / USDC, hourly | loop | 4.33× | 5.3 % | 4.8 % | 5.2 % | 4.9 % | **+9.7 %** | -0.4 % | — | — |
|  | + Boros | 3.90× |  | | |  | +9.7 % | -0.4 % | 0.10 | +120 |

Reading the numbers:

* **Leverage pays when the spread is wide.** Autumn 2025 (NOV2025 PTs at
  9.3–9.4 % implied against 2.8 % USDS borrow) returned +34 % / +53 %
  held to redemption; FEB2026 +19 % / +23 % (USDe/USDC borrow was near
  zero); MAY2026 +6.5 % on a 0.8 pp spread; the live sUSDe market is
  mark-to-market negative (implied APY rose 4.1 → 4.9 %). No run was
  liquidated (minimum health factor 1.18).
* **The overlay is small by design.** The regression of the PT's
  mark-to-market on a yield unit's (both per unit of notional, daily,
  eight markets, 278 market-days) gives a pooled hedge ratio of −0.10
  with R² 0.04: the sign is right (long units offset a PT mark-down) but
  the two marks share little variance, so the ratio-optimal overlay is a
  tenth of the PT value and removes at most 4–14 % of the mark-to-market
  variance. Larger ratios add variance instead of removing it.
* **What it costs.** With the unit held to its own maturity the leg pays
  `fixed − funding` until then: on the FEB2026 markets the MAR2026 unit
  ran seven weeks past the PT at 3.7–4.0 % fixed against 1.3–2.3 %
  funding, −285 / −464 on the parked margin. On the live markets the
  implied APR rose while the unit was on and the leg added +350 / +120.
* **Borrow rates are not funding.** The Morpho borrow APY regressed on
  ETH or BTC funding gives R² 0.03–0.11 within markets with an unstable
  sign, so a yield unit cannot fix the loan's floating cost; the
  `"debt"` sizing is kept for that reading but is not the default.

## Modelling notes

* Yield units are in ETH or BTC; the target is `HEDGE_BETA × PT value /
  price` (or `HEDGE_RATIO × debt` under `"debt"` sizing), capped at what
  the parked margin supports under Boros's initial-margin rule with a
  10 % buffer. Each re-size pays the taker fee; `HEDGE_REBALANCE_THRESHOLD`
  (5 %) limits how often the price drift alone triggers one.
* Funding is Binance's 8-hour rate summed per bar and applied as one
  settlement per bar; `settlementApr` on Boros equals that rate × 1095
  (verified live). The settlement fee (0.1 %/yr) is charged by the entity.
* After the PT is redeemed the observations carry only the Boros state,
  so the unit keeps settling until the entity closes it at maturity at
  zero cost.
* The loop is `MorphoLeveragedPT` unchanged (weekly-smoothed carry gate,
  band 0.70–0.88, repay sized on the PT entity's own quote); the
  `"none"` variant reproduces `examples/pendle_pt_backtests` to the
  last digit.
