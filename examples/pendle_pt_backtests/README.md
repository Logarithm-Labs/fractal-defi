# Pendle PT Backtests: Leveraged Looping and Hedged Carry

Two strategies on real Pendle, Morpho Blue and Boros data, with the
numbers checked against closed forms.

* **Leveraged PT** (`MorphoLeveragedPT`): buy PT, post it as collateral
  on a Morpho market, borrow, buy more PT, repeat; hold to expiry and
  redeem at par. Carry ≈ `L·y_pt − (L−1)·r_borrow`.
* **Hedged PT** (`PerpHedgedPT`): long PT of a volatile underlying,
  short the underlying on a perp so only the fixed yield is kept;
  optionally sell the same notional of Boros yield units to swap the
  perp's floating funding for a fixed rate.

All data comes from keyless public endpoints (Pendle, Morpho, Boros,
Binance). Pendle serves hourly history for roughly the last 60 days
and daily history for a market's whole life, so long windows run on
daily bars.

---

## Files

| File | Purpose |
|---|---|
| `markets.json` | Registry of the markets: Pendle market and Morpho market ids, LLTV, window, bar size, AMM parameters (`scalar_root`, `ln_fee_rate_root` from `readState`) and the impact model. |
| `helpers.py` | Registry loading, observation builders that join the loaders on one grid, the synthetic expiry bar, closed-form carry and summaries. |
| `leveraged_pt_backtest.py` | Runs `MorphoLeveragedPT` on every leveraged market; writes `results/leveraged_<market>.csv` and a row in `results/validation.csv`. |
| `hedged_pt_backtest.py` | Runs `PerpHedgedPT` with and without Boros; writes `results/hedged_<market>_{perp,boros}.csv` and validation rows. |
| `results/` | Reference trajectories and `validation.csv` (also the fixtures of the `slow` real-data tests). |

## Setup and run

```bash
cp .env.example .env            # optional: DATA_PATH for the loader cache
python leveraged_pt_backtest.py  # all leveraged markets, or pass keys: python leveraged_pt_backtest.py usde_25sep2025
python hedged_pt_backtest.py
```

## Markets

| Key | Pendle | Lender / hedge | Window | Why |
|---|---|---|---|---|
| `usde_25sep2025` | PT-USDe-25SEP2025 | Morpho PT-USDe / USDe (LLTV 0.915) | 2025-05-22 → expiry, daily | Full life to redemption; the loan is the accounting asset, so the loop needs no swap. |
| `susde_26nov2026` | PT-sUSDe-26NOV2026 | Morpho PT-sUSDE / USDC (LLTV 0.915) | last 55 days, hourly | Live market on hourly bars with the exact AMM replay (`impact_model="amm"`). |
| `wsteth_30dec2027` | PT-wstETH-30DEC2027 | Binance ETHUSDT perp; Boros BINANCE-ETHUSDT-25DEC2026 | 2026-03-01 → now, daily | Volatile underlying: the hedge removes the ETH leg; Boros fixes the funding from its first quote (2026-05-12). |

## Results (`results/validation.csv`, runs of 2026-09-11)

| Run | Bars | Leverage | PT APY at entry | Borrow APY (mean) | Realised APY | Closed form | Note |
|---|---|---|---|---|---|---|---|
| `usde_25sep2025` | 127 daily | 4.33× | 8.66 % | 10.42 % | **+6.3 %** | +2.8 % | Held to redemption; no liquidation, min health factor 1.19. Realised beats the closed form because the position was de-levered during the July rate spike (weekly-smoothed gate) and re-levered at a higher implied APY. |
| `susde_26nov2026` | 531 hourly | 4.33× | 4.07 % | 3.07 % | −4.8 % (annualised) | +7.4 % | 22-day mark-to-market window, not held to expiry: the implied APY rose 4.07 → 4.92 %, which marks a 4.3× PT position down by about 1 %. |
| `wsteth_30dec2027_perp` | 194 daily | 2× hedge | 2.38 % | — | **+3.2 %** | — | ETH moved +25.7 % over the window and the equity path stayed flat to it; funding received summed to 1.06 %. |
| `wsteth_30dec2027_boros` | 122 daily | 2× hedge | 2.38 % | — | +1.9 % | — | From Boros's first quote; fixed at 1.99 % APR while realised funding ran near 3.8 % annualised, so locking the rate cost about 1.3 pp here. |

Closed form: `L·y_pt − (L−1)·r_borrow` with the entry implied APY and the
window-mean borrow APY; it ignores swap costs, the accrual timing and
any de-leveraging, so the gap is reported rather than asserted.

## Modelling notes

* PT price is Pendle's compounded convention `(1 + apy)^(−t)` in the
  accounting asset; the USD marks from the API are carried alongside.
  Swap cost is either the `MarketMathCore` replay (needs `scalar_root`
  and `ln_fee_rate_root`, read once from `readState` and pinned in the
  registry) or the rate-spread approximation.
* The lender's oracle price drives health and liquidation; the market
  price drives PnL. The registry's `oracle_model` is `"market"` for
  these runs (the oracle tracks the Pendle price); `"linear"` with a
  `base_discount` reproduces a `PendleSparkLinearDiscountOracle`.
* Morpho rates arrive as effective APYs and are converted to the
  per-bar exponent `ln(1 + apy)·Δt/YEAR` that the entity compounds.
* Morpho borrow rates spike for hours at a time. The carry gates use a
  one-week mean (`CARRY_GATE_LOOKBACK_BARS`); with the raw rate the
  strategy whipsawed in and out of the loop and paid the swap costs on
  every spike (that run lost 8 % on the USDe market).
* Pendle's history ends on the expiry day; the builder appends the
  maturity bar so the strategy redeems at par.
* Binance funding (8-hour) is summed per bar as a cash flow; on the
  daily grid the perp and the Boros yield units settle once a day with
  the day's total. Boros `settlementApr` equals Binance funding × 1095
  at the same timestamps (verified live), so the plain funding feed is
  the floating leg.
* Boros candles before a market's first trade are all-zero rows; the
  loader drops them with a warning, and the Boros run starts at the
  first quote.
