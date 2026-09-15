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
| `sensitivity.py` | Grid over target LTV, loop count, carry-gate smoothing and the oracle model on one leveraged market; writes `results/sensitivity_<market>.csv`. |
| `results/` | Reference trajectories, `validation.csv` and the sensitivity grids (also the fixtures of the `slow` real-data tests). |

## Setup and run

```bash
cp .env.example .env            # optional: DATA_PATH for the loader cache
python leveraged_pt_backtest.py  # all leveraged markets, or pass keys: python leveraged_pt_backtest.py usde_25sep2025
python hedged_pt_backtest.py
python sensitivity.py reusd_10dec2026   # parameter grid on one market
```

## Markets

| Key | Pendle | Lender / hedge | Window | Why |
|---|---|---|---|---|
| `usde_25sep2025` | PT-USDe-25SEP2025 | Morpho PT-USDe / USDe (LLTV 0.915) | 2025-05-22 → expiry, daily | Full life to redemption; the loan is the accounting asset, so the loop needs no swap. |
| `susde_25sep2025_dai` | PT-sUSDe-25SEP2025 | Morpho PT-sUSDE / DAI (LLTV 0.915) | 2025-06-05 → expiry, daily | Full life through the September-2025 carry compression; the loan is not the accounting asset (stable-to-stable). |
| `reusd_10dec2026` | PT-reUSD-10DEC2026 | Morpho PT-reUSD / USDC (LLTV 0.915) | 2026-07-15 → 2026-09-10, hourly | Event replay: the 2026-08-25 implied-APY spike that liquidated $36M on this market. Pendle serves hourly rows for ~60 days, so re-runs after mid-October stretch the older bars from daily (the loader warns). |
| `susde_26nov2026` | PT-sUSDe-26NOV2026 | Morpho PT-sUSDE / USDC (LLTV 0.915) | last 55 days, hourly | Live market on hourly bars with the exact AMM replay (`impact_model="amm"`). |
| `wsteth_30dec2027` | PT-wstETH-30DEC2027 | Binance ETHUSDT perp; Boros BINANCE-ETHUSDT-25DEC2026 | 2026-03-01 → now, daily | Volatile underlying: the hedge removes the ETH leg; Boros fixes the funding from its first quote (2026-05-12). |

## Results (`results/validation.csv`, runs of 2026-09-11)

| Run | Bars | Leverage | PT APY at entry | Borrow APY (mean) | Realised APY | Closed form | Note |
|---|---|---|---|---|---|---|---|
| `usde_25sep2025` | 127 daily | 4.33× | 8.66 % | 10.42 % | **+6.3 %** | +2.8 % | Held to redemption; no liquidation, min health factor 1.19. Realised beats the closed form because the position was de-levered during the July rate spike (weekly-smoothed gate) and re-levered at a higher implied APY. |
| `susde_25sep2025_dai` | 113 daily | 4.33× | 8.11 % | 7.93 % | **+9.8 %** | +8.7 % | Held to redemption; the DAI borrow stayed just under the PT yield, so the loop added 1.7 pp over the unlevered 8.1 %. |
| `reusd_10dec2026` | 1369 hourly | 4.33× | 11.0 % | 8.65 % | **+18.3 %** | +18.8 % | Through the 2026-08-25 spike (hourly close 14.6 %, oracle −1 %): equity dipped 4.4 % intra-event and recovered; LTV peaked at 0.774, never near the 0.915 LLTV. |
| `susde_26nov2026` | 531 hourly | 4.33× | 4.07 % | 3.07 % | −4.8 % (annualised) | +7.4 % | 22-day mark-to-market window, not held to expiry: the implied APY rose 4.07 → 4.92 %, which marks a 4.3× PT position down by about 1 %. |
| `wsteth_30dec2027_perp` | 194 daily | 2× hedge | 2.38 % | — | **+3.6 %** | — | ETH moved +25.7 % over the window and the equity path stayed flat to it; funding received summed to 1.06 % (Binance: 1.07 %). |
| `wsteth_30dec2027_boros` | 122 daily | 2× hedge | 2.38 % | — | +1.8 % | — | From Boros's first quote; fixed at 1.99 % APR while realised funding ran near 3.9 % annualised, so locking the rate cost about 1.8 pp here. |

Closed form: `L·y_pt − (L−1)·r_borrow` with the entry implied APY and the
window-mean borrow APY; it ignores swap costs, the accrual timing and
any de-leveraging, so the gap is reported rather than asserted.

## Sensitivity (`results/sensitivity_<market>.csv`)

`python sensitivity.py <market>` varies one parameter at a time around
the base run (target LTV 0.80, 8 loops, weekly-smoothed carry gate,
market oracle). Realised APY, 2026-09-11:

| Variant | `usde_25sep2025` (PT 8.7 %, borrow 10.4 %) | `reusd_10dec2026` (PT 11.0 %, borrow 8.7 %) |
|---|---|---|
| unlevered (`MAX_LOOPS=0`) | +8.7 % | +10.8 % |
| target LTV 0.50 / 0.60 / 0.70 | +9.0 / +8.7 / +8.0 % | +13.0 / +14.1 / +15.7 % |
| base (0.80) | +6.3 % | +18.3 % |
| target LTV 0.86 (5.3×) | +4.3 % | +20.6 % |
| `MAX_LOOPS` 2 / 4 | +3.3 / +5.8 % | +15.3 / +15.2 % |
| flash (one-shot, 5.1×) | +4.8 % | +20.0 % |
| carry gate on the raw rate | **−16.5 %** (21 debt changes) | **−29.7 %** (12 debt changes) |
| no carry gate | +3.5 % | +18.3 % |
| linear 6 %/yr oracle | +6.1 % | +18.8 % |

Two lessons: leverage only pays when the borrow rate stays below the
PT yield for the whole hold (USDe-25SEP2025 had a 10.4 % mean borrow
against an 8.7 % PT, so every levered variant trails the unlevered
hold), and the carry gate must look at a smoothed borrow rate —
reacting to each hourly Morpho spike pays the swap costs a dozen times.

## Are the numbers realistic?

Checked on 2026-09-11 against public data and published figures
(Morpho and Pendle APIs, Binance funding, Aave governance risk reports,
Pendle Print, the reUSD post-mortems):

* **Inputs reproduce.** The Morpho API gives the same borrow series
  for PT-USDe-25SEP2025/USDe (mean 10.4 %, min 0.9 %, max 46 %), the
  Pendle API the same implied APYs (8.66 % on 2025-05-22, 4.07 → 4.92 %
  on sUSDe-26NOV2026, 2.38 % on wstETH), and Binance funding sums to
  1.07 % over the hedged window (ours 1.06 %).
* **The USDe loop is arithmetically right but not what practitioners
  earned.** The /USDe market never held more than $2.6 M of debt; the
  trade of 2025 was PT-USDe/USDC on Morpho (peak $348 M, 10.9 % mean
  borrow, 8–27 % range) or PT-USDe/USDe on Aave e-mode (91 % LTV,
  5.5–7.5 % USDe borrow), entered when the implied APY was 13–16 %
  (late July / August), not on the cheapest day of May. Chaos Labs put
  those September-PT loops at 30–50 % net annualised; commentators at
  25–30 % for 3–5×. Replaying that setup with the same strategy —
  PT-USDe-25SEP2025 on the /USDC market, entry 2025-08-12 at 15.8 %
  implied, 9.9 % mean borrow, held to redemption — gives **+37.6 %**
  at 0.80 LTV (4.33×), **+44.6 %** at 0.86 and **+49.1 %** at 0.89
  (5.9×, min health factor 1.10); entry on 2025-07-26 at 14.9 %
  implied gives +24.5 % / +27.8 %. Those are the published ranges.
* **The sUSDe mark-to-market run matches the arithmetic.** Carry at
  4.33× on a ~1 pp spread is about +0.4 % over 22 days, and the +0.85
  pp implied move on a 77-day PT costs about 0.8 % at that leverage —
  net −0.35 %, which is what the run shows. Advertised numbers (14.5 %
  at 7× with USDe borrowed at 2.4 %) are hold-to-maturity carry on a
  cheaper debt leg, not three-week MtM.
* **The hedged carry sits at the low end of the range.** PT-wstETH
  implied 2.38 % ≈ stETH 2.4 %, so the trade is "ETH staking plus
  funding" with a third of the equity idle as margin; sUSDe paid
  3.7–4.8 % over the same months. Locking funding on Boros at 1.99 %
  while Binance realised 3.9 % annualised costs about the 1.8 pp seen.
* **reUSD, 2026-08-25.** The event fits inside 25 minutes (11 swaps,
  implied 11 % → above 20 %, PT −3 %, 15-minute TWAP 0.9647); hourly
  closes only show 14.6 %. Accounts with health factor below 1.03
  (LTV ≈ 0.89+) were liquidated, none at 0.80 LTV — the same outcome
  as the replay (LTV peak 0.774, no liquidation). Reproducing the
  liquidations themselves needs the swap tape or the TWAP series, not
  hourly bars.

Practitioner settings to prefer over the defaults used here: target
LTV 0.85–0.89 (health factor 1.03–1.10), entry only when
`implied − borrow ≥ 5 pp`, hold to maturity and roll, exit when the
borrow rate crosses the implied APY, rates from the deepest market of
the PT, and sub-hour data for any liquidation study.

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
