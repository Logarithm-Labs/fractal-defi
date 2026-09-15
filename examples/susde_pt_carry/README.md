# sUSDe PT Carry: Leveraged Loans With and Without a Floating-Rate Leg

The strategy of PR #83 in three variants on real data, with a parameter
grid and an analysis notebook.

* **Loop** (`RATE_HEDGE="none"`): buy PT-sUSDe, post it on Morpho, borrow
  the stablecoin, buy more PT; hold to expiry. Fixed PT yield against a
  floating borrow rate.
* **Loop + Boros** (`"boros"`): the same loop plus long yield units on
  Boros BINANCE-ETHUSDT, sized to the debt: pay the fixed implied APR,
  receive the venue's funding. Turns the loan's floating cost into
  `fixed + (borrow − funding)`.
* **Loop + basis leg** (`"perp"`): the floating leg built as a
  delta-neutral position (long ETH spot, short ETHUSDT perp). Receiving
  funding on the whole debt would need spot capital equal to the debt,
  so the leg is sized by the capital parked in it and the coverage is
  reported.

All data is keyless: Pendle, Morpho, Boros, Binance. The Boros market of
2025 lists on 2025-07-31, so on the full-life market the leg opens
lazily half-way through the hold.

## Files

| File | Purpose |
|---|---|
| `markets.json` | Two markets: PT-sUSDe-25SEP2025 / DAI (daily, full life, Boros ETHUSDT-26SEP2025) and PT-sUSDe-26NOV2026 / USDC (hourly, live, Boros ETHUSDT-25DEC2026). |
| `helpers.py` | Frame builder (Pendle + Morpho + Binance price/funding + Boros mark) and per-variant observations; reuses the builders of `examples/pendle_pt_backtests`. |
| `run.py` | The three variants on every market → `results/<market>_<variant>.csv`, `results/validation.csv`. |
| `grid.py` | Target LTV × hedge kind × margin share × hedge ratio → `results/grid_<market>.csv`. |
| `make_notebook.py` / `analysis.ipynb` | Equity curves, PnL decomposition, APY / drawdown, costs, grid heatmaps, hedge-leg breakdown. The notebook is generated, then executed in place. |

```bash
python run.py            # or: python run.py susde_26nov2026
python grid.py
python make_notebook.py && python -m nbconvert --to notebook --execute --inplace analysis.ipynb
```

## Results (2026-09-15, `INITIAL_BALANCE` 100k, target LTV 0.80, hedge share 20 %, ratio 1)

| Market | Variant | Leverage | PT APY | Borrow | Funding (ann.) | Boros mark | Realised APY | Max DD | Coverage | Hedge PnL |
|---|---|---|---|---|---|---|---|---|---|---|
| sUSDe-25SEP2025 / DAI, 113 daily | loop | 4.33× | 8.1 % | 7.9 % | 5.8 % | 6.6 % | **+9.8 %** | −3.4 % | — | — |
| | + Boros | 3.46× | | | | | +4.8 % | −2.8 % | 0.52 | −952 |
| | + basis | 3.46× | | | | | +7.9 % | −2.8 % | 0.05 | −71 |
| sUSDe-26NOV2026 / USDC, 643 hourly | loop | 4.33× | 4.1 % | 3.2 % | 5.6 % | 5.7 % | −2.4 % | −0.6 % | — | — |
| | + Boros | 3.46× | | | | | **+21.8 %** | −2.3 % | 0.99 | +1541 |
| | + basis | 3.46× | | | | | −2.0 % | −0.5 % | 0.05 | −16 |

Reading the numbers:

* **The Boros leg is a rate position, not a free hedge.** In 2025 the
  fixed implied APR (6.6–9.6 %) sat above realised Binance funding
  (5.8 %), so the long YU paid more than it received: −952 on the
  margin parked, −5 pp of APY. On the live market the implied APR rose
  from 2.7 % to 5.3 % while the position was on, and the mark-to-maturity
  gain of the long YU (`N · Δmark · TTM`) dominates the +21.8 %; the
  settlement leg alone is close to flat. Section 6 of the notebook
  splits the two.
* **Coverage is capped by margin.** With 20 % of equity in the leg and
  Boros's 1.55× cap, the YU covers the whole debt only when the mark is
  near the 6 % rate floor; in 2025 the leg reached 52 % of the debt (and
  only from 2025-07-31).
* **The basis leg cannot hedge a loan.** 20 % of equity at 2× buys spot
  worth 13 % of equity, i.e. 2–5 % of the debt; after spot and perp fees
  and the re-margining it earns less than the funding it collects.
* **Leverage helps only with a positive spread**: on the live market
  the plain loop is mark-to-market negative (implied APY rose from 4.1 to
  4.9 %) at every LTV in the grid; on the 2025 market it adds 1.7 pp at
  0.80 and 1.9 pp at 0.86.

## Modelling notes

* Boros yield units are in ETH; the leg is sized as
  `HEDGE_RATIO × debt / ETH price` and re-synced after every loop action
  and when it drifts by more than `HEDGE_REBALANCE_THRESHOLD`. Its
  margin is isolated (`HEDGE_MARGIN_SHARE`); a liquidated leg stays
  closed.
* Funding is Binance's 8-hour rate summed per bar and applied as one
  settlement per bar on both the YU and the perp; `settlementApr` on
  Boros equals that rate × 1095 (verified live).
* The loop is `MorphoLeveragedPT` unchanged (weekly-smoothed carry gate,
  band 0.70–0.88, repay sized on the PT entity's own quote); the
  `"none"` variant reproduces `examples/pendle_pt_backtests` to the
  last digit.
