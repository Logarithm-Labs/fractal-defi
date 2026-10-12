# Morpho Blue LP rotation

Research example: rotate one loan-token supply position across isolated
Morpho Blue markets, on the research recorded in `research/morpho_lp/CONTEXT.md`.

## What the research says (read this before interpreting any number)

RQ2 was **negative for the obvious strategy**: entering a market whose current
rate is in the top decile realises *less* over 7 days (median −56.8 bp) and is
statistically indistinguishable from that market's own trailing 30-day average.
The entry snapshot is a local maximum; chasing it earns nothing.

What did survive is the cross-sectional *level*: 52.2% of top-decile markets are
still top-decile a week later against 10% by chance. So the strategy ranks a
**lagged, smoothed** rate level, not the snapshot — and its expected return
should be judged against the markets' trailing averages, never the peak.

RQ3 (exit deferral) and RQ4 (self-dilution) are the other two ways this thesis
dies; both are measured, not assumed.

## Scripts

```bash
# 1. enumerate the universe -> registry/<chain_id>.csv  (REVIEW THEM)
python discover.py --chains 1,8453

# 2. validate the model against live data (RQ1: formula, IRM curve, liquidity)
python validate_rq1.py 120

# 3. backtest the rotation
python backtest.py --chains 1 --days 90 --top 8
python backtest.py --chains 1 --no-own-impact --cost-bps 5   # exogenous replay
```

All three hit the live Morpho API and are **never run by the test suite**.

## Reading the output

- `results/equity_curve.csv`, `results/summary.json` — gitignored artefacts.
- The static reference in `backtest.py` is a naive equal-weight number, not an
  implementable benchmark. The implementable alternative (a MetaMorpho vault)
  is not modelled.
- Small windows are noise: the same 6-market config gave **+1.10%** excess over
  30 days and **−0.63%** over 7 days. Anything read off a single run of this
  example is a smoke test, not a result.

## Files

| File | Role |
| --- | --- |
| `discover.py` | universe sweep → address-pinned registry CSVs for human review |
| `validate_rq1.py` | live RQ1 checks with median/p95 residuals in bp |
| `helpers.py` | registry → listings → hourly panels → observations |
| `backtest.py` | rotation run, metrics, static reference, deferral report |
| `registry/`, `results/` | local artefacts (gitignored) |
