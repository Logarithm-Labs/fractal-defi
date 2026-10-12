"""Backtest the Morpho Blue LP rotation on real hourly data.

One position, rotated across the largest liquid USDC/USDT markets of the
reviewed registry, with the exit gate, the signal lag and the own-position
impact all applied. Prints the strategy's metrics next to the naive static
reference and writes the equity curve into ``results/`` (gitignored).

The honest reading of RQ2 (CONTEXT §8 M1b): the *timing* edge does not exist —
entering on a high snapshot realises the market's own trailing average — while
the *level* ranking does persist. This example therefore measures whether
holding the structurally-higher markets clears costs, and reports the exit
deferrals (RQ3) and the self-dilution (RQ4) alongside it.

Usage::

    python backtest.py --chains 1,8453 --days 90 --top 8
    python backtest.py --no-own-impact --cost-bps 5      # exogenous replay, costs on

Live API; for the default (offline) test suite nothing here runs.
"""
import argparse
import json
import os
from datetime import datetime, timedelta, timezone

from helpers import CASH_NAME, load_panels, load_registry, observations, select, static_reference_apy

from fractal.core.entities.simple.spot import SimpleSpotExchange
from fractal.strategies.morpho_lp_rotation import MorphoLpRotationParams, MorphoLpRotationStrategy

HERE = os.path.dirname(os.path.abspath(__file__))
UTC = timezone.utc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chains", default="1", help="comma-separated chain ids (default: 1)")
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--top", type=int, default=8, help="markets to trade (largest liquid first)")
    parser.add_argument("--min-liquidity-usd", type=float, default=250_000.0)
    parser.add_argument("--initial", type=float, default=1_000_000.0)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-lltv", type=float, default=0.915)
    parser.add_argument("--min-edge-bps", type=float, default=10.0)
    parser.add_argument("--cost-bps", type=float, default=2.0, help="cost per leg, in basis points")
    parser.add_argument("--signal-window-bars", type=int, default=24)
    parser.add_argument("--signal-lag-bars", type=int, default=1)
    parser.add_argument("--no-own-impact", dest="own_impact", action="store_false")
    parser.add_argument("--max-hold-bars", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chain_ids = [int(chain) for chain in args.chains.split(",")]
    end = datetime.now(UTC) - timedelta(hours=2)
    start = end - timedelta(days=args.days)

    listings = select(load_registry(chain_ids), args.top, args.min_liquidity_usd)
    print(f"trading {len(listings)} markets on chains {chain_ids} over {start:%Y-%m-%d}..{end:%Y-%m-%d}")
    panels = load_panels(listings, start, end)
    stream = observations(panels)
    print(f"observations: {len(stream)} bars x {len(panels)} markets")

    params = MorphoLpRotationParams(
        TOP_K=args.top_k, MAX_LLTV=args.max_lltv, MIN_EDGE_BPS=args.min_edge_bps,
        SIGNAL_WINDOW_BARS=args.signal_window_bars, SIGNAL_LAG_BARS=args.signal_lag_bars,
        MAX_HOLD_BARS=args.max_hold_bars, OWN_IMPACT=args.own_impact, COST_BPS_PER_LEG=args.cost_bps,
    )
    strategy = MorphoLpRotationStrategy(
        markets=[panel.listing for panel in panels],
        cash_entity=SimpleSpotExchange(trading_fee=0.0),
        params=params,
    )
    strategy.get_entity(CASH_NAME).action_deposit(args.initial)

    result = strategy.run(stream)
    metrics = result.get_default_metrics()
    reference = static_reference_apy(panels)
    report = strategy.deferral_report()

    print("\n-- strategy ------------------------------------------------")
    print(f"  accumulated return {metrics.accumulated_return:8.2%}   apy {metrics.apy:8.2%}   cagr {metrics.cagr:8.2%}")
    print(f"  max drawdown       {metrics.max_drawdown:8.2%}   sharpe {metrics.sharpe:7.2f}")
    print(f"  fees paid          {metrics.fees_paid:12.2f}   turnover {metrics.turnover:6.2f}x   fee drag {metrics.fee_drag:6.3%}")
    print(f"  own impact         {'on (state-carrying)' if args.own_impact else 'off (exogenous replay)'}")
    print("\n-- exit gate (RQ3) ----------------------------------------")
    print(f"  blocked episodes   {report['events']:d}   deferred notional {report['deferred_notional']:.0f}")
    print(f"  bars blocked       max {report['max_bars_blocked']:d}   mean {report['mean_bars_blocked']:.1f}")
    print("\n-- naive static reference ---------------------------------")
    print(f"  equal-weight mean realised supply APY {reference:8.2%}  (no costs, no gate, no rotation)")
    print(f"  excess over static {metrics.apy - reference:+8.2%}")

    balances = result.to_dataframe()
    results_dir = os.path.join(HERE, "results")
    os.makedirs(results_dir, exist_ok=True)
    balances.to_csv(os.path.join(results_dir, "equity_curve.csv"))
    with open(os.path.join(results_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump({"params": strategy.params, "metrics": metrics.__dict__, "reference_apy": reference,
                   "deferrals": report, "markets": [panel.name for panel in panels]}, handle,
                  indent=2, default=str)
    print(f"\nwrote {os.path.relpath(results_dir, HERE)}/equity_curve.csv and summary.json")


if __name__ == "__main__":
    main()
