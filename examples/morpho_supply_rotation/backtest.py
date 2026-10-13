"""Backtest one Morpho Blue supply position on provider-backed hourly data.

Reads a human-reviewed registry and writes local results. Universe selection
uses current liquidity and the current market fee is applied retrospectively;
this is a survivorship-biased smoke example, not point-in-time research evidence.
The static reference ignores costs, liquidity constraints and position impact.

Usage::

    python backtest.py --chains 1 --days 30 --top 8
    python backtest.py --no-own-impact --cost-bps 5

Live API calls are opt-in; nothing here runs in the offline test suite.
"""
import argparse
import json
import os
from datetime import datetime, timedelta, timezone

from helpers import CASH_NAME, load_panels, load_registry, observations, select, static_reference_apy

from fractal.core.entities.simple.spot import SimpleSpotExchange
from fractal.strategies.morpho_supply_rotation import MorphoSupplyRotationParams, MorphoSupplyRotationStrategy

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
    parser.add_argument("--absorption", type=float, default=0.0,
                        help="share of our supply borrowers take up (0 = pure dilution, gate cannot bind; "
                             "1 = fully absorbed, gate = historical liquidity)")
    parser.add_argument("--rank-on-headline", dest="rank_on_projected", action="store_false",
                        help="rank on the market's headline APY instead of the APY our own size would earn")
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

    params = MorphoSupplyRotationParams(
        TOP_K=args.top_k, MAX_LLTV=args.max_lltv, MIN_EDGE_BPS=args.min_edge_bps,
        SIGNAL_WINDOW_BARS=args.signal_window_bars, SIGNAL_LAG_BARS=args.signal_lag_bars,
        MAX_HOLD_BARS=args.max_hold_bars, OWN_IMPACT=args.own_impact, ABSORPTION=args.absorption,
        RANK_ON_PROJECTED_APY=args.rank_on_projected, COST_BPS_PER_LEG=args.cost_bps,
    )
    strategy = MorphoSupplyRotationStrategy(
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
    mode = f"on (state-carrying, absorption {args.absorption:.2f})" if args.own_impact else "off (exogenous replay)"
    print(f"  own impact         {mode}")
    print(f"  ranking            {'projected APY at our size' if args.rank_on_projected else 'headline APY'}")
    print("\n-- exit gate ----------------------------------------")
    if args.own_impact and args.absorption == 0.0:
        print("  NOTE: at absorption 0 our own supply is always withdrawable; exit deferrals are zero by")
        print("        construction. Use --absorption > 0 or --no-own-impact to model blocked exits.")
    print(f"  blocked exits      {report['events']:d}   deferred notional {report['deferred_notional']:.0f}"
          f"   still open {report['open_exit_deferrals']:d}")
    print(f"  bars blocked       max {report['max_bars_blocked']:d}   mean {report['mean_bars_blocked']:.1f}")
    print(f"  refused entries    {report['entry_episodes']:d} episodes, {report['entry_blocked_bars']:d} bars")
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
