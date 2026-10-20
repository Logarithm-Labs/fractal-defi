"""Backtest Morpho Blue supply strategies on provider-backed hourly data.

``--strategy rotation`` (default) holds one position and rotates on a rate edge;
``--strategy allocation`` spreads capital across markets in ``average`` or
``marginal`` mode.

Reads a human-reviewed registry and writes local results. Universe selection
uses current liquidity and is applied retrospectively;
this is a survivorship-biased smoke example, not point-in-time research evidence.
The static reference ignores costs, liquidity constraints and position impact.

Usage::

    python backtest.py --chains 1 --days 30 --top 8
    python backtest.py --end 2026-04-01 --days 30
    python backtest.py --no-own-impact --gas-usd-per-leg 1.0
    python backtest.py --strategy allocation --mode marginal --target-apy 0 --hold-horizon-bars 24

Costs: Morpho supply and withdraw carry no protocol fee or slippage, so the
default cost is gas only, a fixed USD amount per leg from the illustrative
per-chain table in ``helpers.py``. ``--gas-usd-per-leg`` replaces the table with
one figure for every chain. ``--cost-bps`` is a proportional cost for swaps or
bridges, which this example does not model; it defaults to zero.

Live API calls are opt-in; nothing here runs in the offline test suite.
"""
import argparse
import json
import os
from datetime import datetime, timedelta, timezone

from helpers import (
    CASH_NAME,
    DEFAULT_GAS_USD_PER_LEG,
    GAS_USD_BY_CHAIN,
    load_panels,
    load_registry,
    observations,
    select,
    static_reference_apy,
)

from fractal.core.entities.simple.spot import SimpleSpotExchange
from fractal.strategies.morpho_supply_allocation import MorphoSupplyAllocationParams, MorphoSupplyAllocationStrategy
from fractal.strategies.morpho_supply_rotation import MorphoSupplyRotationParams, MorphoSupplyRotationStrategy

HERE = os.path.dirname(os.path.abspath(__file__))
UTC = timezone.utc


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--strategy", choices=("rotation", "allocation"), default="rotation")
    parser.add_argument("--chains", default="1", help="comma-separated chain ids (default: 1)")
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--end", type=str, default=None,
                        help="window end as YYYY-MM-DD UTC (default: now minus 2 hours)")
    parser.add_argument("--top", type=int, default=8, help="markets to trade (largest liquid first)")
    parser.add_argument("--min-liquidity-usd", type=float, default=250_000.0)
    parser.add_argument("--initial", type=float, default=1_000_000.0)
    parser.add_argument("--max-lltv", type=float, default=0.915)
    parser.add_argument("--cost-bps", type=float, default=0.0,
                        help="proportional cost per leg in basis points, for swaps or bridges (default: 0)")
    parser.add_argument("--gas-usd-per-leg", type=float, default=None,
                        help="fixed USD gas per leg on every chain (default: the per-chain table in helpers.py)")
    parser.add_argument("--signal-window-bars", type=int, default=None,
                        help="signal averaging window (default: 24 for rotation, 1 for allocation)")
    parser.add_argument("--signal-lag-bars", type=int, default=1)
    parser.add_argument("--no-own-impact", dest="own_impact", action="store_false")
    parser.add_argument("--absorption", type=float, default=0.0,
                        help="share of our supply borrowers take up (0 = pure dilution, gate cannot bind; "
                             "1 = fully absorbed, gate = historical liquidity)")
    parser.add_argument("--rank-on-headline", dest="rank_on_projected", action="store_false",
                        help="rank on the market's headline APY instead of the APY our own size would earn")
    rotation = parser.add_argument_group("rotation")
    rotation.add_argument("--top-k", type=int, default=3)
    rotation.add_argument("--min-edge-bps", type=float, default=10.0)
    rotation.add_argument("--max-hold-bars", type=int, default=0)
    allocation = parser.add_argument_group("allocation")
    allocation.add_argument("--mode", choices=("average", "marginal"), default="average")
    allocation.add_argument("--target-apy", type=float, default=0.06,
                            help="hurdle APY below which capital stays in cash")
    allocation.add_argument("--rebalance-bars", type=int, default=1)
    allocation.add_argument("--min-rebalance-fraction", type=float, default=0.05,
                            help="average mode: smallest resize, as a fraction of NAV")
    allocation.add_argument("--hold-horizon-bars", type=int, default=6, help="marginal mode: income horizon")
    return parser.parse_args(argv)


def build_strategy(args: argparse.Namespace, markets: list) -> MorphoSupplyRotationStrategy | MorphoSupplyAllocationStrategy:
    """The configured strategy on ``markets``, with an empty cash entity."""
    shared = {
        "MAX_LLTV": args.max_lltv, "SIGNAL_LAG_BARS": args.signal_lag_bars, "OWN_IMPACT": args.own_impact,
        "ABSORPTION": args.absorption, "COST_BPS_PER_LEG": args.cost_bps,
    }
    if args.gas_usd_per_leg is None:
        shared.update(GAS_USD_PER_LEG=DEFAULT_GAS_USD_PER_LEG, GAS_USD_BY_CHAIN=dict(GAS_USD_BY_CHAIN))
    else:
        shared["GAS_USD_PER_LEG"] = args.gas_usd_per_leg
    if args.signal_window_bars is not None:
        shared["SIGNAL_WINDOW_BARS"] = args.signal_window_bars
    cash = SimpleSpotExchange(trading_fee=0.0)
    if args.strategy == "allocation":
        params = MorphoSupplyAllocationParams(
            ALLOCATION_MODE=args.mode, TARGET_APY=args.target_apy, REBALANCE_BARS=args.rebalance_bars,
            MIN_REBALANCE_FRACTION=args.min_rebalance_fraction, HOLD_HORIZON_BARS=args.hold_horizon_bars, **shared,
        )
        return MorphoSupplyAllocationStrategy(markets=markets, cash_entity=cash, params=params)
    params = MorphoSupplyRotationParams(
        TOP_K=args.top_k, MIN_EDGE_BPS=args.min_edge_bps, MAX_HOLD_BARS=args.max_hold_bars,
        RANK_ON_PROJECTED_APY=args.rank_on_projected, **shared,
    )
    return MorphoSupplyRotationStrategy(markets=markets, cash_entity=cash, params=params)


def print_rotation_report(strategy: MorphoSupplyRotationStrategy, args: argparse.Namespace) -> dict:
    report = strategy.deferral_report()
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
    return report


def print_allocation_report(strategy: MorphoSupplyAllocationStrategy, args: argparse.Namespace) -> dict:
    report = strategy.allocation_report()
    lam = report["lambda_apy"]
    print(f"  mode               {args.mode}   target APY {args.target_apy:.2%}   "
          f"own impact {'on' if args.own_impact else 'off'} (absorption {args.absorption:.2f})")
    print("\n-- last allocation ----------------------------------")
    print(f"  positions          {report['positions']:d}   cash {report['cash_fraction']:6.2%}   "
          f"lambda APY {'-' if lam is None else f'{lam:.2%}'}   gated legs {report['gated_legs']:d}")
    for name, target in sorted(report["targets"].items(), key=lambda item: -item[1])[:10]:
        print(f"  {target:14,.0f}  {name}")
    return report


def main() -> None:
    args = parse_args()
    chain_ids = [int(chain) for chain in args.chains.split(",")]
    if args.end is None:
        end = datetime.now(UTC) - timedelta(hours=2)
    else:
        end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=UTC)
    start = end - timedelta(days=args.days)

    listings = select(load_registry(chain_ids), args.top, args.min_liquidity_usd)
    print(f"trading {len(listings)} markets on chains {chain_ids} over {start:%Y-%m-%d}..{end:%Y-%m-%d}")
    panels = load_panels(listings, start, end)
    stream = observations(panels)
    print(f"observations: {len(stream)} bars x {len(panels)} markets")

    strategy = build_strategy(args, [panel.listing for panel in panels])
    strategy.get_entity(CASH_NAME).action_deposit(args.initial)

    result = strategy.run(stream)
    metrics = result.get_default_metrics()
    reference = static_reference_apy(panels)

    print(f"\n-- strategy: {args.strategy} " + "-" * (46 - len(args.strategy)))
    print(f"  accumulated return {metrics.accumulated_return:8.2%}   apy {metrics.apy:8.2%}   cagr {metrics.cagr:8.2%}")
    print(f"  max drawdown       {metrics.max_drawdown:8.2%}   sharpe {metrics.sharpe:7.2f}")
    print(f"  fees paid          {metrics.fees_paid:12.2f}   turnover {metrics.turnover:6.2f}x   fee drag {metrics.fee_drag:6.3%}")
    if args.strategy == "allocation":
        report = print_allocation_report(strategy, args)
    else:
        report = print_rotation_report(strategy, args)
    print("\n-- naive static reference ---------------------------------")
    print(f"  equal-weight mean realised supply APY {reference:8.2%}  (no costs, no gate, no rebalancing)")
    print(f"  excess over static {metrics.apy - reference:+8.2%}")

    balances = result.to_dataframe()
    results_dir = os.path.join(HERE, "results")
    os.makedirs(results_dir, exist_ok=True)
    balances.to_csv(os.path.join(results_dir, "equity_curve.csv"))
    with open(os.path.join(results_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump({"params": strategy.params, "metrics": metrics.__dict__, "reference_apy": reference,
                   "strategy": args.strategy, "report": report, "markets": [panel.name for panel in panels]}, handle,
                  indent=2, default=str)
    print(f"\nwrote {os.path.relpath(results_dir, HERE)}/equity_curve.csv and summary.json")


if __name__ == "__main__":
    main()
