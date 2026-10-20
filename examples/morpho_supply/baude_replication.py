"""Replicate Baude, Danos & El Khalloufi (arXiv:2608.24206) with the Morpho supply allocator.

Two experiments, both on provider-backed hourly Morpho history:

``paper``
    The paper's own window (2026-01-01 .. 2026-04-01), Ethereum USDC, market
    sets and protocol: daily re-solve on one-day averages, no gas, borrows held
    fixed (our dilution model, absorption 0), budgets $100k and $100M, and the
    whole budget in USDC/cbBTC as the benchmark. The paper names its markets by
    Morpho-API ids the current API no longer exposes; each is mapped to the
    largest Ethereum USDC market with that collateral on 2026-01-01. The weETH
    and syrupUSDC mappings are uncertain (both markets were below $1M then).
    Each row prints the paper's published value next to ours.

``ladder``
    A recent window on the reviewed registry, relaxing one design choice at a
    time: hourly re-solve, all Ethereum USDC markets (PT collateral allowed),
    USDT, all chains, then per-chain gas. Needs ``registry/<chain_id>.csv``
    from ``discover.py``.

Usage::

    python baude_replication.py paper
    python baude_replication.py ladder --days 90

This is an in-sample replication of a published backtest, not evidence of
returns. Live API calls are opt-in; nothing here runs in the offline test suite.
"""
import argparse
import os
from datetime import datetime, timedelta, timezone

import pandas as pd
from helpers import (
    CASH_NAME,
    DEFAULT_GAS_USD_PER_LEG,
    GAS_USD_BY_CHAIN,
    MarketPanel,
    load_panels,
    load_registry,
    observations,
    select,
)

from fractal.core.entities.simple.spot import SimpleSpotExchange
from fractal.loaders.morpho_markets import MorphoMarketListing
from fractal.strategies.morpho_supply_allocation import MorphoSupplyAllocationParams, MorphoSupplyAllocationStrategy

HERE = os.path.dirname(os.path.abspath(__file__))
UTC = timezone.utc
PAPER_START, PAPER_END = datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 4, 1, tzinfo=UTC)
USDC_ETHEREUM = "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"
PAPER_MARKETS = {  # collateral: (marketId, lltv, oracle)
    "cbBTC": ("0x64d65c9a2d91c36d56fbc42d69e979335320169b3df63bf92789e2c8883fcc64", 0.86, "ChainlinkOracleV2"),
    "WBTC": ("0x3a85e619751152991742810df6ec69ce473daef99e28a64ab2340d7b7ccfee49", 0.86, "ChainlinkOracle"),
    "wstETH": ("0xb323495f7e4148be5643a4ea4a8221eef163e4bccfdedc2a6f4696baacbc86cc", 0.86, "ChainlinkOracle"),
    "AA_FalconXUSDC": ("0xe83d72fa5b00dcd46d9e0e860d95aa540d5ec106da5833108a9f826f21f36f52", 0.77, "ChainlinkOracleV2"),
    "siUSD": ("0xbbf7ce1b40d32d3e3048f5cf27eeaa6de8cb27b80194690aab191a63381d8c99", 0.915, "ChainlinkOracleV2"),
    "stcUSD": ("0xeb17955ea422baeddbfb0b8d8c9086c5be7a9cfdefb292119a102e981a30062e", 0.915, "ChainlinkOracleV2"),
    "LBTC": ("0xbf02d6c6852fa0b8247d5514d0c91e6c1fbde9a168ac3fd2033028b5ee5ce6d0", 0.86, "ChainlinkOracleV2"),
    "weETH": ("0x61765602144e91e5ac9f9e98b8584eae308f9951596fd7f5e0f59f21cd2bf664", 0.86, "ChainlinkOracleV2"),
    "tBTC": ("0xe4cfbee9af4ad713b41bf79f009ca02b17c001a0c0e7bd2e6a89b1111b3d3f08", 0.77, "ChainlinkOracleV2"),
    "syrupUSDC": ("0x729badf297ee9f2f6b3f717b96fd355fc6ec00422284ce1968e76647b258cf44", 0.915, "ChainlinkOracleV2"),
}
PRIME = ("cbBTC", "WBTC", "wstETH")
# The paper's published APYs, %, keyed by (market set, budget).
PAPER_APY = {("cbBTC", 1e5): 3.3, ("prime", 1e5): 3.3, ("core", 1e5): 5.5,
             ("cbBTC", 1e8): 2.2, ("prime", 1e8): 2.4, ("core", 1e8): 2.8}
BUDGETS = (1e5, 1e8)
MODES = ("marginal", "average")
CHAINLINK = ("ChainlinkOracle", "ChainlinkOracleV2")


def paper_listings() -> list[MorphoMarketListing]:
    """The ten mapped Ethereum USDC markets of the paper's "core" set."""
    return [
        MorphoMarketListing(chain_id=1, market_id=market_id, loan_address=USDC_ETHEREUM, loan_symbol="USDC",
                            loan_decimals=6, collateral_address="", collateral_symbol=symbol,
                            collateral_decimals=18, lltv=lltv, oracle_type=oracle, listed=True)
        for symbol, (market_id, lltv, oracle) in PAPER_MARKETS.items()
    ]


def allocation_params(mode: str, daily: bool, gas: bool, paper_filters: bool = False) -> MorphoSupplyAllocationParams:
    """The paper's protocol (``daily``, no gas) or ours (hourly, per-chain gas), always without a target floor."""
    filters = {"MAX_LLTV": 1.0, "MIN_LIQUIDITY_MULTIPLE": 0.0} if paper_filters else {}
    return MorphoSupplyAllocationParams(
        TARGET_APY=0.0, ALLOCATION_MODE=mode, OWN_IMPACT=True, ABSORPTION=0.0, MIN_POOL_SIZE_USD=0.0,
        REBALANCE_BARS=24 if daily else 1, SIGNAL_WINDOW_BARS=24 if daily else 1, SIGNAL_LAG_BARS=1,
        MIN_REBALANCE_FRACTION=0.0 if daily else 0.05, HOLD_HORIZON_BARS=24 if daily else 6,
        GAS_USD_PER_LEG=DEFAULT_GAS_USD_PER_LEG if gas else 0.0, GAS_USD_BY_CHAIN=dict(GAS_USD_BY_CHAIN) if gas else {},
        **filters,
    )


def run_allocation(panels: list[MarketPanel], stream: list, budget: float,
                   params: MorphoSupplyAllocationParams) -> tuple[float, float]:
    """APY and fees paid for one allocator run on prepared panels and observations."""
    strategy = MorphoSupplyAllocationStrategy(markets=[panel.listing for panel in panels],
                                              cash_entity=SimpleSpotExchange(trading_fee=0.0), params=params)
    strategy.get_entity(CASH_NAME).action_deposit(budget)
    metrics = strategy.run(stream).get_default_metrics()
    return metrics.apy, metrics.fees_paid


def ladder_universe(listings: list[MorphoMarketListing], symbols: set[str], top: int,
                    no_pt: bool) -> list[MorphoMarketListing]:
    """Chainlink-priced markets with LLTV <= 91.5%, optionally without PT collateral, largest liquidity first."""
    out = [listing for listing in listings
           if listing.loan_symbol in symbols and listing.oracle_type in CHAINLINK and listing.lltv <= 0.915
           and (not no_pt or not listing.collateral_symbol.upper().startswith("PT-"))]
    return sorted(out, key=lambda listing: -(listing.liquidity_assets_usd or 0.0))[:top]


def run_paper() -> pd.DataFrame:
    panels = load_panels(paper_listings(), PAPER_START, PAPER_END, fetch_fee=False)
    by_symbol = {panel.listing.collateral_symbol: panel for panel in panels}
    print(f"loaded {len(panels)}/{len(PAPER_MARKETS)} markets: {sorted(by_symbol)}")
    sets = {"cbBTC": [by_symbol[s] for s in ("cbBTC",) if s in by_symbol],
            "prime": [by_symbol[s] for s in PRIME if s in by_symbol],
            "core": list(by_symbol.values())}
    rows = []
    for name, set_panels in sets.items():
        if not set_panels:
            continue
        stream = observations(set_panels)
        for budget in BUDGETS:
            for mode in MODES:
                for daily, gas, protocol in ((True, False, "paper: daily, no gas"),
                                             (False, True, "ours: hourly, gas by chain")):
                    if name == "cbBTC" and (mode == "average" or not daily):
                        continue  # one market: the benchmark has a single protocol and allocation
                    params = allocation_params(mode, daily, gas, paper_filters=True)
                    apy, fees = run_allocation(set_panels, stream, budget, params)
                    paper = PAPER_APY.get((name, budget)) if daily else None
                    rows.append({"set": name, "markets": len(set_panels), "budget": budget, "mode": mode,
                                 "protocol": protocol, "apy_pct": apy * 100, "fees": fees, "paper_apy_pct": paper})
                    print(f"  {name:6s} n={len(set_panels):2d} ${budget:>13,.0f} {mode:8s} {protocol:27s} "
                          f"apy={apy:7.2%} paper={'-' if paper is None else f'{paper:.1f}%'}", flush=True)
    return pd.DataFrame(rows)


def run_ladder(days: int, end: datetime) -> pd.DataFrame:
    start = end - timedelta(days=days)
    registry = load_registry(None)
    universes = {
        "eth_usdc_core": ([1], {"USDC"}, 10, True),
        "eth_usdc_all": ([1], {"USDC"}, 300, False),
        "eth_all": ([1], {"USDC", "USDT"}, 300, False),
        "all": (None, {"USDC", "USDT"}, 300, False),
    }
    variants = [  # (label, universe, daily, gas)
        ("1 paper-like: Ethereum USDC core, daily, no gas", "eth_usdc_core", True, False),
        ("2 + hourly re-solve", "eth_usdc_core", False, False),
        ("3 + all Ethereum USDC markets (PT allowed)", "eth_usdc_all", False, False),
        ("4 + USDT on Ethereum", "eth_all", False, False),
        ("5 + all chains", "all", False, False),
        ("6 + gas by chain", "all", False, True),
    ]
    prepared = {}
    rows = []
    for label, key, daily, gas in variants:
        if key not in prepared:
            chains, symbols, top, no_pt = universes[key]
            listings = select([listing for listing in registry if chains is None or listing.chain_id in chains],
                              300, 100_000.0)
            universe = ladder_universe(listings, symbols, top, no_pt)
            try:
                panels = load_panels(universe, start, end, fetch_fee=False) if universe else []
            except ValueError as exc:      # every market in the universe failed to load
                print(f"  {key}: {exc}")
                panels = []
            prepared[key] = (panels, observations(panels) if panels else [])
        panels, stream = prepared[key]
        if not panels:
            print(f"  {label}: no usable markets in the registry, skipped")
            continue
        for budget in BUDGETS:
            for mode in MODES:
                apy, fees = run_allocation(panels, stream, budget, allocation_params(mode, daily, gas))
                rows.append({"variant": label, "markets": len(panels), "budget": budget, "mode": mode,
                             "apy_pct": apy * 100, "fees": fees})
                print(f"  {label:48s} n={len(panels):3d} ${budget:>13,.0f} {mode:8s} apy={apy:7.2%} "
                      f"fees=${fees:9,.0f}", flush=True)
    return pd.DataFrame(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="experiment", required=True)
    sub.add_parser("paper", help="the paper's window, market sets and protocol")
    ladder = sub.add_parser("ladder", help="a recent window, relaxing one design choice at a time")
    ladder.add_argument("--days", type=int, default=90)
    ladder.add_argument("--end", type=lambda text: datetime.fromisoformat(text).replace(tzinfo=UTC), default=None,
                        help="UTC end date, YYYY-MM-DD (default: two hours ago)")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.experiment == "paper":
        table = run_paper()
    else:
        table = run_ladder(args.days, args.end or datetime.now(UTC) - timedelta(hours=2))
    results_dir = os.path.join(HERE, "results")
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"baude_{args.experiment}.csv")
    table.to_csv(path, index=False)
    print(f"\nwrote {os.path.relpath(path, HERE)}")


if __name__ == "__main__":
    main()
