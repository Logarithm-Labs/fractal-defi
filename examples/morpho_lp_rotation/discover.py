"""Enumerate every Morpho Blue market for {USDC, USDT} and write registry CSVs.

Per A5 the universe is ``{USDC, USDT}`` loan assets × every chain the API serves
× listed markets. The CSVs this writes are **address-pinned and must be reviewed
by a human before a backtest reads them** — the points on a registry are market
ids and LLTVs, and a wrong entry is a wrong strategy, not a data glitch.

``registry/<chain_id>.csv`` is blanket-gitignored (``*.csv``), which is the
intent: the reviewed registry is a local artefact, like ``results/``.

Usage::

    python discover.py [--all-chains | --chains 1,8453]

Default is ``--all-chains``. Chain names come from the API's own chain list.
"""
import argparse
import os

import pandas as pd

from fractal.loaders.morpho_markets import MorphoMarketsLoader, discover_chains

HERE = os.path.dirname(os.path.abspath(__file__))
COLUMNS = ["chain_id", "market_id", "loan_symbol", "loan_address", "collateral_symbol", "collateral_address",
           "lltv", "oracle_type", "creation_timestamp", "liquidity_assets_usd", "supply_assets_usd",
           "borrow_assets_usd"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chains", default=None, help="comma-separated chain ids (default: every chain)")
    parser.add_argument("--registry-dir", default=os.path.join(HERE, "registry"))
    parser.add_argument("--lltv-max", type=float, default=None,
                        help="optional download cap on LLTV (not the strategy's risk boundary)")
    args = parser.parse_args()

    chain_ids = None if args.chains is None else [int(chain) for chain in args.chains.split(",")]
    chains = dict(discover_chains())
    listings = MorphoMarketsLoader(chain_ids=chain_ids, lltv_max=args.lltv_max).read(with_run=True)
    if not listings:
        raise SystemExit("discovery returned no markets - refusing to write an empty registry")

    frame = pd.DataFrame([listing.__dict__ for listing in listings])[COLUMNS]
    os.makedirs(args.registry_dir, exist_ok=True)
    print(f"{len(frame)} listed {{USDC, USDT}} markets across {frame['chain_id'].nunique()} chain(s)")
    for chain_id, group in frame.groupby("chain_id"):
        path = os.path.join(args.registry_dir, f"{chain_id}.csv")
        group.sort_values("liquidity_assets_usd", ascending=False).to_csv(path, index=False)
        liquid = group[group["liquidity_assets_usd"].fillna(0) > 0]
        idle = group[group["lltv"] == 0]
        oracle_mix = ", ".join(f"{k}:{v}" for k, v in group["oracle_type"].value_counts().items())
        print(f"  {chain_id:>8} {chains.get(chain_id, '?'):<14} n={len(group):4d} liquid={len(liquid):4d} "
              f"idle={len(idle):3d} lltv[{group['lltv'].min():.3f}-{group['lltv'].max():.3f}] created>="
              f"{pd.to_datetime(group['creation_timestamp'], unit='s', utc=True).min():%Y-%m-%d} "
              f"| {oracle_mix} | -> {os.path.relpath(path, HERE)}")
    print("\nreview these CSV files before any backtest reads them (A5): the entries are the universe.")
    print("lltv == 0 marks an idle market (no IRM, nothing borrowable) - it earns ~0 and must not be "
          "mistaken for a safe low-LLTV market.")


if __name__ == "__main__":
    main()
