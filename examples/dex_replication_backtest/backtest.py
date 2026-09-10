"""Replicate real on-chain Uniswap V3 LP positions and validate the fees.

Per position in ``positions.json``: load native hourly TheGraph
snapshots (mint hour → now), anchor the feeGrowth window to the chain
(mint-block first-bar correction + not-yet-indexed head growth), open
the position 1:1 (exact mint amounts, tick bounds,
mint price) with ``fee_model="fee_growth"``, then compare the accrued
fees per token leg against the un-floored on-chain ``feeGrowthInside``
truth. Artifacts under ``results/``: per-bar trajectories +
``summary.csv``.

Data loading is split from the run (``load_replication_data`` /
``run_replication``) so the methods grid can reuse one pull for several
fee models; positions with no post-mint bars yet are skipped.

Usage: ``python backtest.py`` — requires ``THE_GRAPH_API_KEY`` in the
environment (optional ``BASE_RPC_URL``, default mainnet.base.org).
"""
import os
import time
from typing import Optional

import pandas as pd
from helpers import (
    anchor_first_bar,
    append_head_bar,
    build_observations,
    get_w3,
    load_positions,
    position_ground_truth,
    to_notional_frame,
)

from fractal.core.entities import UniswapV3LPEntity
from fractal.loaders.base_loader import LoaderType
from fractal.loaders.thegraph.uniswap_v3 import UniswapV3BasePoolHourDataLoader
from fractal.strategies import FixedRangeLiquidityProvision, FixedRangeLiquidityProvisionParams

USDC = "0x833589fcd6edb6e08f4c7c32d4f71b54bda02913"
# USD price source for non-USDC quote tokens: token address -> USDC-quoted
# ref pool (its hourly price series gives USD per quote unit).
USD_REFS = {
    "0xcbb7c0000ab88b473b1f5afd9ef808440eed33bf":  # cbBTC
        ("0xfbb6eed8e7aa03b138556eedaf5d271a5e1e43ef", 6, 8, True),
}
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
SUMMARY_COLUMNS = [
    "replicated_position_id", "pair", "pool_address", "fee_tier",
    "fees_accrued_usd", "fees_accrued_token0", "fees_accrued_token1",
    "fees_accrued_usd_onchain", "fees_accrued_token0_onchain",
    "fees_accrued_token1_onchain",
    "usd_ratio", "token0_ratio", "token1_ratio",
]


def load_replication_data(position: dict, api_key: str, w3) -> Optional[dict]:
    """Pull everything one backtest needs: hourly history with the
    mint-block first-bar correction, observations, the legs' USD prices
    (from the loaded data, no extra RPC) and the on-chain ground truth.
    Returns ``None`` (with a warning) when the pool has no post-mint
    hourly rows yet.
    """
    pool = position["pool_address"]
    d0, d1 = position["decimals0"], position["decimals1"]
    # quote-side policy: USDC when present, else token1 (WETH/cbBTC)
    notional_side = "token0" if position["token0_address"] == USDC else "token1"
    mint_dt = pd.Timestamp(position["mint_ts"], unit="s", tz="UTC")

    loader = UniswapV3BasePoolHourDataLoader(
        api_key=api_key, pool=pool, loader_type=LoaderType.CSV,
        start_time=mint_dt.floor("1h"), decimals=d0 - d1,
    )
    history = loader.read(with_run=True)
    after = anchor_first_bar(loader, history[history.index > mint_dt], position, w3)
    after = append_head_bar(loader, after, position, w3)
    if after.empty:
        print(f"WARNING: no hourly data after mint yet for position "
              f"{position['token_id']} ({position['pair']}) — skipped")
        return None

    # Non-USDC-notional pools (WETH/cbBTC): hourly USD-per-notional from
    # the quote token's reference pool.
    notional_usd = None
    notional_addr = position["token1_address"] if notional_side == "token1" else position["token0_address"]
    if notional_addr != USDC:
        ref_pool, rd0, rd1, invert = USD_REFS[notional_addr]
        ref = UniswapV3BasePoolHourDataLoader(
            api_key=api_key, pool=ref_pool, loader_type=LoaderType.CSV,
            start_time=mint_dt.floor("1h"), decimals=rd0 - rd1,
        ).read(with_run=True)
        notional_usd = 1.0 / ref["price"] if invert else ref["price"]

    # Synthetic bar at the mint itself: entering at the exact mint price
    # makes the entity compute the same L as the on-chain mint; zero
    # fees/deltas — accrual starts from the next (anchored) bar.
    synth = pd.DataFrame([{
        "tvl": after["tvl"].iloc[0], "volume": 0.0, "fees": 0.0,
        "liquidity": after["liquidity"].iloc[0],
        "price": position["mint_price_std"],
        "fee_growth0": 0.0, "fee_growth1": 0.0,
    }], index=[mint_dt])
    frame = to_notional_frame(pd.concat([synth, after]), notional_side, notional_usd)
    observations = build_observations(frame)

    # USD prices of the two legs at the end of the window, from the data
    # already loaded: the notional leg via notional_usd (1.0 for USDC),
    # the other leg via the pool's own price.
    usd_notional = 1.0 if notional_usd is None else float(notional_usd.dropna().iloc[-1])
    price_end = float(frame["price"].iloc[-1])  # notional per non-notional
    usd_non_notional = price_end * usd_notional
    if notional_side == "token0":
        p0_usd, p1_usd = usd_notional, usd_non_notional
    else:
        p0_usd, p1_usd = usd_non_notional, usd_notional

    truth = position_ground_truth(position, w3)
    return {
        "observations": observations,
        "notional_side": notional_side,
        "usd_notional": usd_notional,
        "token0_usd_price": p0_usd,
        "token1_usd_price": p1_usd,
        "truth": truth,
    }


def run_replication(
    data: dict,
    position: dict,
    fee_model: str = "fee_growth",
    trajectories_dir: Optional[str] = None,
) -> dict:
    """Replicate one position on pre-loaded data; returns fees (backtest
    + on-chain, USD and per leg), their ratios and the pure compute
    time. ``trajectories_dir`` saves the per-bar trajectory as
    ``trajectory_<position_id>.csv``.
    """
    d0, d1 = position["decimals0"], position["decimals1"]
    notional_side = data["notional_side"]

    std_lower = UniswapV3LPEntity.tick_to_price(position["tick_lower"], d0, d1)
    std_upper = UniswapV3LPEntity.tick_to_price(position["tick_upper"], d0, d1)
    if notional_side == "token0":
        price_lower, price_upper = 1.0 / std_upper, 1.0 / std_lower
    else:
        price_lower, price_upper = std_lower, std_upper

    strategy = FixedRangeLiquidityProvision(
        params=FixedRangeLiquidityProvisionParams(
            PRICE_LOWER=price_lower,
            PRICE_UPPER=price_upper,
            TOKEN0_AMOUNT=position["amount0_human"],
            TOKEN1_AMOUNT=position["amount1_human"],
        ),
        token0_decimals=d0,
        token1_decimals=d1,
        notional_side=notional_side,
        pool_fee_rate=0.0,  # a real mint from pre-held tokens: no swap
        fee_model=fee_model,
        protocol_fee=position["protocol_fee"],  # used by the aggregate model only
    )
    # Pure backtest compute time: the strategy loop only — data loading
    # and observation preparation are excluded.
    t0 = time.perf_counter()
    result = strategy.run(data["observations"])
    backtest_seconds = time.perf_counter() - t0
    df = result.to_dataframe()

    position_id = position["token_id"]
    if trajectories_dir is not None:
        os.makedirs(trajectories_dir, exist_ok=True)
        df.to_csv(os.path.join(trajectories_dir, f"trajectory_{position_id}.csv"), index=False)

    p0_usd, p1_usd = data["token0_usd_price"], data["token1_usd_price"]
    fees_token0 = float(df["UNISWAP_V3_fees_token0"].iloc[-1])
    fees_token1 = float(df["UNISWAP_V3_fees_token1"].iloc[-1])
    if fee_model == "aggregate":
        # The aggregate model cannot attribute fees to a leg — value the
        # accrued notional cash instead (pair-mode entry leaves ~0 cash).
        fees_usd = float(df["UNISWAP_V3_cash"].iloc[-1]) * data["usd_notional"]
    else:
        fees_usd = fees_token0 * p0_usd + fees_token1 * p1_usd

    truth = data["truth"]
    truth_usd = truth["fees_token0"] * p0_usd + truth["fees_token1"] * p1_usd

    return {
        "replicated_position_id": position_id,
        "pair": position["pair"],
        "pool_address": position["pool_address"],
        "fee_tier": position["fee_tier"],
        "fee_model": fee_model,
        "observations": len(df),
        "backtest_seconds": round(backtest_seconds, 4),
        "fees_accrued_usd": fees_usd,
        "fees_accrued_token0": fees_token0,
        "fees_accrued_token1": fees_token1,
        "fees_accrued_usd_onchain": truth_usd,
        "fees_accrued_token0_onchain": truth["fees_token0"],
        "fees_accrued_token1_onchain": truth["fees_token1"],
        # NaN only when the on-chain reference is zero/missing; the
        # aggregate model cannot attribute legs at all
        "usd_ratio": fees_usd / truth_usd if truth_usd else float("nan"),
        "token0_ratio": fees_token0 / truth["fees_token0"]
        if truth["fees_token0"] and fee_model != "aggregate" else float("nan"),
        "token1_ratio": fees_token1 / truth["fees_token1"]
        if truth["fees_token1"] and fee_model != "aggregate" else float("nan"),
        "token0_usd_price": p0_usd,
        "token1_usd_price": p1_usd,
    }


def replicate_position(
    position: dict,
    api_key: str,
    w3,
    fee_model: str = "fee_growth",
    trajectories_dir: Optional[str] = None,
) -> Optional[dict]:
    """Single-position pipeline: load data, run one replication."""
    data = load_replication_data(position, api_key, w3)
    if data is None:
        return None
    return run_replication(data, position, fee_model, trajectories_dir)


if __name__ == '__main__':
    api_key = os.environ['THE_GRAPH_API_KEY']
    w3 = get_w3()
    rows = [
        replicate_position(position, api_key, w3, trajectories_dir=RESULTS_DIR)
        for position in load_positions()
    ]
    rows = [row for row in rows if row is not None]
    summary = pd.DataFrame(rows)[SUMMARY_COLUMNS]
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, 'summary.csv')
    summary.to_csv(out, index=False)
    pd.set_option('display.width', 240)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:12.8f}"))
    print(f"\nsaved {out} and per-position trajectories in {RESULTS_DIR}")
