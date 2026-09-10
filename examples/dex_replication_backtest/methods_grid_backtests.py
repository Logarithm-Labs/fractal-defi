"""Compare fee-accounting methods over the replicated positions:
``fee_growth`` (entity, counter deltas), ``aggregate`` (entity, feesUSD
share net of protocol_fee; no per-leg attribution) and ``per_event``
(event-replay reference via ``UniswapV3SwapsLoader``, tick-gated,
``fee = gross_input × tier/1e6``, net of protocol_fee). All vs the same on-chain
``feeGrowthInside`` truth; inputs are loaded once per position and
shared by both Fractal models. Ratios are reported in USD and per
token leg; timings as pure compute and full wall time.

Usage: ``python methods_grid_backtests.py`` — requires
``THE_GRAPH_API_KEY`` and a ``BASE_RPC_URL`` with ``eth_getLogs``.
Output: results/methods_grid_results.csv + a printed summary.
"""
import os
import time

import pandas as pd
from backtest import RESULTS_DIR, load_replication_data, run_replication
from helpers import get_w3, load_positions

from fractal.core.entities import UniswapV3LPEntity
from fractal.core.entities.models.uniswap_v3_fees import get_liquidity_delta
from fractal.loaders import LoaderType, UniswapV3SwapsLoader


def per_event_position(position: dict, rpc_url: str) -> dict:
    """Reconstruct the position's fees swap-by-swap from event logs:
    for every tick-in-range Swap, ``fee = gross_input × tier/1e6``
    (event amounts include the fee) × ``1 − protocol_fee`` ×
    ``L_pos / L_swap`` with the pool liquidity from the event itself.
    """
    swaps = UniswapV3SwapsLoader(
        rpc_url=rpc_url,
        pool=position["pool_address"],
        token0_decimals=position["decimals0"],
        token1_decimals=position["decimals1"],
        start_block=position["mint_block"],
        loader_type=LoaderType.CSV,
    ).read(with_run=True)

    # Pure compute time: the per-swap fee aggregation only (the event
    # pull above is data loading and excluded, like in the other modes).
    t0 = time.perf_counter()
    in_range = swaps[
        (swaps["liquidity"] > 0)
        & (swaps["tick"] >= position["tick_lower"])
        & (swaps["tick"] < position["tick_upper"])
    ]
    # On-chain L of the position, reconstructed from its mint facts.
    d0, d1 = position["decimals0"], position["decimals1"]
    l_pos = get_liquidity_delta(
        P=position["mint_price_std"],
        lower_price=UniswapV3LPEntity.tick_to_price(position["tick_lower"], d0, d1),
        upper_price=UniswapV3LPEntity.tick_to_price(position["tick_upper"], d0, d1),
        amount0=position["amount0_human"],
        amount1=position["amount1_human"],
        token0_decimal=d0,
        token1_decimal=d1,
    )
    tier = position["fee_tier"]
    k = tier / 1e6 * (1 - position["protocol_fee"]) * l_pos
    input0 = in_range["amount0"] > 0  # token0 is the input side
    fees_token0 = (in_range.loc[input0, "amount0"] * k / in_range.loc[input0, "liquidity"]).sum()
    fees_token1 = (in_range.loc[~input0, "amount1"] * k / in_range.loc[~input0, "liquidity"]).sum()
    compute_seconds = time.perf_counter() - t0
    return {
        "fees_token0": float(fees_token0),
        "fees_token1": float(fees_token1),
        "swaps_in_range": len(in_range),
        "compute_seconds": round(compute_seconds, 4),
    }


def main() -> None:
    api_key = os.environ["THE_GRAPH_API_KEY"]
    rpc_url = os.getenv("BASE_RPC_URL", "https://mainnet.base.org")
    w3 = get_w3()

    rows = []
    for position in load_positions():
        # One data load per position, shared by both Fractal models.
        t = time.perf_counter()
        data = load_replication_data(position, api_key, w3)
        load_seconds = time.perf_counter() - t
        if data is None:
            continue

        growth = run_replication(data, position, fee_model="fee_growth")
        aggregate = run_replication(data, position, fee_model="aggregate")
        t = time.perf_counter()
        event = per_event_position(position, rpc_url)
        event_total = time.perf_counter() - t

        truth = data["truth"]
        event_usd = (event["fees_token0"] * data["token0_usd_price"]
                     + event["fees_token1"] * data["token1_usd_price"])

        rows.append({
            "replicated_position_id": position["token_id"],
            "pair": position["pair"],
            "pool_address": position["pool_address"],
            "fee_tier": position["fee_tier"],
            "fees_accrued_usd_onchain": growth["fees_accrued_usd_onchain"],
            # Fractal, feeGrowth mode
            "growth_usd_ratio": growth["usd_ratio"],
            "growth_token0_ratio": growth["token0_ratio"],
            "growth_token1_ratio": growth["token1_ratio"],
            # Fractal, aggregate mode (no leg attribution -> NaN legs)
            "aggregate_usd_ratio": aggregate["usd_ratio"],
            "aggregate_token0_ratio": aggregate["token0_ratio"],
            "aggregate_token1_ratio": aggregate["token1_ratio"],
            # Event replay reference (NaN when the reference is zero)
            "event_usd_ratio": event_usd / growth["fees_accrued_usd_onchain"]
            if growth["fees_accrued_usd_onchain"] else float("nan"),
            "event_token0_ratio": event["fees_token0"] / truth["fees_token0"]
            if truth["fees_token0"] else float("nan"),
            "event_token1_ratio": event["fees_token1"] / truth["fees_token1"]
            if truth["fees_token1"] else float("nan"),
            "observations": growth["observations"],
            "swaps_in_range": event["swaps_in_range"],
            # pure compute time (no data loading / preparation)
            "growth_seconds": growth["backtest_seconds"],
            "aggregate_seconds": aggregate["backtest_seconds"],
            "event_seconds": event["compute_seconds"],
            # full backtest wall time, incl. data pull and preparation
            # (the shared load is attributed to both Fractal models)
            "growth_total_seconds": round(load_seconds + growth["backtest_seconds"], 2),
            "aggregate_total_seconds": round(load_seconds + aggregate["backtest_seconds"], 2),
            "event_total_seconds": round(event_total, 2),
        })
        r = rows[-1]
        print(f"{r['pair']:>10} {r['fee_tier']:>5}  "
              f"growth=({r['growth_usd_ratio']:.4f}|{r['growth_token0_ratio']:.4f},{r['growth_token1_ratio']:.4f})  "
              f"aggregate={r['aggregate_usd_ratio']:.4f}  "
              f"event=({r['event_usd_ratio']:.4f}|{r['event_token0_ratio']:.4f},{r['event_token1_ratio']:.4f})  "
              f"swaps={r['swaps_in_range']} "
              f"total=({r['growth_total_seconds']:.1f}s/{r['aggregate_total_seconds']:.1f}s/"
              f"{r['event_total_seconds']:.1f}s)",
              flush=True)

    grid = pd.DataFrame(rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, "methods_grid_results.csv")
    grid.to_csv(out, index=False)

    ratio_cols = [c for c in grid.columns if c.endswith("_ratio")]
    summary = grid[ratio_cols].agg(["median", "min", "max"]).T
    timing = grid[["growth_seconds", "aggregate_seconds", "event_seconds",
                   "growth_total_seconds", "aggregate_total_seconds",
                   "event_total_seconds"]].agg(["median", "sum"]).T
    pd.set_option("display.width", 260)
    print("\n" + grid.to_string(index=False, float_format=lambda x: f"{x:9.4f}"))
    print("\nmethod summary (ratio vs on-chain accrual):")
    print(summary.to_string(float_format=lambda x: f"{x:8.4f}"))
    print("\ncompute time, seconds (pure backtest vs full run incl. data):")
    print(timing.to_string(float_format=lambda x: f"{x:8.4f}"))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
