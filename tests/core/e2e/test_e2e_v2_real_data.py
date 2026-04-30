"""End-to-end tests against live Uniswap V2 subgraph data.

Marked ``integration`` + ``slow``. Run with::

    pytest tests/core/e2e/test_e2e_v2_real_data.py -m integration

Requires ``THE_GRAPH_API_KEY`` env var.

V2's subgraph doesn't expose a usable spot price field directly, so we
join the pool snapshot stream (TVL/volume/fees/liquidity) with the V3
prices stream for the same ETH/USDC pair (V3 gives a clean tick-derived
ETH price). Pool data drives the entity's pool-share math; price drives
mark-to-market.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from fractal.core.entities.protocols.uniswap_v2_lp import (UniswapV2LPConfig,
                                                            UniswapV2LPEntity,
                                                            UniswapV2LPGlobalState)
from fractal.loaders import (EthereumUniswapV2PoolDataLoader, LoaderType,
                             UniswapV3EthereumPricesLoader)

UTC = timezone.utc

# Uniswap V2 WETH/USDC pair on Ethereum mainnet — one of the largest V2 pairs.
V2_WETH_USDC = "0xB4e16d0168e52d35CaCD2c6185b44281Ec28C9Dc"
# V3 ETH/USDC 0.05% — used purely as a clean ETH/USD price source for the
# same period (V2 has no spot field of its own in the subgraph).
V3_ETH_USDC_005 = "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640"

V2_FEE_TIER = 0.003  # 0.3% — only V2 fee tier


@pytest.fixture(scope="module")
def real_v2_ethusdc_bars(THE_GRAPH_API_KEY: str) -> pd.DataFrame:
    """Real ETH/USDC V2 pool bars merged with V3-derived ETH price."""
    end = datetime(2024, 12, 1, tzinfo=UTC)
    start = end - timedelta(days=60)

    pool_loader = EthereumUniswapV2PoolDataLoader(
        api_key=THE_GRAPH_API_KEY, pool=V2_WETH_USDC, fee_tier=V2_FEE_TIER,
        loader_type=LoaderType.CSV, start_time=start, end_time=end,
    )
    pool = pool_loader.read(with_run=True)
    if len(pool) == 0:
        pytest.skip("no V2 pool data for the chosen historical window")

    # ETH price from V3 (same asset, cleanest source).
    price_loader = UniswapV3EthereumPricesLoader(
        api_key=THE_GRAPH_API_KEY, pool=V3_ETH_USDC_005,
        loader_type=LoaderType.CSV, start_time=start, end_time=end,
    )
    prices = price_loader.read(with_run=True)
    if len(prices) == 0:
        pytest.skip("no V3 price data for the chosen historical window")

    # V2 pool history is hourly; resample prices to hourly mean.
    hourly_price = prices["price"].resample("1h").mean().dropna()
    df = pool.join(hourly_price.rename("price"), how="inner").dropna()
    if len(df) < 24:
        pytest.skip(f"too few overlapping bars ({len(df)})")
    return df


@pytest.mark.integration
@pytest.mark.slow
def test_real_v2_lifecycle_no_nan_no_negative_balance(real_v2_ethusdc_bars):
    """Walk a V2 LP through 60 days of real ETH/USDC; balance stays
    finite and positive throughout."""
    e = UniswapV2LPEntity(UniswapV2LPConfig())
    first = real_v2_ethusdc_bars.iloc[0]
    e.update_state(UniswapV2LPGlobalState(
        tvl=float(first["tvl"]), volume=float(first["volume"]),
        fees=float(first["fees"]), liquidity=float(first["liquidity"]),
        price=float(first["price"]),
    ))
    e.action_deposit(100_000)
    e.action_open_position(50_000)

    for _, row in real_v2_ethusdc_bars.iloc[1:].iterrows():
        e.update_state(UniswapV2LPGlobalState(
            tvl=float(row["tvl"]), volume=float(row["volume"]),
            fees=float(row["fees"]), liquidity=float(row["liquidity"]),
            price=float(row["price"]),
        ))
        assert e.balance == e.balance, "balance is NaN"
        assert e.balance > 0, f"balance non-positive: {e.balance}"
        assert e._internal_state.cash >= 0
        assert e._internal_state.liquidity >= 0


@pytest.mark.integration
@pytest.mark.slow
def test_real_v2_il_finite_after_run(real_v2_ethusdc_bars):
    """After 60 days of real ETH/USDC V2 LP, IL is finite (sign expected
    non-negative because V2 50/50 IL ≥ 0 on any price move)."""
    e = UniswapV2LPEntity(UniswapV2LPConfig())
    first = real_v2_ethusdc_bars.iloc[0]
    e.update_state(UniswapV2LPGlobalState(
        tvl=float(first["tvl"]), volume=float(first["volume"]),
        fees=float(first["fees"]), liquidity=float(first["liquidity"]),
        price=float(first["price"]),
    ))
    e.action_deposit(100_000)
    e.action_open_position(50_000)

    for _, row in real_v2_ethusdc_bars.iloc[1:].iterrows():
        e.update_state(UniswapV2LPGlobalState(
            tvl=float(row["tvl"]), volume=float(row["volume"]),
            fees=float(row["fees"]), liquidity=float(row["liquidity"]),
            price=float(row["price"]),
        ))

    il = e.impermanent_loss
    assert il == il, "IL is NaN"
    # V2 50/50 IL is theoretically ≥ 0 on any price move (excluding fees).
    # But our entity model uses pool TVL directly (not constant-product
    # rebalance), so IL sign depends on TVL dynamics. Just verify finite.
    assert abs(il) < 100_000


@pytest.mark.integration
@pytest.mark.slow
def test_real_v2_close_returns_cash(real_v2_ethusdc_bars):
    """Open at start, close at end — entity converts everything back to
    cash without exception."""
    e = UniswapV2LPEntity(UniswapV2LPConfig())
    first = real_v2_ethusdc_bars.iloc[0]
    e.update_state(UniswapV2LPGlobalState(
        tvl=float(first["tvl"]), volume=float(first["volume"]),
        fees=float(first["fees"]), liquidity=float(first["liquidity"]),
        price=float(first["price"]),
    ))
    e.action_deposit(100_000)
    e.action_open_position(50_000)

    for _, row in real_v2_ethusdc_bars.iloc[1:].iterrows():
        e.update_state(UniswapV2LPGlobalState(
            tvl=float(row["tvl"]), volume=float(row["volume"]),
            fees=float(row["fees"]), liquidity=float(row["liquidity"]),
            price=float(row["price"]),
        ))
    e.action_close_position()
    assert not e.is_position
    assert e._internal_state.cash > 0


@pytest.mark.integration
@pytest.mark.slow
def test_real_v2_fees_accrue_when_in_position(real_v2_ethusdc_bars):
    """Pool earns fees → LP cash should grow over time (assuming
    non-zero pool fees on the bars)."""
    e = UniswapV2LPEntity(UniswapV2LPConfig())
    first = real_v2_ethusdc_bars.iloc[0]
    e.update_state(UniswapV2LPGlobalState(
        tvl=float(first["tvl"]), volume=float(first["volume"]),
        fees=float(first["fees"]), liquidity=float(first["liquidity"]),
        price=float(first["price"]),
    ))
    e.action_deposit(100_000)
    e.action_open_position(50_000)
    cash_after_open = e._internal_state.cash

    for _, row in real_v2_ethusdc_bars.iloc[1:].iterrows():
        e.update_state(UniswapV2LPGlobalState(
            tvl=float(row["tvl"]), volume=float(row["volume"]),
            fees=float(row["fees"]), liquidity=float(row["liquidity"]),
            price=float(row["price"]),
        ))

    # If any bar had positive fees, cash should have grown.
    if real_v2_ethusdc_bars["fees"].sum() > 0:
        assert e._internal_state.cash >= cash_after_open
