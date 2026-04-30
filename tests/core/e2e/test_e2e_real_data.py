"""End-to-end tests against live Uniswap V3 subgraph data.

Marked ``integration`` + ``slow`` — skipped by default in CI. Run with::

    pytest tests/core/e2e/test_e2e_real_data.py -m integration

Requires ``THE_GRAPH_API_KEY`` env var (provided by the project's conftest).
The point of these is to exercise the entity over *actually historical*
shape data (real fee/TVL/liquidity series and real price walks) — they
catch bugs that synthetic data can't, e.g. unusual TVL crashes, liquidity
spikes, or edge-case dates where one of the subgraph fields is null.
"""
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from fractal.core.entities.protocols.uniswap_v3_lp import (UniswapV3LPConfig,
                                                           UniswapV3LPEntity,
                                                           UniswapV3LPGlobalState)
from fractal.loaders import (LoaderType,
                             UniswapV3EthereumPoolDayDataLoader,
                             UniswapV3EthereumPricesLoader)

UTC = timezone.utc

# ETH/USDC 0.05% fee tier on Ethereum mainnet — high-liquidity, stable subgraph.
ETH_USDC_005 = "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640"


@pytest.fixture(scope="module")
def real_ethusdc_bars(THE_GRAPH_API_KEY: str) -> pd.DataFrame:
    """Real daily ETH/USDC bars merged with end-of-day spot price.

    Uses a 60-day fixed historical window so the fixture is stable
    (same data on each run, network permitting).
    """
    end = datetime(2024, 12, 1, tzinfo=UTC)
    start = end - timedelta(days=60)

    pool_loader = UniswapV3EthereumPoolDayDataLoader(
        api_key=THE_GRAPH_API_KEY, pool=ETH_USDC_005,
        loader_type=LoaderType.CSV, start_time=start, end_time=end,
    )
    pool = pool_loader.read(with_run=True)
    if len(pool) == 0:
        pytest.skip("no pool data for the chosen historical window")

    price_loader = UniswapV3EthereumPricesLoader(
        api_key=THE_GRAPH_API_KEY, pool=ETH_USDC_005,
        loader_type=LoaderType.CSV, start_time=start, end_time=end,
    )
    prices = price_loader.read(with_run=True)
    if len(prices) == 0:
        pytest.skip("no price data for the chosen historical window")

    # Resample hourly prices to daily (last value of day) and merge.
    daily_price = prices["price"].resample("1D").last().dropna()
    df = pool.join(daily_price.rename("price"), how="inner").dropna()
    if len(df) < 10:
        pytest.skip(f"too few overlapping bars ({len(df)}) for a meaningful E2E test")
    return df


@pytest.mark.integration
@pytest.mark.slow
def test_real_data_v3_lifecycle_no_nan_no_negative_balance(real_ethusdc_bars):
    """Walk a V3 LP through 60 days of real ETH/USDC data; balance stays
    finite and positive throughout."""
    e = UniswapV3LPEntity(UniswapV3LPConfig())
    first = real_ethusdc_bars.iloc[0]
    e.update_state(UniswapV3LPGlobalState(
        tvl=float(first["tvl"]), volume=float(first["volume"]),
        fees=float(first["fees"]), liquidity=float(first["liquidity"]),
        price=float(first["price"]),
    ))
    e.action_deposit(100_000)
    p0 = float(first["price"])
    e.action_open_position(50_000, price_lower=p0 * 0.85, price_upper=p0 * 1.15)

    for _, row in real_ethusdc_bars.iloc[1:].iterrows():
        e.update_state(UniswapV3LPGlobalState(
            tvl=float(row["tvl"]), volume=float(row["volume"]),
            fees=float(row["fees"]), liquidity=float(row["liquidity"]),
            price=float(row["price"]),
        ))
        # Sanity invariants on every bar:
        assert e.balance == e.balance, "balance is NaN"
        assert e.balance > 0, f"balance non-positive: {e.balance}"
        assert e._internal_state.cash >= 0, f"cash negative: {e._internal_state.cash}"
        assert e._internal_state.liquidity >= 0


@pytest.mark.integration
@pytest.mark.slow
def test_real_data_v3_il_finite_after_run(real_ethusdc_bars):
    """After 60 days of real ETH/USDC, IL is finite and the sign is sensible."""
    e = UniswapV3LPEntity(UniswapV3LPConfig())
    first = real_ethusdc_bars.iloc[0]
    e.update_state(UniswapV3LPGlobalState(
        tvl=float(first["tvl"]), volume=float(first["volume"]),
        fees=float(first["fees"]), liquidity=float(first["liquidity"]),
        price=float(first["price"]),
    ))
    e.action_deposit(100_000)
    p0 = float(first["price"])
    e.action_open_position(50_000, price_lower=p0 * 0.85, price_upper=p0 * 1.15)

    for _, row in real_ethusdc_bars.iloc[1:].iterrows():
        e.update_state(UniswapV3LPGlobalState(
            tvl=float(row["tvl"]), volume=float(row["volume"]),
            fees=float(row["fees"]), liquidity=float(row["liquidity"]),
            price=float(row["price"]),
        ))

    il = e.impermanent_loss
    assert il == il, "IL is NaN"
    # IL is non-negative when in range (V3 concentrated). Out of range it can
    # become negative iff the position was on the "wrong" side and price moved
    # back in (rare and our run window is too short for that). We still want
    # the magnitude reasonable: less than the original notional.
    assert abs(il) < 100_000


@pytest.mark.integration
@pytest.mark.slow
def test_real_data_v3_close_returns_cash(real_ethusdc_bars):
    """Open at the start, close at the end — entity converts everything to cash
    without exception. Final cash positive."""
    e = UniswapV3LPEntity(UniswapV3LPConfig())
    first = real_ethusdc_bars.iloc[0]
    e.update_state(UniswapV3LPGlobalState(
        tvl=float(first["tvl"]), volume=float(first["volume"]),
        fees=float(first["fees"]), liquidity=float(first["liquidity"]),
        price=float(first["price"]),
    ))
    e.action_deposit(100_000)
    p0 = float(first["price"])
    e.action_open_position(50_000, price_lower=p0 * 0.85, price_upper=p0 * 1.15)

    for _, row in real_ethusdc_bars.iloc[1:].iterrows():
        e.update_state(UniswapV3LPGlobalState(
            tvl=float(row["tvl"]), volume=float(row["volume"]),
            fees=float(row["fees"]), liquidity=float(row["liquidity"]),
            price=float(row["price"]),
        ))
    e.action_close_position()
    assert not e.is_position
    assert e._internal_state.cash > 0
