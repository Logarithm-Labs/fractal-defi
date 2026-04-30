"""Real-API tests for the Aave V3 GraphQL loaders.

The legacy ``aave-api-v2.aave.com/data/rates-history`` REST endpoint was
sunset in 2025; the loader now goes through ``api.v3.aave.com/graphql``.
``AaveV2EthereumLoader`` is preserved as a deprecated alias for the V3
Ethereum mainnet loader (the V2 backend is gone).
"""
import warnings
from datetime import datetime, timedelta, timezone

import pytest

from fractal.loaders import (AaveV2EthereumLoader, AaveV3ArbitrumLoader,
                             AaveV3EthereumLoader, LendingHistory,
                             LoaderType)

UTC = timezone.utc

# Token addresses commonly used in tests.
WETH_ARBITRUM = "0x82af49447d8a07e3bd95bd0d56f35241523fbab1"
WETH_ETHEREUM = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
DAI_ETHEREUM = "0x6b175474e89094c44da98b954eedeac495271d0f"


@pytest.mark.integration
def test_aave_v3_arbitrum():
    end = datetime.now(UTC)
    start = end - timedelta(days=7)
    loader = AaveV3ArbitrumLoader(
        asset_address=WETH_ARBITRUM,
        loader_type=LoaderType.CSV,
        start_time=start,
        end_time=end,
        resolution=24,
    )
    data: LendingHistory = loader.read(with_run=True)
    assert isinstance(data, LendingHistory)
    assert len(data) > 0
    assert data["borrowing_rate"].dtype == "float64"
    assert data["lending_rate"].dtype == "float64"
    assert data["lending_rate"].iloc[-1] > 0  # supply rate: positive ⇒ you earn
    assert data["borrowing_rate"].iloc[0] > 0  # borrow rate: positive ⇒ debt grows
    assert data.index.tz is not None


@pytest.mark.integration
def test_aave_v3_ethereum():
    end = datetime.now(UTC)
    start = end - timedelta(days=7)
    loader = AaveV3EthereumLoader(
        asset_address=WETH_ETHEREUM,
        loader_type=LoaderType.CSV,
        start_time=start,
        end_time=end,
        resolution=24,
    )
    data: LendingHistory = loader.read(with_run=True)
    assert len(data) > 0
    assert data["lending_rate"].iloc[-1] > 0
    assert data["borrowing_rate"].iloc[0] > 0


@pytest.mark.integration
def test_aave_v2_ethereum_emits_deprecation_warning_and_works():
    """The deprecated alias must keep returning data (V3 mainnet) while
    surfacing a DeprecationWarning so callers can migrate."""
    end = datetime.now(UTC)
    start = end - timedelta(days=7)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loader = AaveV2EthereumLoader(
            asset_address=DAI_ETHEREUM,
            loader_type=LoaderType.CSV,
            start_time=start,
            end_time=end,
            resolution=24,
        )
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
    data: LendingHistory = loader.read(with_run=True)
    assert len(data) > 0
    assert data["lending_rate"].dtype == "float64"


@pytest.mark.integration
def test_aave_v3_arbitrum_empty_window_returns_empty_history():
    """A window strictly before any Aave V3 data must produce an empty,
    well-shaped LendingHistory rather than crashing."""
    far_past_start = datetime(2000, 1, 1, tzinfo=UTC)
    far_past_end = datetime(2000, 1, 2, tzinfo=UTC)
    loader = AaveV3ArbitrumLoader(
        asset_address=WETH_ARBITRUM,
        start_time=far_past_start,
        end_time=far_past_end,
        resolution=24,
    )
    data: LendingHistory = loader.read(with_run=True)
    assert isinstance(data, LendingHistory)
    assert len(data) == 0
    assert list(data.columns) == ["lending_rate", "borrowing_rate"]


@pytest.mark.integration
def test_aave_v3_arbitrum_cache_round_trip():
    end = datetime.now(UTC)
    start = end - timedelta(days=7)
    loader = AaveV3ArbitrumLoader(
        asset_address=WETH_ARBITRUM,
        start_time=start, end_time=end, resolution=24,
    )
    fresh = loader.read(with_run=True)
    again = AaveV3ArbitrumLoader(
        asset_address=WETH_ARBITRUM,
        start_time=start, end_time=end, resolution=24,
    ).read()
    assert len(fresh) == len(again)
