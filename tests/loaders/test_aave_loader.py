from datetime import datetime

import pytest

from fractal.loaders import (AaveV2EthereumLoader, AaveV3ArbitrumLoader,
                             LendingHistory, LoaderType)


@pytest.mark.integration
def test_aave_v2_ethereum():
    loader: AaveV2EthereumLoader = AaveV2EthereumLoader(
        asset_address="0x6b175474e89094c44da98b954eedeac495271d0f",  # DAI
        loader_type=LoaderType.CSV,
        start_time=datetime(2025, 1, 1),
        resolution=24,
    )
    data: LendingHistory = loader.read(with_run=True)
    assert len(data) > 0
    assert data["borrowing_rate"].dtype == "float64"
    assert data["lending_rate"].dtype == "float64"
    assert data["lending_rate"].iloc[-1] > 0
    assert data["borrowing_rate"].iloc[0] < 0


@pytest.mark.integration
def test_aave_v3_arbitrum():
    loader: AaveV3ArbitrumLoader = AaveV3ArbitrumLoader(
        asset_address="0x82aF49447D8a07e3bd95BD0d56f35241523fBab1",  # WETH
        loader_type=LoaderType.CSV,
        start_time=datetime(2025, 1, 1),
        resolution=24,
    )
    data: LendingHistory = loader.read(with_run=True)
    assert len(data) > 0
    assert data["borrowing_rate"].dtype == "float64"
    assert data["lending_rate"].dtype == "float64"
    assert data["lending_rate"].iloc[-1] > 0
    assert data["borrowing_rate"].iloc[0] < 0
