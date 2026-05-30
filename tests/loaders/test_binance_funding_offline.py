"""Offline tests for ``BinanceFundingLoader.transform()`` — no network."""
import pandas as pd
import pytest

from fractal.loaders.binance.binance_funding_rates import BinanceFundingLoader


def _loader_with_data(df: pd.DataFrame) -> BinanceFundingLoader:
    loader = BinanceFundingLoader(ticker="BTCUSDT")
    loader._data = df
    return loader


@pytest.mark.core
def test_transform_rejects_non_numeric_funding_rates():
    df = pd.DataFrame({
        "fundingTime": pd.to_datetime(["2024-01-01", "2024-01-02"], utc=True),
        "fundingRate": [0.0001, "bad"],
        "ticker": ["BTCUSDT", "BTCUSDT"],
    })
    loader = _loader_with_data(df)
    with pytest.raises(ValueError, match="missing or non-numeric"):
        loader.transform()
