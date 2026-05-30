"""Offline tests for ``AaveV3RatesLoader.transform()`` — no network.

Regression tests for the borrow-rate sign convention (H1, codex review):
``borrowing_rate > 0`` must mean *debt grows per step*, mirroring
``SimpleLendingEntity`` and the docstring at the top of
``fractal/loaders/aave.py``.
"""
from datetime import datetime, timezone

import pandas as pd
import pytest

from fractal.loaders.aave import AaveV3ArbitrumLoader

UTC = timezone.utc

# Token used only as a constructor placeholder; transform never reads it.
WETH = "0x82af49447d8a07e3bd95bd0d56f35241523fbab1"


def _loader_with_data(df: pd.DataFrame, resolution: int = 1) -> AaveV3ArbitrumLoader:
    loader = AaveV3ArbitrumLoader(asset_address=WETH, resolution=resolution)
    loader._data = df  # bypass extract(); transform() consumes _data directly
    return loader


@pytest.mark.core
def test_transform_keeps_borrowing_rate_positive():
    """Lock-in for H1: positive annual APY must produce a positive
    per-period borrowing_rate (downstream expects ``balance *= 1 + rate``)."""
    df = pd.DataFrame({
        "date": [datetime(2024, 1, 1, tzinfo=UTC)],
        "lending_rate": [0.05],    # 5% APY supply
        "borrowing_rate": [0.10],  # 10% APY borrow
    })
    loader = _loader_with_data(df, resolution=24)  # daily steps
    loader.transform()
    assert loader._data["borrowing_rate"].iloc[0] > 0
    assert loader._data["lending_rate"].iloc[0] > 0


@pytest.mark.core
def test_transform_per_period_scaling():
    """At ``resolution=24`` (daily), per-period rate is annual/365."""
    df = pd.DataFrame({
        "date": [datetime(2024, 1, 1, tzinfo=UTC)],
        "lending_rate": [0.0365],
        "borrowing_rate": [0.0876],
    })
    loader = _loader_with_data(df, resolution=24)
    loader.transform()
    assert loader._data["lending_rate"].iloc[0] == pytest.approx(0.0365 / 365, rel=1e-9)
    assert loader._data["borrowing_rate"].iloc[0] == pytest.approx(0.0876 / 365, rel=1e-9)


@pytest.mark.core
def test_transform_empty_dataframe_returns_well_shaped_columns():
    loader = _loader_with_data(pd.DataFrame())
    loader.transform()
    assert list(loader._data.columns) == ["date", "lending_rate", "borrowing_rate"]
    assert loader._data.empty


@pytest.mark.core
def test_read_raises_on_missing_rates_after_merge():
    """Outer merge can leave NaN on one leg; must not silently become 0% APY."""
    df = pd.DataFrame({
        "date": [datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 2, tzinfo=UTC)],
        "lending_rate": [0.05, float("nan")],
        "borrowing_rate": [float("nan"), 0.10],
    })
    loader = _loader_with_data(df, resolution=24)
    loader.transform()
    loader._read = lambda _key: None  # _data already set; skip CSV cache lookup
    with pytest.raises(ValueError, match="missing lending/borrowing"):
        loader.read()


@pytest.mark.core
def test_window_for_warns_when_span_exceeds_last_year():
    from fractal.loaders.aave import _window_for

    start = datetime(2022, 1, 1, tzinfo=UTC)
    end = datetime(2024, 6, 1, tzinfo=UTC)
    with pytest.warns(UserWarning, match="exceeds Aave GraphQL max window"):
        assert _window_for(start, end) == "LAST_YEAR"
