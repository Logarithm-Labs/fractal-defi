"""Offline tests for :func:`fractal.loaders.missing_bars`: grid bounds, off-grid points and inputs."""
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from fractal.loaders import PriceHistory, missing_bars

START = datetime(2026, 3, 1, tzinfo=timezone.utc)


@pytest.mark.core
def test_missing_bars_lists_absent_grid_bars_only():
    times = [START + timedelta(hours=h) for h in (0, 1, 3)] + [START + timedelta(hours=2, minutes=30)]
    end = START + timedelta(hours=5)
    missing = missing_bars(times, START, end)
    assert missing.equals(pd.DatetimeIndex([START + timedelta(hours=h) for h in (2, 4, 5)], name="time"))
    half_open = missing_bars(times, START, end, inclusive="left")  # [start, end): the research panel convention
    assert half_open.equals(pd.DatetimeIndex([START + timedelta(hours=h) for h in (2, 4)], name="time"))
    assert str(missing.tz) == "UTC"


@pytest.mark.core
def test_missing_bars_accepts_structs_naive_utc_and_daily_frequency():
    days = pd.date_range(START, periods=4, freq="D").delete(1)
    prices = PriceHistory(prices=[1.0, 2.0, 3.0], time=days)
    naive_end = (START + timedelta(days=3)).replace(tzinfo=None)  # naive datetimes are UTC
    assert missing_bars(prices, START, naive_end, freq="1D").equals(
        pd.DatetimeIndex([START + timedelta(days=1)], name="time"))
    assert missing_bars([], START, START + timedelta(hours=1)).size == 2
    assert missing_bars(prices, START, START + timedelta(hours=1), freq="1D").empty


@pytest.mark.core
def test_missing_bars_validates_arguments():
    with pytest.raises(ValueError, match="end"):
        missing_bars([], START + timedelta(hours=1), START)
    with pytest.raises(ValueError, match="inclusive"):
        missing_bars([], START, START, inclusive="right")
