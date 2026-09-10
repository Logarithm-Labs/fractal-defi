"""Offline tests for the native-hourly pool loader's transform stage.

``transform`` needs no network: construct the loader with explicit
``decimals`` (skips the subgraph lookup) and inject raw rows into
``_data`` the way ``extract`` would.
"""
import pytest

from fractal.loaders import LoaderType, UniswapV3BasePoolHourDataLoader

POOL = "0xd0b53d9277642d899df5c87a3966a349a798f224"


def make_loader() -> UniswapV3BasePoolHourDataLoader:
    return UniswapV3BasePoolHourDataLoader(
        api_key="dummy", pool=POOL, loader_type=LoaderType.CSV, decimals=12,
    )


def raw_row(ts: int, fg0: int, fg1: int, tvl: str = "1000.0") -> dict:
    return {
        "periodStartUnix": ts,
        "volumeUSD": "100.0",
        "tvlUSD": tvl,
        "feesUSD": "0.05",
        "liquidity": "1000000000000000000",
        "tick": "-201688",
        "feeGrowthGlobal0X128": str(fg0),
        "feeGrowthGlobal1X128": str(fg1),
    }


@pytest.mark.core
def test_fee_growth_deltas_from_cumulative_counters():
    loader = make_loader()
    q128 = 1 << 128
    loader._data = __import__("pandas").DataFrame([
        raw_row(3600, 10 * q128, 100 * q128),
        raw_row(7200, 13 * q128, 100 * q128),
        raw_row(10800, 14 * q128, 105 * q128),
    ])
    loader.transform()
    df = loader._data
    assert list(df["fee_growth0"]) == [0.0, 3.0, 1.0]  # first bar has no seed
    assert list(df["fee_growth1"]) == [0.0, 0.0, 5.0]


@pytest.mark.integration
@pytest.mark.slow
def test_hourly_end_time_boundary_is_inclusive(THE_GRAPH_API_KEY: str):
    """A bar starting exactly at end_time must be returned (the docstring
    promises an inclusive window; WETH/USDC 0.05% trades every hour)."""
    from datetime import UTC, datetime, timedelta

    import pandas as pd

    end = datetime(2026, 7, 3, 12, 0, tzinfo=UTC)
    loader = UniswapV3BasePoolHourDataLoader(
        api_key=THE_GRAPH_API_KEY, pool=POOL, loader_type=LoaderType.CSV,
        start_time=end - timedelta(hours=6), end_time=end, decimals=12,
    )
    history = loader.read(with_run=True)
    assert history.index.max() == pd.Timestamp(end)


@pytest.mark.core
def test_negative_counter_delta_clamped_with_warning():
    """The counter is monotonic on-chain; a dip is a subgraph anomaly —
    the delta is clamped to 0 and a warning names the pool and bar."""
    loader = make_loader()
    q128 = 1 << 128
    loader._data = __import__("pandas").DataFrame([
        raw_row(3600, 10 * q128, 0),
        raw_row(7200, 8 * q128, 0),   # dip: cumulative decreased
        raw_row(10800, 12 * q128, 0),
    ])
    with pytest.warns(UserWarning, match="non-monotonic"):
        loader.transform()
    df = loader._data
    assert df["fee_growth0"].iloc[1] == 0.0            # clamped
    assert df["fee_growth0"].iloc[2] == pytest.approx(4.0)  # spans the dip


@pytest.mark.core
def test_negative_tvl_clamped_with_warning():
    """The subgraph's derived ``tvlUSD`` can dip below zero (seen on Base
    V3 pools); the LP entity rejects negative ``tvl``, so the loader
    clamps it to 0 and warns. Fee-growth deltas are unaffected."""
    loader = make_loader()
    q128 = 1 << 128
    loader._data = __import__("pandas").DataFrame([
        raw_row(3600, 10 * q128, 0),
        raw_row(7200, 13 * q128, 0, tvl="-740.15"),
        raw_row(10800, 14 * q128, 0, tvl="-836.98"),
        raw_row(14400, 15 * q128, 0),
    ])
    with pytest.warns(UserWarning, match="negative tvlUSD"):
        loader.transform()
    df = loader._data
    assert list(df["tvl"]) == [1000.0, 0.0, 0.0, 1000.0]
    assert list(df["fee_growth0"]) == [0.0, 3.0, 1.0, 1.0]
