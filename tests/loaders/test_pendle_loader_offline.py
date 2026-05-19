"""Offline tests for ``PendleMarketLoader`` — no network.

Verify the linear PT-pricing convention, the parallel-array payload
parsing, the optional Pendle API fields (baseApy / underlyingApy /
maxApy), the EVM-address validation on construction, the deterministic
cache-key that includes hour-level epoch seconds, the typed-struct
contract on both cache paths, and the UTC-seconds index conversion.
"""
from datetime import datetime, timezone

import pandas as pd
import pytest

from fractal.loaders.pendle import (
    PendleMarketLoader,
    _compute_pt_price_linear,
    _rebuild_from_cache,
)
from fractal.loaders.structs import PendleMarketHistory

UTC = timezone.utc

_STUB_MARKET = "0xa36b60a14a1a5247912584768c6e53e1a269a9f7"
_STUB_EXPIRY = 1758758400  # 25 Sep 2025
_STUB_START = datetime(2025, 7, 27, tzinfo=UTC)
_STUB_END = datetime(2025, 9, 25, tzinfo=UTC)


def _stub_loader() -> PendleMarketLoader:
    return PendleMarketLoader(
        market_address=_STUB_MARKET,
        expiry_timestamp=_STUB_EXPIRY,
        start_time=_STUB_START,
        end_time=_STUB_END,
    )


@pytest.mark.core
def test_compute_pt_price_linear_at_expiry_is_one():
    assert _compute_pt_price_linear(0.14, 0.0) == 1.0
    assert _compute_pt_price_linear(0.14, -100.0) == 1.0


@pytest.mark.core
def test_compute_pt_price_linear_matches_formula():
    secs = 90 * 24 * 3600
    tau = secs / (365.25 * 24 * 3600)
    expected = 1.0 - 0.10 * tau
    assert abs(_compute_pt_price_linear(0.10, secs) - expected) < 1e-12


@pytest.mark.core
def test_compute_pt_price_clamps_to_unit_interval():
    five_years = 5 * 365.25 * 24 * 3600
    assert _compute_pt_price_linear(1.0, five_years) == 0.0
    assert _compute_pt_price_linear(-0.10, 30 * 24 * 3600) == 1.0


@pytest.mark.core
def test_transform_returns_expected_columns():
    loader = _stub_loader()
    loader._raw = {
        "timestamp": [_STUB_EXPIRY - 86400 * i for i in range(3, 0, -1)],
        "impliedApy": [0.12, 0.13, 0.14],
        "tvl": [1_000_000.0, 1_100_000.0, 1_050_000.0],
    }
    loader.transform()
    expected = {
        "pt_price",
        "implied_yield",
        "seconds_to_expiry",
        "pool_liquidity",
        "base_apy",
        "underlying_apy",
        "max_apy",
    }
    assert set(loader._data.columns) == expected
    assert isinstance(loader._data, PendleMarketHistory)


@pytest.mark.core
def test_transform_unpacks_optional_api_fields():
    loader = _stub_loader()
    loader._raw = {
        "timestamp": [_STUB_EXPIRY - 86400 * 2, _STUB_EXPIRY - 86400],
        "impliedApy": [0.10, 0.11],
        "tvl": [1.0e6, 1.1e6],
        "baseApy": [0.07, 0.075],
        "underlyingApy": [0.06, 0.065],
        "maxApy": [0.20, 0.21],
    }
    loader.transform()
    assert loader._data["base_apy"].iloc[0] == pytest.approx(0.07)
    assert loader._data["underlying_apy"].iloc[1] == pytest.approx(0.065)
    assert loader._data["max_apy"].iloc[1] == pytest.approx(0.21)


@pytest.mark.core
def test_transform_missing_optional_fields_yield_nan():
    loader = _stub_loader()
    loader._raw = {
        "timestamp": [_STUB_EXPIRY - 86400],
        "impliedApy": [0.10],
        "tvl": [1.0e6],
    }
    loader.transform()
    assert pd.isna(loader._data["base_apy"].iloc[0])
    assert pd.isna(loader._data["underlying_apy"].iloc[0])
    assert pd.isna(loader._data["max_apy"].iloc[0])


@pytest.mark.core
def test_transform_seconds_to_expiry_strictly_decreasing():
    loader = _stub_loader()
    loader._raw = {
        "timestamp": [_STUB_EXPIRY - 86400 * i for i in range(3, 0, -1)],
        "impliedApy": [0.12, 0.13, 0.14],
        "tvl": [1_000_000.0, 1_100_000.0, 1_050_000.0],
    }
    loader.transform()
    secs = loader._data["seconds_to_expiry"].to_numpy()
    assert (secs[:-1] > secs[1:]).all()


@pytest.mark.core
def test_transform_index_is_utc_seconds_aware():
    """Epoch seconds must land in the 2025 timeline, not near 1970."""
    loader = _stub_loader()
    loader._raw = {
        "timestamp": [_STUB_EXPIRY - 86400],
        "impliedApy": [0.10],
        "tvl": [1.0e6],
    }
    loader.transform()
    idx = loader._data.index
    assert isinstance(idx, pd.DatetimeIndex)
    assert idx.tz is not None  # tz-aware
    # 2025-09-24
    assert idx[0].year == 2025
    assert idx[0].month == 9
    assert idx[0].day == 24


@pytest.mark.core
def test_transform_handles_empty_payload():
    loader = _stub_loader()
    loader._raw = {}
    loader.transform()
    assert isinstance(loader._data, PendleMarketHistory)
    assert loader._data.empty


@pytest.mark.core
def test_transform_skips_malformed_rows():
    loader = _stub_loader()
    loader._raw = {
        "timestamp": [_STUB_EXPIRY - 7200, "not-a-number", _STUB_EXPIRY - 3600],
        "impliedApy": [0.10, 0.12, 0.11],
        "tvl": [1_000.0, 1_100.0, 1_050.0],
    }
    loader.transform()
    assert len(loader._data) == 2


@pytest.mark.core
def test_constructor_rejects_invalid_market_address():
    with pytest.raises(Exception):  # GraphLoaderException
        PendleMarketLoader(
            market_address="not-an-address",
            expiry_timestamp=_STUB_EXPIRY,
            start_time=_STUB_START,
            end_time=_STUB_END,
        )


@pytest.mark.core
def test_constructor_rejects_path_traversal_in_address():
    """Cache filename safety: reject anything that would escape the cache dir."""
    with pytest.raises(Exception):
        PendleMarketLoader(
            market_address="0x../etc/passwd",
            expiry_timestamp=_STUB_EXPIRY,
            start_time=_STUB_START,
            end_time=_STUB_END,
        )


@pytest.mark.core
def test_constructor_rejects_unsupported_chain():
    with pytest.raises(ValueError):
        PendleMarketLoader(
            market_address=_STUB_MARKET,
            expiry_timestamp=_STUB_EXPIRY,
            start_time=_STUB_START,
            end_time=_STUB_END,
            chain_id=10,  # Optimism not supported
        )


@pytest.mark.core
def test_constructor_rejects_reversed_window():
    with pytest.raises(ValueError):
        PendleMarketLoader(
            market_address=_STUB_MARKET,
            expiry_timestamp=_STUB_EXPIRY,
            start_time=_STUB_END,
            end_time=_STUB_START,
        )


@pytest.mark.core
def test_cache_key_uses_full_epoch_seconds_not_dates():
    """Two windows that share the date but differ by hours must not collide."""
    a = PendleMarketLoader(
        market_address=_STUB_MARKET,
        expiry_timestamp=_STUB_EXPIRY,
        start_time=datetime(2025, 7, 27, 0, tzinfo=UTC),
        end_time=datetime(2025, 7, 27, 6, tzinfo=UTC),
    )
    b = PendleMarketLoader(
        market_address=_STUB_MARKET,
        expiry_timestamp=_STUB_EXPIRY,
        start_time=datetime(2025, 7, 27, 12, tzinfo=UTC),
        end_time=datetime(2025, 7, 27, 18, tzinfo=UTC),
    )
    assert a._cache_key() != b._cache_key()


@pytest.mark.core
def test_rebuild_from_cache_restores_typed_history():
    """Cache rehydration must return ``PendleMarketHistory`` with UTC index."""
    df = pd.DataFrame(
        {
            "pt_price": [0.98, 0.99],
            "implied_yield": [0.10, 0.09],
            "seconds_to_expiry": [3600.0, 1800.0],
            "pool_liquidity": [1e6, 1e6],
            "base_apy": [0.07, 0.07],
            "underlying_apy": [0.06, 0.06],
            "max_apy": [0.20, 0.20],
        },
        index=pd.to_datetime([_STUB_EXPIRY - 3600, _STUB_EXPIRY - 1800], unit="s", utc=True),
    )
    df.index.name = "time"
    rebuilt = _rebuild_from_cache(df)
    assert isinstance(rebuilt, PendleMarketHistory)
    assert rebuilt.index.tz is not None
    assert len(rebuilt) == 2


@pytest.mark.core
def test_rebuild_from_cache_handles_empty_dataframe():
    rebuilt = _rebuild_from_cache(pd.DataFrame())
    assert isinstance(rebuilt, PendleMarketHistory)
    assert rebuilt.empty


@pytest.mark.core
def test_empty_history_constructor_accepts_required_and_optional_args():
    """Regression guard: the empty-payload helper must construct a valid
    ``PendleMarketHistory`` without ``TypeError``. The 5 required
    parameters (``pt_prices``, ``implied_yields``, ``seconds_to_expiry``,
    ``pool_liquidity``, ``time``) plus the 3 optional API-extra columns
    are all explicit keyword arguments, so adding new optional columns
    later does not silently change empty-branch behaviour.
    """
    h = PendleMarketHistory(
        pt_prices=[],
        implied_yields=[],
        seconds_to_expiry=[],
        pool_liquidity=[],
        time=[],
        base_apy=[],
        underlying_apy=[],
        max_apy=[],
    )
    assert isinstance(h, PendleMarketHistory)
    assert h.empty
    assert set(h.columns) >= {
        "pt_price",
        "implied_yield",
        "seconds_to_expiry",
        "pool_liquidity",
        "base_apy",
        "underlying_apy",
        "max_apy",
    }
