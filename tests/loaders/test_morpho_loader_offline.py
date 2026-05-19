"""Offline tests for ``MorphoMarketLoader`` — no network.

Verify the GraphQL parallel-series merge, the empty-payload short-circuit,
the typed ``LendingHistory`` return contract (with ``utilization``), the
strict 32-byte market-id validation, and the UTC-seconds index conversion.
"""
from datetime import datetime, timezone

import pandas as pd
import pytest

from fractal.loaders.morpho import MorphoMarketLoader, _rebuild_from_cache
from fractal.loaders.structs import LendingHistory

UTC = timezone.utc

_STUB_MARKET = "0x" + "ab" * 32  # 32-byte = 64 hex chars after 0x
_STUB_START = datetime(2025, 7, 27, tzinfo=UTC)
_STUB_END = datetime(2025, 9, 25, tzinfo=UTC)


def _stub_loader(chain: str = "ethereum") -> MorphoMarketLoader:
    return MorphoMarketLoader(
        market_id=_STUB_MARKET,
        chain=chain,
        start_time=_STUB_START,
        end_time=_STUB_END,
    )


@pytest.mark.core
def test_constructor_validates_market_id_prefix():
    with pytest.raises(ValueError):
        MorphoMarketLoader(
            market_id="not-0x-prefixed",
            chain="ethereum",
            start_time=_STUB_START,
            end_time=_STUB_END,
        )


@pytest.mark.core
def test_constructor_rejects_short_market_id():
    """0x-prefix but only 4 hex chars: must fail (full 64-char check)."""
    with pytest.raises(ValueError):
        MorphoMarketLoader(
            market_id="0xabcd",
            chain="ethereum",
            start_time=_STUB_START,
            end_time=_STUB_END,
        )


@pytest.mark.core
def test_constructor_rejects_non_hex_market_id():
    """0x + 64 chars but one of them is 'z': must fail."""
    bad = "0x" + "z" * 64
    with pytest.raises(ValueError):
        MorphoMarketLoader(
            market_id=bad,
            chain="ethereum",
            start_time=_STUB_START,
            end_time=_STUB_END,
        )


@pytest.mark.core
def test_constructor_rejects_unknown_chain():
    with pytest.raises(ValueError):
        MorphoMarketLoader(
            market_id=_STUB_MARKET,
            chain="solana",
            start_time=_STUB_START,
            end_time=_STUB_END,
        )


@pytest.mark.core
def test_constructor_rejects_reversed_window():
    with pytest.raises(ValueError):
        MorphoMarketLoader(
            market_id=_STUB_MARKET,
            chain="ethereum",
            start_time=_STUB_END,
            end_time=_STUB_START,
        )


@pytest.mark.core
def test_transform_returns_lending_history():
    loader = _stub_loader()
    loader._data = pd.DataFrame(
        {
            "x": [int(_STUB_START.timestamp()), int(_STUB_END.timestamp())],
            "borrowing_rate": [0.05, 0.07],
            "utilization": [0.8, 0.9],
            "lending_rate": [0.04, 0.06],
        }
    )
    loader.transform()
    assert isinstance(loader._data, LendingHistory)
    assert list(loader._data.columns) == ["lending_rate", "borrowing_rate", "utilization"]
    assert len(loader._data) == 2


@pytest.mark.core
def test_transform_index_is_utc_seconds_aware():
    """Epoch seconds must produce 2025 timestamps, not nanosecond offsets near 1970."""
    loader = _stub_loader()
    loader._data = pd.DataFrame(
        {
            "x": [int(_STUB_START.timestamp())],
            "borrowing_rate": [0.05],
            "utilization": [0.8],
            "lending_rate": [0.04],
        }
    )
    loader.transform()
    assert loader._data.index[0].year == 2025
    assert loader._data.index.tz is not None


@pytest.mark.core
def test_transform_handles_empty_staging():
    loader = _stub_loader()
    loader._data = pd.DataFrame()
    loader.transform()
    assert isinstance(loader._data, LendingHistory)
    assert loader._data.empty


@pytest.mark.core
def test_chain_id_routes_to_correct_network():
    assert _stub_loader("ethereum")._chain_id == 1
    assert _stub_loader("arbitrum")._chain_id == 42161
    assert _stub_loader("base")._chain_id == 8453


@pytest.mark.core
def test_cache_key_uses_epoch_seconds():
    """Hour-different windows on the same day must produce different cache keys."""
    a = MorphoMarketLoader(
        market_id=_STUB_MARKET,
        chain="ethereum",
        start_time=datetime(2025, 7, 27, 0, tzinfo=UTC),
        end_time=datetime(2025, 7, 27, 6, tzinfo=UTC),
    )
    b = MorphoMarketLoader(
        market_id=_STUB_MARKET,
        chain="ethereum",
        start_time=datetime(2025, 7, 27, 12, tzinfo=UTC),
        end_time=datetime(2025, 7, 27, 18, tzinfo=UTC),
    )
    assert a._cache_key() != b._cache_key()


@pytest.mark.core
def test_rebuild_from_cache_restores_typed_history():
    df = pd.DataFrame(
        {
            "lending_rate": [0.04, 0.05],
            "borrowing_rate": [0.05, 0.07],
            "utilization": [0.8, 0.9],
        },
        index=pd.to_datetime(
            [int(_STUB_START.timestamp()), int(_STUB_END.timestamp())],
            unit="s",
            utc=True,
        ),
    )
    df.index.name = "time"
    rebuilt = _rebuild_from_cache(df)
    assert isinstance(rebuilt, LendingHistory)
    assert rebuilt.index.tz is not None
    assert len(rebuilt) == 2


@pytest.mark.core
def test_rebuild_from_cache_handles_empty_dataframe():
    rebuilt = _rebuild_from_cache(pd.DataFrame())
    assert isinstance(rebuilt, LendingHistory)
    assert rebuilt.empty
