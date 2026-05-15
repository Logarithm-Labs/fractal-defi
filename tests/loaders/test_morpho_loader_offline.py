"""Offline tests for ``MorphoMarketLoader.transform()`` — no network.

Verify the GraphQL parallel-series merge, the empty-payload short-circuit
and the typed-struct return contract.
"""
from datetime import datetime, timezone

import pandas as pd
import pytest

from fractal.loaders.morpho import MorphoMarketLoader
from fractal.loaders.structs import MorphoMarketHistory

UTC = timezone.utc

_STUB_MARKET = "0x" + "ab" * 32
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
def test_transform_returns_morpho_market_history():
    loader = _stub_loader()
    # Staged frame as if `extract` had merged the three series.
    loader._data = pd.DataFrame(
        {
            "x": [int(_STUB_START.timestamp()), int(_STUB_END.timestamp())],
            "borrowing_rate": [0.05, 0.07],
            "utilization": [0.8, 0.9],
            "supply_apy": [0.04, 0.06],
        }
    )
    loader.transform()
    assert isinstance(loader._data, MorphoMarketHistory)
    assert list(loader._data.columns) == ["borrowing_rate", "utilization", "supply_apy"]
    assert len(loader._data) == 2


@pytest.mark.core
def test_transform_handles_empty_staging():
    loader = _stub_loader()
    loader._data = pd.DataFrame()
    loader.transform()
    assert isinstance(loader._data, MorphoMarketHistory)
    assert loader._data.empty


@pytest.mark.core
def test_chain_id_routes_to_correct_network():
    eth = _stub_loader("ethereum")
    arb = _stub_loader("arbitrum")
    base = _stub_loader("base")
    assert eth._chain_id == 1
    assert arb._chain_id == 42161
    assert base._chain_id == 8453
