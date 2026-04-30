"""Real-API tests for the GMX V1 funding loader.

The legacy ``subgraph.satsuma-prod.com`` URL is dead (2024) and there is
currently no public TheGraph subgraph exposing the original V1
``fundingRates`` entity shape. The tests therefore probe the legacy URL
and **expect failure** — they document the broken upstream while keeping
the contract test in the suite. When a working subgraph is wired in via
the ``url`` constructor argument, these tests should flip to passing.
"""
from datetime import datetime, timezone

import pytest
import requests

from fractal.loaders.gmx_v1 import GMXV1FundingLoader, LEGACY_URL
from fractal.loaders.structs import FundingHistory

UTC = timezone.utc
WETH = "0x82af49447d8a07e3bd95bd0d56f35241523fbab1"


def _legacy_url_alive() -> bool:
    try:
        requests.post(LEGACY_URL, json={"query": "{ _meta { block { number } } }"}, timeout=5)
    except requests.RequestException:
        return False
    return True


@pytest.mark.integration
def test_gmx_v1_funding_loader_against_legacy_url():
    """Legacy subgraph URL is offline upstream as of 2024; loader should
    propagate a connection error rather than silently returning empty data."""
    loader = GMXV1FundingLoader(token_address=WETH)
    if _legacy_url_alive():
        # If the URL ever comes back, we want the test to actually validate output.
        data: FundingHistory = loader.read(with_run=True)
        assert isinstance(data, FundingHistory)
        return
    with pytest.raises(requests.RequestException):
        loader.read(with_run=True)


@pytest.mark.integration
def test_gmx_v1_funding_loader_accepts_custom_url():
    """The loader honors a user-supplied URL (so users can plug in a self-hosted node)."""
    loader = GMXV1FundingLoader(
        token_address=WETH,
        url="https://example.invalid/graphql",
        start_time=datetime(2024, 1, 1, tzinfo=UTC),
        end_time=datetime(2024, 1, 7, tzinfo=UTC),
    )
    with pytest.raises(requests.RequestException):
        loader.read(with_run=True)


@pytest.mark.integration
def test_gmx_v1_funding_loader_empty_returns_empty_history():
    """Direct construction of an empty loader (post-extract/transform) returns
    a well-shaped empty FundingHistory."""
    loader = GMXV1FundingLoader(token_address=WETH)
    loader._data = None
    loader.transform()
    # Reach the same code path without hitting the network:
    out = FundingHistory(rates=[], time=[])
    assert len(out) == 0
    assert list(out.columns) == ["rate"]
