"""Real-API tests for the Lido stETH staking-rate loader."""
from datetime import datetime, timedelta, timezone

import pytest

from fractal.loaders import LoaderType, RateHistory, StETHLoader

UTC = timezone.utc


@pytest.mark.integration
@pytest.mark.slow
def test_steth_loader(THE_GRAPH_API_KEY: str):
    loader = StETHLoader(api_key=THE_GRAPH_API_KEY, loader_type=LoaderType.CSV)
    data: RateHistory = loader.read(with_run=True)
    assert isinstance(data, RateHistory)
    assert len(data) > 0
    assert data["rate"].dtype == "float64"
    assert data.index.tz is not None


@pytest.mark.integration
def test_steth_loader_with_window(THE_GRAPH_API_KEY: str):
    end = datetime(2025, 2, 1, tzinfo=UTC)
    start = end - timedelta(days=14)
    loader = StETHLoader(
        api_key=THE_GRAPH_API_KEY, start_time=start, end_time=end,
    )
    data: RateHistory = loader.read(with_run=True)
    assert len(data) > 0
    assert data.index.min() >= start
    assert data.index.max() <= end


@pytest.mark.integration
def test_steth_loader_empty_window_returns_empty(THE_GRAPH_API_KEY: str):
    """Far-past window before Lido inception → empty RateHistory."""
    far_past_start = datetime(2000, 1, 1, tzinfo=UTC)
    far_past_end = datetime(2000, 1, 2, tzinfo=UTC)
    loader = StETHLoader(
        api_key=THE_GRAPH_API_KEY,
        start_time=far_past_start, end_time=far_past_end,
    )
    data = loader.read(with_run=True)
    assert isinstance(data, RateHistory)
    assert len(data) == 0
    assert list(data.columns) == ["rate"]


@pytest.mark.integration
def test_steth_loader_rejects_missing_api_key():
    from fractal.loaders.thegraph.base_graph_loader import GraphLoaderException
    with pytest.raises(GraphLoaderException):
        StETHLoader(api_key="", loader_type=LoaderType.CSV)
