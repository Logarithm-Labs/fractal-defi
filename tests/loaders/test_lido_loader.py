import pytest

from fractal.loaders import LoaderType, StETHLoader


@pytest.mark.integration
@pytest.mark.slow
def test_steth_loader(THE_GRAPH_API_KEY: str):
    loader = StETHLoader(
        api_key=THE_GRAPH_API_KEY,
        loader_type=LoaderType.CSV,
    )
    data = loader.read(with_run=True)
    assert data is not None
    assert len(data) > 0
