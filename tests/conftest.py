import os

import pytest
from core.hodler import Hodler, HodlerParams, HodlerStrategy


@pytest.fixture(scope='module', autouse=True)
def hodler() -> Hodler:
    return Hodler()

@pytest.fixture(scope='module', autouse=True)
def hodler_strategy() -> HodlerStrategy:
    strategy = HodlerStrategy(debug=True, params=HodlerParams())
    return strategy

@pytest.fixture(scope='module', autouse=True)
def THE_GRAPH_API_KEY() -> str:
    api_key = os.getenv('THE_GRAPH_API_KEY')
    if not api_key:
        raise ValueError('THE_GRAPH_API_KEY environment variable is not set')
    return api_key
