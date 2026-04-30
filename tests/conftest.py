import os

import pytest
from core.hodler import Hodler, HodlerParams, HodlerStrategy


@pytest.fixture(scope='module', autouse=True)
def hodler() -> Hodler:
    return Hodler()


@pytest.fixture(scope='function', autouse=True)
def hodler_strategy() -> HodlerStrategy:
    # Function-scoped: each test gets a fresh strategy. Shared scope was
    # making tests order-dependent (mutations from test_register_entity
    # leaked into test_step under STRICT_OBSERVATIONS).
    strategy = HodlerStrategy(debug=True, params=HodlerParams())
    return strategy


@pytest.fixture(scope='module', autouse=False)
def THE_GRAPH_API_KEY() -> str:
    """Skip the test if the required env var is missing instead of erroring.

    Tests that depend on this fixture also carry ``@pytest.mark.integration``;
    the default ``pytest`` invocation excludes them already (see
    ``pytest.ini::addopts``). This fallback covers explicit ``-m
    integration`` runs without secrets configured.
    """
    api_key = os.getenv('THE_GRAPH_API_KEY')
    if not api_key:
        pytest.skip("THE_GRAPH_API_KEY environment variable is not set")
    return api_key
