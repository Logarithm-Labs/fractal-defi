import pytest
from hodler import Hodler, HodlerParams, HodlerStrategy


@pytest.fixture(scope='module', autouse=True)
def hodler() -> Hodler:
    return Hodler()

@pytest.fixture(scope='module', autouse=True)
def hodler_strategy() -> HodlerStrategy:
    strategy = HodlerStrategy(debug=True, params=HodlerParams())
    return strategy
