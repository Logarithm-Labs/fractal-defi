from dataclasses import dataclass
from datetime import datetime

import pytest

from fractal.core.base.entity import GlobalState
from fractal.core.base.observations import (Observation,
                                            SQLiteObservationsStorage)


@dataclass
class SomeState(GlobalState):
    price: float
    volume: float


@pytest.fixture
def observation():
    return Observation(
        timestamp=datetime.now(),
        states={
            'exchange':  SomeState(price=100.0, volume=1000.0),
        }
    )


@pytest.fixture
def storage():
    return SQLiteObservationsStorage('')


@pytest.mark.skip(reason="This function is for live testing only")
def test_read_all(storage):
    data = storage.read()
    assert len(data) > 0
    print(data)


@pytest.mark.core
def test_write_read(storage, observation):
    storage.write(observation)
    observations = storage.read()
    assert len(observations) == 1
    assert observations[0] == observation
