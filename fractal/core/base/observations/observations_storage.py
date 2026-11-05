from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import datetime

from fractal.core.base.observations.observation import Observation


class ObservationsStorage(ABC):
    """
    Observation Storage Interface.
    """

    @abstractmethod
    def write(self, observation: Observation):
        raise NotImplementedError

    @abstractmethod
    def read(self, start_time: datetime | None = None,
             end_time: datetime | None = None) -> Sequence[Observation]:
        raise NotImplementedError
