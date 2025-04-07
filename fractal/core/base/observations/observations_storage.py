from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, Sequence

from fractal.core.base.observations.observation import Observation


class ObservationsStorage(ABC):
    """
    Observation Storage Interface.
    """

    @abstractmethod
    def write(self, observation: Observation):
        raise NotImplementedError

    @abstractmethod
    def read(self, start_time: Optional[datetime] = None,
             end_time: Optional[datetime] = None) -> Sequence[Observation]:
        raise NotImplementedError
