from datetime import datetime
from typing import Dict

from fractal.core.base.entity import GlobalState


class Observation:
    """
    Observation is a snapshot of the entities states at a given timestamp.
    """
    def __init__(self, timestamp: datetime, states: Dict[str, GlobalState]):
        """
        Args:
            timestamp (datetime): Timestamp of the observation.
            states (Dict[str, GlobalState]): States of the entities
             per each registered entity in strategy.
             Key is the entity name in strategy entities registry.
        """
        self.timestamp: datetime = timestamp
        self.states: Dict[str, GlobalState] = states
        self.__validate_types()

    def __validate_types(self):
        """
        Check if each field is of the correct type.
        """
        for entity_name, state in self.states.items():
            for field, value in state.__dict__.items():
                if not isinstance(value, type(getattr(self.states[entity_name], field))):
                    raise ValueError(f"Field {field} of entity\
                                     {entity_name} must be of type {type(value)}.")

    def __repr__(self) -> str:
        return f"Observation(timestamp={self.timestamp}, states={self.states})"
