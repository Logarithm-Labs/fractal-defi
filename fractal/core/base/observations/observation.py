import json
from datetime import datetime
from typing import Dict

from fractal.core.base.entity import GlobalState


class Observation:
    """A snapshot of all entity global-states at a single timestamp.

    Validation is intentionally cheap: it confirms the structural
    contract (non-empty dict, string keys, ``GlobalState`` values) but
    does **not** introspect dataclass fields. Domain-specific validation
    (e.g. positive prices, valid funding rates) belongs to each
    entity's :meth:`update_state`.
    """
    def __init__(self, timestamp: datetime, states: Dict[str, GlobalState]):
        """
        Args:
            timestamp: Timestamp of the observation.
            states: Mapping ``entity_name -> GlobalState`` for every
                entity that has a state at this step.
        """
        self.timestamp: datetime = timestamp
        self.states: Dict[str, GlobalState] = states
        self.__validate()

    def __validate(self) -> None:
        """Cheap structural check.

        Catches the realistic mistakes:
          * empty observation (almost certainly a bug);
          * non-string entity keys;
          * a value that is not a ``GlobalState`` (e.g. raw dict, str).
        """
        if not isinstance(self.states, dict) or not self.states:
            raise ValueError(
                "Observation.states must be a non-empty Dict[str, GlobalState]"
            )
        for name, state in self.states.items():
            if not isinstance(name, str) or not name:
                raise ValueError(
                    f"entity name must be a non-empty str, got {name!r}"
                )
            if not isinstance(state, GlobalState):
                raise ValueError(
                    f"state for {name!r} must be a GlobalState instance, "
                    f"got {type(state).__name__}"
                )

    def __repr__(self) -> str:
        return f"Observation(timestamp={self.timestamp}, states={self.states})"

    def __json__(self):
        data = {entity_name: state.__dict__ for entity_name, state in self.states.items()}
        return json.dumps(data)

    def to_json(self) -> str:
        return self.__json__()

    def __hash__(self):
        # Include timestamp so two snapshots at different points in time
        # do not collide in sets/dicts even if their states match.
        return hash((self.timestamp, self.__json__()))

    def __eq__(self, other):
        if not isinstance(other, Observation):
            return NotImplemented
        return self.timestamp == other.timestamp and self.__json__() == other.__json__()
