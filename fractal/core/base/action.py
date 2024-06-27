from dataclasses import dataclass, field
from typing import Dict


@dataclass
class Action:
    """
    Action to be executed by the simulation engine.
    It contains the action name and arguments that matching
    for method name and it's argumetns.

    action: str - action name to execute (method name)
    args: Dict - action arguments (method arguments)

    Example:
    --------
    ```
    action = Action("buy", {"price": 100, "quantity": 10})
    some_entity.execute(action) # it will call some_entity.action_buy(price=100, quantity=10)
    ```
    """
    action: str
    args: Dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"Action({self.action}, {self.args})"
