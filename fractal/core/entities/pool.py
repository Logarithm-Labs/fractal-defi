from abc import abstractmethod

from fractal.core.base.entity import BaseEntity


class BasePoolEntity(BaseEntity):
    """
    Base class for Pool entity.

    Pool entity is responsible for managing the position in the pool.
    It can open and close the position in the pool.
    """
    @abstractmethod
    def action_open_position(self, amount_in_notional: float, *args, **kwargs):
        """
        Open position in pool.

        Args:
            amount_in_notional (float): amount in notional
        """
        raise NotImplementedError

    @abstractmethod
    def action_close_position(self):
        """
        Close position in pool.
        """
        raise NotImplementedError
