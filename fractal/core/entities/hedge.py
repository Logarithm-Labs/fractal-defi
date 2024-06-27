from abc import abstractmethod

from fractal.core.base.entity import BaseEntity


class BaseHedgeEntity(BaseEntity):
    """
    Base class for Hedge entities.

    Hedge entities are entities that can open positions on the exchange.
    """
    @abstractmethod
    def action_open_position(self, amount_in_product: float):
        """
        Opens a position on the protocol.

        Args:
            amount_in_product (float, optional): The amount to open in product value.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def leverage(self) -> float:
        """
        Returns the leverage of the entity.

        Returns:
            float: The leverage of the entity.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def size(self) -> float:
        """
        Returns the size of the entity.

        Returns:
            float: The size of the entity.
        """
        raise NotImplementedError
