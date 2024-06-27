from abc import abstractmethod

from fractal.core.base.entity import BaseEntity


class BaseSpotEntity(BaseEntity):
    """
    Base class for Spot entity.

    Spot entity is responsible for managing the position in the spot market.
    We can buy and sell the product in the spot market.
    """

    @abstractmethod
    def action_buy(self, amount_in_notional: float):
        """
        Executes a buy action on the protocol.

        Args:
            amount_in_notional (float, optional): The amount to buy in notional value.

        Raises:
            EntityException: If there is not enough cash to buy.
        """
        raise NotImplementedError

    @abstractmethod
    def action_sell(self, amount_in_product: float):
        """
        Executes a sell action on the protocol.

        Args:
            amount_in_product (float, optional): The amount to sell in product value.

        Raises:
            EntityException: If there is not enough product to sell.
        """
        raise NotImplementedError
