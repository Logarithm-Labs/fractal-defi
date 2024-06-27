from abc import abstractmethod

from fractal.core.base.entity import BaseEntity


class BaseLendingEntity(BaseEntity):
    """
    Base class for Lending entities.

    Lending entities are entities that can lend and borrow on the protocol.
    """
    @abstractmethod
    def action_redeem(self, amount_in_product: float):
        """
        Redeems an amount on the protocol.

        Args:
            amount_in_product (float, optional): The amount to redeem in product value.
        """
        raise NotImplementedError

    @abstractmethod
    def action_borrow(self, amount_in_product: float):
        """
        Borrows an amount on the protocol.

        Args:
            amount_in_product (float, optional): The amount to borrow in product value.
        """
        raise NotImplementedError
