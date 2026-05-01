from abc import abstractmethod

from fractal.core.base.entity import BaseEntity


class BaseLendingEntity(BaseEntity):
    """
    Base class for Lending entities.

    Lending entities can borrow and repay debt on the protocol. Collateral
    deposit/withdraw goes through the inherited
    :meth:`BaseEntity.action_deposit` / :meth:`BaseEntity.action_withdraw`.
    """
    @abstractmethod
    def action_repay(self, amount_in_product: float):
        """Repay borrowed product on the protocol.

        Args:
            amount_in_product: Amount of product debt to repay.
        """
        raise NotImplementedError

    @abstractmethod
    def action_borrow(self, amount_in_product: float):
        """Borrow product against the deposited collateral.

        Args:
            amount_in_product: Amount of product to borrow.
        """
        raise NotImplementedError
