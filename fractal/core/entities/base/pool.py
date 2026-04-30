from abc import abstractmethod
from dataclasses import dataclass

from fractal.core.base.entity import BaseEntity, GlobalState


@dataclass
class BasePoolGlobalState(GlobalState):
    """Common pool snapshot fields shared by V2-style and V3-style LP entities.

    Concrete entity ``GlobalState`` classes (e.g. ``UniswapV2LPGlobalState``,
    ``UniswapV3LPGlobalState``, ``SimplePoolGlobalState``) inherit from this
    so strategies can rely on the same shape regardless of pool variant.

    Attributes:
        tvl: Total value locked in the pool, in notional units.
        volume: Trading volume during the previous bar.
        fees: Fees collected by the pool during the previous bar.
        liquidity: Total LP tokens / liquidity outstanding (units depend
            on the protocol — LP-token count for V2, ``L`` parameter for V3).
        price: Pool spot price (entity-specific convention — see the
            concrete subclass docstring).
    """
    tvl: float = 0.0
    volume: float = 0.0
    fees: float = 0.0
    liquidity: float = 0.0
    price: float = 0.0


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
