"""Base class for liquid-staking-token (LST) entities.

A liquid-staking token is a spot-traded asset whose underlying balance
**rebases** automatically: on each :meth:`update_state` the held quantity
grows (or shrinks, in slashing scenarios) by a per-step staking rate.

LSTs inherit the full :class:`BaseSpotEntity` contract — buy/sell with
notional/product semantics, ``amount`` + ``cash`` internal state,
``current_price`` — and add :attr:`staking_rate` so strategies can
inspect the current accrual rate polymorphically.
"""
from abc import abstractmethod

from fractal.core.entities.base.spot import BaseSpotEntity


class BaseLiquidStakingToken(BaseSpotEntity):
    """Spot-tradeable token with an auto-accruing underlying balance."""

    @property
    @abstractmethod
    def staking_rate(self) -> float:
        """Per-step staking accrual rate.

        Used by ``update_state`` (or its concrete equivalent) to rebase
        ``internal_state.amount`` upward. Strategies can read this
        property to compare yield across LSTs without knowing the
        underlying global-state field name.
        """
        raise NotImplementedError
