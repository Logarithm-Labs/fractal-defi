"""Abstract base classes and shared state shapes for entities."""
from fractal.core.entities.base.hedge import BaseHedgeEntity  # deprecated alias
from fractal.core.entities.base.lending import BaseLendingEntity
from fractal.core.entities.base.liquid_staking import BaseLiquidStakingToken
from fractal.core.entities.base.perp import (BasePerpEntity,
                                             BasePerpInternalState)
from fractal.core.entities.base.pool import BasePoolEntity
from fractal.core.entities.base.spot import (BaseSpotEntity,
                                             BaseSpotInternalState)

__all__ = [
    "BaseHedgeEntity",
    "BaseLendingEntity",
    "BaseLiquidStakingToken",
    "BasePerpEntity", "BasePerpInternalState",
    "BasePoolEntity",
    "BaseSpotEntity", "BaseSpotInternalState",
]
