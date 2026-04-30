"""Minimal generic entity implementations.

These classes capture the essential mechanics of each entity category
without protocol-specific quirks. Use them for tests, examples, quick
prototypes, and as the simplest concrete entities to learn the
framework. For production runs against a specific DeFi protocol, use
the corresponding class from :mod:`fractal.core.entities.protocols`.
"""
from fractal.core.entities.simple.lending import (SimpleLendingEntity,
                                                  SimpleLendingGlobalState,
                                                  SimpleLendingInternalState)
from fractal.core.entities.simple.liquid_staking import (
    SimpleLiquidStakingToken, SimpleLiquidStakingTokenGlobalState,
    SimpleLiquidStakingTokenInternalState)
from fractal.core.entities.simple.perp import (SimplePerpEntity,
                                               SimplePerpGlobalState,
                                               SimplePerpInternalState)
from fractal.core.entities.simple.pool import (SimplePoolEntity,
                                               SimplePoolGlobalState,
                                               SimplePoolInternalState)
from fractal.core.entities.simple.spot import (SimpleSpotExchange,
                                               SimpleSpotExchangeGlobalState,
                                               SimpleSpotExchangeInternalState)

__all__ = [
    "SimpleLendingEntity", "SimpleLendingGlobalState", "SimpleLendingInternalState",
    "SimpleLiquidStakingToken", "SimpleLiquidStakingTokenGlobalState",
    "SimpleLiquidStakingTokenInternalState",
    "SimplePerpEntity", "SimplePerpGlobalState", "SimplePerpInternalState",
    "SimplePoolEntity", "SimplePoolGlobalState", "SimplePoolInternalState",
    "SimpleSpotExchange", "SimpleSpotExchangeGlobalState", "SimpleSpotExchangeInternalState",
]
