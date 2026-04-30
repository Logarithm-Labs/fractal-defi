"""Public entity API.

Three logical levels:

* :mod:`fractal.core.entities.base` — abstract interfaces and shared
  state shapes (``BasePerpEntity``, ``BaseSpotInternalState``, …).
* :mod:`fractal.core.entities.simple` — minimal generic implementations
  (``SimplePerpEntity``, ``SimpleSpotExchange``, ``SimplePoolEntity``,
  ``SimpleLendingEntity``). Use for tests, examples, prototypes.
* :mod:`fractal.core.entities.protocols` — protocol-specific
  implementations (``HyperliquidEntity``, ``AaveEntity``,
  ``UniswapV2LPEntity``, ``UniswapV3LPEntity``, ``UniswapV3SpotEntity``,
  ``StakedETHEntity``).
"""
# Bases
from fractal.core.entities.base import (BaseHedgeEntity,  # deprecated alias
                                        BaseLendingEntity,
                                        BaseLiquidStakingToken,
                                        BasePerpEntity, BasePerpInternalState,
                                        BasePoolEntity, BaseSpotEntity,
                                        BaseSpotInternalState)
# Protocols
from fractal.core.entities.protocols import (AaveEntity, AaveGlobalState,
                                             HyperliquidEntity,
                                             HyperLiquidGlobalState,
                                             HyperLiquidInternalState,
                                             StakedETHEntity,
                                             StakedETHGlobalState,
                                             UniswapV2LPConfig,
                                             UniswapV2LPEntity,
                                             UniswapV2LPGlobalState,
                                             UniswapV3LPConfig,
                                             UniswapV3LPEntity,
                                             UniswapV3LPGlobalState,
                                             UniswapV3SpotEntity,
                                             UniswapV3SpotGlobalState)
# Simple
from fractal.core.entities.simple import (SimpleLendingEntity,
                                          SimpleLendingGlobalState,
                                          SimpleLendingInternalState,
                                          SimpleLiquidStakingToken,
                                          SimpleLiquidStakingTokenGlobalState,
                                          SimpleLiquidStakingTokenInternalState,
                                          SimplePerpEntity,
                                          SimplePerpGlobalState,
                                          SimplePerpInternalState,
                                          SimplePoolEntity,
                                          SimplePoolGlobalState,
                                          SimplePoolInternalState,
                                          SimpleSpotExchange,
                                          SimpleSpotExchangeGlobalState,
                                          SimpleSpotExchangeInternalState)
# Deprecated alias (top-level back-compat)
from fractal.core.entities.single_spot_exchange import (
    SingleSpotExchange, SingleSpotExchangeGlobalState)

__all__ = [
    # base
    "BasePerpEntity", "BasePerpInternalState",
    "BaseLendingEntity",
    "BaseLiquidStakingToken",
    "BasePoolEntity",
    "BaseSpotEntity", "BaseSpotInternalState",
    "BaseHedgeEntity",  # deprecated alias of BasePerpEntity
    # simple
    "SimplePerpEntity", "SimplePerpGlobalState", "SimplePerpInternalState",
    "SimpleSpotExchange", "SimpleSpotExchangeGlobalState", "SimpleSpotExchangeInternalState",
    "SimplePoolEntity", "SimplePoolGlobalState", "SimplePoolInternalState",
    "SimpleLendingEntity", "SimpleLendingGlobalState", "SimpleLendingInternalState",
    "SimpleLiquidStakingToken", "SimpleLiquidStakingTokenGlobalState",
    "SimpleLiquidStakingTokenInternalState",
    # protocols
    "AaveEntity", "AaveGlobalState",
    "HyperliquidEntity", "HyperLiquidGlobalState", "HyperLiquidInternalState",
    "StakedETHEntity", "StakedETHGlobalState",
    "UniswapV2LPConfig", "UniswapV2LPEntity", "UniswapV2LPGlobalState",
    "UniswapV3LPConfig", "UniswapV3LPEntity", "UniswapV3LPGlobalState",
    "UniswapV3SpotEntity", "UniswapV3SpotGlobalState",
    # deprecated aliases
    "SingleSpotExchange", "SingleSpotExchangeGlobalState",
]
