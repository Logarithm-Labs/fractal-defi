"""Protocol-specific entity implementations.

These classes encode real-world DeFi protocol mechanics (Hyperliquid
funding model, GMX V2 two-sided rates, Aave LTV/liquidation thresholds,
Uniswap V2/V3 reserve math, Lido staking rewards). Use them to backtest
strategies against actual protocol behaviour.
"""
from fractal.core.entities.protocols.aave import AaveEntity, AaveGlobalState
from fractal.core.entities.protocols.gmx_v2 import (GMXV2Entity,
                                                    GMXV2GlobalState,
                                                    GMXV2Position)
from fractal.core.entities.protocols.hyperliquid import (HyperliquidEntity,
                                                         HyperLiquidGlobalState,
                                                         HyperLiquidInternalState)
from fractal.core.entities.protocols.steth import (StakedETHEntity,
                                                   StakedETHGlobalState)
from fractal.core.entities.protocols.uniswap_v2_lp import (UniswapV2LPConfig,
                                                           UniswapV2LPEntity,
                                                           UniswapV2LPGlobalState)
from fractal.core.entities.protocols.uniswap_v3_lp import (UniswapV3LPConfig,
                                                           UniswapV3LPEntity,
                                                           UniswapV3LPGlobalState)
from fractal.core.entities.protocols.uniswap_v3_spot import (
    UniswapV3SpotEntity, UniswapV3SpotGlobalState)

__all__ = [
    "AaveEntity", "AaveGlobalState",
    "GMXV2Entity", "GMXV2GlobalState", "GMXV2Position",
    "HyperliquidEntity", "HyperLiquidGlobalState", "HyperLiquidInternalState",
    "StakedETHEntity", "StakedETHGlobalState",
    "UniswapV2LPConfig", "UniswapV2LPEntity", "UniswapV2LPGlobalState",
    "UniswapV3LPConfig", "UniswapV3LPEntity", "UniswapV3LPGlobalState",
    "UniswapV3SpotEntity", "UniswapV3SpotGlobalState",
]
