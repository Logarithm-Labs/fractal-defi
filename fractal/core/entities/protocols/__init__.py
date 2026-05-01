"""Protocol-specific entity implementations.

These classes encode real-world DeFi protocol mechanics (Hyperliquid
funding model, Aave LTV/liquidation thresholds, Uniswap V2/V3 reserve
math, Lido staking rewards). Use them to backtest strategies against
actual protocol behaviour.
"""
from fractal.core.entities.protocols.aave import AaveEntity, AaveGlobalState
from fractal.core.entities.protocols.hyperliquid import (  # Pre-1.3.0 aliases — re-exported for back-compat.
    HyperliquidEntity,
    HyperliquidGlobalState,
    HyperLiquidGlobalState,
    HyperliquidInternalState,
    HyperLiquidInternalState,
    HyperliquidPosition,
    HyperLiquidPosition,
)
from fractal.core.entities.protocols.steth import StakedETHEntity, StakedETHGlobalState
from fractal.core.entities.protocols.uniswap_v2_lp import UniswapV2LPConfig, UniswapV2LPEntity, UniswapV2LPGlobalState
from fractal.core.entities.protocols.uniswap_v3_lp import UniswapV3LPConfig, UniswapV3LPEntity, UniswapV3LPGlobalState
from fractal.core.entities.protocols.uniswap_v3_spot import UniswapV3SpotEntity, UniswapV3SpotGlobalState

__all__ = [
    "AaveEntity", "AaveGlobalState",
    "HyperliquidEntity",
    "HyperliquidGlobalState", "HyperliquidInternalState", "HyperliquidPosition",
    # Pre-1.3.0 aliases (deprecated; will be removed in a future major release).
    "HyperLiquidGlobalState", "HyperLiquidInternalState", "HyperLiquidPosition",
    "StakedETHEntity", "StakedETHGlobalState",
    "UniswapV2LPConfig", "UniswapV2LPEntity", "UniswapV2LPGlobalState",
    "UniswapV3LPConfig", "UniswapV3LPEntity", "UniswapV3LPGlobalState",
    "UniswapV3SpotEntity", "UniswapV3SpotGlobalState",
]
