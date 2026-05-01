"""Exhaustive public-API import smoke check.

Runs against an INSTALLED ``fractal-defi`` package (wheel or sdist),
not the editable repo. Any import that fails here means the published
wheel is missing a class, dropped a re-export, or has a runtime-only
import error that ``import fractal`` would otherwise catch silently.

Run via the orchestrator:

    bash scripts/smoke/run.sh

Or standalone in a venv that already has ``fractal-defi`` installed:

    python scripts/smoke/imports.py
"""
from __future__ import annotations

import sys


def section(name: str) -> None:
    print(f"\n[{name}]")


def ok(label: str) -> None:
    print(f"  ✓ {label}")


def main() -> int:
    section("top-level package")
    import fractal  # noqa: F401
    ok(f"fractal — {getattr(fractal, '__file__', '<no __file__>')}")

    section("core.base contracts")
    from fractal.core.base import (  # noqa: F401
        Action,
        ActionToTake,
        BaseEntity,
        BaseStrategy,
        BaseStrategyParams,
        EntityException,
        GlobalState,
        InternalState,
        NamedEntity,
        Observation,
        ObservationsStorage,
    )
    from fractal.core.base.observations import (  # noqa: F401
        SQLiteObservationsStorage,
    )
    from fractal.core.base.strategy import (  # noqa: F401
        StrategyMetrics,
        StrategyResult,
    )
    ok("BaseStrategy / BaseEntity / Action / Observation / state types")

    section("entities — base contracts")
    from fractal.core.entities import (  # noqa: F401
        BaseLendingEntity,
        BaseLiquidStakingToken,
        BasePerpEntity,
        BasePerpInternalState,
        BasePoolEntity,
        BaseSpotEntity,
        BaseSpotInternalState,
    )
    ok("BasePerp / BaseLending / BasePool / BaseSpot / BaseLiquidStaking")

    section("entities — protocol implementations")
    from fractal.core.entities import (  # noqa: F401
        AaveEntity,
        AaveGlobalState,
        HyperliquidEntity,
        HyperliquidGlobalState,
        HyperliquidInternalState,
        HyperliquidPosition,
        StakedETHEntity,
        StakedETHGlobalState,
        UniswapV2LPConfig,
        UniswapV2LPEntity,
        UniswapV2LPGlobalState,
        UniswapV3LPConfig,
        UniswapV3LPEntity,
        UniswapV3LPGlobalState,
        UniswapV3SpotEntity,
        UniswapV3SpotGlobalState,
    )
    ok("Aave / Hyperliquid / Uniswap V2/V3 LP+spot / stETH")

    section("entities — back-compat aliases")
    from fractal.core.entities import (  # noqa: F401
        HyperLiquidGlobalState,
        HyperLiquidInternalState,
        HyperLiquidPosition,
    )
    ok("Pre-1.3.0 HyperLiquid* aliases still resolvable")

    section("entities — simple/generic implementations")
    from fractal.core.entities import (  # noqa: F401
        SimpleLendingEntity,
        SimpleLendingGlobalState,
        SimpleLendingInternalState,
        SimpleLiquidStakingToken,
        SimpleLiquidStakingTokenGlobalState,
        SimplePerpEntity,
        SimplePerpGlobalState,
        SimplePoolEntity,
        SimplePoolGlobalState,
        SimpleSpotExchange,
        SimpleSpotExchangeGlobalState,
        SimpleSpotExchangeInternalState,
    )
    ok("SimpleLending / SimplePerp / SimplePool / SimpleSpot / SimpleLST")

    section("loaders — base + types")
    from fractal.loaders import (  # noqa: F401
        FundingHistory,
        KlinesHistory,
        LendingHistory,
        Loader,
        LoaderType,
        PoolHistory,
        PriceHistory,
        RateHistory,
        TrajectoryBundle,
    )
    ok("Loader ABC + typed return structs")

    section("loaders — concrete data sources")
    from fractal.loaders import (  # noqa: F401
        AaveV2EthereumLoader,
        AaveV3ArbitrumLoader,
        AaveV3EthereumLoader,
        AaveV3RatesLoader,
        BinanceDayPriceLoader,
        BinanceFundingLoader,
        BinanceHourPriceLoader,
        BinanceKlinesLoader,
        BinancePriceLoader,
        ConstantFundingsLoader,
        GMXV1FundingLoader,
        HyperliquidFundingRatesLoader,
        HyperliquidPerpsKlinesLoader,
        HyperliquidPerpsPricesLoader,
        MonteCarloHourPriceLoader,
        MonteCarloPriceLoader,
        UniswapV3ArbitrumPoolDayDataLoader,
        UniswapV3ArbitrumPoolHourDataLoader,
        UniswapV3ArbitrumPricesLoader,
        UniswapV3EthereumPoolDayDataLoader,
        UniswapV3EthereumPoolHourDataLoader,
        UniswapV3EthereumPoolMinuteDataLoader,
        UniswapV3EthereumPricesLoader,
    )
    ok("Aave / Binance / GMX / Hyperliquid / TheGraph / MonteCarlo")

    section("loaders — back-compat alias")
    from fractal.loaders import HyperLiquidPerpsPricesLoader  # noqa: F401
    ok("Pre-1.3.0 HyperLiquidPerpsPricesLoader alias still resolvable")

    section("strategies")
    from fractal.strategies.basis_trading_strategy import (  # noqa: F401
        BasisTradingStrategy,
        BasisTradingStrategyHyperparams,
    )
    from fractal.strategies.hyperliquid_basis import (  # noqa: F401
        HyperliquidBasis,
        HyperliquidBasisParams,
    )
    from fractal.strategies.tau_reset_strategy import (  # noqa: F401
        TauResetParams,
        TauResetStrategy,
    )
    ok("BasisTrading / HyperliquidBasis / TauReset")

    section("pipeline")
    from fractal.core.pipeline import (  # noqa: F401
        DefaultPipeline,
        ExperimentConfig,
        MLflowConfig,
        Pipeline,
    )
    from fractal.core.pipeline import MLFlowConfig  # noqa: F401
    ok("DefaultPipeline / MLflowConfig / Pipeline + back-compat MLFlowConfig")

    print("\nAll public-API imports resolved against the installed wheel.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
