"""Lock-in: config attributes must be set BEFORE ``super().__init__()``.

``BaseEntity.__init__`` calls ``self._initialize_states()`` as a hook —
so a subclass that overrides ``_initialize_states`` and reads
``self.<config>`` inside it would crash with ``AttributeError`` if the
parent set config AFTER ``super`` returned.

This file verifies each of the 7 protocol entities sets its config
BEFORE delegating to ``super``. Each test defines a subclass whose
``_initialize_states`` reads a known config attribute; constructing
the subclass must not raise.
"""
import pytest

from fractal.core.entities.protocols.aave import (AaveEntity, AaveGlobalState,
                                                  AaveInternalState)
from fractal.core.entities.protocols.hyperliquid import (HyperliquidEntity,
                                                         HyperLiquidGlobalState,
                                                         HyperLiquidInternalState)
from fractal.core.entities.protocols.steth import (StakedETHEntity,
                                                   StakedETHGlobalState,
                                                   StakedETHInternalState)
from fractal.core.entities.protocols.uniswap_v2_lp import (UniswapV2LPConfig,
                                                            UniswapV2LPEntity,
                                                            UniswapV2LPGlobalState,
                                                            UniswapV2LPInternalState)
from fractal.core.entities.protocols.uniswap_v3_lp import (UniswapV3LPConfig,
                                                            UniswapV3LPEntity,
                                                            UniswapV3LPGlobalState,
                                                            UniswapV3LPInternalState)
from fractal.core.entities.protocols.uniswap_v3_spot import (UniswapV3SpotEntity,
                                                              UniswapV3SpotGlobalState,
                                                              UniswapV3SpotInternalState)


@pytest.mark.core
def test_aave_config_available_in_initialize_states():
    captured = {}

    class _Sub(AaveEntity):
        def _initialize_states(self):
            captured["max_ltv"] = self.max_ltv
            captured["liq_thr"] = self.liq_thr
            self._internal_state = AaveInternalState()
            self._global_state = AaveGlobalState()

    _Sub(max_ltv=0.7, liq_thr=0.9)
    assert captured == {"max_ltv": 0.7, "liq_thr": 0.9}


@pytest.mark.core
def test_hyperliquid_config_available_in_initialize_states():
    captured = {}

    class _Sub(HyperliquidEntity):
        def _initialize_states(self):
            captured["TRADING_FEE"] = self.TRADING_FEE
            captured["MAX_LEVERAGE"] = self.MAX_LEVERAGE
            self._internal_state = HyperLiquidInternalState()
            self._global_state = HyperLiquidGlobalState()

    _Sub(trading_fee=0.001, max_leverage=20)
    assert captured == {"TRADING_FEE": 0.001, "MAX_LEVERAGE": 20}


@pytest.mark.core
def test_steth_config_available_in_initialize_states():
    captured = {}

    class _Sub(StakedETHEntity):
        def _initialize_states(self):
            captured["trading_fee"] = self.trading_fee
            self._internal_state = StakedETHInternalState()
            self._global_state = StakedETHGlobalState()

    _Sub(trading_fee=0.005)
    assert captured == {"trading_fee": 0.005}


@pytest.mark.core
def test_uniswap_v2_config_available_in_initialize_states():
    captured = {}

    class _Sub(UniswapV2LPEntity):
        def _initialize_states(self):
            captured["pool_fee_rate"] = self.pool_fee_rate
            captured["slippage_pct"] = self.slippage_pct
            captured["token0_decimals"] = self.token0_decimals
            captured["token1_decimals"] = self.token1_decimals
            self._internal_state = UniswapV2LPInternalState()
            self._global_state = UniswapV2LPGlobalState()

    cfg = UniswapV2LPConfig(pool_fee_rate=0.005, slippage_pct=0.001,
                            token0_decimals=6, token1_decimals=18)
    _Sub(cfg)
    assert captured == {
        "pool_fee_rate": 0.005,
        "slippage_pct": 0.001,
        "token0_decimals": 6,
        "token1_decimals": 18,
    }


@pytest.mark.core
def test_uniswap_v3_lp_config_available_in_initialize_states():
    captured = {}

    class _Sub(UniswapV3LPEntity):
        def _initialize_states(self):
            captured["pool_fee_rate"] = self.pool_fee_rate
            captured["slippage_pct"] = self.slippage_pct
            captured["token0_decimals"] = self.token0_decimals
            captured["token1_decimals"] = self.token1_decimals
            self._internal_state = UniswapV3LPInternalState()
            self._global_state = UniswapV3LPGlobalState()

    cfg = UniswapV3LPConfig(pool_fee_rate=0.005, slippage_pct=0.001,
                            token0_decimals=6, token1_decimals=18)
    _Sub(cfg)
    assert captured == {
        "pool_fee_rate": 0.005,
        "slippage_pct": 0.001,
        "token0_decimals": 6,
        "token1_decimals": 18,
    }


@pytest.mark.core
def test_uniswap_v3_spot_config_available_in_initialize_states():
    captured = {}

    class _Sub(UniswapV3SpotEntity):
        def _initialize_states(self):
            captured["trading_fee"] = self.trading_fee
            self._internal_state = UniswapV3SpotInternalState()
            self._global_state = UniswapV3SpotGlobalState()

    _Sub(trading_fee=0.005)
    assert captured == {"trading_fee": 0.005}
