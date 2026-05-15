# fractal/strategies/panoptic_straddle.py

from collections import deque
from dataclasses import dataclass
from typing import Deque, List

import numpy as np

from fractal.core.base import (
    Action, ActionToTake, BaseStrategy,
    BaseStrategyParams, NamedEntity,
)
from fractal.core.base.entity import (
    BaseEntity, EntityException, GlobalState, InternalState,
)
from fractal.core.entities.models.uniswap_v3_fees import (
    estimate_fee, get_liquidity_delta,
)
from fractal.core.entities.protocols.uniswap_v3_lp import (
    UniswapV3LPConfig, UniswapV3LPEntity,
)


# ── States ────────────────────────────────────────────────────────────────────

@dataclass
class PanopticPoolGlobalState(GlobalState):
    """Pool snapshot passed to PanopticStraddleEntity each bar."""
    price: float = 0.0
    fees: float = 0.0
    liquidity: float = 0.0
    volume: float = 0.0
    tvl: float = 0.0
    iv_annual: float = 0.0

    @property
    def implied_volatility(self) -> float:
        return self.iv_annual


@dataclass
class PanopticStraddleInternalState(InternalState):
    """Internal state of the long straddle position."""
    is_open: bool = False
    entry_price: float = 0.0
    notional: float = 0.0
    collateral: float = 0.0
    accumulated_premium: float = 0.0
    bars_held: int = 0
    cash: float = 0.0


# ── Entity ────────────────────────────────────────────────────────────────────

class PanopticStraddleEntity(BaseEntity):
    """Long straddle on Panoptic over Uniswap V3.

    Implements streaming-premium mechanics from the Panoptic whitepaper
    (Lambert & Kristensen, 2023, §III.A):

        premium_bar = estimate_fee(L_ours, L_pool, pool_fees_bar)

    PnL = intrinsic_value - accumulated_premium
        = |S_t - K|        - sum(premium_bar)
    """

    TOKEN0_DECIMALS: int = 18    # WETH
    TOKEN1_DECIMALS: int = 6     # USDC
    TICK_SPACING_PCT: float = 0.006  # 0.3%-fee pool tick spacing

    def _initialize_states(self) -> None:
        self._global_state = PanopticPoolGlobalState()
        self._internal_state = PanopticStraddleInternalState()

    def update_state(self, state: PanopticPoolGlobalState) -> None:
        self._global_state = state
        if not self._internal_state.is_open:
            return
        self._internal_state.accumulated_premium += (
            self._streaming_premium(state)
        )
        self._internal_state.bars_held += 1

    def _streaming_premium(self, state: PanopticPoolGlobalState) -> float:
        if state.liquidity <= 0 or state.fees <= 0:
            return 0.0
        our_L = self._our_liquidity(state)
        if our_L <= 0:
            return 0.0
        return estimate_fee(
            liquidity_delta=int(our_L),
            liquidity=int(state.liquidity),
            fees=state.fees,
        )

    def _our_liquidity(self, state: PanopticPoolGlobalState) -> float:
        if state.price <= 0:
            return 0.0
        K = self._internal_state.entry_price
        half = self._internal_state.notional / 2
        try:
            L_put = get_liquidity_delta(
                P=state.price,
                lower_price=K * (1 - self.TICK_SPACING_PCT),
                upper_price=K,
                amount0=0.0, amount1=half,
                token0_decimal=self.TOKEN0_DECIMALS,
                token1_decimal=self.TOKEN1_DECIMALS,
            )
            L_call = get_liquidity_delta(
                P=state.price,
                lower_price=K,
                upper_price=K * (1 + self.TICK_SPACING_PCT),
                amount0=half / state.price, amount1=0.0,
                token0_decimal=self.TOKEN0_DECIMALS,
                token1_decimal=self.TOKEN1_DECIMALS,
            )
            return float(L_put + L_call)
        except (ValueError, ZeroDivisionError):
            return (self._internal_state.notional / state.price ** 0.5) * 1e9

    @property
    def intrinsic_value(self) -> float:
        if not self._internal_state.is_open:
            return 0.0
        return abs(self._global_state.price - self._internal_state.entry_price)

    @property
    def balance(self) -> float:
        pnl = self.intrinsic_value - self._internal_state.accumulated_premium
        return self._internal_state.cash + self._internal_state.collateral + pnl

    # ── actions ───────────────────────────────────────────────────────────────

    def action_deposit(self, amount: float) -> None:
        if amount < 0:
            raise EntityException(f"deposit must be >= 0, got {amount}")
        self._internal_state.cash += amount

    def action_open(
        self, notional: float, collateral: float,
        commission: float, gas_usd: float,
    ) -> None:
        if self._internal_state.is_open:
            raise EntityException("Position already open")
        cost = collateral + commission + gas_usd
        if self._internal_state.cash < cost:
            raise EntityException(
                f"Insufficient cash: need {cost:.2f}, have "
                f"{self._internal_state.cash:.2f}"
            )
        self._internal_state.cash -= cost
        self._internal_state.is_open = True
        self._internal_state.entry_price = self._global_state.price
        self._internal_state.notional = notional
        self._internal_state.collateral = collateral
        self._internal_state.accumulated_premium = commission + gas_usd
        self._internal_state.bars_held = 0

    def action_close(self, commission: float, gas_usd: float) -> None:
        if not self._internal_state.is_open:
            raise EntityException("No open position")
        pnl = self.intrinsic_value - self._internal_state.accumulated_premium
        self._internal_state.cash += (
            self._internal_state.collateral + pnl - commission - gas_usd
        )
        self._internal_state.is_open = False
        self._internal_state.entry_price = 0.0
        self._internal_state.notional = 0.0
        self._internal_state.collateral = 0.0
        self._internal_state.accumulated_premium = 0.0
        self._internal_state.bars_held = 0


# ── Strategy ──────────────────────────────────────────────────────────────────

@dataclass
class PanopticStraddleParams(BaseStrategyParams):
    """Hyperparameters for PanopticStraddleStrategy."""
    INITIAL_BALANCE: float = 10_000.0
    NOTIONAL_FRACTION: float = 0.80
    COLLATERAL_PCT: float = 0.10
    LP_RANGE_PCT: float = 0.10
    IV_ENTRY_PERCENTILE: float = 30.0
    LOOKBACK_BARS: int = 14 * 24
    TAKE_PROFIT_MULT: float = 1.5
    STOP_LOSS_BUDGET_PCT: float = 0.05
    MAX_HOLD_BARS: int = 24
    PANOPTIC_COMMISSION_PCT: float = 0.0
    GAS_USD: float = 0.0


class PanopticStraddleStrategy(BaseStrategy):
    """Panoptic long-straddle strategy with passive Uniswap V3 LP baseline.

    Entry rule: open a long straddle when the 168-hour rolling realised
    volatility drops below the IV_ENTRY_PERCENTILE-th percentile of its
    own 14-day history (low-IV regime).

    Exit rules (first triggered wins):
        * Take Profit  — intrinsic >= TAKE_PROFIT_MULT * total_costs
        * Stop Loss    — total_costs >= STOP_LOSS_BUDGET_PCT * INITIAL_BALANCE
        * Time Stop    — bars_held >= MAX_HOLD_BARS

    References:
        Lambert & Kristensen (2023). Panoptic whitepaper v1.3.1.
        https://docs.panoptic.xyz/
    """

    PARAMS_CLS = PanopticStraddleParams

    def set_up(self) -> None:
        self.register_entity(NamedEntity(
            entity_name='STRADDLE',
            entity=PanopticStraddleEntity(),
        ))
        self.register_entity(NamedEntity(
            entity_name='LP',
            entity=UniswapV3LPEntity(config=UniswapV3LPConfig(
                pool_fee_rate=0.003,
                slippage_pct=0.001,
                token0_decimals=18,
                token1_decimals=6,
                notional_side='token1',
            )),
        ))
        self._iv_history: Deque[float] = deque(
            maxlen=self._params.LOOKBACK_BARS
        )
        self._initialized: bool = False

    def predict(self) -> List[ActionToTake]:
        straddle: PanopticStraddleEntity = self.get_entity('STRADDLE')
        state: PanopticPoolGlobalState = straddle.global_state

        if not self._initialized:
            return self._initialize(state)

        self._iv_history.append(state.implied_volatility)
        if len(self._iv_history) < self._params.LOOKBACK_BARS:
            return []

        if not straddle.internal_state.is_open:
            return self._maybe_open(state, straddle)
        return self._maybe_close(straddle)

    def _initialize(self, state: PanopticPoolGlobalState) -> List[ActionToTake]:
        self._initialized = True
        half = self._params.INITIAL_BALANCE / 2
        return [
            ActionToTake('STRADDLE', Action('deposit', {'amount': half})),
            ActionToTake('LP', Action('deposit', {'amount_in_notional': half})),
            ActionToTake('LP', Action('open_position', {
                'amount_in_notional': half,
                'price_lower': state.price * (1 - self._params.LP_RANGE_PCT),
                'price_upper': state.price * (1 + self._params.LP_RANGE_PCT),
            })),
        ]

    def _maybe_open(
        self, state: PanopticPoolGlobalState,
        straddle: PanopticStraddleEntity,
    ) -> List[ActionToTake]:
        iv_arr = np.array(self._iv_history)
        threshold = np.percentile(iv_arr, self._params.IV_ENTRY_PERCENTILE)
        if state.implied_volatility >= threshold:
            return []
        notional = straddle.internal_state.cash * self._params.NOTIONAL_FRACTION
        if straddle.internal_state.cash < notional * self._params.COLLATERAL_PCT:
            return []
        return [ActionToTake('STRADDLE', Action('open', {
            'notional':   notional,
            'collateral': notional * self._params.COLLATERAL_PCT,
            'commission': notional * self._params.PANOPTIC_COMMISSION_PCT,
            'gas_usd':    self._params.GAS_USD,
        }))]

    def _maybe_close(
        self, straddle: PanopticStraddleEntity,
    ) -> List[ActionToTake]:
        s = straddle.internal_state
        commission = s.notional * self._params.PANOPTIC_COMMISSION_PCT
        total_costs = s.accumulated_premium + commission + self._params.GAS_USD

        if (total_costs > 0
                and straddle.intrinsic_value
                >= self._params.TAKE_PROFIT_MULT * total_costs):
            reason = 'TAKE_PROFIT'
        elif total_costs >= (self._params.INITIAL_BALANCE
                             * self._params.STOP_LOSS_BUDGET_PCT):
            reason = 'STOP_LOSS'
        elif s.bars_held >= self._params.MAX_HOLD_BARS:
            reason = 'TIME_STOP'
        else:
            return []

        return [ActionToTake('STRADDLE', Action('close', {
            'commission': commission,
            'gas_usd':    self._params.GAS_USD,
        }))]