from dataclasses import dataclass
from typing import List, Optional

from fractal.core.base import Action, ActionToTake, BaseStrategy, BaseStrategyParams, NamedEntity
from fractal.core.entities import UniswapV3LPConfig, UniswapV3LPEntity


class FixedRangeLiquidityProvisionException(Exception):
    pass


@dataclass
class FixedRangeLiquidityProvisionParams(BaseStrategyParams):
    """
    Parameters for the fixed-range liquidity-provision strategy.

    PRICE_LOWER: float — lower bound of the LP range, in the entity's
        price convention (**notional per non-notional unit**, same as
        ``GlobalState.price``).
    PRICE_UPPER: float — upper bound of the LP range, same convention.
    INITIAL_BALANCE: float — notional amount deposited and zapped into
        the position on the first observation (zap-in mode).
    TOKEN0_AMOUNT / TOKEN1_AMOUNT: Optional[float] — when both are set,
        the position is minted from these exact on-chain token amounts
        instead of zapping in (``INITIAL_BALANCE`` is then ignored).
        Use this to replicate a real position 1:1 — same amounts, same
        range, same resulting ``L``.
    """
    PRICE_LOWER: float
    PRICE_UPPER: float
    INITIAL_BALANCE: float = 0.0
    TOKEN0_AMOUNT: Optional[float] = None
    TOKEN1_AMOUNT: Optional[float] = None


class FixedRangeLiquidityProvision(BaseStrategy[FixedRangeLiquidityProvisionParams]):
    """
    Passive fixed-range LP: open a position over
    ``[PRICE_LOWER, PRICE_UPPER]`` on the first observation and hold it.
    Mirrors a plain on-chain mint held untouched — the natural harness
    for validating the fee model against real positions (exact-pair
    mode + tick-derived bounds).

    ``token0_decimals``/``token1_decimals`` are required;
    ``pool_fee_rate``, ``slippage_pct``, ``protocol_fee``, ``fee_model``
    and ``notional_side`` pass through to :class:`UniswapV3LPConfig`.
    ``pool_fee_rate`` defaults to ``0.0`` — a real mint from pre-held
    tokens pays no swap fee.
    """

    def __init__(
        self,
        params: FixedRangeLiquidityProvisionParams,
        *args,
        debug: bool = False,
        token0_decimals: Optional[int] = None,
        token1_decimals: Optional[int] = None,
        pool_fee_rate: float = 0.0,
        slippage_pct: float = 0.0,
        protocol_fee: float = 0.0,
        fee_model: str = "auto",
        notional_side: str = "token0",
        **kwargs,
    ):
        if token0_decimals is None or token1_decimals is None:
            raise FixedRangeLiquidityProvisionException(
                "FixedRangeLiquidityProvision needs token0_decimals and "
                "token1_decimals constructor kwargs."
            )
        self._token0_decimals = token0_decimals
        self._token1_decimals = token1_decimals
        self._pool_fee_rate = pool_fee_rate
        self._slippage_pct = slippage_pct
        self._protocol_fee = protocol_fee
        self._fee_model = fee_model
        self._notional_side = notional_side
        super().__init__(params=params, debug=debug, *args, **kwargs)
        # Instance attribute (NOT class-level) so independent strategy
        # instances run side-by-side without sharing the entry flag.
        self.entered = False

    def set_up(self):
        """Register the Uniswap V3 LP entity holding the fixed-range position."""
        self.register_entity(NamedEntity(
            entity_name='UNISWAP_V3',
            entity=UniswapV3LPEntity(
                UniswapV3LPConfig(
                    token0_decimals=self._token0_decimals,
                    token1_decimals=self._token1_decimals,
                    pool_fee_rate=self._pool_fee_rate,
                    slippage_pct=self._slippage_pct,
                    protocol_fee=self._protocol_fee,
                    fee_model=self._fee_model,
                    notional_side=self._notional_side,
                )
            )
        ))

    @property
    def _exact_pair_mode(self) -> bool:
        return (
            self._params.TOKEN0_AMOUNT is not None
            and self._params.TOKEN1_AMOUNT is not None
        )

    def predict(self) -> List[ActionToTake]:
        """Open the position once; afterwards do nothing."""
        entity: UniswapV3LPEntity = self.get_entity('UNISWAP_V3')
        if entity.is_position or self.entered:
            return []
        self.entered = True
        if self._exact_pair_mode:
            self._debug(
                f"Minting position [{self._params.PRICE_LOWER}, {self._params.PRICE_UPPER}] "
                f"from exact pair ({self._params.TOKEN0_AMOUNT}, {self._params.TOKEN1_AMOUNT})."
            )
            return [
                ActionToTake(
                    entity_name='UNISWAP_V3',
                    action=Action(action='open_position_from_pair', args={
                        'token0_amount': self._params.TOKEN0_AMOUNT,
                        'token1_amount': self._params.TOKEN1_AMOUNT,
                        'price_lower': self._params.PRICE_LOWER,
                        'price_upper': self._params.PRICE_UPPER,
                    }),
                ),
            ]
        self._debug(
            f"Opening fixed-range position [{self._params.PRICE_LOWER}, "
            f"{self._params.PRICE_UPPER}] with {self._params.INITIAL_BALANCE} notional."
        )
        return [
            ActionToTake(
                entity_name='UNISWAP_V3',
                action=Action(action='deposit', args={
                    'amount_in_notional': self._params.INITIAL_BALANCE,
                }),
            ),
            ActionToTake(
                entity_name='UNISWAP_V3',
                action=Action(action='open_position', args={
                    # Delegate: resolves at execute time, after the deposit
                    # in this same step has credited the cash.
                    'amount_in_notional': lambda strategy: strategy.get_entity('UNISWAP_V3').internal_state.cash,
                    'price_lower': self._params.PRICE_LOWER,
                    'price_upper': self._params.PRICE_UPPER,
                }),
            ),
        ]
