from dataclasses import dataclass

from fractal.core.base.strategy import NamedEntity
from fractal.core.entities import HyperliquidEntity, UniswapV3SpotEntity
from fractal.strategies.basis_trading_strategy import (
    BasisTradingStrategy, BasisTradingStrategyHyperparams)


@dataclass
class HyperliquidBasisParams(BasisTradingStrategyHyperparams):
    """Hyperparams for :class:`HyperliquidBasis`.

    Adds ``EXECUTION_COST`` — the total round-trip basis spread paid
    when opening or closing the basis position. In a basis trade the
    dominant cost is the funding-vs-spot spread, not the venue fees,
    so a single consolidated number is exposed rather than two
    per-venue fee fields. ``set_up`` splits it equally across the
    HEDGE and SPOT legs (``trading_fee = EXECUTION_COST / 2`` on each).
    """
    EXECUTION_COST: float = 0.0005


class HyperliquidBasis(BasisTradingStrategy):
    """``BasisTradingStrategy`` wired to Hyperliquid (HEDGE) + UniV3 spot (SPOT).

    ``self.MAX_LEVERAGE`` (class attribute, default 10) configures the
    physical Hyperliquid margin cap used by ``HyperliquidEntity``. It
    is intentionally distinct from ``self._params.MAX_LEVERAGE`` — the
    strategy-level rebalance-trigger upper bound. They overlap in name
    only.
    """

    # Explicit ``PARAMS_CLS`` because ``set_up`` reads
    # ``self._params.EXECUTION_COST`` which lives on the extended
    # ``HyperliquidBasisParams``, not the base hyperparams. Without
    # this override a dict-shaped grid cell would be coerced through
    # the parent class and reject the extra field.
    PARAMS_CLS = HyperliquidBasisParams

    #: Hyperliquid per-asset max leverage cap. Override per instrument
    #: (``HyperliquidBasis.MAX_LEVERAGE = 25``) before constructing the
    #: strategy. Default of 10 covers most majors; without a value here
    #: a forgotten override would hit ``AttributeError`` inside
    #: ``set_up``.
    MAX_LEVERAGE: float = 10.0

    def set_up(self):
        """Register HEDGE / SPOT entities with the per-leg fee.

        ``EXECUTION_COST`` represents the **total** round-trip spread
        (the dominant cost in basis trades — funding-vs-spot spread,
        not venue fees), so we split it equally between the two legs
        (``leg_fee = EXECUTION_COST / 2`` charged independently on each
        ``action_open_position`` call). Sum across both legs equals the
        configured ``EXECUTION_COST`` on a single open/close round.
        """
        leg_fee = self._params.EXECUTION_COST / 2.0
        self.register_entity(
            NamedEntity(entity_name='HEDGE', entity=HyperliquidEntity(
                trading_fee=leg_fee,
                max_leverage=self.MAX_LEVERAGE,
            )))
        self.register_entity(
            NamedEntity(entity_name='SPOT',
                        entity=UniswapV3SpotEntity(trading_fee=leg_fee)))
        super().set_up()
