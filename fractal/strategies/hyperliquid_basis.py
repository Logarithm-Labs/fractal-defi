from fractal.core.base.strategy import NamedEntity
from fractal.core.entities import HyperliquidEntity, UniswapV3SpotEntity
from fractal.strategies.basis_trading_strategy import BasisTradingStrategy


class HyperliquidBasis(BasisTradingStrategy):
    """``BasisTradingStrategy`` wired to Hyperliquid (HEDGE) + UniV3 spot (SPOT).

    Note on naming: ``self.MAX_LEVERAGE`` (class attribute, set
    PER-INSTRUMENT, default 10) configures the **physical Hyperliquid
    margin cap** used by ``HyperliquidEntity``. It is intentionally
    distinct from ``self._params.MAX_LEVERAGE`` (rebalance-trigger
    upper bound). They overlap in name only — see S-X1 in
    ``strategies_review.md``.
    """

    #: Hyperliquid per-asset max leverage cap. Class-level so callers can
    #: override per instrument (``HyperliquidBasis.MAX_LEVERAGE = 25``)
    #: before constructing the strategy. Default of 10 is a conservative
    #: safe value covering most majors; without it, a forgotten override
    #: would hit ``AttributeError`` deep inside ``set_up`` (HB-1).
    MAX_LEVERAGE: float = 10.0

    def set_up(self):
        """
        Set up the strategy by registering the hedge and spot entities.
        """
        # include execution cost for spread
        self.register_entity(
            NamedEntity(entity_name='HEDGE', entity=HyperliquidEntity(
                trading_fee=self._params.EXECUTION_COST,
                max_leverage=self.MAX_LEVERAGE,
            )))
        self.register_entity(
            NamedEntity(entity_name='SPOT',
                        entity=UniswapV3SpotEntity(trading_fee=self._params.EXECUTION_COST)))
        super().set_up()
