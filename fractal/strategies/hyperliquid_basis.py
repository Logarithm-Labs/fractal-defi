from fractal.core.base.strategy import NamedEntity
from fractal.core.entities import HyperliquidEntity, UniswapV3SpotEntity
from fractal.strategies.basis_trading_strategy import BasisTradingStrategy


class HyperliquidBasis(BasisTradingStrategy):

    MAX_LEVERAGE: float  # Hyperliquid MAX_LEVERAGE

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
