from fractal.core.base.strategy import NamedEntity
from fractal.core.entities import GMXV2Entity, UniswapV3SpotEntity
from fractal.strategies.basis_trading_strategy import BasisTradingStrategy


class GMXV2UniswapV3Basis(BasisTradingStrategy):

    def set_up(self):
        """
        Set up the strategy by registering the hedge and spot entities.
        """
        self.register_entity(NamedEntity(entity_name='HEDGE', entity=GMXV2Entity()))
        self.register_entity(NamedEntity(entity_name='SPOT', entity=UniswapV3SpotEntity()))
        super().set_up()
