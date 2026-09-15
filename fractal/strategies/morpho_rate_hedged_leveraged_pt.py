"""``RateHedgedLeveragedPTStrategy`` wired to Pendle + Morpho Blue with a Boros or a
Hyperliquid-style perp (plus a simple spot venue) as the floating-rate leg."""
from dataclasses import dataclass

from fractal.core.base.strategy import NamedEntity
from fractal.core.entities import BorosEntity, HyperliquidEntity, SimpleSpotExchange
from fractal.strategies.morpho_leveraged_pt import MorphoLeveragedPT, MorphoLeveragedPTParams
from fractal.strategies.rate_hedged_leveraged_pt import RateHedgedLeveragedPTParams, RateHedgedLeveragedPTStrategy


@dataclass
class MorphoRateHedgedLeveragedPTParams(RateHedgedLeveragedPTParams, MorphoLeveragedPTParams):
    """Venue parameters for the hedge legs on top of the Morpho loop parameters.

    BOROS_*: :class:`BorosEntity` margining and fees (live Binance markets by default).
    PERP_TRADING_FEE / PERP_MAX_LEVERAGE: the perp venue; SPOT_TRADING_FEE: the spot venue.
    """
    BOROS_MAX_LEVERAGE: float = 1.55
    BOROS_MM_TO_IM_RATIO: float = 0.5
    BOROS_RATE_FLOOR: float = 0.06
    BOROS_TAKER_FEE_RATE: float = 0.0005
    BOROS_SETTLE_FEE_RATE: float = 0.001
    PERP_TRADING_FEE: float = 0.00035
    PERP_MAX_LEVERAGE: float = 10.0
    SPOT_TRADING_FEE: float = 0.0005


class MorphoRateHedgedLeveragedPT(RateHedgedLeveragedPTStrategy, MorphoLeveragedPT):
    """PT looping on Morpho Blue with an optional Boros / perp floating-rate leg."""

    PARAMS_CLS = MorphoRateHedgedLeveragedPTParams

    def set_up(self):
        params: MorphoRateHedgedLeveragedPTParams = self._params
        if params.RATE_HEDGE == "boros":
            self.register_entity(NamedEntity(entity_name="BOROS", entity=BorosEntity(
                max_leverage=params.BOROS_MAX_LEVERAGE, mm_to_im_ratio=params.BOROS_MM_TO_IM_RATIO,
                rate_floor=params.BOROS_RATE_FLOOR, taker_fee_rate=params.BOROS_TAKER_FEE_RATE,
                settle_fee_rate=params.BOROS_SETTLE_FEE_RATE,
            )))
        elif params.RATE_HEDGE == "perp":
            self.register_entity(NamedEntity(entity_name="SPOT", entity=SimpleSpotExchange(
                trading_fee=params.SPOT_TRADING_FEE)))
            self.register_entity(NamedEntity(entity_name="PERP", entity=HyperliquidEntity(
                trading_fee=params.PERP_TRADING_FEE, max_leverage=params.PERP_MAX_LEVERAGE)))
        super().set_up()
