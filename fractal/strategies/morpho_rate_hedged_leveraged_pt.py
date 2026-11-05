"""``RateHedgedLeveragedPTStrategy`` wired to Pendle + Morpho Blue with a Boros yield unit."""
from dataclasses import dataclass

from fractal.core.base.strategy import NamedEntity
from fractal.core.entities import BorosEntity
from fractal.strategies.morpho_leveraged_pt import MorphoLeveragedPT, MorphoLeveragedPTParams
from fractal.strategies.rate_hedged_leveraged_pt import RateHedgedLeveragedPTParams, RateHedgedLeveragedPTStrategy


@dataclass
class MorphoRateHedgedLeveragedPTParams(RateHedgedLeveragedPTParams, MorphoLeveragedPTParams):
    """Boros venue parameters on top of the Morpho loop and overlay parameters.

    BOROS_*: :class:`BorosEntity` margining and fees (live Binance markets by default).
    """
    BOROS_MAX_LEVERAGE: float = 1.55
    BOROS_MM_TO_IM_RATIO: float = 0.5
    BOROS_RATE_FLOOR: float = 0.06
    BOROS_TAKER_FEE_RATE: float = 0.0005
    BOROS_SETTLE_FEE_RATE: float = 0.001


class MorphoRateHedgedLeveragedPT(RateHedgedLeveragedPTStrategy, MorphoLeveragedPT):
    """PT looping on Morpho Blue with an optional Boros yield-unit overlay."""

    PARAMS_CLS = MorphoRateHedgedLeveragedPTParams

    def set_up(self):
        params: MorphoRateHedgedLeveragedPTParams = self._params
        if params.RATE_HEDGE == "boros":
            self.register_entity(NamedEntity(entity_name="BOROS", entity=BorosEntity(
                max_leverage=params.BOROS_MAX_LEVERAGE, mm_to_im_ratio=params.BOROS_MM_TO_IM_RATIO,
                rate_floor=params.BOROS_RATE_FLOOR, taker_fee_rate=params.BOROS_TAKER_FEE_RATE,
                settle_fee_rate=params.BOROS_SETTLE_FEE_RATE,
            )))
        super().set_up()
