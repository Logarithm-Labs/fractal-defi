"""``HedgedPTStrategy`` wired to a Pendle PT, a perp hedge and (optionally) Boros."""
from dataclasses import dataclass

from fractal.core.base.strategy import NamedEntity
from fractal.core.entities import BorosEntity, HyperliquidEntity, PendlePTConfig, PendlePTEntity, SimplePerpEntity
from fractal.strategies.hedged_pt import HedgedPTException, HedgedPTParams, HedgedPTStrategy

_VENUES = ("hyperliquid", "simple")


@dataclass
class PerpHedgedPTParams(HedgedPTParams):
    """Adds the venue configuration to :class:`HedgedPTParams`.

    HEDGE_VENUE: ``"hyperliquid"`` (default) or ``"simple"`` perp entity.
    PERP_TRADING_FEE / PERP_MAX_LEVERAGE: perp entity config.
    PT_IMPACT_MODEL / PT_FEE_LN_RATE / PT_IMPACT_LN_RATE_PER_SHARE: see :class:`PendlePTConfig`.
    BOROS_MAX_LEVERAGE / BOROS_MM_TO_IM_RATIO / BOROS_RATE_FLOOR / BOROS_TAKER_FEE_RATE /
    BOROS_SETTLE_FEE_RATE / BOROS_COIN_MARGINED: :class:`BorosEntity` config (used when ``USE_BOROS``).
    """
    HEDGE_VENUE: str = "hyperliquid"
    PERP_TRADING_FEE: float = 0.00035
    PERP_MAX_LEVERAGE: float = 10.0
    PT_IMPACT_MODEL: str = "amm"
    PT_FEE_LN_RATE: float = 0.001
    PT_IMPACT_LN_RATE_PER_SHARE: float = 0.075
    BOROS_MAX_LEVERAGE: float = 1.55
    BOROS_MM_TO_IM_RATIO: float = 0.5
    BOROS_RATE_FLOOR: float = 0.06
    BOROS_TAKER_FEE_RATE: float = 0.0005
    BOROS_SETTLE_FEE_RATE: float = 0.001
    BOROS_COIN_MARGINED: bool = False


class PerpHedgedPT(HedgedPTStrategy):
    """Hedged PT carry on a perp venue, optionally with a Boros funding lock."""

    PARAMS_CLS = PerpHedgedPTParams

    def set_up(self):
        params: PerpHedgedPTParams = self._params
        if params.HEDGE_VENUE not in _VENUES:
            raise HedgedPTException(f"HEDGE_VENUE must be one of {_VENUES}, got {params.HEDGE_VENUE!r}")
        self.register_entity(NamedEntity(entity_name="PT", entity=PendlePTEntity(PendlePTConfig(
            impact_model=params.PT_IMPACT_MODEL,
            fee_ln_rate=params.PT_FEE_LN_RATE,
            impact_ln_rate_per_share=params.PT_IMPACT_LN_RATE_PER_SHARE,
        ))))
        perp_cls = HyperliquidEntity if params.HEDGE_VENUE == "hyperliquid" else SimplePerpEntity
        self.register_entity(NamedEntity(entity_name="HEDGE", entity=perp_cls(
            trading_fee=params.PERP_TRADING_FEE, max_leverage=params.PERP_MAX_LEVERAGE,
        )))
        if params.USE_BOROS:
            self.register_entity(NamedEntity(entity_name="BOROS", entity=BorosEntity(
                max_leverage=params.BOROS_MAX_LEVERAGE, mm_to_im_ratio=params.BOROS_MM_TO_IM_RATIO,
                rate_floor=params.BOROS_RATE_FLOOR, taker_fee_rate=params.BOROS_TAKER_FEE_RATE,
                settle_fee_rate=params.BOROS_SETTLE_FEE_RATE, coin_margined=params.BOROS_COIN_MARGINED,
            )))
        super().set_up()
