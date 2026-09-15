"""``LeveragedPTStrategy`` wired to a Pendle PT market and a Morpho Blue market."""
from dataclasses import dataclass
from typing import Optional

from fractal.core.base.strategy import NamedEntity
from fractal.core.entities import MorphoEntity, PendlePTConfig, PendlePTEntity
from fractal.strategies.leveraged_pt import LeveragedPTParams, LeveragedPTStrategy


@dataclass
class MorphoLeveragedPTParams(LeveragedPTParams):
    """Adds the venue configuration to :class:`LeveragedPTParams`.

    LLTV: Morpho market liquidation LTV (``0.915`` on the live PT-sUSDe/USDC market).
    MAX_LTV: strategy-side borrow cap on the market (defaults to ``LLTV``).
    PT_IMPACT_MODEL / PT_FEE_LN_RATE / PT_IMPACT_LN_RATE_PER_SHARE: see :class:`PendlePTConfig`.
    """
    LLTV: float = 0.915
    MAX_LTV: Optional[float] = None
    PT_IMPACT_MODEL: str = "amm"
    PT_FEE_LN_RATE: float = 0.001
    PT_IMPACT_LN_RATE_PER_SHARE: float = 0.075


class MorphoLeveragedPT(LeveragedPTStrategy):
    """PT looping on Morpho Blue."""

    # ``set_up`` reads the extended params, so dict-shaped grid cells must
    # be coerced through the venue params class, not the base one.
    PARAMS_CLS = MorphoLeveragedPTParams

    def set_up(self):
        params: MorphoLeveragedPTParams = self._params
        self.register_entity(NamedEntity(entity_name="PT", entity=PendlePTEntity(PendlePTConfig(
            impact_model=params.PT_IMPACT_MODEL,
            fee_ln_rate=params.PT_FEE_LN_RATE,
            impact_ln_rate_per_share=params.PT_IMPACT_LN_RATE_PER_SHARE,
        ))))
        self.register_entity(NamedEntity(entity_name="LENDING", entity=MorphoEntity(
            lltv=params.LLTV, max_ltv=params.MAX_LTV, collateral_is_volatile=True,
        )))
        super().set_up()
