"""Shared plumbing of the Morpho supply strategies.

Entity naming, the shared parameter block, entity construction and the
eligibility filters used by both the single-position rotation
(:mod:`fractal.strategies.morpho_supply_rotation`) and the multi-market
allocator (:mod:`fractal.strategies.morpho_supply_allocation`). One home so
the two strategies cannot drift apart.
"""
import math
from dataclasses import dataclass, field

from fractal.core.base.strategy import BaseStrategyParams
from fractal.core.base.time import SECONDS_PER_YEAR
from fractal.core.entities.protocols.morpho_supply import MorphoSupplyEntity
from fractal.loaders.morpho_markets import MorphoMarketListing

__all__ = [
    "MorphoSupplyBaseParams",
    "build_supply_entity",
    "is_selectable",
    "market_entity_name",
    "validate_common_params",
]


def market_entity_name(listing: MorphoMarketListing) -> str:
    """Registry name of the entity backing ``listing``.

    Public because observation builders must produce states under exactly these
    names; deriving them separately in an example would be a silent mismatch.
    """
    return f"SUPPLY_{listing.chain_id}_{listing.market_id[:10]}"


def validate_common_params(params: "MorphoSupplyBaseParams") -> None:
    """Validate the shared parameter block: signal integers, LLTV range, liquidity multiple."""
    for field_name in ("SIGNAL_LAG_BARS", "SIGNAL_WINDOW_BARS"):
        value = getattr(params, field_name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{field_name} must be an integer >= 1, got {value!r}")
    if not 0.0 < params.MAX_LLTV <= 1.0:
        raise ValueError(f"MAX_LLTV must be in (0, 1], got {params.MAX_LLTV!r}")
    if not math.isfinite(params.MIN_LIQUIDITY_MULTIPLE) or params.MIN_LIQUIDITY_MULTIPLE < 0.0:
        raise ValueError(
            f"MIN_LIQUIDITY_MULTIPLE must be a finite non-negative ratio, got {params.MIN_LIQUIDITY_MULTIPLE!r}")


def _gas_per_leg(params: "MorphoSupplyBaseParams", listing: MorphoMarketListing) -> float:
    """USD gas per leg on ``listing``'s chain (``GAS_USD_BY_CHAIN``, else ``GAS_USD_PER_LEG``)."""
    gas = float(params.GAS_USD_BY_CHAIN.get(listing.chain_id, params.GAS_USD_PER_LEG))
    if gas < 0 or not math.isfinite(gas):
        raise ValueError(
            f"gas per leg must be a finite non-negative USD amount, got {gas} for chain {listing.chain_id}")
    return gas


def build_supply_entity(params: "MorphoSupplyBaseParams", listing: MorphoMarketListing) -> MorphoSupplyEntity:
    """One supply entity for ``listing``, configured from the shared parameter block."""
    return MorphoSupplyEntity(
        own_impact=params.OWN_IMPACT,
        absorption=params.ABSORPTION,
        irm_feedback=params.IRM_FEEDBACK,
        use_rewards=params.USE_REWARDS,
        reward_weight=params.REWARD_WEIGHT,
        cost_bps_per_leg=params.COST_BPS_PER_LEG,
        gas_usd_per_leg=_gas_per_leg(params, listing),
        bars_per_year=params.BARS_PER_YEAR,
    )


def is_selectable(params: "MorphoSupplyBaseParams", listing: MorphoMarketListing, state) -> bool:
    """Static (LLTV, oracle, loan asset) and current-bar (pool size) eligibility of a market."""
    return (
        listing.lltv <= params.MAX_LLTV
        and listing.oracle_type in params.ORACLE_TYPES
        and listing.loan_symbol in params.LOAN_SYMBOLS
        and state.supply_assets_usd >= params.MIN_POOL_SIZE_USD
    )


@dataclass
class MorphoSupplyBaseParams(BaseStrategyParams):
    """Parameters shared by the Morpho supply strategies.

    Attributes:
        MAX_LLTV: inclusive LLTV ceiling for eligibility.
        ORACLE_TYPES: allowed oracle types; anything else is not selectable.
        LOAN_SYMBOLS: loan assets the strategy may supply (exact symbols).
        MIN_POOL_SIZE_USD: a market must have at least this ``supply_assets_usd``
            on the current bar to be selectable (size is read at execution
            time, like the liquidity gate).
        MIN_LIQUIDITY_MULTIPLE: required ratio of a target's withdrawable
            liquidity to the position being entered. ``1.0`` means "we could
            get out immediately"; ``0`` disables the check.
        SIGNAL_LAG_BARS: bars of observation lag in the signal (``>= 1``).
        SIGNAL_WINDOW_BARS: bars averaged to form the signal level; the
            default here suits the rotation's steady holding, the allocation
            overrides it with ``1`` to react bar-by-bar.
        CASH_ENTITY_NAME: registry name of the entity holding idle capital.
        OWN_IMPACT / ABSORPTION / IRM_FEEDBACK / COST_BPS_PER_LEG /
            GAS_USD_PER_LEG / BARS_PER_YEAR: passed straight to each
            :class:`MorphoSupplyEntity` — whether the position's own size
            moves the rate, the share of it borrowers take up, whether the
            adaptive-rate drift is simulated, the per-leg cost in basis
            points, the fixed gas per leg in USD, and the bar length
            convention (``SECONDS_PER_YEAR / 3600``: hourly bars, ACT/365).
        USE_REWARDS / REWARD_WEIGHT: credit market reward campaigns, diluted
            by our own supply (see :class:`MorphoSupplyEntity`).
        GAS_USD_BY_CHAIN: ``{chain_id: USD gas per leg}``; overrides
            ``GAS_USD_PER_LEG`` for the markets on that chain.
    """
    MAX_LLTV: float = 0.915
    ORACLE_TYPES: tuple[str, ...] = ("ChainlinkOracle", "ChainlinkOracleV2")
    LOAN_SYMBOLS: tuple[str, ...] = ("USDC", "USDT")
    MIN_POOL_SIZE_USD: float = 1_000_000.0
    MIN_LIQUIDITY_MULTIPLE: float = 1.0
    SIGNAL_LAG_BARS: int = 1
    SIGNAL_WINDOW_BARS: int = 24
    CASH_ENTITY_NAME: str = "CASH"
    OWN_IMPACT: bool = True
    ABSORPTION: float = 0.0
    IRM_FEEDBACK: bool = False
    USE_REWARDS: bool = False
    REWARD_WEIGHT: float = 1.0
    COST_BPS_PER_LEG: float = 0.0
    GAS_USD_PER_LEG: float = 0.0
    GAS_USD_BY_CHAIN: dict[int, float] = field(default_factory=dict)
    BARS_PER_YEAR: float = SECONDS_PER_YEAR / 3600.0
