"""API-surface parity of the Pendle/Morpho/Boros entities with their
siblings: ``MorphoEntity`` vs ``SimpleLendingEntity``, ``BorosEntity`` vs
``SimplePerpEntity``, ``PendlePTEntity`` vs ``SimpleSpotExchange``.

Programmatic drift guards, like ``test_perp_api_parity``: every member
that exists on one side only is enumerated with its justification, so a
strategy written against the sibling keeps working on the new entity.
"""
import pytest

from fractal.core.entities import (
    BorosEntity,
    MorphoEntity,
    PendlePTEntity,
    SimpleLendingEntity,
    SimplePerpEntity,
    SimpleSpotExchange,
)
from fractal.core.entities.base import BaseFixedTermEntity, BaseLendingEntity, BasePerpEntity, BaseSpotEntity

# Every public member of the sibling that the new entity must also expose.
SHARED_LENDING_API = {
    "action_deposit", "action_withdraw", "action_borrow", "action_repay",
    "update_state", "internal_state", "global_state", "execute", "get_available_actions",
    "balance", "collateral_value", "debt_value", "ltv", "health_factor", "max_borrow_amount",
    "liquidation_price", "calculate_repay", "max_ltv", "liq_thr", "collateral_is_volatile",
}
MORPHO_ONLY = {
    "lltv",  # Morpho's single liquidation LTV (``liq_thr`` aliases it for the shared API)
    "lif",  # liquidation incentive factor, derived from the LLTV
    "collateral_market_value",  # PnL mark when the lender's oracle differs from the market
}

SHARED_PERP_API = {
    "action_deposit", "action_withdraw", "action_open_position", "action_close_position",
    "update_state", "internal_state", "global_state", "execute", "get_available_actions",
    "balance", "size", "leverage", "pnl", "maintenance_margin", "trading_fee", "max_leverage",
}
SIMPLE_PERP_ONLY = {
    "liquidation_price",  # a rate instrument has no price of liquidation; Boros exposes ``health_ratio``
    "MAX_LEVERAGE", "TRADING_FEE",  # legacy upper-case aliases
}
BOROS_ONLY = {
    "seconds_to_expiry", "years_to_expiry", "is_matured",  # BaseFixedTermEntity
    "entry_rate", "health_ratio", "initial_margin",  # fixed-rate position readouts
    "k_im", "k_mm", "rate_floor", "time_floor_years", "taker_fee_rate", "settle_fee_rate",  # Boros margining
    "coin_margined", "slippage_rate",
}

SHARED_SPOT_API = {
    "action_deposit", "action_withdraw", "action_buy", "action_sell",
    "action_inject_product", "action_remove_product",
    "update_state", "internal_state", "global_state", "execute", "get_available_actions",
    "balance", "current_price",
}
SIMPLE_SPOT_ONLY = {"trading_fee", "effective_fee_rate"}  # PT fees are spreads on the implied rate
PT_ONLY = {
    "seconds_to_expiry", "years_to_expiry", "is_matured",  # BaseFixedTermEntity
    "action_redeem", "implied_apy", "pt_price_asset", "redeem_haircut",
    "quote_buy", "quote_sell",  # side-effect-free quotes the strategies size trades on
    "impact_model", "fee_ln_rate", "impact_ln_rate_per_share", "max_pool_share",
}


def _public(instance) -> set:
    return {m for m in dir(instance) if not m.startswith("_")}


@pytest.mark.core
@pytest.mark.parametrize("new,sibling,shared,new_only,sibling_only", [
    (MorphoEntity, SimpleLendingEntity, SHARED_LENDING_API, MORPHO_ONLY, set()),
    (BorosEntity, SimplePerpEntity, SHARED_PERP_API, BOROS_ONLY, SIMPLE_PERP_ONLY),
    (PendlePTEntity, SimpleSpotExchange, SHARED_SPOT_API, PT_ONLY, SIMPLE_SPOT_ONLY),
])
def test_public_surface_matches_the_sibling_up_to_the_enumerated_extras(
        new, sibling, shared, new_only, sibling_only):
    new_members, sibling_members = _public(new()), _public(sibling())
    assert shared <= new_members, f"{new.__name__} misses shared members {shared - new_members}"
    assert shared <= sibling_members, f"{sibling.__name__} misses shared members {shared - sibling_members}"
    assert new_members - shared == new_only, f"unexpected {new.__name__} extras: {new_members - shared - new_only}"
    assert sibling_members - shared == sibling_only, (
        f"unexpected {sibling.__name__} extras: {sibling_members - shared - sibling_only}")


@pytest.mark.core
def test_base_classes():
    assert issubclass(MorphoEntity, BaseLendingEntity) and not issubclass(MorphoEntity, BaseFixedTermEntity)
    assert issubclass(BorosEntity, BasePerpEntity) and issubclass(BorosEntity, BaseFixedTermEntity)
    assert issubclass(PendlePTEntity, BaseSpotEntity) and issubclass(PendlePTEntity, BaseFixedTermEntity)


@pytest.mark.core
def test_morpho_aliases_the_shared_lending_thresholds():
    entity = MorphoEntity(lltv=0.86, max_ltv=0.8)
    assert entity.liq_thr == entity.lltv == 0.86
    assert entity.max_ltv == 0.8
