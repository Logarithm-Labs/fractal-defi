"""API-surface parity between HyperliquidEntity and SimplePerpEntity.

Counterpart to ``test_api_parity.py`` (which covers V2/V3 LP) for the
perp paradigm. Verifies both concrete perp entities expose the same
public surface so polymorphic strategy code (typing
``perp: BasePerpEntity``) can rely on a uniform API.

Hyperliquid-specific extras (e.g. ``positions`` list state) are
explicitly enumerated and excluded from the parity check.
"""
from dataclasses import fields

import pytest

from fractal.core.entities.protocols.hyperliquid import (HyperliquidEntity,
                                                         HyperLiquidGlobalState,
                                                         HyperLiquidInternalState)
from fractal.core.entities.simple.perp import (SimplePerpEntity,
                                                SimplePerpGlobalState,
                                                SimplePerpInternalState)


# Methods/properties that BOTH perp entities must expose with the same name.
SHARED_PUBLIC_API = {
    # Inherited from BaseEntity machinery
    "execute",
    "get_available_actions",
    "internal_state",
    "global_state",
    # Lifecycle
    "update_state",
    # Account flow
    "action_deposit",
    "action_withdraw",
    # Position flow
    "action_open_position",
    "action_close_position",
    # Readouts
    "balance",
    "size",
    "leverage",
    "pnl",
    "maintenance_margin",
    "liquidation_price",
}

# Members allowed only on Hyperliquid (none expected currently — but we
# enumerate so additions are explicit).
HL_ONLY_PUBLIC: set[str] = set()

# Members allowed only on SimplePerp.
SP_ONLY_PUBLIC: set[str] = set()


def _public_members(cls) -> set[str]:
    if cls is HyperliquidEntity:
        instance = cls()
    else:
        instance = cls()
    return {m for m in dir(instance) if not m.startswith("_") and not m.isupper()}


# ============================================================ public API parity
@pytest.mark.core
def test_hl_exposes_all_shared_public_api():
    members = _public_members(HyperliquidEntity)
    missing = SHARED_PUBLIC_API - members
    assert not missing, f"Hyperliquid missing public members: {missing}"


@pytest.mark.core
def test_sp_exposes_all_shared_public_api():
    members = _public_members(SimplePerpEntity)
    missing = SHARED_PUBLIC_API - members
    assert not missing, f"SimplePerp missing public members: {missing}"


@pytest.mark.core
def test_no_unexpected_public_members_hl():
    members = _public_members(HyperliquidEntity)
    extras = members - SHARED_PUBLIC_API - HL_ONLY_PUBLIC
    assert not extras, (
        f"Hyperliquid has unexpected public members: {extras}. "
        f"If intentional, update SHARED_PUBLIC_API or HL_ONLY_PUBLIC."
    )


@pytest.mark.core
def test_no_unexpected_public_members_sp():
    members = _public_members(SimplePerpEntity)
    extras = members - SHARED_PUBLIC_API - SP_ONLY_PUBLIC
    assert not extras, (
        f"SimplePerp has unexpected public members: {extras}. "
        f"If intentional, update the whitelist."
    )


# ============================================================ subclass relations
@pytest.mark.core
def test_both_subclass_base_perp_entity():
    from fractal.core.entities import BasePerpEntity
    assert issubclass(HyperliquidEntity, BasePerpEntity)
    assert issubclass(SimplePerpEntity, BasePerpEntity)


# ============================================================ internal-state shared base fields
@pytest.mark.core
def test_internal_states_share_collateral_field():
    """Both internal states inherit ``collateral`` from BasePerpInternalState."""
    hl_fields = {f.name for f in fields(HyperLiquidInternalState)}
    sp_fields = {f.name for f in fields(SimplePerpInternalState)}
    assert "collateral" in hl_fields
    assert "collateral" in sp_fields


@pytest.mark.core
def test_internal_states_have_distinct_position_models():
    """Hyperliquid keeps ``positions: List``, SimplePerp keeps scalars."""
    hl_fields = {f.name for f in fields(HyperLiquidInternalState)}
    sp_fields = {f.name for f in fields(SimplePerpInternalState)}
    assert "positions" in hl_fields
    assert "size" in sp_fields and "entry_price" in sp_fields


# ============================================================ global-state shared fields
@pytest.mark.core
def test_global_states_share_mark_price_and_funding_rate():
    hl_fields = {f.name for f in fields(HyperLiquidGlobalState)}
    sp_fields = {f.name for f in fields(SimplePerpGlobalState)}
    assert "mark_price" in hl_fields
    assert "mark_price" in sp_fields
    assert "funding_rate" in hl_fields
    assert "funding_rate" in sp_fields


# ============================================================ behavioural API parity
@pytest.mark.core
def test_both_perps_have_zero_initial_state():
    hl = HyperliquidEntity()
    sp = SimplePerpEntity()
    for e in (hl, sp):
        assert e.size == 0
        assert e.pnl == 0
        assert e.balance == 0
        assert e.leverage == 0
        assert e.maintenance_margin == 0


@pytest.mark.core
def test_both_perps_liquidation_price_nan_when_flat():
    import math
    hl = HyperliquidEntity()
    sp = SimplePerpEntity()
    assert math.isnan(hl.liquidation_price)
    assert math.isnan(sp.liquidation_price)


@pytest.mark.core
def test_both_perps_init_validate_trading_fee():
    """Both reject negative ``trading_fee``."""
    with pytest.raises(Exception):
        HyperliquidEntity(trading_fee=-0.01)
    with pytest.raises(Exception):
        SimplePerpEntity(trading_fee=-0.01)


@pytest.mark.core
def test_both_perps_init_validate_max_leverage():
    """Both reject non-positive ``max_leverage``."""
    with pytest.raises(Exception):
        HyperliquidEntity(max_leverage=0)
    with pytest.raises(Exception):
        SimplePerpEntity(max_leverage=0)
    with pytest.raises(Exception):
        HyperliquidEntity(max_leverage=-5)
    with pytest.raises(Exception):
        SimplePerpEntity(max_leverage=-5)


# ============================================================ liquidation_price contract: long below entry, short above
@pytest.mark.core
def test_both_perps_long_liquidation_below_entry():
    hl = HyperliquidEntity(trading_fee=0.0)
    sp = SimplePerpEntity(trading_fee=0.0)
    hl.update_state(HyperLiquidGlobalState(mark_price=3000))
    sp.update_state(SimplePerpGlobalState(mark_price=3000))
    hl.action_deposit(1000)
    sp.action_deposit(1000)
    hl.action_open_position(1.0)
    sp.action_open_position(1.0)
    assert hl.liquidation_price < 3000
    assert sp.liquidation_price < 3000


@pytest.mark.core
def test_both_perps_short_liquidation_above_entry():
    hl = HyperliquidEntity(trading_fee=0.0)
    sp = SimplePerpEntity(trading_fee=0.0)
    hl.update_state(HyperLiquidGlobalState(mark_price=3000))
    sp.update_state(SimplePerpGlobalState(mark_price=3000))
    hl.action_deposit(1000)
    sp.action_deposit(1000)
    hl.action_open_position(-1.0)
    sp.action_open_position(-1.0)
    assert hl.liquidation_price > 3000
    assert sp.liquidation_price > 3000


# ============================================================ MM scaling with mark_price
@pytest.mark.core
def test_both_perps_maintenance_margin_scales_with_mark_price():
    """MM is mark-price-based on both entities."""
    hl = HyperliquidEntity(trading_fee=0.0)
    sp = SimplePerpEntity(trading_fee=0.0)
    hl.update_state(HyperLiquidGlobalState(mark_price=3000))
    sp.update_state(SimplePerpGlobalState(mark_price=3000))
    hl.action_deposit(1000)
    sp.action_deposit(1000)
    hl.action_open_position(1.0)
    sp.action_open_position(1.0)
    mm_hl_at_3000 = hl.maintenance_margin
    mm_sp_at_3000 = sp.maintenance_margin
    # Bump mark price (avoid liquidation by going up)
    hl._global_state.mark_price = 4000
    sp._global_state.mark_price = 4000
    # MM scales linearly: 4/3 of original (size unchanged)
    assert hl.maintenance_margin == pytest.approx(mm_hl_at_3000 * 4 / 3)
    assert sp.maintenance_margin == pytest.approx(mm_sp_at_3000 * 4 / 3)
