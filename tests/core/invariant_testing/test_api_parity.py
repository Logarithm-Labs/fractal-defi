"""API-surface parity between UniswapV2LPEntity and UniswapV3LPEntity.

These tests are programmatic guards against drift: the moment someone adds
a method/property to one entity but not the other (without justification),
a test fails. V3-specific extensions are explicitly enumerated.
"""
from dataclasses import fields

import pytest

from fractal.core.entities.base.pool import BasePoolGlobalState
from fractal.core.entities.protocols.uniswap_v2_lp import (UniswapV2LPConfig,
                                                           UniswapV2LPEntity,
                                                           UniswapV2LPGlobalState,
                                                           UniswapV2LPInternalState)
from fractal.core.entities.protocols.uniswap_v3_lp import (UniswapV3LPConfig,
                                                           UniswapV3LPEntity,
                                                           UniswapV3LPGlobalState,
                                                           UniswapV3LPInternalState)


# Methods/properties that BOTH V2 and V3 must expose with the same name.
SHARED_PUBLIC_API = {
    # actions
    "action_deposit",
    "action_withdraw",
    "action_open_position",
    "action_close_position",
    # state
    "update_state",
    "internal_state",
    "global_state",
    # base entity machinery (inherited from BaseEntity)
    "execute",
    "get_available_actions",
    # readouts
    "balance",
    "is_position",
    "stable_amount",
    "volatile_amount",
    "entry_stable_amount",
    "entry_volatile_amount",
    "hodl_value",
    "impermanent_loss",
    "calculate_fees",
    "effective_fee_rate",
    # config-derived
    "pool_fee_rate",
    "slippage_pct",
    "token0_decimals",
    "token1_decimals",
    "notional_side",
}

# Members allowed only on V3 (range/tick math).
V3_ONLY_PUBLIC = {
    "is_in_range",
    "price_to_tick",
    "tick_to_price",
}


def _public_members(cls):
    """Public members exposed on a class instance.

    Instantiates with default config so instance-level attributes (config-derived)
    show up alongside class-level methods/properties.
    """
    if cls is UniswapV2LPEntity:
        instance = cls(UniswapV2LPConfig())
    else:
        instance = cls(UniswapV3LPConfig())
    return {m for m in dir(instance) if not m.startswith("_") and not m.isupper()}


@pytest.mark.core
def test_v2_exposes_all_shared_public_api():
    members = _public_members(UniswapV2LPEntity)
    missing = SHARED_PUBLIC_API - members
    assert not missing, f"V2 missing public members: {missing}"


@pytest.mark.core
def test_v3_exposes_all_shared_public_api():
    members = _public_members(UniswapV3LPEntity)
    missing = SHARED_PUBLIC_API - members
    assert not missing, f"V3 missing public members: {missing}"


@pytest.mark.core
def test_v3_extras_are_only_on_v3():
    v2_members = _public_members(UniswapV2LPEntity)
    v3_only_in_v2 = V3_ONLY_PUBLIC & v2_members
    assert not v3_only_in_v2, (
        f"V3-only members leaked into V2: {v3_only_in_v2}"
    )


@pytest.mark.core
def test_v3_has_all_v3_specific_members():
    members = _public_members(UniswapV3LPEntity)
    missing = V3_ONLY_PUBLIC - members
    assert not missing, f"V3 missing V3-only members: {missing}"


@pytest.mark.core
def test_no_unexpected_public_api_members_v2():
    """Catch silent additions to V2's public surface."""
    members = _public_members(UniswapV2LPEntity)
    # Allow shared API + standard Python/object members.
    extras = members - SHARED_PUBLIC_API - V3_ONLY_PUBLIC
    # Whitelist is empty; if you add a public member, update SHARED_PUBLIC_API.
    assert not extras, (
        f"V2 has unexpected public members: {extras}. "
        f"If intentional, add them to SHARED_PUBLIC_API or document why V2-only."
    )


@pytest.mark.core
def test_no_unexpected_public_api_members_v3():
    members = _public_members(UniswapV3LPEntity)
    extras = members - SHARED_PUBLIC_API - V3_ONLY_PUBLIC
    assert not extras, (
        f"V3 has unexpected public members: {extras}. "
        f"If intentional, update the whitelist."
    )


@pytest.mark.core
def test_v2_v3_configs_have_same_fields():
    """Both configs must expose identical field names. V3 may have extras
    (currently none — kept symmetric)."""
    v2_fields = {f.name for f in fields(UniswapV2LPConfig)}
    v3_fields = {f.name for f in fields(UniswapV3LPConfig)}
    assert v2_fields == v3_fields, (
        f"Config fields differ — V2-only: {v2_fields - v3_fields}, "
        f"V3-only: {v3_fields - v2_fields}"
    )


@pytest.mark.core
def test_configs_have_same_defaults():
    """Default values for shared fields must match (so users get the same
    behaviour by default regardless of which entity they instantiate)."""
    v2_defaults = {f.name: f.default for f in fields(UniswapV2LPConfig)}
    v3_defaults = {f.name: f.default for f in fields(UniswapV3LPConfig)}
    common = set(v2_defaults) & set(v3_defaults)
    for k in common:
        assert v2_defaults[k] == v3_defaults[k], (
            f"Default for {k!r} differs: V2={v2_defaults[k]}, V3={v3_defaults[k]}"
        )


SHARED_INTERNAL_FIELDS = {
    "token0_amount",
    "token1_amount",
    "entry_token0_amount",
    "entry_token1_amount",
    "price_init",
    "liquidity",
    "cash",
}

V3_ONLY_INTERNAL_FIELDS = {
    "price_lower",
    "price_upper",
}


@pytest.mark.core
def test_v2_internal_state_fields_match_shared_set():
    v2_fields = {f.name for f in fields(UniswapV2LPInternalState)}
    assert v2_fields == SHARED_INTERNAL_FIELDS, (
        f"V2 internal state fields differ from shared set — "
        f"missing: {SHARED_INTERNAL_FIELDS - v2_fields}, "
        f"extra: {v2_fields - SHARED_INTERNAL_FIELDS}"
    )


@pytest.mark.core
def test_v3_internal_state_fields_match_shared_plus_range():
    v3_fields = {f.name for f in fields(UniswapV3LPInternalState)}
    expected = SHARED_INTERNAL_FIELDS | V3_ONLY_INTERNAL_FIELDS
    assert v3_fields == expected, (
        f"V3 internal state fields don't match expected — "
        f"missing: {expected - v3_fields}, "
        f"extra: {v3_fields - expected}"
    )


@pytest.mark.core
def test_global_states_inherit_from_base():
    assert issubclass(UniswapV2LPGlobalState, BasePoolGlobalState)
    assert issubclass(UniswapV3LPGlobalState, BasePoolGlobalState)


@pytest.mark.core
def test_v2_v3_global_states_have_same_fields():
    """Both inherit from BasePoolGlobalState and add nothing — so fields match."""
    v2_fields = {f.name for f in fields(UniswapV2LPGlobalState)}
    v3_fields = {f.name for f in fields(UniswapV3LPGlobalState)}
    assert v2_fields == v3_fields


@pytest.mark.core
def test_both_entities_produce_same_balance_for_zero_position():
    """Just-deposited entities should return identical balance (cash only)."""
    v2 = UniswapV2LPEntity(UniswapV2LPConfig())
    v3 = UniswapV3LPEntity(UniswapV3LPConfig())
    v2.action_deposit(500)
    v3.action_deposit(500)
    assert v2.balance == v3.balance == 500


@pytest.mark.core
def test_both_entities_have_zero_il_when_no_position():
    v2 = UniswapV2LPEntity(UniswapV2LPConfig())
    v3 = UniswapV3LPEntity(UniswapV3LPConfig())
    assert v2.impermanent_loss == 0
    assert v3.impermanent_loss == 0


@pytest.mark.core
def test_both_entities_have_zero_fees_when_no_position():
    v2 = UniswapV2LPEntity(UniswapV2LPConfig())
    v3 = UniswapV3LPEntity(UniswapV3LPConfig())
    assert v2.calculate_fees() == 0
    assert v3.calculate_fees() == 0


@pytest.mark.core
def test_both_entities_effective_fee_rate_combines_pool_and_slippage():
    """Identity ``effective_fee_rate == pool_fee_rate + slippage_pct``."""
    v2 = UniswapV2LPEntity(UniswapV2LPConfig(pool_fee_rate=0.003, slippage_pct=0.002))
    v3 = UniswapV3LPEntity(UniswapV3LPConfig(pool_fee_rate=0.003, slippage_pct=0.002))
    assert v2.effective_fee_rate == 0.005
    assert v3.effective_fee_rate == 0.005


@pytest.mark.core
def test_both_entities_have_pair_helpers():
    """Both expose ``_open_from_pair`` and ``_close_to_pair`` for advanced use."""
    assert hasattr(UniswapV2LPEntity, "_open_from_pair")
    assert hasattr(UniswapV3LPEntity, "_open_from_pair")
    assert hasattr(UniswapV2LPEntity, "_close_to_pair")
    assert hasattr(UniswapV3LPEntity, "_close_to_pair")


@pytest.mark.core
def test_both_entities_have_slot_set_helpers():
    assert hasattr(UniswapV2LPEntity, "_set_position_amounts")
    assert hasattr(UniswapV3LPEntity, "_set_position_amounts")
    assert hasattr(UniswapV2LPEntity, "_set_entry_amounts")
    assert hasattr(UniswapV3LPEntity, "_set_entry_amounts")
