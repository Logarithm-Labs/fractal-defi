"""Tests for protocol_fee (slot0.feeProtocol split) and exact-pair entry.

Motivated by Base mainnet, where V3 pools run with the protocol-fee
switch ON (``slot0.feeProtocol = 4|4`` → the protocol takes 25%):
a fee model that ignores it overstates LP income by 1/0.75.
"""
import pytest

from fractal.core.base.entity import EntityException
from fractal.core.entities.protocols.uniswap_v3_lp import UniswapV3LPConfig, UniswapV3LPEntity, UniswapV3LPGlobalState


def make_entity(**config_kwargs) -> UniswapV3LPEntity:
    return UniswapV3LPEntity(config=UniswapV3LPConfig(**config_kwargs))


@pytest.mark.core
@pytest.mark.parametrize("bad_fee", [-0.25, 1.0, 1.5])
def test_protocol_fee_validation(bad_fee):
    with pytest.raises(EntityException):
        make_entity(protocol_fee=bad_fee)


@pytest.mark.core
def test_protocol_fee_scales_accrued_fees():
    """Same position, same pool bars — fees scale with 1 − protocol_fee."""
    accrued = {}
    for share in (1.0, 0.75, 0.5):
        entity = make_entity(pool_fee_rate=0.0, protocol_fee=1 - share)
        entity.update_state(UniswapV3LPGlobalState(price=1.0, tvl=1_000_000))
        entity.action_deposit(1000)
        entity.action_open_position(1000, 0.9, 1.1)
        cash_after_open = entity.internal_state.cash
        entity.update_state(UniswapV3LPGlobalState(
            price=1.0, tvl=1_000_000, fees=100.0, liquidity=1_000_000,
        ))
        accrued[share] = entity.internal_state.cash - cash_after_open

    assert accrued[1.0] > 0
    assert accrued[0.75] == pytest.approx(accrued[1.0] * 0.75, rel=1e-12)
    assert accrued[0.5] == pytest.approx(accrued[1.0] * 0.5, rel=1e-12)


@pytest.mark.core
def test_protocol_fee_default_keeps_legacy_behavior():
    """Default config (protocol_fee=0.0) must accrue exactly as before."""
    legacy = make_entity(pool_fee_rate=0.0)
    assert legacy.protocol_fee == 0.0


@pytest.mark.core
def test_open_position_from_pair_exact_mint():
    """Pair-mode mint replicates a real mint: same amounts, no swap, no fee.

    In-range at p=1.0 over [0.9, 1.1] the V3 split is symmetric, so equal
    token amounts enter fully with ~zero leftover.
    """
    entity = make_entity(pool_fee_rate=0.003)  # fee must NOT apply on mint
    entity.update_state(UniswapV3LPGlobalState(price=1.0, tvl=1_000_000))
    entity.action_open_position_from_pair(
        token0_amount=100.0, token1_amount=100.0,
        price_lower=1.0 * 1.0001 ** -1000, price_upper=1.0 * 1.0001 ** 1000,
    )
    assert entity.is_position
    assert entity.internal_state.token0_amount == pytest.approx(100.0, rel=1e-9)
    assert entity.internal_state.token1_amount == pytest.approx(100.0, rel=1e-9)
    assert entity.internal_state.cash == pytest.approx(0.0, abs=1e-9)
    # Balance = both legs at p=1.0, nothing lost to fees on the mint.
    assert entity.balance == pytest.approx(200.0, rel=1e-9)


@pytest.mark.core
def test_fee_growth_mode_per_leg_accrual():
    """feeGrowth deltas accrue per-leg fees: delta × L_onchain diluted by
    the pool share L_pool/(L_pool+L_onchain), scaled to human units,
    cumulated in fees_token0/1 and cashed at the bar price. protocol_fee
    must NOT apply (counter is already LP-net)."""
    entity = make_entity(
        pool_fee_rate=0.0, protocol_fee=0.5,  # must be ignored
        token0_decimals=6, token1_decimals=6, notional_side="token1",
    )
    entity.update_state(UniswapV3LPGlobalState(price=1.0, tvl=1.0))
    entity.action_open_position_from_pair(
        token0_amount=100.0, token1_amount=100.0,
        price_lower=1.0 * 1.0001 ** -1000, price_upper=1.0 * 1.0001 ** 1000,
    )
    liq_onchain = entity.internal_state.liquidity * 10 ** 6  # 10^((6+6)/2)
    pool_liq = 1e18
    dilution = pool_liq / (pool_liq + liq_onchain)
    fg0, fg1 = 3e-4, 5e-4  # raw token per unit L
    entity.update_state(UniswapV3LPGlobalState(
        price=1.0, tvl=1.0, fees=999.0, liquidity=pool_liq,  # aggregate must be ignored
        fee_growth0=fg0, fee_growth1=fg1,
    ))
    expected0 = fg0 * liq_onchain / 10 ** 6 * dilution
    expected1 = fg1 * liq_onchain / 10 ** 6 * dilution
    assert entity.internal_state.fees_token0 == pytest.approx(expected0, rel=1e-12)
    assert entity.internal_state.fees_token1 == pytest.approx(expected1, rel=1e-12)
    # notional = token1: cash = fees1 + fees0 × price
    assert entity.internal_state.cash == pytest.approx(expected1 + expected0 * 1.0, rel=1e-12)


@pytest.mark.core
def test_fee_growth_dilution_bounds_large_position():
    """A position comparable to the pool cannot earn more than the
    counterfactual pool-share of the bar's fees: delta × L is diluted by
    L_pool/(L_pool+L_pos), halving the naive credit when L_pos == L_pool."""
    entity = make_entity(pool_fee_rate=0.0, token0_decimals=6, token1_decimals=6)
    entity.update_state(UniswapV3LPGlobalState(price=1.0, tvl=1.0))
    entity.action_open_position_from_pair(
        token0_amount=100.0, token1_amount=100.0,
        price_lower=1.0 * 1.0001 ** -1000, price_upper=1.0 * 1.0001 ** 1000,
    )
    liq_onchain = entity.internal_state.liquidity * 10 ** 6
    fg0 = 1e-3
    entity.update_state(UniswapV3LPGlobalState(
        price=1.0, tvl=1.0, liquidity=liq_onchain,  # pool L == position L
        fee_growth0=fg0, fee_growth1=0.0,
    ))
    naive = fg0 * liq_onchain / 10 ** 6
    assert entity.internal_state.fees_token0 == pytest.approx(naive / 2, rel=1e-12)


@pytest.mark.core
def test_fee_growth_zero_delta_bar_stays_on_growth_path():
    """A bar that CARRIES feeGrowth data with 0.0 deltas must not fall
    back to the aggregate estimate in auto mode (presence-based switch):
    the pool earned nothing that bar, so the position accrues nothing."""
    entity = make_entity(pool_fee_rate=0.0, token0_decimals=6, token1_decimals=6)
    entity.update_state(UniswapV3LPGlobalState(price=1.0, tvl=1.0))
    entity.action_deposit(1000)
    entity.action_open_position(1000, 0.9, 1.1)
    cash0 = entity.internal_state.cash
    entity.update_state(UniswapV3LPGlobalState(
        price=1.0, tvl=1.0, fees=100.0, liquidity=1_000_000,  # aggregate bait
        fee_growth0=0.0, fee_growth1=0.0,  # data present, zero accrual
    ))
    assert entity.internal_state.cash == cash0


@pytest.mark.core
def test_negative_fee_growth_rejected_before_mutation():
    """Negative deltas are invalid data: EntityException from the
    validate-then-mutate block, with the entity left untouched."""
    entity = make_entity(pool_fee_rate=0.0, token0_decimals=6, token1_decimals=6)
    entity.update_state(UniswapV3LPGlobalState(price=1.0, tvl=1.0))
    entity.action_open_position_from_pair(
        token0_amount=100.0, token1_amount=100.0,
        price_lower=1.0 * 1.0001 ** -1000, price_upper=1.0 * 1.0001 ** 1000,
    )
    price_before = entity.global_state.price
    amounts_before = (entity.internal_state.token0_amount, entity.internal_state.token1_amount)
    with pytest.raises(EntityException):
        entity.update_state(UniswapV3LPGlobalState(
            price=2.0, tvl=1.0, fee_growth0=-1e-9, fee_growth1=1e-9,
        ))
    assert entity.global_state.price == price_before  # state not swapped
    assert (entity.internal_state.token0_amount,
            entity.internal_state.token1_amount) == amounts_before


@pytest.mark.core
def test_fee_growth_mode_no_accrual_out_of_range():
    entity = make_entity(pool_fee_rate=0.0, token0_decimals=6, token1_decimals=6)
    entity.update_state(UniswapV3LPGlobalState(price=1.0, tvl=1.0))
    entity.action_open_position_from_pair(
        token0_amount=100.0, token1_amount=100.0,
        price_lower=1.0 * 1.0001 ** -1000, price_upper=1.0 * 1.0001 ** 1000,
    )
    cash_after_open = entity.internal_state.cash  # mint float-dust leftover
    entity.update_state(UniswapV3LPGlobalState(
        price=2.0, tvl=1.0, fee_growth0=1e-3, fee_growth1=1e-3,  # price above range
    ))
    assert entity.internal_state.fees_token0 == 0.0
    assert entity.internal_state.fees_token1 == 0.0
    assert entity.internal_state.cash == cash_after_open


@pytest.mark.core
def test_tick_price_statics_decimals_aware():
    """Entity tick/price converters are static and accept decimals;
    defaults keep the legacy raw (decimals-less) behavior."""
    assert UniswapV3LPEntity.tick_to_price(0) == pytest.approx(1.0)
    assert UniswapV3LPEntity.tick_to_price(0, 18, 6) == pytest.approx(1e12)
    assert UniswapV3LPEntity.price_to_tick(1.0) == 0
    assert UniswapV3LPEntity.price_to_tick(1e12, 18, 6) == 0


@pytest.mark.core
def test_fee_model_validation():
    with pytest.raises(EntityException):
        make_entity(fee_model="bogus")


@pytest.mark.core
def test_fee_model_aggregate_ignores_growth_fields():
    """fee_model='aggregate' must use the fees-share estimate even when
    the observation carries feeGrowth deltas."""
    entity = make_entity(pool_fee_rate=0.0, fee_model="aggregate",
                         token0_decimals=6, token1_decimals=6)
    entity.update_state(UniswapV3LPGlobalState(price=1.0, tvl=1.0))
    entity.action_deposit(1000)
    entity.action_open_position(1000, 0.9, 1.1)
    cash0 = entity.internal_state.cash
    entity.update_state(UniswapV3LPGlobalState(
        price=1.0, tvl=1.0, fees=100.0, liquidity=1_000_000,
        fee_growth0=1e-3, fee_growth1=1e-3,
    ))
    assert entity.internal_state.fees_token0 == 0.0  # growth path not taken
    assert entity.internal_state.cash > cash0        # aggregate path accrued


@pytest.mark.core
def test_fee_model_fee_growth_ignores_aggregate():
    """fee_model='fee_growth' accrues nothing on bars without deltas,
    even if aggregate ``fees`` are present."""
    entity = make_entity(pool_fee_rate=0.0, fee_model="fee_growth",
                         token0_decimals=6, token1_decimals=6)
    entity.update_state(UniswapV3LPGlobalState(price=1.0, tvl=1.0))
    entity.action_deposit(1000)
    entity.action_open_position(1000, 0.9, 1.1)
    cash0 = entity.internal_state.cash
    entity.update_state(UniswapV3LPGlobalState(
        price=1.0, tvl=1.0, fees=100.0, liquidity=1_000_000,
    ))
    assert entity.internal_state.cash == cash0  # no growth data -> no accrual


@pytest.mark.core
def test_no_fee_growth_falls_back_to_aggregate_model():
    """Without feeGrowth deltas the legacy fees-share path still accrues."""
    entity = make_entity(pool_fee_rate=0.0)
    entity.update_state(UniswapV3LPGlobalState(price=1.0, tvl=1.0))
    entity.action_deposit(1000)
    entity.action_open_position(1000, 0.9, 1.1)
    entity.update_state(UniswapV3LPGlobalState(
        price=1.0, tvl=1.0, fees=100.0, liquidity=1_000_000,
    ))
    assert entity.internal_state.cash > 0
    # legacy path cannot attribute legs
    assert entity.internal_state.fees_token0 == 0.0
    assert entity.internal_state.fees_token1 == 0.0


@pytest.mark.core
def test_open_position_from_pair_leftover_to_cash():
    """Unbalanced amounts: the excess leg that can't enter at the current
    ratio returns to cash (volatile leftover converts at price, with fee)."""
    entity = make_entity(pool_fee_rate=0.0, notional_side="token0")
    entity.update_state(UniswapV3LPGlobalState(price=1.0, tvl=1_000_000))
    entity.action_open_position_from_pair(
        token0_amount=200.0, token1_amount=100.0,
        price_lower=1.0 * 1.0001 ** -1000, price_upper=1.0 * 1.0001 ** 1000,
    )
    # Symmetric range → position takes ~100/100; ~100 token0 goes to cash.
    assert entity.internal_state.token1_amount == pytest.approx(100.0, rel=1e-9)
    assert entity.internal_state.cash == pytest.approx(100.0, rel=1e-6)
    assert entity.balance == pytest.approx(300.0, rel=1e-6)
