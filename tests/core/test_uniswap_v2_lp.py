"""Functional tests for UniswapV2LPEntity.

Pool fee model (post-2026 refactor):
* ``pool_fee_rate`` — pool's swap-fee tier; charged on the **swapped**
  portion only (half of notional in V2).
* ``slippage_pct`` — additional execution-cost handwave; default 0.
* Old ``trading_fee`` field is removed (was applied to the full deposit
  on both open AND close — incorrect modeling).
"""
import pytest

from fractal.core.base.entity import EntityException
from fractal.core.entities.protocols.uniswap_v2_lp import UniswapV2LPConfig, UniswapV2LPEntity, UniswapV2LPGlobalState


@pytest.fixture
def uniswap_lp_entity():
    config = UniswapV2LPConfig(pool_fee_rate=0.003, slippage_pct=0.0,
                               token0_decimals=6, token1_decimals=18)
    entity = UniswapV2LPEntity(config=config)
    entity.update_state(UniswapV2LPGlobalState(tvl=10_000, liquidity=10_000,
                                               fees=0, price=1000, volume=0))
    return entity


@pytest.mark.core
def test_action_deposit(uniswap_lp_entity):
    uniswap_lp_entity.action_deposit(1000)
    assert uniswap_lp_entity.balance == 1000


@pytest.mark.core
def test_action_withdraw(uniswap_lp_entity):
    uniswap_lp_entity.action_deposit(1000)
    uniswap_lp_entity.action_withdraw(500)
    assert uniswap_lp_entity.balance == 500


@pytest.mark.core
def test_action_open_position(uniswap_lp_entity):
    """Zap-in: half stays as stable, half swaps to volatile (pays fee on the swap).

    With ``pool_fee_rate=0.003`` on a 500-notional zap-in into a pool with
    ``tvl=10_000, liquidity=10_000, price=1000``:
    * volatile_at_mint = (250 / 1000) * 0.997 = 0.24925 token1
    * stable_used (limited by volatile) = 0.24925 * 1000 = 249.25 (notional)
    * stable_leftover (returned to cash) = 250 - 249.25 = 0.75
    * cash after open = 1000 - 500 + 0.75 = 500.75
    * share = 0.24925 / 5 = 0.04985 → liquidity = 0.04985 * 10000 = 498.5
    * balance = 249.25 + 0.24925 * 1000 + 500.75 = 999.25
    Effective execution cost = 0.75 = pool_fee × half (only the swapped portion).
    """
    uniswap_lp_entity.action_deposit(1000)
    uniswap_lp_entity.action_open_position(500)
    assert uniswap_lp_entity.is_position is True
    assert uniswap_lp_entity._internal_state.token0_amount == pytest.approx(249.25)
    assert uniswap_lp_entity._internal_state.token1_amount == pytest.approx(0.24925)
    assert uniswap_lp_entity._internal_state.price_init == 1000
    assert uniswap_lp_entity._internal_state.liquidity == pytest.approx(498.5)
    assert uniswap_lp_entity.balance == pytest.approx(999.25)


@pytest.mark.core
def test_action_close_position(uniswap_lp_entity):
    """Zap-out: stable side returns at full value, volatile swaps back at pool fee.

    Position from zap-in: stable=249.25, volatile=0.24925, cash=500.75.
    On close: stable_back=249.25 (no swap), volatile_proceeds = 0.24925 * 1000 * 0.997 = 248.50225.
    Cash after close = 500.75 + 249.25 + 248.50225 = 998.50225.
    Round-trip cost = 1000 - 998.50225 = 1.49775 = 0.75 (open: half × fee) + 0.74775 (close: volatile_value × fee).
    """
    uniswap_lp_entity.action_deposit(1000)
    uniswap_lp_entity.action_open_position(500)
    uniswap_lp_entity.action_close_position()
    assert uniswap_lp_entity.is_position is False
    assert uniswap_lp_entity.balance == pytest.approx(998.50225)


@pytest.mark.core
def test_update_state(uniswap_lp_entity):
    state = UniswapV2LPGlobalState(tvl=20_000, liquidity=20_000, price=2000, fees=50, volume=0)
    uniswap_lp_entity.update_state(state)
    assert uniswap_lp_entity._global_state == state


@pytest.mark.core
def test_balance(uniswap_lp_entity):
    uniswap_lp_entity.action_deposit(1000)
    assert uniswap_lp_entity.balance == 1000


@pytest.mark.core
def test_calculate_fees(uniswap_lp_entity):
    uniswap_lp_entity.action_deposit(1000)
    uniswap_lp_entity.action_open_position(500)
    updated_state = UniswapV2LPGlobalState(tvl=10_000, liquidity=10_000, price=1000, fees=100, volume=0)
    uniswap_lp_entity.update_state(updated_state)
    fees_earned = uniswap_lp_entity.calculate_fees()
    assert fees_earned >= 0
    assert fees_earned <= updated_state.fees
    share = uniswap_lp_entity._internal_state.liquidity / updated_state.liquidity
    expected_fees = share * updated_state.fees
    assert fees_earned == pytest.approx(expected_fees)


# ─── fees_compounding_model tests ──────────────────────────────────

def _open_lender(model: str = "cash") -> UniswapV2LPEntity:
    """Helper: open a 500-notional V2 LP under the chosen fee model."""
    cfg = UniswapV2LPConfig(
        pool_fee_rate=0.003, slippage_pct=0.0,
        token0_decimals=6, token1_decimals=18,
        fees_compounding_model=model,
    )
    entity = UniswapV2LPEntity(config=cfg)
    entity.update_state(UniswapV2LPGlobalState(
        tvl=10_000, liquidity=10_000, fees=0, price=1000, volume=0,
    ))
    entity.action_deposit(1000)
    entity.action_open_position(500)
    return entity


@pytest.mark.core
def test_fees_compounding_model_default_is_cash():
    """Backward compat: default config keeps the legacy cash-flow behaviour."""
    cfg = UniswapV2LPConfig()
    assert cfg.fees_compounding_model == "cash"


@pytest.mark.core
def test_fees_compounding_model_invalid_raises():
    cfg = UniswapV2LPConfig(fees_compounding_model="foo")  # type: ignore[arg-type]
    with pytest.raises(EntityException, match="fees_compounding_model"):
        UniswapV2LPEntity(config=cfg)


@pytest.mark.core
def test_cash_mode_routes_fees_to_cash():
    """cash mode: fees accumulate in ``cash``; position tokens unchanged
    relative to a fees=0 update.
    """
    e = _open_lender(model="cash")
    cash_before = e.internal_state.cash
    t0_before = e.internal_state.token0_amount
    t1_before = e.internal_state.token1_amount
    # Pre-fee tvl=10_000, fees=100 ⇒ on-chain post-fee reserves = 10_100,
    # but the entity sees pre-fee tvl per the loader contract.
    e.update_state(UniswapV2LPGlobalState(
        tvl=10_000, liquidity=10_000, fees=100, price=1000, volume=0,
    ))
    expected_fee_share = e.internal_state.liquidity / 10_000 * 100
    assert e.internal_state.cash == pytest.approx(cash_before + expected_fee_share)
    # Token amounts stay at the share-based formula (no compound buffer).
    assert e.internal_state.token0_amount == pytest.approx(t0_before)
    assert e.internal_state.token1_amount == pytest.approx(t1_before)
    assert e.internal_state.compounded_token0_amount == 0.0
    assert e.internal_state.compounded_token1_amount == 0.0


@pytest.mark.core
def test_compound_mode_routes_fees_to_position():
    """compound mode: fees grow ``token0/token1_amount`` (split 50/50 by
    value) and live in the cumulative buffer; ``cash`` is unchanged.
    """
    e = _open_lender(model="compound")
    cash_before = e.internal_state.cash
    e.update_state(UniswapV2LPGlobalState(
        tvl=10_000, liquidity=10_000, fees=100, price=1000, volume=0,
    ))
    bar_fee_value = e.internal_state.liquidity / 10_000 * 100
    expected_t0_buffer = bar_fee_value / 2                  # stable side (token0)
    expected_t1_buffer = (bar_fee_value / 2) / 1000          # volatile side (token1) at price=1000
    assert e.internal_state.cash == pytest.approx(cash_before)
    assert e.internal_state.compounded_token0_amount == pytest.approx(expected_t0_buffer)
    assert e.internal_state.compounded_token1_amount == pytest.approx(expected_t1_buffer)
    # Liquidity (LP-token count) unchanged — compound is implicit, not a mint.
    assert e.internal_state.liquidity == pytest.approx(498.5)


@pytest.mark.core
def test_total_balance_identical_across_modes_at_constant_price():
    """The total ``balance`` matches between cash and compound modes —
    fees just live in different buckets. Verified across 50 bars at a
    constant price + constant per-bar fees.
    """
    cash_e = _open_lender(model="cash")
    comp_e = _open_lender(model="compound")
    state = UniswapV2LPGlobalState(
        tvl=10_000, liquidity=10_000, fees=10, price=1000, volume=0,
    )
    for _ in range(50):
        cash_e.update_state(state)
        comp_e.update_state(state)
    assert cash_e.balance == pytest.approx(comp_e.balance, rel=1e-9)


@pytest.mark.core
def test_compound_mode_buffer_accumulates_across_bars():
    """Compound buffer grows monotonically with each bar's fees."""
    e = _open_lender(model="compound")
    state = UniswapV2LPGlobalState(
        tvl=10_000, liquidity=10_000, fees=10, price=1000, volume=0,
    )
    snapshots = []
    for _ in range(5):
        e.update_state(state)
        snapshots.append(e.internal_state.compounded_token0_amount)
    # Strictly monotonically increasing — each bar adds positive fees.
    assert all(b > a for a, b in zip(snapshots, snapshots[1:]))


@pytest.mark.core
def test_close_position_resets_compound_buffer():
    """``action_close_position`` must zero the compound buffer alongside
    everything else, so a subsequent ``open_position`` starts clean.
    """
    e = _open_lender(model="compound")
    e.update_state(UniswapV2LPGlobalState(tvl=10_000, liquidity=10_000,
                                          fees=100, price=1000, volume=0))
    assert e.internal_state.compounded_token0_amount > 0
    e.action_close_position()
    assert e.internal_state.compounded_token0_amount == 0.0
    assert e.internal_state.compounded_token1_amount == 0.0
    assert e.internal_state.liquidity == 0.0


@pytest.mark.core
def test_compound_mode_il_smaller_than_cash_mode():
    """Under fee accrual, compound mode reports a SMALLER ``impermanent_loss``
    than cash mode — pool yield offsets the price-divergence cost in compound,
    while cash mode keeps ``impermanent_loss`` purely as price divergence.
    """
    cash_e = _open_lender(model="cash")
    comp_e = _open_lender(model="compound")
    # Bar 1: price moves up + non-zero fees.
    state = UniswapV2LPGlobalState(
        tvl=10_000, liquidity=10_000, fees=50, price=1500, volume=0,
    )
    cash_e.update_state(state)
    comp_e.update_state(state)
    # Cash IL is the textbook hodl-vs-LP gap, untouched by fees.
    # Compound IL has the bar's fees absorbed into the position, lowering it.
    assert comp_e.impermanent_loss < cash_e.impermanent_loss
