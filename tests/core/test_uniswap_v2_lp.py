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


def _stream_states(R_0: float, bar_fees: float, bars: int, price: float = 1000):
    """Yield N states modelling the loader contract: bar n's ``tvl`` is
    pre-fee for the bar = ``R_0 + sum(fees over bars 0..n-1)``. This is
    what real loaders emit — without it tests degenerate to constant
    state, masking the cumulative-fee semantics under test.
    """
    for n in range(bars):
        yield UniswapV2LPGlobalState(
            tvl=R_0 + n * bar_fees,
            liquidity=10_000, fees=bar_fees, price=price, volume=0,
        )


@pytest.mark.core
def test_cash_mode_position_constant_at_constant_price():
    """In cash mode, position USD value stays equal to its no-fee
    baseline at constant price — fee yield lives in cash, not in the
    position. After 10 bars, position value is unchanged from open-time
    (only price divergence would move it).
    """
    e = _open_lender(model="cash")
    pos_at_open = e.stable_amount + e.volatile_amount * e._global_state.price
    for state in _stream_states(R_0=10_000, bar_fees=10, bars=10):
        e.update_state(state)
    pos_after = e.stable_amount + e.volatile_amount * e._global_state.price
    assert pos_after == pytest.approx(pos_at_open, rel=1e-9)
    # Cumulative tracker = sum(share * fees) over the bars.
    expected_cumulative = 10 * (e.internal_state.liquidity / 10_000) * 10
    assert e.internal_state.cumulative_position_fees == pytest.approx(expected_cumulative)


@pytest.mark.core
def test_compound_mode_position_grows_each_bar():
    """In compound mode, position USD grows each bar by ``share * fees``;
    cash and cumulative-fee tracker remain at their open-time values.
    """
    e = _open_lender(model="compound")
    cash_at_open = e.internal_state.cash
    pos_history = []
    for state in _stream_states(R_0=10_000, bar_fees=10, bars=5):
        e.update_state(state)
        pos_history.append(e.stable_amount + e.volatile_amount * e._global_state.price)
    # Strictly monotonically increasing — each bar adds share * fees to position.
    assert all(b > a for a, b in zip(pos_history, pos_history[1:]))
    # Cash + cumulative tracker untouched (no separate accounting in compound).
    assert e.internal_state.cash == pytest.approx(cash_at_open)
    assert e.internal_state.cumulative_position_fees == 0.0


@pytest.mark.core
def test_both_modes_match_onchain_truth_at_constant_price():
    """The crucial double-count regression test.

    Pre-Path-B implementation drifted upward by ``share * sum_prior_fees``
    each bar — the bug the reviewer flagged. Path B's invariant: balance
    after N bars = balance-at-open + N · share · bar_fees, exact, in both
    modes (the per-bar fee yield is the only source of growth at
    constant price).
    """
    bars = 50
    bar_fees = 10.0
    R_0 = 10_000.0
    cash_e = _open_lender(model="cash")
    comp_e = _open_lender(model="compound")
    open_balance = cash_e.balance     # both modes start identical at open
    assert comp_e.balance == pytest.approx(open_balance, rel=1e-12)

    for state in _stream_states(R_0=R_0, bar_fees=bar_fees, bars=bars):
        cash_e.update_state(state)
        comp_e.update_state(state)

    share = cash_e.internal_state.liquidity / 10_000     # ≈ 0.04985
    expected_growth = bars * share * bar_fees
    expected_balance = open_balance + expected_growth

    assert cash_e.balance == pytest.approx(expected_balance, rel=1e-9)
    assert comp_e.balance == pytest.approx(expected_balance, rel=1e-9)
    # Modes match each other exactly — the deeper invariant.
    assert cash_e.balance == pytest.approx(comp_e.balance, rel=1e-12)


@pytest.mark.core
def test_close_position_resets_cumulative_fees():
    """``action_close_position`` must zero ``cumulative_position_fees``
    so a subsequent ``open_position`` starts clean.
    """
    e = _open_lender(model="cash")
    e.update_state(UniswapV2LPGlobalState(
        tvl=10_000, liquidity=10_000, fees=100, price=1000, volume=0,
    ))
    assert e.internal_state.cumulative_position_fees > 0
    e.action_close_position()
    assert e.internal_state.cumulative_position_fees == 0.0
    assert e.internal_state.liquidity == 0.0


@pytest.mark.core
def test_compound_mode_il_smaller_than_cash_mode_with_fee_accrual():
    """Under fee accrual + price move, compound IL < cash IL: in
    compound mode fees grow position tokens directly, narrowing the
    hodl-vs-LP gap; in cash mode fees flow to cash so position stays
    purely price-divergence-driven and IL is the textbook (larger)
    figure.
    """
    cash_e = _open_lender(model="cash")
    comp_e = _open_lender(model="compound")
    state = UniswapV2LPGlobalState(
        tvl=10_000, liquidity=10_000, fees=50, price=1500, volume=0,
    )
    cash_e.update_state(state)
    comp_e.update_state(state)
    assert comp_e.impermanent_loss < cash_e.impermanent_loss
