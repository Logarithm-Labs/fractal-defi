"""Tests for ``BasisTradingStrategy`` — Layers 1 (unit) and 2 (synthetic
end-to-end).

The strategy hedges a long *spot* position with a short *perp* position
to harvest funding while staying market-neutral. ``_rebalance`` has four
branches (liquidated, delta_spot>0, delta_spot<0, no-op) so we cover
each one independently before adding integration scenarios.

Hedge entity: :class:`SimplePerpEntity` (clean liquidation math).
Spot entity:  :class:`UniswapV3SpotEntity` (its global state has the
``price`` field that ``BasisTradingStrategy`` reads directly).
"""
from datetime import datetime, timedelta
from typing import List

import pytest

from fractal.core.base import NamedEntity, Observation
from fractal.core.entities.protocols.uniswap_v3_spot import (
    UniswapV3SpotEntity, UniswapV3SpotGlobalState,
)
from fractal.core.entities.simple.perp import (
    SimplePerpEntity, SimplePerpGlobalState,
)
from fractal.strategies.basis_trading_strategy import (
    BasisTradingStrategy, BasisTradingStrategyException,
    BasisTradingStrategyHyperparams,
)


# ============================================================ test scaffolding


class _TestableBasis(BasisTradingStrategy):
    """Concrete subclass that registers HEDGE / SPOT entities.

    ``BasisTradingStrategy.set_up`` only asserts on registered entities;
    real subclasses (e.g. ``HyperliquidBasis``) register before calling
    ``super().set_up()``. Mirror that pattern with deterministic, simple
    entities for tests.
    """

    HEDGE_TRADING_FEE: float = 0.0
    SPOT_TRADING_FEE: float = 0.0
    HEDGE_MAX_LEVERAGE: float = 20.0

    def set_up(self):
        self.register_entity(NamedEntity(
            'HEDGE',
            SimplePerpEntity(
                trading_fee=self.HEDGE_TRADING_FEE,
                max_leverage=self.HEDGE_MAX_LEVERAGE,
            ),
        ))
        self.register_entity(NamedEntity(
            'SPOT',
            UniswapV3SpotEntity(trading_fee=self.SPOT_TRADING_FEE),
        ))
        super().set_up()


def _make_strategy(
    *,
    target_lev: float = 3.0,
    min_lev: float = 1.0,
    max_lev: float = 5.0,
    initial: float = 100_000.0,
    hedge_fee: float = 0.0,
    spot_fee: float = 0.0,
    hedge_max_lev: float = 20.0,
) -> _TestableBasis:
    cls = type(
        '_S', (_TestableBasis,),
        {'HEDGE_TRADING_FEE': hedge_fee, 'SPOT_TRADING_FEE': spot_fee,
         'HEDGE_MAX_LEVERAGE': hedge_max_lev},
    )
    return cls(params=BasisTradingStrategyHyperparams(
        MIN_LEVERAGE=min_lev, TARGET_LEVERAGE=target_lev,
        MAX_LEVERAGE=max_lev, INITIAL_BALANCE=initial,
    ))


def _obs(t: datetime, price: float, funding: float = 0.0) -> Observation:
    return Observation(
        timestamp=t,
        states={
            'SPOT': UniswapV3SpotGlobalState(price=price),
            'HEDGE': SimplePerpGlobalState(mark_price=price, funding_rate=funding),
        },
    )


def _seed_initial_position(s: _TestableBasis, price: float = 3000.0) -> None:
    """Run the deposit step to bring the strategy into a steady state."""
    t0 = datetime(2024, 1, 1)
    s.step(_obs(t0, price))


# ============================================================ Layer 1.A — _deposit_into_strategy

@pytest.mark.core
def test_deposit_into_strategy_returns_four_actions_in_correct_order():
    s = _make_strategy(target_lev=3.0, initial=100_000.0)
    actions = s._deposit_into_strategy()
    assert len(actions) == 4
    kinds = [(a.entity_name, a.action.action) for a in actions]
    assert kinds == [
        ('SPOT', 'deposit'),
        ('HEDGE', 'deposit'),
        ('SPOT', 'buy'),
        ('HEDGE', 'open_position'),
    ]


@pytest.mark.core
def test_deposit_into_strategy_splits_initial_balance_by_target_leverage():
    """SPOT gets B·L/(1+L); HEDGE gets B/(1+L)."""
    L = 3.0
    B = 100_000.0
    s = _make_strategy(target_lev=L, initial=B)
    actions = s._deposit_into_strategy()
    spot_dep, hedge_dep, spot_buy, hedge_open = actions
    expected_spot = B - B / (1 + L)
    expected_hedge = B / (1 + L)
    assert spot_dep.action.args['amount_in_notional'] == pytest.approx(expected_spot)
    assert hedge_dep.action.args['amount_in_notional'] == pytest.approx(expected_hedge)
    # buy uses the same spot share as the deposit
    assert spot_buy.action.args['amount_in_notional'] == pytest.approx(expected_spot)
    # open_position is delegated (lambda) — resolved at execute time
    assert callable(hedge_open.action.args['amount_in_product'])


@pytest.mark.core
def test_deposit_into_strategy_yields_target_leverage_after_step():
    """After running deposit step on a fresh strategy, hedge leverage
    must equal TARGET_LEVERAGE (within rounding)."""
    s = _make_strategy(target_lev=3.0, initial=100_000.0,
                       hedge_fee=0.0, spot_fee=0.0)
    _seed_initial_position(s, price=3000.0)
    hedge = s.get_entity('HEDGE')
    spot = s.get_entity('SPOT')
    assert hedge.leverage == pytest.approx(3.0, rel=1e-3)
    # And the basis hedge invariant: hedge.size ≈ -spot.amount
    assert hedge.size + spot.internal_state.amount == pytest.approx(0, abs=1e-9)


# ============================================================ Layer 1.B — predict() dispatch

@pytest.mark.core
def test_predict_calls_deposit_when_both_balances_zero():
    s = _make_strategy()
    # Set price so global states are valid.
    s.get_entity('SPOT').update_state(UniswapV3SpotGlobalState(price=3000.0))
    s.get_entity('HEDGE').update_state(SimplePerpGlobalState(mark_price=3000.0))
    actions = s.predict()
    # Same shape as _deposit_into_strategy returns (4 actions).
    assert len(actions) == 4
    assert actions[0].action.action == 'deposit'


@pytest.mark.core
def test_predict_returns_empty_when_leverage_in_bounds():
    s = _make_strategy(min_lev=1.0, target_lev=3.0, max_lev=5.0)
    _seed_initial_position(s, price=3000.0)
    # Right after deposit, leverage ≈ TARGET = 3, which is in [MIN, MAX].
    actions = s.predict()
    assert actions == []


@pytest.mark.core
def test_predict_triggers_rebalance_when_leverage_above_max():
    """A small downward price tick keeps the short hedge alive but
    raises its leverage. Set MAX low enough to trigger."""
    s = _make_strategy(min_lev=2.5, target_lev=3.0, max_lev=3.05,
                      hedge_fee=0.0, spot_fee=0.0)
    _seed_initial_position(s, price=3000.0)
    # Move price up by 1% — short loses, balance shrinks → leverage rises.
    s.get_entity('SPOT').update_state(UniswapV3SpotGlobalState(price=3030.0))
    s.get_entity('HEDGE').update_state(SimplePerpGlobalState(mark_price=3030.0))
    assert s.get_entity('HEDGE').leverage > 3.05
    actions = s.predict()
    assert actions, "expected non-empty rebalance"
    # delta_spot < 0 branch (price up): SPOT.sell first.
    assert actions[0].entity_name == 'SPOT'
    assert actions[0].action.action == 'sell'


@pytest.mark.core
def test_predict_triggers_rebalance_when_hedge_liquidated_and_spot_alive():
    s = _make_strategy(target_lev=3.0, hedge_fee=0.0, spot_fee=0.0)
    _seed_initial_position(s, price=3000.0)
    hedge = s.get_entity('HEDGE')
    # Force a liquidated state (hedge balance == 0 but spot still alive).
    hedge._internal_state.collateral = 0.0
    hedge._internal_state.size = 0.0
    hedge._internal_state.entry_price = 0.0
    assert hedge.balance == 0
    assert s.get_entity('SPOT').balance > 0
    actions = s.predict()
    assert actions, "expected non-empty rebalance after liquidation"


# ============================================================ Layer 1.C — _rebalance branches

def _force_liquidated_state(s: _TestableBasis, price: float = 3000.0) -> None:
    """Bring strategy to: hedge.balance=0 (liquidated), spot.balance>0."""
    _seed_initial_position(s, price=price)
    hedge = s.get_entity('HEDGE')
    hedge._internal_state.collateral = 0.0
    hedge._internal_state.size = 0.0
    hedge._internal_state.entry_price = 0.0


@pytest.mark.core
def test_rebalance_liquidated_branch_returns_four_actions():
    s = _make_strategy(target_lev=3.0)
    _force_liquidated_state(s, price=3000.0)
    actions = s._rebalance()
    assert len(actions) == 4
    spot_actions = [a for a in actions if a.entity_name == 'SPOT']
    hedge_actions = [a for a in actions if a.entity_name == 'HEDGE']
    assert len(spot_actions) == 2  # sell + withdraw
    assert len(hedge_actions) == 2  # deposit + open_position


@pytest.mark.core
def test_rebalance_liquidated_branch_orders_hedge_deposit_before_spot_withdraw():
    """**Lock-in for B-1** — the same delegate ``lambda obj: spot.cash`` is
    used for ``SPOT.withdraw`` and ``HEDGE.deposit``. If ``withdraw``
    runs first, ``spot.cash`` is 0 by the time ``deposit`` resolves the
    delegate, and the deposit fires with amount=0 — a silent equity loss.

    The sibling branch (``delta_spot < 0`` non-liquidated) already orders
    deposit before withdraw with an explanatory comment. The liquidated
    branch must do the same.

    EXPECTED TO FAIL on current code; will turn green after fix.
    """
    s = _make_strategy(target_lev=3.0)
    _force_liquidated_state(s, price=3000.0)
    actions = s._rebalance()

    def _idx(entity: str, action_name: str) -> int:
        for i, a in enumerate(actions):
            if a.entity_name == entity and a.action.action == action_name:
                return i
        raise AssertionError(f"no {entity}.{action_name} in {actions}")

    spot_withdraw_idx = _idx('SPOT', 'withdraw')
    hedge_deposit_idx = _idx('HEDGE', 'deposit')
    assert hedge_deposit_idx < spot_withdraw_idx, (
        "Same delegate reads SPOT.cash; deposit on HEDGE must come BEFORE "
        "withdraw on SPOT, otherwise the deposit sees cash=0."
    )


@pytest.mark.core
def test_rebalance_delta_spot_positive_branch_orders_actions_correctly():
    """Price went down → spot value too low → buy more spot.

    Order: HEDGE.withdraw → SPOT.deposit → SPOT.buy → HEDGE.open_position.
    Amounts are static floats here (not delegates) so order is purely
    sequential dependence on cash flow.
    """
    s = _make_strategy(min_lev=2.5, target_lev=3.0, max_lev=3.05)
    _seed_initial_position(s, price=3000.0)
    # Drop price by 1% → spot value drops, hedge gains, leverage drops below MIN.
    s.get_entity('SPOT').update_state(UniswapV3SpotGlobalState(price=2970.0))
    s.get_entity('HEDGE').update_state(SimplePerpGlobalState(mark_price=2970.0))
    actions = s._rebalance()
    kinds = [(a.entity_name, a.action.action) for a in actions]
    assert kinds == [
        ('HEDGE', 'withdraw'),
        ('SPOT', 'deposit'),
        ('SPOT', 'buy'),
        ('HEDGE', 'open_position'),
    ]


@pytest.mark.core
def test_rebalance_delta_spot_negative_branch_orders_deposit_before_withdraw():
    """Price went up → spot too valuable → sell some, move to hedge.

    Order: SPOT.sell → HEDGE.deposit → SPOT.withdraw → HEDGE.open_position.
    The deposit must precede the withdraw because both share a delegate
    that reads SPOT.cash.
    """
    s = _make_strategy(min_lev=2.5, target_lev=3.0, max_lev=3.05)
    _seed_initial_position(s, price=3000.0)
    # Bump price up → triggers delta_spot < 0 path.
    s.get_entity('SPOT').update_state(UniswapV3SpotGlobalState(price=3030.0))
    s.get_entity('HEDGE').update_state(SimplePerpGlobalState(mark_price=3030.0))
    actions = s._rebalance()
    kinds = [(a.entity_name, a.action.action) for a in actions]
    assert kinds == [
        ('SPOT', 'sell'),
        ('HEDGE', 'deposit'),
        ('SPOT', 'withdraw'),
        ('HEDGE', 'open_position'),
    ]


@pytest.mark.core
def test_rebalance_returns_empty_when_no_delta():
    """Right after deposit, equity is split exactly per TARGET_LEVERAGE
    so ``delta_spot == 0`` and rebalance must be a no-op."""
    s = _make_strategy(target_lev=3.0)
    _seed_initial_position(s, price=3000.0)
    actions = s._rebalance()
    assert actions == []


# ============================================================ Layer 2 — synthetic end-to-end

@pytest.mark.core
def test_e2e_flat_price_no_rebalance():
    """At constant price with zero funding, equity holds and no rebalances fire."""
    s = _make_strategy(target_lev=3.0, hedge_fee=0.0, spot_fee=0.0)
    t0 = datetime(2024, 1, 1)
    obs = [_obs(t0 + timedelta(hours=i), 3000.0) for i in range(5)]
    initial_total = s._params.INITIAL_BALANCE
    s.run(obs)
    # Equity preserved (no fees, no funding, flat price).
    final_total = sum(e.balance for e in s.get_all_available_entities().values())
    assert final_total == pytest.approx(initial_total, rel=1e-9)


@pytest.mark.core
def test_e2e_uptrend_triggers_rebalance_and_keeps_leverage_bounded():
    """Slow uptrend forces rebalance once leverage breaks above MAX."""
    s = _make_strategy(min_lev=2.0, target_lev=3.0, max_lev=4.0,
                       hedge_fee=0.0, spot_fee=0.0)
    t0 = datetime(2024, 1, 1)
    # +0.5% steps for 20 ticks — drives short underwater gradually.
    obs = [_obs(t0 + timedelta(hours=i), 3000.0 * (1 + 0.005 * i))
           for i in range(20)]
    s.run(obs)
    hedge = s.get_entity('HEDGE')
    # Final leverage should be within [MIN, MAX] (right after rebalance,
    # leverage snaps back to TARGET).
    assert s._params.MIN_LEVERAGE <= hedge.leverage <= s._params.MAX_LEVERAGE * 1.05


@pytest.mark.core
def test_e2e_downtrend_triggers_rebalance_when_leverage_drops_below_min():
    """Slow downtrend lifts hedge balance, lowers leverage below MIN → rebalance."""
    s = _make_strategy(min_lev=2.5, target_lev=3.0, max_lev=4.0,
                       hedge_fee=0.0, spot_fee=0.0)
    t0 = datetime(2024, 1, 1)
    obs = [_obs(t0 + timedelta(hours=i), 3000.0 * (1 - 0.005 * i))
           for i in range(20)]
    s.run(obs)
    hedge = s.get_entity('HEDGE')
    spot = s.get_entity('SPOT')
    # Basis hedge invariant after final rebalance.
    assert abs(hedge.size + spot.internal_state.amount) / abs(hedge.size) < 0.01


@pytest.mark.core
def test_e2e_hedge_liquidation_then_recovery_no_silent_equity_loss():
    """**Companion to B-1 lock-in.** A 60% price spike liquidates the
    short hedge; the post-liq rebalance must not lose equity beyond the
    realized liquidation loss.

    Math (zero fees, TARGET_LEV=3, INITIAL=100K, price 3000→4800):

    * Step 1: deposit. SPOT 75K → 25 ETH. HEDGE 25K, short 25 ETH.
      Total = 100K.
    * Step 2: spike to 4800. HEDGE pnl = -25·(4800-3000) = -45K →
      balance = -20K → liquidates → balance=0, size=0. SPOT value =
      25·4800 = 120K. **Total = 120K** (post-liquidation).
    * Step 3: liquidated rebalance fires. Strategy must preserve total.

    With B-1 bug, the same delegate reads SPOT.cash, but withdraw
    drains it before deposit on HEDGE resolves — 30K of cash escapes
    into the void. Final total with bug ≈ 90K.

    EXPECTED TO FAIL on current code; will turn green after fix.
    """
    s = _make_strategy(target_lev=3.0, max_lev=5.0,
                       hedge_fee=0.0, spot_fee=0.0)
    t0 = datetime(2024, 1, 1)
    obs = [
        _obs(t0, 3000.0),                       # initial deposit
        _obs(t0 + timedelta(hours=1), 3000.0),  # steady
        _obs(t0 + timedelta(hours=2), 4800.0),  # spike → hedge liquidates
        _obs(t0 + timedelta(hours=3), 4800.0),  # rebalance fires here
        _obs(t0 + timedelta(hours=4), 4800.0),  # settle
    ]
    s.run(obs)
    total = s.total_balance
    # Spot still holds 25 ETH at $4800 = $120K post-liquidation;
    # post-rebalance must preserve that (give or take a few %).
    expected_min = 0.95 * 120_000.0
    assert total >= expected_min, (
        f"post-liquidation rebalance leaked equity: total={total}, "
        f"expected >= {expected_min}"
    )


# ============================================================ S-X3 — invariants raise (not silent under -O)


@pytest.mark.core
def test_set_up_raises_when_hedge_is_not_a_perp_entity():
    """**Lock-in for S-X3.** ``BasisTradingStrategy.set_up`` must
    explicitly raise (not ``assert``) when the registered HEDGE entity
    is the wrong type — ``python -O`` strips asserts.
    """
    from fractal.core.base import NamedEntity
    from fractal.core.entities.protocols.uniswap_v3_spot import UniswapV3SpotEntity

    class _BadBasis(BasisTradingStrategy):
        def set_up(self):
            # HEDGE wired to a spot entity — wrong category.
            self.register_entity(NamedEntity('HEDGE', UniswapV3SpotEntity(trading_fee=0.0)))
            self.register_entity(NamedEntity('SPOT', UniswapV3SpotEntity(trading_fee=0.0)))
            super().set_up()

    with pytest.raises(BasisTradingStrategyException, match="HEDGE must be a BasePerpEntity"):
        _BadBasis(params=BasisTradingStrategyHyperparams(
            MIN_LEVERAGE=1.0, TARGET_LEVERAGE=3.0,
            MAX_LEVERAGE=5.0, INITIAL_BALANCE=100_000.0,
        ))


@pytest.mark.core
def test_set_up_raises_when_spot_is_not_a_spot_entity():
    """Companion to above — wrong SPOT type also raises."""
    from fractal.core.base import NamedEntity
    from fractal.core.entities.simple.perp import SimplePerpEntity

    class _BadBasis(BasisTradingStrategy):
        def set_up(self):
            self.register_entity(NamedEntity('HEDGE', SimplePerpEntity(trading_fee=0.0)))
            # SPOT wired to a perp — wrong category.
            self.register_entity(NamedEntity('SPOT', SimplePerpEntity(trading_fee=0.0)))
            super().set_up()

    with pytest.raises(BasisTradingStrategyException, match="SPOT must be a BaseSpotEntity"):
        _BadBasis(params=BasisTradingStrategyHyperparams(
            MIN_LEVERAGE=1.0, TARGET_LEVERAGE=3.0,
            MAX_LEVERAGE=5.0, INITIAL_BALANCE=100_000.0,
        ))


@pytest.mark.core
def test_rebalance_raises_when_basis_hedge_invariant_violated():
    """**Lock-in for S-X3.** If hedge.size and -spot.amount drift apart
    (e.g. someone closed only one leg outside the strategy), the
    rebalance must raise — not silently produce a bad action plan.
    """
    s = _make_strategy(min_lev=2.5, target_lev=3.0, max_lev=3.05)
    _seed_initial_position(s, price=3000.0)
    # Force drift: kill the hedge short without touching spot.
    hedge = s.get_entity('HEDGE')
    hedge._internal_state.size = -1.0  # was -25, now mismatched
    # Bump price up to trigger rebalance path.
    s.get_entity('SPOT').update_state(UniswapV3SpotGlobalState(price=3030.0))
    s.get_entity('HEDGE').update_state(SimplePerpGlobalState(mark_price=3030.0))
    with pytest.raises(BasisTradingStrategyException, match="basis hedge mismatch"):
        s._rebalance()


@pytest.mark.core
def test_rebalance_basis_invariant_does_not_blow_up_on_zero_zero_state():
    """Edge case the prior ``np.abs / np.abs`` ratio mishandled:
    when hedge.size and spot.amount are both effectively 0,
    ``math.isclose(0, 0)`` is True and the function reaches a clean
    no-op return rather than ``nan <= 1e-6 = False`` → AssertionError.
    """
    s = _make_strategy(target_lev=3.0)
    s.get_entity('SPOT').update_state(UniswapV3SpotGlobalState(price=3000.0))
    s.get_entity('HEDGE').update_state(SimplePerpGlobalState(mark_price=3000.0))
    # Both balances zero AND we set non-zero deposit so the dispatcher
    # in predict() does not route us to ``_deposit_into_strategy``.
    # Direct call to ``_rebalance`` from this state used to produce nan.
    hedge = s.get_entity('HEDGE')
    hedge._internal_state.collateral = 100.0  # break the predict guard
    # spot.amount = 0, hedge.size = 0 → 0/0 in old formula.
    actions = s._rebalance()  # must not raise
    assert isinstance(actions, list)


@pytest.mark.core
def test_rebalance_invariants_are_real_raises_not_asserts():
    """Make sure the lock-ins above test ``raise`` rather than ``assert``
    — by reading the source and confirming no top-level ``assert``
    statements remain in ``_rebalance`` / ``set_up``.

    ``python -O`` strips asserts; raises survive. Catches accidental
    regressions to the old pattern.
    """
    import inspect
    src_rebalance = inspect.getsource(BasisTradingStrategy._rebalance)
    src_set_up = inspect.getsource(BasisTradingStrategy.set_up)
    # Allow ``assert`` only inside string literals or comments — but the
    # crude check is fine because the file currently has none of those.
    assert "assert " not in src_rebalance, (
        "found bare ``assert`` in _rebalance; replace with a raise so "
        "python -O does not strip the safety net"
    )
    assert "assert " not in src_set_up


# ============================================================ B-3 — re-deposit thin-air guard


@pytest.mark.core
def test_full_wipe_after_initial_deposit_raises_instead_of_re_funding():
    """**Lock-in for B-3.** If both balances reach 0 *after* the initial
    deposit, the strategy must refuse to ``_deposit_into_strategy`` again
    — that would silently print ``INITIAL_BALANCE`` worth of equity.
    """
    s = _make_strategy(target_lev=3.0, initial=100_000.0)
    _seed_initial_position(s, price=3000.0)
    # Force a full wipe directly (HEDGE liquidated AND SPOT empty).
    hedge = s.get_entity('HEDGE')
    spot = s.get_entity('SPOT')
    hedge._internal_state.collateral = 0.0
    hedge._internal_state.size = 0.0
    spot._internal_state.amount = 0.0
    spot._internal_state.cash = 0.0
    assert hedge.balance == 0 and spot.balance == 0
    assert s._deposited is True
    with pytest.raises(BasisTradingStrategyException, match="fully wiped"):
        s.predict()


@pytest.mark.core
def test_initial_deposit_flag_flips_to_true_on_first_predict():
    """The ``_deposited`` flag flips on the first ``_deposit_into_strategy`` call."""
    s = _make_strategy()
    assert s._deposited is False
    s.get_entity('SPOT').update_state(UniswapV3SpotGlobalState(price=3000.0))
    s.get_entity('HEDGE').update_state(SimplePerpGlobalState(mark_price=3000.0))
    s.predict()  # both balances 0 → deposit
    assert s._deposited is True


# ============================================================ B-4 — free-margin pre-check on HEDGE.withdraw


@pytest.mark.core
def test_rebalance_raises_clear_error_when_hedge_lacks_free_margin():
    """**Lock-in for B-4.** When delta_spot>0 rebalance would withdraw
    more than the hedge's free margin allows, surface a strategy-level
    error pointing at the configuration — not the generic perp-entity
    "drop below maintenance margin" message.

    Construction: tight ``hedge_max_leverage=2`` makes MM heavy relative
    to balance, then we force a basis-preserved state where the
    rebalance must withdraw more than free margin.
    """
    s = _make_strategy(min_lev=2.5, target_lev=3.0, max_lev=4.0,
                       hedge_fee=0.0, spot_fee=0.0,
                       hedge_max_lev=2.0)  # tight cap → MM dominant
    hedge = s.get_entity('HEDGE')
    spot = s.get_entity('SPOT')
    # Steer state directly: basis hedge holds (size = -amount), but
    # hedge.balance is large enough to overshoot target_hedge while MM
    # keeps free margin below the required withdrawal.
    spot._internal_state.amount = 10.0
    spot._internal_state.cash = 0.0
    spot.update_state(UniswapV3SpotGlobalState(price=3000.0))
    hedge._internal_state.size = -10.0
    hedge._internal_state.entry_price = 3000.0
    hedge._internal_state.collateral = 20_000.0
    hedge.update_state(SimplePerpGlobalState(mark_price=3000.0))
    # Sanity: MM = 10·3000/2 = 15_000; free margin = 5_000.
    assert hedge.maintenance_margin == pytest.approx(15_000.0)
    assert hedge.balance - hedge.maintenance_margin == pytest.approx(5_000.0)
    # Math: equity = 30K (spot) + 20K (hedge) = 50K, target_hedge =
    # 12.5K, delta_hedge = 12.5K - 20K = -7.5K — the rebalance would
    # withdraw 7_500 from a 5_000 free margin.
    with pytest.raises(BasisTradingStrategyException,
                       match=r"free margin is only"):
        s._rebalance()


# ============================================================ B-5/B-6 — abstract signatures


@pytest.mark.core
def test_set_up_signature_matches_base_strategy():
    """B-5: ``set_up`` takes only ``self`` (no ``*args``/``**kwargs``)."""
    import inspect
    params = list(inspect.signature(BasisTradingStrategy.set_up).parameters)
    assert params == ['self']


@pytest.mark.core
def test_predict_signature_matches_base_strategy():
    """B-6: ``predict`` takes only ``self`` (no ``*args``/``**kwargs``)."""
    import inspect
    params = list(inspect.signature(BasisTradingStrategy.predict).parameters)
    assert params == ['self']


@pytest.mark.core
def test_e2e_hedge_liquidation_post_rebalance_size_invariant():
    """After post-liquidation rebalance, hedge.size ≈ -spot.amount again
    (basis hedge restored)."""
    s = _make_strategy(target_lev=3.0, max_lev=5.0,
                       hedge_fee=0.0, spot_fee=0.0)
    t0 = datetime(2024, 1, 1)
    obs = [
        _obs(t0, 3000.0),
        _obs(t0 + timedelta(hours=1), 3000.0),
        _obs(t0 + timedelta(hours=2), 4800.0),  # liquidate
        _obs(t0 + timedelta(hours=3), 4800.0),  # rebalance
        _obs(t0 + timedelta(hours=4), 4800.0),  # settle
    ]
    s.run(obs)
    hedge = s.get_entity('HEDGE')
    spot = s.get_entity('SPOT')
    # Allow generous tolerance — the recovery rebalance restores hedge,
    # then the next steady step lets the basis equality reassert.
    if abs(hedge.size) > 1e-6:
        ratio = abs(hedge.size + spot.internal_state.amount) / abs(hedge.size)
        assert ratio < 0.05, f"basis hedge not restored, ratio={ratio}"
