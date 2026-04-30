"""Unit and synthetic end-to-end tests for ``BasisTradingStrategy``."""
from datetime import datetime, timedelta

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


class _TestableBasis(BasisTradingStrategy):
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


def _make_strategy(*, target_lev=3.0, min_lev=1.0, max_lev=5.0,
                   initial=100_000.0, hedge_fee=0.0, spot_fee=0.0,
                   hedge_max_lev=20.0):
    cls = type('_S', (_TestableBasis,), {
        'HEDGE_TRADING_FEE': hedge_fee,
        'SPOT_TRADING_FEE': spot_fee,
        'HEDGE_MAX_LEVERAGE': hedge_max_lev,
    })
    return cls(params=BasisTradingStrategyHyperparams(
        MIN_LEVERAGE=min_lev, TARGET_LEVERAGE=target_lev,
        MAX_LEVERAGE=max_lev, INITIAL_BALANCE=initial,
    ))


def _obs(t, price, funding=0.0):
    return Observation(
        timestamp=t,
        states={
            'SPOT': UniswapV3SpotGlobalState(price=price),
            'HEDGE': SimplePerpGlobalState(mark_price=price, funding_rate=funding),
        },
    )


def _seed_initial_position(s, price=3000.0):
    s.step(_obs(datetime(2024, 1, 1), price))


def _force_liquidated_state(s, price=3000.0):
    _seed_initial_position(s, price=price)
    hedge = s.get_entity('HEDGE')
    hedge._internal_state.collateral = 0.0
    hedge._internal_state.size = 0.0
    hedge._internal_state.entry_price = 0.0


@pytest.mark.core
def test_deposit_into_strategy_returns_four_actions_in_correct_order():
    s = _make_strategy(target_lev=3.0, initial=100_000.0)
    actions = s._deposit_into_strategy()
    kinds = [(a.entity_name, a.action.action) for a in actions]
    assert kinds == [
        ('SPOT', 'deposit'),
        ('HEDGE', 'deposit'),
        ('SPOT', 'buy'),
        ('HEDGE', 'open_position'),
    ]


@pytest.mark.core
def test_deposit_into_strategy_splits_initial_balance_by_target_leverage():
    L, B = 3.0, 100_000.0
    s = _make_strategy(target_lev=L, initial=B)
    actions = s._deposit_into_strategy()
    spot_dep, hedge_dep, spot_buy, hedge_open = actions
    expected_spot = B - B / (1 + L)
    expected_hedge = B / (1 + L)
    assert spot_dep.action.args['amount_in_notional'] == pytest.approx(expected_spot)
    assert hedge_dep.action.args['amount_in_notional'] == pytest.approx(expected_hedge)
    assert spot_buy.action.args['amount_in_notional'] == pytest.approx(expected_spot)
    assert callable(hedge_open.action.args['amount_in_product'])


@pytest.mark.core
def test_deposit_into_strategy_yields_target_leverage_after_step():
    s = _make_strategy(target_lev=3.0, initial=100_000.0)
    _seed_initial_position(s, price=3000.0)
    hedge = s.get_entity('HEDGE')
    spot = s.get_entity('SPOT')
    assert hedge.leverage == pytest.approx(3.0, rel=1e-3)
    assert hedge.size + spot.internal_state.amount == pytest.approx(0, abs=1e-9)


@pytest.mark.core
def test_predict_calls_deposit_when_both_balances_zero():
    s = _make_strategy()
    s.get_entity('SPOT').update_state(UniswapV3SpotGlobalState(price=3000.0))
    s.get_entity('HEDGE').update_state(SimplePerpGlobalState(mark_price=3000.0))
    actions = s.predict()
    assert len(actions) == 4
    assert actions[0].action.action == 'deposit'


@pytest.mark.core
def test_predict_returns_empty_when_leverage_in_bounds():
    s = _make_strategy(min_lev=1.0, target_lev=3.0, max_lev=5.0)
    _seed_initial_position(s, price=3000.0)
    assert s.predict() == []


@pytest.mark.core
def test_predict_triggers_rebalance_when_leverage_above_max():
    s = _make_strategy(min_lev=2.5, target_lev=3.0, max_lev=3.05)
    _seed_initial_position(s, price=3000.0)
    s.get_entity('SPOT').update_state(UniswapV3SpotGlobalState(price=3030.0))
    s.get_entity('HEDGE').update_state(SimplePerpGlobalState(mark_price=3030.0))
    assert s.get_entity('HEDGE').leverage > 3.05
    actions = s.predict()
    assert actions
    assert actions[0].entity_name == 'SPOT'
    assert actions[0].action.action == 'sell'


@pytest.mark.core
def test_predict_triggers_rebalance_when_hedge_liquidated_and_spot_alive():
    s = _make_strategy(target_lev=3.0)
    _force_liquidated_state(s)
    assert s.get_entity('HEDGE').balance == 0
    assert s.get_entity('SPOT').balance > 0
    assert s.predict()


@pytest.mark.core
def test_rebalance_liquidated_branch_returns_four_actions():
    s = _make_strategy(target_lev=3.0)
    _force_liquidated_state(s)
    actions = s._rebalance()
    spot_actions = [a for a in actions if a.entity_name == 'SPOT']
    hedge_actions = [a for a in actions if a.entity_name == 'HEDGE']
    assert len(spot_actions) == 2
    assert len(hedge_actions) == 2


@pytest.mark.core
def test_rebalance_liquidated_branch_orders_hedge_deposit_before_spot_withdraw():
    # Both actions share the SPOT.cash delegate; withdraw must come AFTER deposit (B-1).
    s = _make_strategy(target_lev=3.0)
    _force_liquidated_state(s)
    actions = s._rebalance()
    idx = {(a.entity_name, a.action.action): i for i, a in enumerate(actions)}
    assert idx[('HEDGE', 'deposit')] < idx[('SPOT', 'withdraw')]


@pytest.mark.core
def test_rebalance_delta_spot_positive_branch_orders_actions_correctly():
    s = _make_strategy(min_lev=2.5, target_lev=3.0, max_lev=3.05)
    _seed_initial_position(s, price=3000.0)
    s.get_entity('SPOT').update_state(UniswapV3SpotGlobalState(price=2970.0))
    s.get_entity('HEDGE').update_state(SimplePerpGlobalState(mark_price=2970.0))
    kinds = [(a.entity_name, a.action.action) for a in s._rebalance()]
    assert kinds == [
        ('HEDGE', 'withdraw'),
        ('SPOT', 'deposit'),
        ('SPOT', 'buy'),
        ('HEDGE', 'open_position'),
    ]


@pytest.mark.core
def test_rebalance_delta_spot_negative_branch_orders_deposit_before_withdraw():
    s = _make_strategy(min_lev=2.5, target_lev=3.0, max_lev=3.05)
    _seed_initial_position(s, price=3000.0)
    s.get_entity('SPOT').update_state(UniswapV3SpotGlobalState(price=3030.0))
    s.get_entity('HEDGE').update_state(SimplePerpGlobalState(mark_price=3030.0))
    kinds = [(a.entity_name, a.action.action) for a in s._rebalance()]
    assert kinds == [
        ('SPOT', 'sell'),
        ('HEDGE', 'deposit'),
        ('SPOT', 'withdraw'),
        ('HEDGE', 'open_position'),
    ]


@pytest.mark.core
def test_rebalance_returns_empty_when_no_delta():
    s = _make_strategy(target_lev=3.0)
    _seed_initial_position(s, price=3000.0)
    assert s._rebalance() == []


@pytest.mark.core
def test_set_up_raises_when_hedge_is_not_a_perp_entity():
    class _Bad(BasisTradingStrategy):
        def set_up(self):
            self.register_entity(NamedEntity('HEDGE', UniswapV3SpotEntity(trading_fee=0.0)))
            self.register_entity(NamedEntity('SPOT', UniswapV3SpotEntity(trading_fee=0.0)))
            super().set_up()

    with pytest.raises(BasisTradingStrategyException, match="HEDGE must be a BasePerpEntity"):
        _Bad(params=BasisTradingStrategyHyperparams(
            MIN_LEVERAGE=1.0, TARGET_LEVERAGE=3.0, MAX_LEVERAGE=5.0, INITIAL_BALANCE=100_000.0,
        ))


@pytest.mark.core
def test_set_up_raises_when_spot_is_not_a_spot_entity():
    class _Bad(BasisTradingStrategy):
        def set_up(self):
            self.register_entity(NamedEntity('HEDGE', SimplePerpEntity(trading_fee=0.0)))
            self.register_entity(NamedEntity('SPOT', SimplePerpEntity(trading_fee=0.0)))
            super().set_up()

    with pytest.raises(BasisTradingStrategyException, match="SPOT must be a BaseSpotEntity"):
        _Bad(params=BasisTradingStrategyHyperparams(
            MIN_LEVERAGE=1.0, TARGET_LEVERAGE=3.0, MAX_LEVERAGE=5.0, INITIAL_BALANCE=100_000.0,
        ))


@pytest.mark.core
def test_rebalance_raises_when_basis_hedge_invariant_violated():
    s = _make_strategy(min_lev=2.5, target_lev=3.0, max_lev=3.05)
    _seed_initial_position(s, price=3000.0)
    s.get_entity('HEDGE')._internal_state.size = -1.0
    s.get_entity('SPOT').update_state(UniswapV3SpotGlobalState(price=3030.0))
    s.get_entity('HEDGE').update_state(SimplePerpGlobalState(mark_price=3030.0))
    with pytest.raises(BasisTradingStrategyException, match="basis hedge mismatch"):
        s._rebalance()


@pytest.mark.core
def test_rebalance_basis_invariant_does_not_blow_up_on_zero_zero_state():
    # ``math.isclose(0, 0)`` is True; old ratio formula produced nan here.
    s = _make_strategy(target_lev=3.0)
    s.get_entity('SPOT').update_state(UniswapV3SpotGlobalState(price=3000.0))
    s.get_entity('HEDGE').update_state(SimplePerpGlobalState(mark_price=3000.0))
    s.get_entity('HEDGE')._internal_state.collateral = 100.0
    assert isinstance(s._rebalance(), list)


@pytest.mark.core
def test_rebalance_invariants_are_real_raises_not_asserts():
    # ``python -O`` strips ``assert``; raises survive.
    import inspect
    src = inspect.getsource(BasisTradingStrategy._rebalance)
    src += inspect.getsource(BasisTradingStrategy.set_up)
    assert "assert " not in src


@pytest.mark.core
def test_full_wipe_after_initial_deposit_raises_instead_of_re_funding():
    s = _make_strategy(target_lev=3.0, initial=100_000.0)
    _seed_initial_position(s, price=3000.0)
    hedge, spot = s.get_entity('HEDGE'), s.get_entity('SPOT')
    hedge._internal_state.collateral = 0.0
    hedge._internal_state.size = 0.0
    spot._internal_state.amount = 0.0
    spot._internal_state.cash = 0.0
    assert s._deposited is True
    with pytest.raises(BasisTradingStrategyException, match="fully wiped"):
        s.predict()


@pytest.mark.core
def test_initial_deposit_flag_flips_to_true_on_first_predict():
    s = _make_strategy()
    assert s._deposited is False
    s.get_entity('SPOT').update_state(UniswapV3SpotGlobalState(price=3000.0))
    s.get_entity('HEDGE').update_state(SimplePerpGlobalState(mark_price=3000.0))
    s.predict()
    assert s._deposited is True


@pytest.mark.core
def test_rebalance_raises_clear_error_when_hedge_lacks_free_margin():
    s = _make_strategy(min_lev=2.5, target_lev=3.0, max_lev=4.0, hedge_max_lev=2.0)
    hedge, spot = s.get_entity('HEDGE'), s.get_entity('SPOT')
    spot._internal_state.amount = 10.0
    spot._internal_state.cash = 0.0
    spot.update_state(UniswapV3SpotGlobalState(price=3000.0))
    hedge._internal_state.size = -10.0
    hedge._internal_state.entry_price = 3000.0
    hedge._internal_state.collateral = 20_000.0
    hedge.update_state(SimplePerpGlobalState(mark_price=3000.0))
    assert hedge.maintenance_margin == pytest.approx(15_000.0)
    with pytest.raises(BasisTradingStrategyException, match="free margin is only"):
        s._rebalance()


@pytest.mark.core
def test_set_up_signature_matches_base_strategy():
    import inspect
    assert list(inspect.signature(BasisTradingStrategy.set_up).parameters) == ['self']


@pytest.mark.core
def test_predict_signature_matches_base_strategy():
    import inspect
    assert list(inspect.signature(BasisTradingStrategy.predict).parameters) == ['self']


@pytest.mark.core
def test_e2e_flat_price_no_rebalance():
    s = _make_strategy(target_lev=3.0)
    obs = [_obs(datetime(2024, 1, 1) + timedelta(hours=i), 3000.0) for i in range(5)]
    s.run(obs)
    final_total = sum(e.balance for e in s.get_all_available_entities().values())
    assert final_total == pytest.approx(s._params.INITIAL_BALANCE, rel=1e-9)


@pytest.mark.core
def test_e2e_uptrend_triggers_rebalance_and_keeps_leverage_bounded():
    s = _make_strategy(min_lev=2.0, target_lev=3.0, max_lev=4.0)
    t0 = datetime(2024, 1, 1)
    obs = [_obs(t0 + timedelta(hours=i), 3000.0 * (1 + 0.005 * i)) for i in range(20)]
    s.run(obs)
    hedge = s.get_entity('HEDGE')
    assert s._params.MIN_LEVERAGE <= hedge.leverage <= s._params.MAX_LEVERAGE * 1.05


@pytest.mark.core
def test_e2e_downtrend_triggers_rebalance_when_leverage_drops_below_min():
    s = _make_strategy(min_lev=2.5, target_lev=3.0, max_lev=4.0)
    t0 = datetime(2024, 1, 1)
    obs = [_obs(t0 + timedelta(hours=i), 3000.0 * (1 - 0.005 * i)) for i in range(20)]
    s.run(obs)
    hedge, spot = s.get_entity('HEDGE'), s.get_entity('SPOT')
    assert abs(hedge.size + spot.internal_state.amount) / abs(hedge.size) < 0.01


@pytest.mark.core
def test_e2e_hedge_liquidation_then_recovery_no_silent_equity_loss():
    # 60% spike liquidates the short. Spot still holds 25 ETH @ 4800 = 120K;
    # post-rebalance must preserve that. With B-1 unfixed total ≈ 50K.
    s = _make_strategy(target_lev=3.0, max_lev=5.0)
    t0 = datetime(2024, 1, 1)
    obs = [
        _obs(t0, 3000.0),
        _obs(t0 + timedelta(hours=1), 3000.0),
        _obs(t0 + timedelta(hours=2), 4800.0),
        _obs(t0 + timedelta(hours=3), 4800.0),
        _obs(t0 + timedelta(hours=4), 4800.0),
    ]
    s.run(obs)
    assert s.total_balance >= 0.95 * 120_000.0


@pytest.mark.core
def test_e2e_hedge_liquidation_post_rebalance_size_invariant():
    s = _make_strategy(target_lev=3.0, max_lev=5.0)
    t0 = datetime(2024, 1, 1)
    obs = [
        _obs(t0, 3000.0),
        _obs(t0 + timedelta(hours=1), 3000.0),
        _obs(t0 + timedelta(hours=2), 4800.0),
        _obs(t0 + timedelta(hours=3), 4800.0),
        _obs(t0 + timedelta(hours=4), 4800.0),
    ]
    s.run(obs)
    hedge, spot = s.get_entity('HEDGE'), s.get_entity('SPOT')
    if abs(hedge.size) > 1e-6:
        ratio = abs(hedge.size + spot.internal_state.amount) / abs(hedge.size)
        assert ratio < 0.05
