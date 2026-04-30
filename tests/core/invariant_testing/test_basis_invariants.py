"""Invariant and randomized property checks for ``BasisTradingStrategy``."""
import math
import random
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
    BasisTradingStrategy, BasisTradingStrategyHyperparams,
)


class _TestableBasis(BasisTradingStrategy):
    HEDGE_TRADING_FEE: float = 0.0
    SPOT_TRADING_FEE: float = 0.0
    HEDGE_MAX_LEVERAGE: float = 50.0

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
                   hedge_max_lev=50.0):
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


def _build_path(prices, start=datetime(2024, 1, 1), step=timedelta(hours=1)):
    return [_obs(start + i * step, p) for i, p in enumerate(prices)]


def _basis_ratio(s):
    hedge = s.get_entity('HEDGE')
    spot = s.get_entity('SPOT')
    return abs(hedge.size + spot.internal_state.amount) / max(abs(hedge.size), 1e-12)


def _random_quiet_path(rng, n=30, start=3000.0, step_pct=0.003):
    prices = [start]
    for _ in range(n - 1):
        prices.append(prices[-1] * (1 + rng.uniform(-step_pct, step_pct)))
    return prices


@pytest.mark.core
def test_basis_invariant_holds_after_initial_deposit():
    s = _make_strategy(target_lev=3.0)
    s.run([_obs(datetime(2024, 1, 1), 3000.0)])
    assert _basis_ratio(s) < 1e-12


@pytest.mark.core
def test_basis_invariant_holds_through_quiet_path_no_rebalance():
    s = _make_strategy(min_lev=1.0, target_lev=3.0, max_lev=10.0)
    prices = [3000.0 * (1 + 0.002 * (i % 3 - 1)) for i in range(15)]
    s.run(_build_path(prices))
    assert _basis_ratio(s) < 1e-12


@pytest.mark.core
def test_basis_invariant_restored_after_rebalance_uptrend():
    s = _make_strategy(min_lev=2.5, target_lev=3.0, max_lev=3.2)
    prices = [3000.0 * (1 + 0.005 * i) for i in range(15)]
    s.run(_build_path(prices))
    assert _basis_ratio(s) < 1e-3


@pytest.mark.core
def test_equity_conservation_under_zero_fee_no_liquidation():
    s = _make_strategy(target_lev=3.0, min_lev=1.0, max_lev=10.0)
    initial = s._params.INITIAL_BALANCE
    prices = [3000.0 * (1 + 0.001 * (i % 5 - 2)) for i in range(20)]
    s.run(_build_path(prices))
    assert s.total_balance == pytest.approx(initial, rel=1e-9)


@pytest.mark.core
def test_equity_conservation_through_rebalance_zero_fee():
    s = _make_strategy(min_lev=2.5, target_lev=3.0, max_lev=3.1)
    initial = s._params.INITIAL_BALANCE
    prices = [3000.0 * (1 + 0.005 * i) for i in range(10)]
    s.run(_build_path(prices))
    assert s.total_balance == pytest.approx(initial, rel=1e-6)


@pytest.mark.core
def test_leverage_returns_to_target_after_rebalance():
    s = _make_strategy(min_lev=2.0, target_lev=3.0, max_lev=3.5)
    prices = [3000.0 * (1 + 0.01 * i) for i in range(8)]
    s.run(_build_path(prices))
    hedge = s.get_entity('HEDGE')
    assert s._params.MIN_LEVERAGE <= hedge.leverage <= s._params.MAX_LEVERAGE * 1.05


@pytest.mark.core
def test_total_balance_never_increases_above_initial_under_zero_funding():
    s = _make_strategy(min_lev=2.0, target_lev=3.0, max_lev=3.5)
    initial = s._params.INITIAL_BALANCE
    prices = [3000.0 * (1 + 0.005 * i) for i in range(12)]
    result = s.run(_build_path(prices))
    for balances in result.balances:
        assert sum(balances.values()) <= initial * (1 + 1e-6)


@pytest.mark.core
def test_rebalance_delta_spot_negative_action_amounts_consistent():
    s = _make_strategy(min_lev=2.5, target_lev=3.0, max_lev=3.05)
    s.run([_obs(datetime(2024, 1, 1), 3000.0)])
    s.get_entity('SPOT').update_state(UniswapV3SpotGlobalState(price=3030.0))
    s.get_entity('HEDGE').update_state(SimplePerpGlobalState(mark_price=3030.0))
    actions = s._rebalance()
    sell_amount_product = actions[0].action.args['amount_in_product']
    open_amount_product = actions[3].action.args['amount_in_product']
    sell_proceeds_notional = sell_amount_product * 3030.0
    open_notional = abs(open_amount_product) * 3030.0
    assert sell_proceeds_notional == pytest.approx(open_notional, rel=1e-9)


@pytest.mark.core
@pytest.mark.parametrize("seed", list(range(20)))
def test_property_basis_invariant_holds_after_random_quiet_path(seed):
    rng = random.Random(seed)
    s = _make_strategy(min_lev=1.0, target_lev=rng.uniform(2.0, 5.0), max_lev=15.0)
    s.run(_build_path(_random_quiet_path(rng, n=30)))
    assert _basis_ratio(s) < 1e-3


@pytest.mark.core
@pytest.mark.parametrize("seed", list(range(20)))
def test_property_equity_preserved_under_random_quiet_path(seed):
    rng = random.Random(seed)
    s = _make_strategy(min_lev=1.0, target_lev=rng.uniform(2.0, 5.0), max_lev=15.0)
    s.run(_build_path(_random_quiet_path(rng, n=30)))
    assert s.total_balance == pytest.approx(s._params.INITIAL_BALANCE, rel=1e-6)


@pytest.mark.core
@pytest.mark.parametrize("seed", list(range(15)))
def test_property_no_nan_or_inf_after_random_path(seed):
    rng = random.Random(seed)
    s = _make_strategy(min_lev=1.0, target_lev=rng.uniform(2.0, 5.0), max_lev=10.0)
    s.run(_build_path(_random_quiet_path(rng, n=25, step_pct=0.005)))
    hedge, spot = s.get_entity('HEDGE'), s.get_entity('SPOT')
    for v in (hedge.balance, hedge.size, hedge.leverage,
              spot.balance, spot.internal_state.amount, spot.internal_state.cash):
        assert math.isfinite(v)


@pytest.mark.core
@pytest.mark.parametrize("seed", list(range(10)))
def test_property_leverage_returns_to_bounds_after_run(seed):
    rng = random.Random(seed)
    s = _make_strategy(min_lev=2.0, target_lev=3.0, max_lev=4.0)
    s.run(_build_path(_random_quiet_path(rng, n=20, step_pct=0.008)))
    hedge = s.get_entity('HEDGE')
    assert hedge.leverage >= s._params.MIN_LEVERAGE * 0.5
    assert hedge.leverage <= s._params.MAX_LEVERAGE * 1.5
