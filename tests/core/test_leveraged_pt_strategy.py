"""L1 tests for :class:`LeveragedPTStrategy` / :class:`MorphoLeveragedPT`:
parameter resolution, entity type checks, the exact action lists each
``predict`` branch emits, and the ``_Once`` delegate."""
import pytest

from fractal.core.base import Observation
from fractal.core.base.strategy import BaseStrategy, NamedEntity
from fractal.core.entities import (
    MorphoEntity,
    MorphoGlobalState,
    PendlePTConfig,
    PendlePTEntity,
    PendlePTGlobalState,
    SimpleLendingEntity,
    SimpleSpotExchange,
)
from fractal.strategies import LeveragedPTException, LeveragedPTParams, LeveragedPTStrategy, MorphoLeveragedPT
from fractal.strategies.leveraged_pt import _Once
from fractal.strategies.morpho_leveraged_pt import MorphoLeveragedPTParams
from tests.core.pendle_synthetic import synthetic_observations


def params(**overrides) -> MorphoLeveragedPTParams:
    base = dict(INITIAL_BALANCE=10_000.0, TARGET_LTV=0.8, MAX_LOOPS=3, REBALANCE_LTV_BAND=(0.7, 0.88),
                PT_IMPACT_MODEL="rate_spread", PT_FEE_LN_RATE=0.0, PT_IMPACT_LN_RATE_PER_SHARE=0.0,
                BAR_HOURS=24, LLTV=0.915, MIN_DAYS_TO_MATURITY_AT_ENTRY=1)
    base.update(overrides)
    return MorphoLeveragedPTParams(**base)


def strategy(**overrides) -> MorphoLeveragedPT:
    return MorphoLeveragedPT(params=params(**overrides))


def names(actions):
    return [(a.entity_name, a.action.action) for a in actions]


LOOP = [("PT", "buy"), ("LENDING", "deposit"), ("PT", "remove_product"), ("LENDING", "borrow"), ("PT", "deposit")]
CLOSE = LOOP[:3]


@pytest.mark.core
def test_params_class_is_the_venue_one():
    assert MorphoLeveragedPT.PARAMS_CLS is MorphoLeveragedPTParams
    assert LeveragedPTStrategy.PARAMS_CLS is LeveragedPTParams
    strat = strategy()
    assert strat.target_ltv == 0.8
    assert isinstance(strat.pt, PendlePTEntity) and isinstance(strat.lending, MorphoEntity)


@pytest.mark.core
def test_target_leverage_maps_to_ltv():
    strat = strategy(TARGET_LTV=None, TARGET_LEVERAGE=4.0, REBALANCE_LTV_BAND=(0.6, 0.9))
    assert strat.target_ltv == pytest.approx(0.75)


@pytest.mark.core
@pytest.mark.parametrize("overrides,exc", [
    ({"TARGET_LTV": None}, LeveragedPTException),                       # neither
    ({"TARGET_LEVERAGE": 3.0}, LeveragedPTException),                   # both
    ({"TARGET_LTV": None, "TARGET_LEVERAGE": 0.5}, LeveragedPTException),
    ({"REBALANCE_LTV_BAND": (0.85, 0.9)}, LeveragedPTException),        # target below lo
    ({"REBALANCE_LTV_BAND": (0.5, 0.7)}, LeveragedPTException),         # target above hi
    ({"MULTIPLY_MODE": "recursive"}, LeveragedPTException),
    ({"ROLL_TO_NEXT_MATURITY": True}, NotImplementedError),
    ({"INITIAL_BALANCE": 0.0}, LeveragedPTException),
])
def test_param_validation(overrides, exc):
    with pytest.raises(exc):
        strategy(**overrides)


@pytest.mark.core
def test_set_up_type_checks_entities():
    class _WrongPT(LeveragedPTStrategy):
        def set_up(self):
            self.register_entity(NamedEntity("PT", SimpleSpotExchange()))
            self.register_entity(NamedEntity("LENDING", MorphoEntity()))
            super().set_up()

    class _WrongLending(LeveragedPTStrategy):
        def set_up(self):
            self.register_entity(NamedEntity("PT", PendlePTEntity(PendlePTConfig())))
            self.register_entity(NamedEntity("LENDING", SimpleSpotExchange()))
            super().set_up()

    class _SimpleLending(LeveragedPTStrategy):
        def set_up(self):
            self.register_entity(NamedEntity("PT", PendlePTEntity(PendlePTConfig())))
            self.register_entity(NamedEntity("LENDING", SimpleLendingEntity()))
            super().set_up()

    base = LeveragedPTParams(INITIAL_BALANCE=1.0, TARGET_LTV=0.7)
    with pytest.raises(LeveragedPTException, match="PT must be"):
        _WrongPT(params=base)
    with pytest.raises(LeveragedPTException, match="LENDING must be"):
        _WrongLending(params=base)
    assert _SimpleLending(params=base).lending is not None  # any lending sibling works


@pytest.mark.core
def test_once_evaluates_the_delegate_a_single_time():
    calls = []
    once = _Once(lambda s: calls.append(1) or 42.0)
    assert once(None) == 42.0 and once(None) == 42.0
    assert len(calls) == 1


@pytest.mark.core
def test_entry_action_list_shape_and_order():
    strat = strategy(MAX_LOOPS=3)
    strat.step(synthetic_observations(days=30)[0])
    # step executed predict → the entry list was consumed; rebuild it to inspect
    actions = strat._enter()
    assert names(actions) == [("PT", "deposit")] + LOOP * 3 + CLOSE
    assert actions[0].action.args == {"amount_in_notional": 10_000.0}


@pytest.mark.core
def test_flash_entry_list():
    strat = strategy(MULTIPLY_MODE="flash", FLASH_FEE=0.0005)
    actions = strat._enter()
    assert names(actions) == [("PT", "deposit"), ("PT", "deposit"), ("PT", "buy"), ("LENDING", "deposit"),
                              ("PT", "remove_product"), ("LENDING", "borrow"), ("PT", "deposit"), ("PT", "withdraw")]
    flash = 10_000.0 * 0.8 / 0.2
    assert actions[1].action.args["amount_in_notional"] == pytest.approx(flash)
    assert actions[5].action.args["amount_in_product"] == pytest.approx(flash * 1.0005)


@pytest.mark.core
def test_maturity_gate_at_entry():
    strat = strategy(MIN_DAYS_TO_MATURITY_AT_ENTRY=30)
    obs = synthetic_observations(days=10)[0]
    with pytest.raises(LeveragedPTException, match="MIN_DAYS_TO_MATURITY_AT_ENTRY"):
        strat.step(obs)


@pytest.mark.core
def test_predict_branches_after_entry():
    strat = strategy(MAX_LOOPS=2)
    obs = synthetic_observations(days=60)
    strat.step(obs[0])
    assert strat._deposited and strat.lending.internal_state.borrowed > 0
    # in band → nothing
    strat.step(obs[1])
    assert strat.predict() == []
    # oracle price drop pushes LTV (0.70 after two loops) above the 0.88 band but below LLTV → repay list
    strat.lending.update_state(MorphoGlobalState(collateral_price=0.78, debt_price=1.0, collateral_market_price=0.78))
    assert 0.88 < strat.lending.ltv < 0.915
    assert names(strat.predict()) == [("LENDING", "repay"), ("LENDING", "withdraw"), ("PT", "inject_product"),
                                      ("PT", "sell"), ("PT", "withdraw")]
    # carry gate → full deleverage list (repay to zero)
    strat.lending.update_state(MorphoGlobalState(collateral_price=1.0, debt_price=1.0, collateral_market_price=1.0,
                                                 borrowing_rate=0.5 / 365))
    actions = strat.predict()
    assert names(actions)[0] == ("LENDING", "repay")
    # matured → unwind with redeem, then exited → []
    strat.lending.update_state(MorphoGlobalState(collateral_price=1.0, debt_price=1.0, collateral_market_price=1.0))
    strat.pt.update_state(PendlePTGlobalState(seconds_to_expiry=0.0, implied_apy=0.1))
    actions = strat.predict()
    assert names(actions) == [("LENDING", "repay"), ("LENDING", "withdraw"), ("PT", "inject_product"),
                              ("PT", "redeem"), ("PT", "withdraw")]
    assert strat._exited and strat.predict() == []


@pytest.mark.core
def test_early_exit_uses_sell_and_wiped_position_raises():
    strat = strategy(EXIT_BEFORE_EXPIRY_DAYS=5)
    obs = synthetic_observations(days=20)
    strat.step(obs[0])
    strat.pt.update_state(PendlePTGlobalState(seconds_to_expiry=4 * 86_400, implied_apy=0.1,
                                              total_pt=1e9, total_sy=1e9, scalar_root=50.0))
    strat.lending.update_state(MorphoGlobalState(collateral_price=0.999, debt_price=1.0, collateral_market_price=0.999))
    actions = strat.predict()
    assert ("PT", "sell") in names(actions) and strat._exited
    wiped = strategy()
    wiped.step(obs[0])
    wiped.lending._internal_state.collateral = 0.0
    wiped.lending._internal_state.borrowed = 0.0
    wiped.pt._internal_state.amount = 0.0
    wiped.pt._internal_state.cash = 0.0
    with pytest.raises(LeveragedPTException, match="wiped"):
        wiped.predict()


@pytest.mark.core
def test_observation_with_missing_entity_is_rejected():
    strat = strategy()
    obs = synthetic_observations(days=10)[0]
    with pytest.raises(ValueError):
        strat.step(Observation(timestamp=obs.timestamp, states={"PT": obs.states["PT"]}))


@pytest.mark.core
def test_strategy_is_a_base_strategy():
    assert issubclass(MorphoLeveragedPT, BaseStrategy)


@pytest.mark.core
def test_carry_gate_uses_the_smoothed_borrow_apy():
    """A one-bar borrow spike does not trip the gate when the lookback averages it away."""
    twitchy = strategy(MAX_LOOPS=2, MAX_BORROW_APY=0.25, MIN_CARRY_SPREAD=-1.0, CARRY_GATE_LOOKBACK_BARS=1)
    smooth = strategy(MAX_LOOPS=2, MAX_BORROW_APY=0.25, MIN_CARRY_SPREAD=-1.0, CARRY_GATE_LOOKBACK_BARS=7)
    obs = synthetic_observations(days=60, borrow_apy=lambda i: 0.05 if i != 10 else 0.80)
    for strat in (twitchy, smooth):
        for o in obs[:11]:
            strat.step(o)
    assert twitchy.lending.internal_state.borrowed == 0.0     # deleveraged on the spike bar
    assert smooth.lending.internal_state.borrowed > 0.0       # spike averaged over a week: still levered
    assert smooth.smoothed_borrow_apy() < 0.25 < max(twitchy._borrow_apy_window)
    twitchy.step(obs[11])                                     # ...and re-levers the very next bar: the whipsaw
    assert twitchy.lending.internal_state.borrowed > 0.0
    with pytest.raises(LeveragedPTException):
        strategy(CARRY_GATE_LOOKBACK_BARS=0)
