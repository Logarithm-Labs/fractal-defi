"""L1 tests for :class:`HedgedPTStrategy` / :class:`PerpHedgedPT`: params,
entity checks, the action lists of each ``predict`` branch."""
from datetime import timedelta

import pytest

from fractal.core.base import Action, Observation
from fractal.core.base.strategy import NamedEntity
from fractal.core.entities import (
    BorosEntity,
    BorosGlobalState,
    HyperliquidEntity,
    HyperliquidGlobalState,
    PendlePTConfig,
    PendlePTEntity,
    PendlePTGlobalState,
    SimplePerpEntity,
    SimpleSpotExchange,
)
from fractal.strategies import HedgedPTException, HedgedPTParams, HedgedPTStrategy, PerpHedgedPT, PerpHedgedPTParams
from tests.core.pendle_synthetic import EXPIRY, synthetic_hedged_observations


def params(**overrides) -> PerpHedgedPTParams:
    base = dict(INITIAL_BALANCE=10_000.0, TARGET_HEDGE_LEVERAGE=2.0, HEDGE_LEVERAGE_BAND=(1.0, 4.0),
                PT_IMPACT_MODEL="rate_spread", PT_FEE_LN_RATE=0.0, PT_IMPACT_LN_RATE_PER_SHARE=0.0,
                PERP_TRADING_FEE=0.0, BOROS_TAKER_FEE_RATE=0.0, BOROS_SETTLE_FEE_RATE=0.0,
                MIN_DAYS_TO_MATURITY_AT_ENTRY=1)
    base.update(overrides)
    return PerpHedgedPTParams(**base)


def strategy(**overrides) -> PerpHedgedPT:
    return PerpHedgedPT(params=params(**overrides))


def names(actions):
    return [(a.entity_name, a.action.action) for a in actions]


def _execute(strat, actions):
    """Resolve delegates and run an action list the way ``BaseStrategy.step`` does."""
    for action in actions:
        args = {k: (v(strat) if callable(v) else v) for k, v in action.action.args.items()}
        strat.get_entity(action.entity_name).execute(Action(action.action.action, args))


@pytest.mark.core
def test_params_class_venues_and_boros_wiring():
    assert PerpHedgedPT.PARAMS_CLS is PerpHedgedPTParams
    assert HedgedPTStrategy.PARAMS_CLS is HedgedPTParams
    assert isinstance(strategy().hedge, HyperliquidEntity)
    assert isinstance(strategy(HEDGE_VENUE="simple").hedge, SimplePerpEntity)
    with_boros = strategy(USE_BOROS=True)
    assert isinstance(with_boros.boros, BorosEntity)
    assert strategy().boros is None
    assert HedgedPTStrategy.STRICT_OBSERVATIONS is False


@pytest.mark.core
@pytest.mark.parametrize("overrides", [
    {"INITIAL_BALANCE": 0.0}, {"HEDGE_LEVERAGE_BAND": (3.0, 4.0)}, {"HEDGE_LEVERAGE_BAND": (1.0, 1.5)},
    {"HEDGE_REBALANCE_THRESHOLD": 0.0}, {"BOROS_MARGIN_SHARE": 1.0}, {"BOROS_MATURITY_POLICY": "roll"},
    {"HEDGE_VENUE": "gmx"},
])
def test_param_validation(overrides):
    with pytest.raises(HedgedPTException):
        strategy(**overrides)


@pytest.mark.core
def test_set_up_type_checks():
    class _WrongPT(HedgedPTStrategy):
        def set_up(self):
            self.register_entity(NamedEntity("PT", SimpleSpotExchange()))
            self.register_entity(NamedEntity("HEDGE", SimplePerpEntity()))
            super().set_up()

    class _WrongHedge(HedgedPTStrategy):
        def set_up(self):
            self.register_entity(NamedEntity("PT", PendlePTEntity(PendlePTConfig())))
            self.register_entity(NamedEntity("HEDGE", SimpleSpotExchange()))
            super().set_up()

    class _StrayBoros(HedgedPTStrategy):
        def set_up(self):
            self.register_entity(NamedEntity("PT", PendlePTEntity(PendlePTConfig())))
            self.register_entity(NamedEntity("HEDGE", SimplePerpEntity()))
            self.register_entity(NamedEntity("BOROS", BorosEntity()))
            super().set_up()

    base = HedgedPTParams(INITIAL_BALANCE=1.0)
    with pytest.raises(HedgedPTException, match="PT must be"):
        _WrongPT(params=base)
    with pytest.raises(HedgedPTException, match="HEDGE must be"):
        _WrongHedge(params=base)
    with pytest.raises(HedgedPTException, match="USE_BOROS is False"):
        _StrayBoros(params=base)


@pytest.mark.core
def test_entry_lists_with_and_without_boros():
    plain = strategy()
    assert names(plain._enter()) == [("PT", "deposit"), ("HEDGE", "deposit"), ("PT", "buy"), ("HEDGE", "open_position")]
    assert plain._enter()[1].action.args["amount_in_notional"] == pytest.approx(10_000 / 3)
    boros = strategy(USE_BOROS=True, BOROS_MARGIN_SHARE=0.1)
    boros.boros.update_state(BorosGlobalState(seconds_to_expiry=60 * 86_400, mark_rate=0.08, underlying_price=2_000.0))
    actions = boros._enter()
    assert names(actions) == [("PT", "deposit"), ("HEDGE", "deposit"), ("BOROS", "deposit"), ("PT", "buy"),
                              ("HEDGE", "open_position"), ("BOROS", "open_position")]
    assert actions[0].action.args["amount_in_notional"] == pytest.approx(9_000 * 2 / 3)
    assert actions[1].action.args["amount_in_notional"] == pytest.approx(9_000 / 3)
    assert actions[2].action.args["amount_in_notional"] == pytest.approx(1_000)


@pytest.mark.core
def test_entry_hedges_the_pt_delta_and_boros_matches_the_hedge():
    strat = strategy(USE_BOROS=True)
    obs = synthetic_hedged_observations(days=60, boros_mark_apr=0.08)
    strat.step(obs[0])
    pt, hedge, boros = strat.pt, strat.hedge, strat.boros
    assert hedge.size == pytest.approx(-pt.internal_state.amount * pt.pt_price_asset)
    assert boros.size == pytest.approx(hedge.size)
    assert strat.hedge_drift() == pytest.approx(0.0, abs=1e-12)
    assert hedge.leverage == pytest.approx(2.0, rel=1e-6)


@pytest.mark.core
def test_maturity_gates():
    strat = strategy(MIN_DAYS_TO_MATURITY_AT_ENTRY=30)
    with pytest.raises(HedgedPTException, match="MIN_DAYS_TO_MATURITY_AT_ENTRY"):
        strat.step(synthetic_hedged_observations(days=10)[0])
    strict = strategy(USE_BOROS=True, BOROS_MATURITY_POLICY="match_pt")
    obs = synthetic_hedged_observations(days=60, boros_mark_apr=0.08)
    obs[0].states["BOROS"].seconds_to_expiry = 10 * 86_400
    with pytest.raises(HedgedPTException, match="match_pt"):
        strict.step(obs[0])


@pytest.mark.core
def test_predict_branches_after_entry():
    strat = strategy()
    obs = synthetic_hedged_observations(days=60)
    strat.step(obs[0])
    strat.step(obs[1])
    assert strat.predict() == []
    # PT accretion makes the hedge drift; force a big drift via the PT price
    strat.pt.update_state(PendlePTGlobalState(seconds_to_expiry=59 * 86_400 - 1, implied_apy=0.05, asset_price=2_000.0,
                                              total_pt=1e9, total_sy=1e9, scalar_root=50.0))
    strat.pt._internal_state.amount *= 1.10
    assert names(strat.predict()) == [("HEDGE", "open_position")]
    strat.pt._internal_state.amount /= 1.10
    # leverage above the band → margin moves PT → HEDGE, then re-size
    strat.hedge.update_state(HyperliquidGlobalState(mark_price=2_600.0))  # +30 %: short loses, leverage 6.5×
    strat.pt.update_state(PendlePTGlobalState(seconds_to_expiry=59 * 86_400 - 2, implied_apy=0.05, asset_price=2_600.0,
                                              total_pt=1e9, total_sy=1e9, scalar_root=50.0))
    assert strat.hedge.leverage > 4.0
    actions = strat.predict()
    assert names(actions)[:1] == [("PT", "sell")]
    assert names(actions)[-3:] == [("HEDGE", "deposit"), ("PT", "withdraw"), ("HEDGE", "open_position")]
    # matured → close hedge and redeem; then exited
    strat.hedge.update_state(HyperliquidGlobalState(mark_price=2_000.0))
    strat.pt.update_state(PendlePTGlobalState(seconds_to_expiry=0.0, implied_apy=0.05, asset_price=2_000.0))
    assert names(strat.predict()) == [("HEDGE", "close_position"), ("PT", "redeem")]
    assert strat._exited and strat.predict() == []


@pytest.mark.core
def test_liquidated_hedge_is_refunded_from_pt():
    strat = strategy(USE_BOROS=True)
    obs = synthetic_hedged_observations(days=60, boros_mark_apr=0.08)
    strat.step(obs[0])
    strat.hedge._internal_state.collateral = 0.0
    strat.hedge._internal_state.positions = []
    strat.boros.update_state(BorosGlobalState(seconds_to_expiry=59 * 86_400, mark_rate=0.08, underlying_price=2_000.0))
    actions = strat.predict()
    assert names(actions) == [("PT", "sell"), ("HEDGE", "deposit"), ("PT", "withdraw"), ("HEDGE", "open_position"),
                              ("BOROS", "close_position"), ("BOROS", "open_position")]


@pytest.mark.core
def test_partial_observations_without_boros_are_accepted():
    strat = strategy(USE_BOROS=True)
    obs = synthetic_hedged_observations(days=30, boros_mark_apr=0.08)
    strat.step(obs[0])
    bare = Observation(timestamp=obs[1].timestamp, states={"PT": obs[1].states["PT"], "HEDGE": obs[1].states["HEDGE"]})
    strat.step(bare)  # no BOROS state this bar → fine
    assert strat.boros.size != 0


@pytest.mark.core
def test_early_exit_closes_a_live_boros_leg():
    """The YU matures after the PT exit: the exit list must close the Boros
    position explicitly (a matured YU is skipped instead)."""
    strat = strategy(USE_BOROS=True, EXIT_BEFORE_EXPIRY_DAYS=10)
    obs = synthetic_hedged_observations(days=30, boros_mark_apr=0.08,
                                        boros_maturity=EXPIRY + timedelta(days=90))
    strat.step(obs[0])
    assert strat.boros.size < 0
    for observation in obs[1:]:
        strat.step(observation)
        if strat._exited:
            break
    assert strat._exited and strat.pt.internal_state.amount == 0
    assert strat.boros.size == 0 and not strat.boros.is_matured
    assert strat.hedge.size == 0

    matured = strategy(USE_BOROS=True)
    obs = synthetic_hedged_observations(days=30, boros_mark_apr=0.08)  # YU matures with the PT
    matured.step(obs[0])
    matured.hedge.update_state(HyperliquidGlobalState(mark_price=2_000.0))
    matured.pt.update_state(PendlePTGlobalState(seconds_to_expiry=0.0, implied_apy=0.05, asset_price=2_000.0))
    matured.boros.update_state(BorosGlobalState(seconds_to_expiry=0.0, mark_rate=0.08, underlying_price=2_000.0))
    assert names(matured.predict()) == [("HEDGE", "close_position"), ("PT", "redeem")]


@pytest.mark.core
def test_boros_leg_opens_lazily_when_the_yu_market_lists_after_entry():
    strat = strategy(USE_BOROS=True)
    obs = synthetic_hedged_observations(days=30, boros_mark_apr=0.08)
    first = Observation(timestamp=obs[0].timestamp, states={"PT": obs[0].states["PT"], "HEDGE": obs[0].states["HEDGE"]})
    strat.step(first)  # no BOROS quote yet: entry must not touch the YU
    assert strat.boros.size == 0 and strat.hedge.size < 0
    assert strat.boros.balance == pytest.approx(1_000.0)  # margin share parked
    strat.step(obs[1])  # first quote → open against the hedge target
    assert strat.boros.size == pytest.approx(strat.target_hedge_size(), rel=1e-9)


@pytest.mark.core
def test_margin_rebalance_lands_on_target_leverage_and_resyncs_boros():
    strat = strategy(USE_BOROS=True, HEDGE_LEVERAGE_BAND=(1.2, 3.5))
    obs = synthetic_hedged_observations(days=60, boros_mark_apr=0.08)
    strat.step(obs[0])
    up = 2_000.0 * 1.35
    strat.hedge.update_state(HyperliquidGlobalState(mark_price=up))
    strat.pt.update_state(PendlePTGlobalState(seconds_to_expiry=59 * 86_400 - 1, implied_apy=0.05, asset_price=up,
                                              total_pt=1e9, total_sy=1e9, scalar_root=50.0))
    strat.boros.update_state(BorosGlobalState(seconds_to_expiry=59 * 86_400 - 1, mark_rate=0.08, underlying_price=up))
    assert strat.hedge.leverage > 3.5
    _execute(strat, strat.predict())
    assert strat.hedge.leverage == pytest.approx(2.0, rel=0.02)
    assert strat.boros.size == pytest.approx(strat.hedge.size, rel=1e-6)
    # a drift re-size keeps both legs equal even when the YU had diverged
    strat.boros._internal_state.size *= 0.5
    strat.pt._internal_state.amount *= 1.10
    _execute(strat, strat.predict())
    assert strat.boros.size == pytest.approx(strat.hedge.size, rel=1e-6)
    assert strat.hedge.size == pytest.approx(strat.target_hedge_size(), rel=1e-6)
