"""L1 + L2 tests for :class:`RateHedgedLeveragedPTStrategy` / :class:`MorphoRateHedgedLeveragedPT`:
params, entity checks, sizing rules, lazy open, exit policies and the
closed-form behaviour of the overlay on synthetic paths."""
import pytest

from fractal.core.base import Observation
from fractal.core.entities import BorosGlobalState
from fractal.strategies import (
    LeveragedPTException,
    MorphoLeveragedPT,
    MorphoLeveragedPTParams,
    MorphoRateHedgedLeveragedPT,
    MorphoRateHedgedLeveragedPTParams,
    RateHedgedLeveragedPTStrategy,
)
from tests.core.pendle_synthetic import EXPIRY, synthetic_rate_hedged_observations

ZERO_FEES = dict(PT_IMPACT_MODEL="rate_spread", PT_FEE_LN_RATE=0.0, PT_IMPACT_LN_RATE_PER_SHARE=0.0,
                 BOROS_TAKER_FEE_RATE=0.0, BOROS_SETTLE_FEE_RATE=0.0)


def params(**overrides) -> MorphoRateHedgedLeveragedPTParams:
    base = dict(INITIAL_BALANCE=10_000.0, TARGET_LTV=0.8, MAX_LOOPS=8, REBALANCE_LTV_BAND=(0.7, 0.88),
                BAR_HOURS=8, LLTV=0.915, MIN_DAYS_TO_MATURITY_AT_ENTRY=1, MIN_CARRY_SPREAD=-1.0,
                MAX_BORROW_APY=10.0, **ZERO_FEES)
    base.update(overrides)
    return MorphoRateHedgedLeveragedPTParams(**base)


def strategy(**overrides) -> MorphoRateHedgedLeveragedPT:
    return MorphoRateHedgedLeveragedPT(params=params(**overrides))


def boros(**overrides):
    return strategy(RATE_HEDGE="boros", HEDGE_MARGIN_SHARE=0.25, **overrides)


@pytest.mark.core
def test_wiring_and_params_class():
    assert MorphoRateHedgedLeveragedPT.PARAMS_CLS is MorphoRateHedgedLeveragedPTParams
    assert issubclass(MorphoRateHedgedLeveragedPT, RateHedgedLeveragedPTStrategy)
    assert issubclass(MorphoRateHedgedLeveragedPT, MorphoLeveragedPT)
    assert strategy().hedge_kind == "none" and strategy().boros is None
    assert boros().boros is not None


@pytest.mark.core
@pytest.mark.parametrize("overrides", [
    {"RATE_HEDGE": "perp"}, {"HEDGE_SIZING": "dv01"}, {"BOROS_EXIT_POLICY": "roll"}, {"HEDGE_MARGIN_SHARE": 1.0},
    {"HEDGE_RATIO": -0.1}, {"HEDGE_BETA": -0.1}, {"HEDGE_MARGIN_BUFFER": 0.9},
    {"RATE_HEDGE": "boros", "HEDGE_MARGIN_SHARE": 0.0},
])
def test_param_validation(overrides):
    with pytest.raises(LeveragedPTException):
        strategy(**overrides)


@pytest.mark.core
def test_plain_loop_matches_leveraged_pt_exactly():
    """``RATE_HEDGE="none"`` must be byte-identical to ``MorphoLeveragedPT``."""
    obs = synthetic_rate_hedged_observations(days=30)
    ours = strategy().run(obs).to_dataframe()
    base = MorphoLeveragedPT(params=MorphoLeveragedPTParams(**{
        k: v for k, v in params().__dict__.items() if k in MorphoLeveragedPTParams.__dataclass_fields__})).run(
        obs).to_dataframe()
    assert ours["net_balance"].tolist() == pytest.approx(base["net_balance"].tolist(), rel=1e-12)


@pytest.mark.core
def test_beta_sizing_scales_with_the_pt_value_and_follows_it():
    strat = boros(HEDGE_BETA=0.10)
    obs = synthetic_rate_hedged_observations(days=60, boros_mark_apr=0.05, coin_price=2_000.0)
    strat.step(obs[0])
    assert strat.pt_value() > 0
    assert strat.hedge_notional() == pytest.approx(0.10 * strat.pt_value(), rel=1e-9)
    assert strat.hedge_coverage() == pytest.approx(0.10, rel=1e-9)
    value = strat.pt_value()
    shocked = obs[1]  # above the band → repay to target → the YU follows the smaller PT value
    shocked.states["LENDING"].collateral_price *= 0.85
    shocked.states["LENDING"].collateral_market_price *= 0.85
    strat.step(shocked)
    assert strat.pt_value() < value
    assert strat.hedge_coverage() == pytest.approx(0.10, rel=1e-6)


@pytest.mark.core
def test_duration_scaled_beta_and_debt_sizing():
    scaled = boros(HEDGE_BETA=0.10, HEDGE_DURATION_SCALED=True)
    late = EXPIRY.replace(month=12, day=31)
    obs = synthetic_rate_hedged_observations(days=60, boros_mark_apr=0.05, boros_maturity=late)
    scaled.step(obs[0])
    ratio = scaled.pt.years_to_expiry / scaled.boros.years_to_expiry
    assert scaled.hedge_notional() == pytest.approx(0.10 * scaled.pt_value() * ratio, rel=1e-9)
    debt = boros(HEDGE_SIZING="debt", HEDGE_RATIO=0.5)
    debt.step(synthetic_rate_hedged_observations(days=60, boros_mark_apr=0.05)[0])
    assert debt.hedge_notional() == pytest.approx(0.5 * debt.lending.debt_value, rel=1e-9)
    assert debt.hedge_coverage() == pytest.approx(0.5, rel=1e-9)


@pytest.mark.core
def test_size_is_capped_by_the_margin_the_leg_holds():
    strat = strategy(RATE_HEDGE="boros", HEDGE_MARGIN_SHARE=0.01, HEDGE_SIZING="debt", HEDGE_RATIO=1.0,
                     BOROS_MAX_LEVERAGE=1.55)
    strat.step(synthetic_rate_hedged_observations(days=120, boros_mark_apr=0.10)[0])
    assert 0 < strat.hedge_coverage() < 1.0
    assert strat.boros.balance >= strat.boros.initial_margin * 1.10 * (1 - 1e-9)


@pytest.mark.core
def test_leg_opens_lazily_and_is_settled_to_its_own_maturity_by_default():
    strat = boros()
    late_maturity = EXPIRY.replace(month=12, day=31)
    obs = synthetic_rate_hedged_observations(days=30, boros_mark_apr=0.05, boros_from_bar=5,
                                             boros_maturity=late_maturity)
    strat.step(obs[0])
    assert strat.boros.size == 0 and strat.boros.balance == pytest.approx(2_500.0)
    for observation in obs[1:5]:
        strat.step(observation)
    assert strat.boros.size == 0
    strat.step(obs[5])
    assert strat.boros.size > 0
    for observation in obs[6:]:
        strat.step(observation)
    assert strat._exited and strat.lending.internal_state.borrowed == 0.0
    assert strat.boros.size > 0 and not strat.boros.is_matured  # "settle": held past the PT's unwind
    assert strat.predict() == []
    closed = boros(BOROS_EXIT_POLICY="close")
    for observation in synthetic_rate_hedged_observations(days=30, boros_mark_apr=0.05, boros_maturity=late_maturity):
        closed.step(observation)
    assert closed._exited and closed.boros.size == 0.0


@pytest.mark.core
def test_long_yu_offsets_the_pt_mark_to_market():
    """Implied APY and the Boros mark jump together: with ``HEDGE_DURATION_SCALED``
    and ``HEDGE_BETA=1`` the yield unit's gain matches the PT's mark-down."""
    days = 60
    jump_bar = 30

    def implied(i):
        return 0.05 if i < jump_bar else 0.08

    def mark(i):
        return 0.05 if i < jump_bar else 0.08

    kw = dict(days=days, implied_apy=implied, borrow_apy=0.0, funding_rate=0.0)
    hedged = strategy(RATE_HEDGE="boros", HEDGE_MARGIN_SHARE=0.30, HEDGE_BETA=1.0, HEDGE_DURATION_SCALED=True,
                      BOROS_MAX_LEVERAGE=50.0, HEDGE_REBALANCE_THRESHOLD=1.0)
    out = hedged.run(synthetic_rate_hedged_observations(boros_mark_apr=mark, **kw)).to_dataframe()
    plain = strategy(RATE_HEDGE="none").run(synthetic_rate_hedged_observations(**kw)).to_dataframe()
    before, after = jump_bar - 1, jump_bar  # ``implied``/``mark`` are indexed by bar
    pt_move = plain["net_balance"].iloc[after] - plain["net_balance"].iloc[before]
    hedged_move = out["net_balance"].iloc[after] - out["net_balance"].iloc[before]
    assert pt_move < 0
    assert hedged_move > pt_move and abs(hedged_move) < 0.5 * abs(pt_move)


@pytest.mark.core
def test_partial_observation_without_boros_is_accepted_after_entry():
    strat = boros()
    obs = synthetic_rate_hedged_observations(days=30, boros_mark_apr=0.05)
    strat.step(obs[0])
    bare = Observation(timestamp=obs[1].timestamp,
                       states={"PT": obs[1].states["PT"], "LENDING": obs[1].states["LENDING"]})
    strat.step(bare)
    assert strat.boros.size > 0 and isinstance(strat.boros.global_state, BorosGlobalState)
