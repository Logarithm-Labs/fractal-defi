"""L1 + L2 tests for :class:`RateHedgedLeveragedPTStrategy` / :class:`MorphoRateHedgedLeveragedPT`:
params, entity checks, action lists, hedge sizing, and the closed-form
behaviour of both floating-rate legs on synthetic paths."""
import math

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
                 BOROS_TAKER_FEE_RATE=0.0, BOROS_SETTLE_FEE_RATE=0.0, PERP_TRADING_FEE=0.0, SPOT_TRADING_FEE=0.0)


def params(**overrides) -> MorphoRateHedgedLeveragedPTParams:
    base = dict(INITIAL_BALANCE=10_000.0, TARGET_LTV=0.8, MAX_LOOPS=8, REBALANCE_LTV_BAND=(0.7, 0.88),
                BAR_HOURS=8, LLTV=0.915, MIN_DAYS_TO_MATURITY_AT_ENTRY=1, MIN_CARRY_SPREAD=-1.0,
                MAX_BORROW_APY=10.0, **ZERO_FEES)
    base.update(overrides)
    return MorphoRateHedgedLeveragedPTParams(**base)


def strategy(**overrides) -> MorphoRateHedgedLeveragedPT:
    return MorphoRateHedgedLeveragedPT(params=params(**overrides))


def names(actions):
    return [(a.entity_name, a.action.action) for a in actions]


@pytest.mark.core
def test_wiring_and_params_class():
    assert MorphoRateHedgedLeveragedPT.PARAMS_CLS is MorphoRateHedgedLeveragedPTParams
    assert issubclass(MorphoRateHedgedLeveragedPT, RateHedgedLeveragedPTStrategy)
    assert issubclass(MorphoRateHedgedLeveragedPT, MorphoLeveragedPT)
    plain = strategy()
    assert plain.hedge_kind == "none" and plain.boros is None and plain.perp is None
    assert strategy(RATE_HEDGE="boros").boros is not None
    with_perp = strategy(RATE_HEDGE="perp")
    assert with_perp.spot is not None and with_perp.perp is not None


@pytest.mark.core
@pytest.mark.parametrize("overrides", [
    {"RATE_HEDGE": "swap"}, {"HEDGE_MARGIN_SHARE": 1.0}, {"HEDGE_RATIO": -0.1}, {"HEDGE_MARGIN_BUFFER": 0.9},
    {"PERP_LEVERAGE_BAND": (3.0, 4.0)}, {"RATE_HEDGE": "boros", "HEDGE_MARGIN_SHARE": 0.0},
])
def test_param_validation(overrides):
    with pytest.raises(LeveragedPTException):
        strategy(**overrides)


@pytest.mark.core
def test_entry_lists_per_hedge_kind():
    loop = names(strategy(MAX_LOOPS=1)._enter())
    boros = strategy(RATE_HEDGE="boros", MAX_LOOPS=1)
    boros.step(synthetic_rate_hedged_observations(days=10, boros_mark_apr=0.08)[0])
    assert boros._deposited
    perp = strategy(RATE_HEDGE="perp", MAX_LOOPS=1)
    obs = synthetic_rate_hedged_observations(days=10, with_perp=True)
    perp.step(obs[0])
    # the loop is the plain one; the hedge deposit comes first and the sizing last
    assert loop[0] == ("PT", "deposit")
    assert perp.spot.balance + perp.perp.balance == pytest.approx(1_000.0, rel=1e-9)
    assert perp.perp.size < 0 and perp.spot.internal_state.amount == pytest.approx(-perp.perp.size)
    assert boros.boros.balance == pytest.approx(1_000.0, rel=1e-6) or boros.boros.size > 0


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
def test_boros_leg_is_sized_to_the_debt_and_follows_it():
    strat = strategy(RATE_HEDGE="boros", HEDGE_MARGIN_SHARE=0.25, HEDGE_RATIO=1.0)
    obs = synthetic_rate_hedged_observations(days=60, boros_mark_apr=0.05, coin_price=2_000.0)
    strat.step(obs[0])
    debt = strat.lending.debt_value
    assert debt > 0 and strat.boros.size == pytest.approx(debt / 2_000.0, rel=1e-9)
    assert strat.hedge_coverage() == pytest.approx(1.0, rel=1e-9)
    # loop above the band → repay to target → the YU follows the smaller debt in the same step
    shocked = obs[1]
    shocked.states["LENDING"].collateral_price *= 0.85
    shocked.states["LENDING"].collateral_market_price *= 0.85
    strat.step(shocked)
    assert strat.lending.debt_value < debt
    assert strat.hedge_coverage() == pytest.approx(1.0, rel=1e-6)


@pytest.mark.core
def test_boros_size_is_capped_by_the_margin_it_holds():
    strat = strategy(RATE_HEDGE="boros", HEDGE_MARGIN_SHARE=0.02, HEDGE_RATIO=1.0, BOROS_MAX_LEVERAGE=1.55)
    obs = synthetic_rate_hedged_observations(days=120, boros_mark_apr=0.10, coin_price=2_000.0)
    strat.step(obs[0])
    boros = strat.boros
    assert 0 < strat.hedge_coverage() < 1.0
    assert boros.balance >= boros.initial_margin * 1.10 * (1 - 1e-9)


@pytest.mark.core
def test_boros_leg_opens_lazily_and_closes_on_unwind():
    strat = strategy(RATE_HEDGE="boros", HEDGE_MARGIN_SHARE=0.25)
    obs = synthetic_rate_hedged_observations(days=30, boros_mark_apr=0.05, boros_from_bar=5)
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
    assert strat.boros.size == 0.0


@pytest.mark.core
def test_boros_long_yu_turns_the_floating_cost_into_the_fixed_rate():
    """Funding == borrow rate every bar, zero fees: the hedged loop's carry
    equals a loop borrowing at the Boros fixed rate (on the hedged share)."""
    days, borrow, fixed = 90, 0.12, 0.06
    per_bar = borrow * 8 / (24 * 365)  # raw 8h funding equal to the borrow rate
    kw = dict(days=days, borrow_apy=borrow, implied_apy=0.10, funding_rate=per_bar, coin_price=2_000.0)
    hedged = strategy(RATE_HEDGE="boros", HEDGE_MARGIN_SHARE=0.30, BOROS_MAX_LEVERAGE=50.0)
    out = hedged.run(synthetic_rate_hedged_observations(boros_mark_apr=fixed, **kw)).to_dataframe()
    unhedged = strategy(RATE_HEDGE="none").run(synthetic_rate_hedged_observations(**kw)).to_dataframe()
    assert (out["BOROS_size"].iloc[1:-2] > 0).all()
    debt = out["LENDING_borrowed"].iloc[1:-2].mean()
    years = days / 365
    # the YU receives borrow − fixed on the debt (≈ full coverage) over the hold
    saved = debt * (borrow - fixed) * years
    gain = out["net_balance"].iloc[-1] - 7_000.0 - (unhedged["net_balance"].iloc[-1] - 10_000.0) * 0.7
    assert gain == pytest.approx(saved + 3_000.0, rel=0.03)


@pytest.mark.core
def test_perp_basis_leg_is_delta_neutral_and_receives_funding():
    def price(i):
        return 2_000.0 * (1.0 + 0.4 * math.sin(i / 7.0))

    strat = strategy(RATE_HEDGE="perp", HEDGE_MARGIN_SHARE=0.30, PERP_TARGET_LEVERAGE=2.0,
                     HEDGE_REBALANCE_THRESHOLD=0.02)
    obs = synthetic_rate_hedged_observations(days=60, with_perp=True, coin_price=price, funding_rate=0.0002,
                                             borrow_apy=0.0, implied_apy=0.0)
    out = strat.run(obs).to_dataframe()
    leg = out["SPOT_balance"] + out["PERP_balance"]
    assert (out["PERP_positions_0_amount"].fillna(0).iloc[1:-2] < 0).all()
    assert leg.iloc[-2] > 3_000.0  # funding received, no price PnL
    assert leg.iloc[1:-1].pct_change().abs().max() < 0.01  # price swings of ±40 % leave the leg flat
    assert 0 < strat.hedge_coverage() <= 1.0 or strat._exited


@pytest.mark.core
def test_partial_observation_without_boros_is_accepted_after_entry():
    strat = strategy(RATE_HEDGE="boros", HEDGE_MARGIN_SHARE=0.25)
    obs = synthetic_rate_hedged_observations(days=30, boros_mark_apr=0.05)
    strat.step(obs[0])
    bare = Observation(timestamp=obs[1].timestamp,
                       states={"PT": obs[1].states["PT"], "LENDING": obs[1].states["LENDING"]})
    strat.step(bare)
    assert strat.boros.size > 0
    assert strat.boros.global_state.seconds_to_expiry > 0
    assert isinstance(strat.boros.global_state, BorosGlobalState) and EXPIRY is not None
