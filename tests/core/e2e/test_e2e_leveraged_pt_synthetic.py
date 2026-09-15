"""L2 synthetic end-to-end runs of :class:`MorphoLeveragedPT`.

Closed form (zero costs, constant rates, hold to expiry): with leverage
``L`` measured after entry, PT bought at ``P0`` and debt growing at the
borrow rate, ``equity_T = L·E0/P0 − (L−1)·E0·growth``.
"""
import math

import pytest

from fractal.core.base.time import SECONDS_PER_YEAR
from fractal.core.entities.models.pendle_math import pt_price_from_apy
from fractal.strategies import MorphoLeveragedPT, MorphoLeveragedPTParams
from tests.core.pendle_synthetic import synthetic_observations


def run(days=90, borrow_apy=0.0, implied_apy=0.10, **overrides):
    base = dict(INITIAL_BALANCE=10_000.0, TARGET_LTV=0.8, MAX_LOOPS=6, REBALANCE_LTV_BAND=(0.7, 0.88),
                PT_IMPACT_MODEL="rate_spread", PT_FEE_LN_RATE=0.0, PT_IMPACT_LN_RATE_PER_SHARE=0.0,
                BAR_HOURS=24, LLTV=0.915, MIN_DAYS_TO_MATURITY_AT_ENTRY=1)
    base.update(overrides)
    strat = MorphoLeveragedPT(params=MorphoLeveragedPTParams(**base))
    obs = synthetic_observations(days=days, borrow_apy=borrow_apy, implied_apy=implied_apy)
    result = strat.run(obs)
    return strat, result, result.to_dataframe()


@pytest.mark.core
def test_hold_to_expiry_matches_closed_form_with_zero_costs():
    strat, _, df = run()
    row0 = df.iloc[0]
    p0 = pt_price_from_apy(0.10, 90 * 86_400 / SECONDS_PER_YEAR)
    equity0 = row0["net_balance"]
    leverage = (row0["LENDING_collateral"] + row0["PT_amount"]) * p0 / equity0
    assert equity0 == pytest.approx(10_000.0)  # entry is cost-free
    assert 3.9 < leverage < 4.1                # L_6 at ℓ = 0.8 → 3.95 (a touch more: borrow is sized at the oracle)
    expected_end = leverage * equity0 / p0 - (leverage - 1) * equity0
    assert df["net_balance"].iloc[-1] == pytest.approx(expected_end, rel=1e-9)
    last = df.iloc[-1]
    assert last["LENDING_borrowed"] == 0.0 and last["LENDING_collateral"] == 0.0 and last["PT_amount"] == 0.0
    assert last["PT_cash"] == pytest.approx(expected_end)
    assert strat._exited
    assert df["net_balance"].is_monotonic_increasing  # PT accretes toward par every bar


@pytest.mark.core
def test_borrow_cost_reduces_the_carry_by_the_debt_growth():
    _, _, free = run(borrow_apy=0.0)
    strat, _, paid = run(borrow_apy=0.05)
    debt0 = paid.iloc[0]["LENDING_borrowed"]
    growth = math.exp(math.log1p(0.05) * 90 / 365)
    lost = debt0 * (growth - 1)
    assert free["net_balance"].iloc[-1] - paid["net_balance"].iloc[-1] == pytest.approx(lost, rel=1e-6)
    assert strat.borrow_apy() == pytest.approx(0.05, rel=1e-9)


@pytest.mark.core
def test_no_leverage_returns_exactly_the_entry_discount():
    _, _, df = run(MAX_LOOPS=0, TARGET_LTV=0.0, REBALANCE_LTV_BAND=(0.0, 0.0))
    p0 = pt_price_from_apy(0.10, 90 * 86_400 / SECONDS_PER_YEAR)
    assert df["net_balance"].iloc[-1] / df["net_balance"].iloc[0] - 1 == pytest.approx(1 / p0 - 1, rel=1e-9)
    assert (df["LENDING_borrowed"] == 0).all()


@pytest.mark.core
def test_oracle_drop_triggers_repay_and_restores_the_band():
    """A 15 % oracle haircut from bar 30 (the lender's feed marks the collateral
    down) pushes LTV above the band but below LLTV; the strategy repays back
    to target inside that bar, so the recorded state is already in band."""
    obs = synthetic_observations(days=90)
    for o in obs[30:]:
        o.states["LENDING"].collateral_price *= 0.85
    strat = MorphoLeveragedPT(params=MorphoLeveragedPTParams(
        INITIAL_BALANCE=10_000.0, TARGET_LTV=0.8, MAX_LOOPS=6, REBALANCE_LTV_BAND=(0.7, 0.85),
        PT_IMPACT_MODEL="rate_spread", PT_FEE_LN_RATE=0.0, PT_IMPACT_LN_RATE_PER_SHARE=0.0,
        BAR_HOURS=24, LLTV=0.915, MIN_DAYS_TO_MATURITY_AT_ENTRY=1))
    df = strat.run(obs).to_dataframe()
    ltv = df["LENDING_borrowed"] / (df["LENDING_collateral"] * df["LENDING_collateral_price"])
    shocked_value = df["LENDING_collateral"].iloc[29] * df["LENDING_collateral_price"].iloc[30]
    pre_shock = df["LENDING_borrowed"].iloc[29] / shocked_value
    assert 0.85 < pre_shock < 0.915                     # the shock alone would breach the band
    assert df["LENDING_borrowed"].iloc[30] < df["LENDING_borrowed"].iloc[29]   # repay fired on that bar
    assert ltv.iloc[30] == pytest.approx(0.8, abs=1e-6)  # back at target
    assert (ltv.iloc[30:].dropna() <= 0.85 + 1e-9).all()  # NaN once unwound at expiry
    assert (df["LENDING_liquidation_count"] == 0).all()


@pytest.mark.core
def test_carry_gate_deleverages_to_zero_debt():
    strat, _, df = run(borrow_apy=lambda i: 0.02 if i < 40 else 0.30, MAX_BORROW_APY=0.25)
    assert df["LENDING_borrowed"].iloc[39] > 0
    assert df["LENDING_borrowed"].iloc[41] == 0.0
    assert (df["net_balance"] > 0).all() and not df.isna().any().any()
    assert strat.lending.internal_state.liquidation_count == 0


@pytest.mark.core
def test_early_exit_sells_before_expiry():
    strat, _, df = run(EXIT_BEFORE_EXPIRY_DAYS=10)
    exited = df.index[df["LENDING_collateral"] == 0.0][0]
    assert 0 < exited < len(df) - 1
    assert strat._exited and df["PT_amount"].iloc[-1] == 0.0
