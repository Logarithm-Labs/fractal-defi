"""L3/L4 invariants of the PT loop: no money printing per block, LTV
respects the target and the band, unwind leaves nothing behind, random
implied-APY paths never produce NaN or a liquidation."""
import random

import pytest

from fractal.core.base import Action
from fractal.strategies import MorphoLeveragedPT, MorphoLeveragedPTParams
from tests.core.pendle_synthetic import synthetic_observations


def make(**overrides) -> MorphoLeveragedPT:
    base = dict(INITIAL_BALANCE=10_000.0, TARGET_LTV=0.8, MAX_LOOPS=6, REBALANCE_LTV_BAND=(0.7, 0.88),
                PT_IMPACT_MODEL="rate_spread", PT_FEE_LN_RATE=0.0, PT_IMPACT_LN_RATE_PER_SHARE=0.0,
                BAR_HOURS=24, LLTV=0.915, MIN_DAYS_TO_MATURITY_AT_ENTRY=1)
    base.update(overrides)
    return MorphoLeveragedPT(params=MorphoLeveragedPTParams(**base))


def execute(strategy, actions):
    for action in actions:
        entity = strategy.get_entity(action.entity_name)
        resolved = {k: (v(strategy) if callable(v) else v) for k, v in action.action.args.items()}
        entity.execute(Action(action.action.action, resolved))


@pytest.mark.core
def test_equity_conserved_after_every_loop_block_and_ltv_at_target():
    strat = make()
    obs = synthetic_observations(days=60)
    strat.pt.update_state(obs[0].states["PT"])
    strat.lending.update_state(obs[0].states["LENDING"])
    execute(strat, [strat._enter()[0]])  # deposit only
    equity0 = strat.equity()
    for _ in range(6):
        execute(strat, strat._loop_block())
        assert strat.equity() == pytest.approx(equity0, abs=1e-6)
        assert strat.lending.ltv <= 0.8 + 1e-9
    execute(strat, strat._close_block())
    assert strat.equity() == pytest.approx(equity0, abs=1e-6)
    assert strat.pt.internal_state.cash == 0.0 and strat.pt.internal_state.amount == 0.0


@pytest.mark.core
def test_flash_entry_conserves_equity_minus_the_flash_fee():
    strat = make(MULTIPLY_MODE="flash", FLASH_FEE=0.0005, TARGET_LTV=0.75, REBALANCE_LTV_BAND=(0.6, 0.85))
    obs = synthetic_observations(days=60)
    strat.pt.update_state(obs[0].states["PT"])
    strat.lending.update_state(obs[0].states["LENDING"])
    execute(strat, strat._enter())
    flash = 10_000.0 * 0.75 / 0.25
    assert strat.equity() == pytest.approx(10_000.0 - flash * 0.0005, rel=1e-9)
    assert strat.pt.internal_state.cash == pytest.approx(0.0, abs=1e-9)
    assert strat.lending.ltv < 0.85


@pytest.mark.core
def test_unwind_leaves_zero_debt_zero_collateral_zero_pt():
    strat = make()
    df = strat.run(synthetic_observations(days=30)).to_dataframe()
    last = df.iloc[-1]
    assert last["LENDING_borrowed"] == 0.0 and last["LENDING_collateral"] == 0.0 and last["PT_amount"] == 0.0
    assert last["PT_cash"] == pytest.approx(last["net_balance"])


@pytest.mark.core
@pytest.mark.parametrize("seed", range(20))
def test_random_implied_apy_paths_stay_inside_the_band_without_liquidation(seed):
    rng = random.Random(seed)
    path = [0.10]
    for _ in range(120):
        path.append(min(0.60, max(0.01, path[-1] * (1 + rng.gauss(0.0, 0.05)))))
    strat = make(REBALANCE_LTV_BAND=(0.7, 0.9), TARGET_LTV=0.8)
    obs = synthetic_observations(days=120, implied_apy=path, borrow_apy=lambda i: 0.03 + 0.02 * rng.random())
    df = strat.run(obs).to_dataframe()
    assert not df.isna().any().any()
    assert (df["LENDING_liquidation_count"] == 0).all()
    assert (df["net_balance"] > 0).all()
    ltv = (df["LENDING_borrowed"] / (df["LENDING_collateral"] * df["LENDING_collateral_price"])).fillna(0.0)
    # after the strategy acted, LTV never sits above the band at the start of the next bar
    assert (ltv.shift(-1).dropna() <= 0.9 + 1e-6).all()
    assert df["LENDING_borrowed"].iloc[-1] == 0.0
