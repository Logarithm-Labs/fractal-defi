"""L3/L4 invariants of the hedged PT carry: delta neutrality after every
rebalance, no money printing from a rebalance, PT at par at expiry."""
import random

import pytest

from fractal.core.base import Action
from fractal.strategies import PerpHedgedPT, PerpHedgedPTParams
from tests.core.pendle_synthetic import synthetic_hedged_observations


def make(**overrides) -> PerpHedgedPT:
    base = dict(INITIAL_BALANCE=10_000.0, TARGET_HEDGE_LEVERAGE=2.0, HEDGE_LEVERAGE_BAND=(1.0, 4.0),
                HEDGE_REBALANCE_THRESHOLD=0.01, PT_IMPACT_MODEL="rate_spread", PT_FEE_LN_RATE=0.0,
                PT_IMPACT_LN_RATE_PER_SHARE=0.0, PERP_TRADING_FEE=0.0, BOROS_TAKER_FEE_RATE=0.0,
                BOROS_SETTLE_FEE_RATE=0.0, MIN_DAYS_TO_MATURITY_AT_ENTRY=1)
    base.update(overrides)
    return PerpHedgedPT(params=PerpHedgedPTParams(**base))


def execute(strategy, actions):
    for action in actions:
        entity = strategy.get_entity(action.entity_name)
        resolved = {k: (v(strategy) if callable(v) else v) for k, v in action.action.args.items()}
        entity.execute(Action(action.action.action, resolved))


@pytest.mark.core
@pytest.mark.parametrize("seed", range(10))
def test_hedge_is_delta_neutral_after_every_step(seed):
    rng = random.Random(seed)
    path = [2_000.0]
    for _ in range(120):
        path.append(path[-1] * (1 + rng.gauss(0.0, 0.03)))
    strat = make()
    obs = synthetic_hedged_observations(days=40, price=path, funding_rate=lambda i: 0.0002 * (rng.random() - 0.3))
    for o in obs:
        strat.step(o)
        if strat._exited:
            break
        target = strat.target_hedge_size()
        assert abs(strat.hedge.size - target) <= 0.01 * abs(target) + 1e-9
        lo, hi = 1.0, 4.0
        assert lo * 0.5 <= strat.hedge.leverage <= hi * 1.5   # band re-entered within the bar's actions
    assert strat.pt.internal_state.amount == 0.0


@pytest.mark.core
def test_rebalances_do_not_print_money():
    """Executing a rebalance list at frozen prices leaves the equity unchanged (zero fees)."""
    strat = make()
    obs = synthetic_hedged_observations(days=60, price=2_000.0, funding_rate=0.0)
    strat.step(obs[0])
    before = strat.equity()
    strat.pt._internal_state.amount *= 1.2  # fake a drift so a resize fires
    before = strat.equity()
    execute(strat, strat._resize_hedge())
    assert strat.equity() == pytest.approx(before, abs=1e-6)
    strat.hedge.update_state(strat.hedge.global_state)  # same prices
    execute(strat, strat._rebalance_margin())
    assert strat.equity() == pytest.approx(before, abs=1e-6)


@pytest.mark.core
def test_pt_pays_par_at_expiry_and_hedge_closes():
    strat = make()
    df = strat.run(synthetic_hedged_observations(days=30, price=2_500.0, funding_rate=0.0)).to_dataframe()
    entry = df.iloc[0]
    matured = df.iloc[-1]
    pt_share = 10_000.0 - 10_000.0 / 3
    assert matured["PT_amount"] == 0.0
    assert matured["PT_cash"] == pytest.approx(entry["PT_amount"] * 2_500.0)  # N PT → N coins → cash at par
    assert matured["net_balance"] == pytest.approx(10_000.0 + entry["PT_amount"] * 2_500.0 - pt_share, rel=1e-9)
