"""L2 synthetic runs of :class:`PerpHedgedPT`: the hedge removes the price
leg, funding flows to the short, Boros converts it to fixed, expiry pays par."""
import random

import pytest

from fractal.strategies import PerpHedgedPT, PerpHedgedPTParams
from tests.core.pendle_synthetic import synthetic_hedged_observations


def run(days=90, use_boros=False, **kwargs):
    strat = PerpHedgedPT(params=PerpHedgedPTParams(
        INITIAL_BALANCE=10_000.0, TARGET_HEDGE_LEVERAGE=2.0, HEDGE_LEVERAGE_BAND=(1.0, 4.0),
        HEDGE_REBALANCE_THRESHOLD=0.01, USE_BOROS=use_boros, PT_IMPACT_MODEL="rate_spread", PT_FEE_LN_RATE=0.0,
        PT_IMPACT_LN_RATE_PER_SHARE=0.0, PERP_TRADING_FEE=0.0, BOROS_TAKER_FEE_RATE=0.0, BOROS_SETTLE_FEE_RATE=0.0,
        MIN_DAYS_TO_MATURITY_AT_ENTRY=1))
    obs = synthetic_hedged_observations(days=days, **kwargs)
    df = strat.run(obs).to_dataframe()
    return strat, df


@pytest.mark.core
def test_flat_price_zero_funding_earns_exactly_the_pt_discount_on_the_pt_share():
    strat, df = run(price=2_000.0, funding_rate=0.0, implied_apy=0.05)
    pt_share = 10_000.0 - 10_000.0 / 3
    p0 = strat.pt.pt_price_asset  # matured → 1.0; use the entry price from the first row instead
    first = df.iloc[0]
    entry_price = pt_share / first["PT_amount"] / 2_000.0
    expected = 10_000.0 + pt_share * (1 / entry_price - 1)
    assert p0 == 1.0
    assert df["net_balance"].iloc[-1] == pytest.approx(expected, rel=1e-9)
    assert df["HEDGE_collateral"].iloc[-1] == pytest.approx(10_000.0 / 3)  # no funding, no PnL at flat price


@pytest.mark.core
def test_price_moves_are_hedged_away():
    """A ±30 % saw-tooth in the coin leaves the equity path within the rebalance tolerance of the flat run."""
    _, flat = run(price=2_000.0, funding_rate=0.0)
    _, wild = run(price=lambda i: 2_000.0 * (1 + 0.3 * ((i * 7919) % 13 - 6) / 6), funding_rate=0.0)
    ratio = wild["net_balance"] / flat["net_balance"]
    assert ratio.iloc[-1] == pytest.approx(1.0, abs=0.03)
    assert ratio.min() > 0.9 and ratio.max() < 1.1
    unhedged = flat["PT_amount"].iloc[0] * 2_000.0 * 0.3  # what an unhedged PT leg would swing by
    assert (wild["net_balance"] - flat["net_balance"]).abs().max() < 0.2 * unhedged


@pytest.mark.core
def test_positive_funding_flows_to_the_short_hedge():
    _, no_funding = run(funding_rate=0.0)
    _, funded = run(funding_rate=0.0002)
    gain = funded["net_balance"].iloc[-1] - no_funding["net_balance"].iloc[-1]
    size = abs(funded["HEDGE_size"].iloc[0]) if "HEDGE_size" in funded else None
    assert gain > 0
    # ~ |size| · mark · f per 8h bar over 270 bars, with the hedge shrinking as PT accretes
    approx = (10_000.0 - 10_000.0 / 3) * 0.0002 * 270
    assert 0.8 * approx < gain < 1.1 * approx
    assert size is None or size > 0


@pytest.mark.core
def test_boros_locks_funding_at_the_fixed_rate():
    """Realised funding below the fixed rate: the YU pays the difference — Boros beats the naked hedge."""
    _, naked = run(funding_rate=0.00005)
    _, locked = run(use_boros=True, funding_rate=0.00005, boros_mark_apr=0.1095)  # fixed 10.95 % ≈ 0.0001 per 8h
    assert locked["net_balance"].iloc[-1] > naked["net_balance"].iloc[-1]
    assert (locked["BOROS_size"].iloc[:-2] < 0).all()  # short YU throughout
    # when realised funding equals the fixed rate the yield units settle to zero net
    _, same = run(use_boros=True, funding_rate=0.0001, boros_mark_apr=0.1095)
    assert abs(same["BOROS_realized_settlements"].iloc[-2]) < 1e-6
    assert same["BOROS_collateral"].iloc[-2] == pytest.approx(1_000.0, abs=1e-6)
    # below the fixed rate the YU keeps paying the short: settlements accumulate positively
    assert locked["BOROS_realized_settlements"].iloc[-2] > 0


@pytest.mark.core
def test_random_path_ends_flat_with_all_cash():
    rng = random.Random(7)
    path = [2_000.0]
    for _ in range(300):
        path.append(path[-1] * (1 + rng.gauss(0.0, 0.02)))
    strat, df = run(price=path, funding_rate=lambda i: 0.0002 * rng.random())
    last = df.iloc[-1]
    assert strat._exited
    assert last["PT_amount"] == 0.0
    assert abs(last["HEDGE_collateral"] - last["HEDGE_balance"]) < 1e-6  # hedge flat: balance == collateral
    scalar_columns = df.loc[:, ~df.columns.str.contains("positions")]  # Hyperliquid's positions list is NaN when flat
    assert not scalar_columns.isna().any().any()
    assert last["net_balance"] > 0.95 * 10_000.0
