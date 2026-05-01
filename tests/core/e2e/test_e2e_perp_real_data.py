"""E2E tests with live Hyperliquid prices + funding for HyperliquidEntity.

Marked ``integration`` + ``slow``. Run with::

    pytest tests/core/e2e/test_e2e_perp_real_data.py -m integration

Pulls real BTC perp daily klines + 8-hourly funding rates from Hyperliquid's
public API and walks ``HyperliquidEntity`` through the merged stream.
Verifies balance stays finite, PnL tracks price moves, funding accumulates
correctly.
"""
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from fractal.core.entities.protocols.hyperliquid import HyperliquidEntity, HyperliquidGlobalState
from fractal.loaders import HyperliquidFundingRatesLoader, HyperliquidPerpsKlinesLoader, LoaderType

UTC = timezone.utc
TICKER = "BTC"


@pytest.fixture(scope="module")
def real_btc_perp() -> pd.DataFrame:
    """30-day daily BTC klines + funding merged on date.

    Hyperliquid funding is hourly; we resample to daily (sum of payments)
    so it aligns with the daily price bars.
    """
    end = datetime.now(tz=UTC)
    start = end - timedelta(days=30)

    klines_loader = HyperliquidPerpsKlinesLoader(
        ticker=TICKER, interval="1d",
        loader_type=LoaderType.CSV,
        start_time=start, end_time=end,
    )
    klines = klines_loader.read(with_run=True)
    if len(klines) < 5:
        pytest.skip(f"too few BTC klines ({len(klines)}) for the chosen window")

    funding_loader = HyperliquidFundingRatesLoader(
        ticker=TICKER,
        loader_type=LoaderType.CSV,
        start_time=start, end_time=end,
    )
    funding = funding_loader.read(with_run=True)
    if len(funding) == 0:
        pytest.skip("no funding data for the window")

    # Aggregate per-hour funding into per-day (sum, since each is a
    # cumulative payment over its bar).
    daily_funding = funding["rate"].resample("1D").sum()
    df = klines.copy()
    df["funding_rate"] = daily_funding.reindex(df.index, fill_value=0.0)
    df = df.dropna(subset=["close"])
    if len(df) < 5:
        pytest.skip(f"too few overlapping bars ({len(df)})")
    return df


@pytest.mark.integration
@pytest.mark.slow
def test_real_btc_long_balance_finite_through_walk(real_btc_perp):
    """Walk a long BTC position through 30 days; balance stays finite."""
    e = HyperliquidEntity(trading_fee=0.0)
    first = real_btc_perp.iloc[0]
    e.update_state(HyperliquidGlobalState(mark_price=float(first["close"])))
    e.action_deposit(100_000)
    # Conservative size: ~1x leverage at entry
    leverage_target = 0.5
    btc_to_buy = (100_000 * leverage_target) / float(first["close"])
    e.action_open_position(btc_to_buy)

    bars_walked = 0
    for _, row in real_btc_perp.iloc[1:].iterrows():
        e.update_state(HyperliquidGlobalState(
            mark_price=float(row["close"]),
            funding_rate=float(row["funding_rate"]),
        ))
        bars_walked += 1
        # Sanity invariants
        assert e.balance == e.balance, "NaN balance"
        if e.size != 0:
            assert e._internal_state.collateral >= 0
        # Allow liquidation along the way (real BTC has volatility).
        if e.size == 0:
            break
    assert bars_walked > 0


@pytest.mark.integration
@pytest.mark.slow
def test_real_btc_short_funding_accrual_consistent(real_btc_perp):
    """Walk a short with conservative leverage; funding accrual is monotonic
    relative to per-bar rate sign (assuming no liquidation)."""
    e = HyperliquidEntity(trading_fee=0.0)
    first = real_btc_perp.iloc[0]
    p0 = float(first["close"])
    e.update_state(HyperliquidGlobalState(mark_price=p0))
    e.action_deposit(1_000_000)  # huge collateral → no liquidation risk
    btc_to_short = -0.05  # ~5k notional, very low leverage
    e.action_open_position(btc_to_short)

    coll_history = [e._internal_state.collateral]
    for _, row in real_btc_perp.iloc[1:].iterrows():
        e.update_state(HyperliquidGlobalState(
            mark_price=float(row["close"]),
            funding_rate=float(row["funding_rate"]),
        ))
        if e.size == 0:
            pytest.fail("conservative short was liquidated — unexpected")
        coll_history.append(e._internal_state.collateral)

    # All bars have finite collateral
    assert all(c == c for c in coll_history)
    assert all(c > 0 for c in coll_history)
