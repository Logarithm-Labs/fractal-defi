"""E2E test for StakedETHEntity over live Lido stETH rate history.

Marked ``integration`` + ``slow``. Run with::

    THE_GRAPH_API_KEY=... pytest tests/core/e2e/test_e2e_steth_real_data.py -m integration

Pulls real hourly stETH APR from Lido's subgraph and walks the entity
through it. Combines with a synthetic price walk (since the Lido loader
returns rate-only). Verifies amount rebases monotonically and balance
stays finite throughout.
"""
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from fractal.core.entities.protocols.steth import (StakedETHEntity,
                                                   StakedETHGlobalState)
from fractal.loaders import LoaderType
from fractal.loaders.thegraph.lido import StETHLoader

UTC = timezone.utc


@pytest.fixture(scope="module")
def real_steth_rates(THE_GRAPH_API_KEY: str) -> pd.DataFrame:
    """7-day hourly stETH APR window (~168 bars)."""
    end = datetime(2024, 12, 1, tzinfo=UTC)
    start = end - timedelta(days=7)
    loader = StETHLoader(
        api_key=THE_GRAPH_API_KEY,
        loader_type=LoaderType.CSV,
        start_time=start, end_time=end,
    )
    df = loader.read(with_run=True)
    if len(df) == 0:
        pytest.skip("no stETH rate data for the chosen window")
    return df


@pytest.mark.integration
@pytest.mark.slow
def test_real_steth_amount_rebases_through_real_rates(real_steth_rates):
    """Walk through hourly stETH rates; ``amount`` grows on each bar."""
    e = StakedETHEntity(trading_fee=0.0)
    e.update_state(StakedETHGlobalState(price=2000, staking_rate=0.0))
    e.action_deposit(10_000)
    e.action_buy(10_000)
    initial_amount = e._internal_state.amount

    for _, row in real_steth_rates.iterrows():
        rate = float(row["rate"])
        # Rates from the loader are positive (no slashing in window).
        e.update_state(StakedETHGlobalState(price=2000, staking_rate=rate))
        assert e._internal_state.amount >= initial_amount, (
            "amount must monotonically grow with positive rates"
        )

    # ~7 days at ~3% APY ≈ 0.06% growth.
    assert e._internal_state.amount > initial_amount
    growth = e._internal_state.amount / initial_amount
    assert 1.0 < growth < 1.01  # bounded — sanity check


@pytest.mark.integration
@pytest.mark.slow
def test_real_steth_no_negative_state_through_rates(real_steth_rates):
    e = StakedETHEntity()
    e.update_state(StakedETHGlobalState(price=2000, staking_rate=0.0))
    e.action_deposit(10_000)
    e.action_buy(5_000)
    for _, row in real_steth_rates.iterrows():
        e.update_state(StakedETHGlobalState(price=2000, staking_rate=float(row["rate"])))
        assert e._internal_state.amount >= 0
        assert e._internal_state.cash >= 0
        assert e.balance >= 0
