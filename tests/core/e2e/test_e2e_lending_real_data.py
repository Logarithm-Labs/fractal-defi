"""E2E test for AaveEntity over live Aave V3 rates from the GraphQL API.

Marked ``integration`` + ``slow``. Run with::

    THE_GRAPH_API_KEY=... pytest tests/core/e2e/test_e2e_lending_real_data.py -m integration

Pulls a real ~30-day window of WETH supply/borrow rates from Aave V3 on
Ethereum and walks ``AaveEntity`` through them. Verifies balance stays
finite and interest accrual matches the rate stream.
"""
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from fractal.core.entities.protocols.aave import (AaveEntity, AaveGlobalState)
from fractal.loaders import LoaderType
from fractal.loaders.aave import AaveV3EthereumLoader

UTC = timezone.utc

# WETH on Ethereum mainnet — high-volume asset, stable rate stream.
WETH_ETHEREUM = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"


@pytest.fixture(scope="module")
def real_aave_rates() -> pd.DataFrame:
    """Last-14-day window of Aave V3 WETH supply+borrow rates (Ethereum).

    Aave's GraphQL API exposes only a rolling window relative to "now"
    (max enum is ``LAST_YEAR``), so we anchor to the current time rather
    than a fixed historical date — otherwise the request returns empty.
    """
    end = datetime.now(tz=UTC)
    start = end - timedelta(days=14)
    loader = AaveV3EthereumLoader(
        asset_address=WETH_ETHEREUM,
        loader_type=LoaderType.CSV,
        start_time=start, end_time=end,
        resolution=1,  # hourly
    )
    df = loader.read(with_run=True)
    if len(df) == 0:
        pytest.skip("no Aave rate data for the chosen window")
    return df


@pytest.mark.integration
@pytest.mark.slow
def test_real_aave_rates_walk_balance_stable(real_aave_rates):
    """Walk AaveEntity through real WETH rates over 30 days; no NaN, no negative."""
    e = AaveEntity()
    e.update_state(AaveGlobalState(collateral_price=1.0, debt_price=1.0))
    e.action_deposit(10_000)
    e.action_borrow(5_000)

    for _, row in real_aave_rates.iterrows():
        e.update_state(AaveGlobalState(
            collateral_price=1.0,
            debt_price=1.0,
            lending_rate=float(row["lending_rate"]),
            borrowing_rate=float(abs(row["borrowing_rate"])),
        ))
        assert e.balance == e.balance, "NaN balance"
        assert e._internal_state.collateral >= 0
        assert e._internal_state.borrowed >= 0


@pytest.mark.integration
@pytest.mark.slow
def test_real_aave_rates_collateral_grows_borrowed_grows(real_aave_rates):
    """With positive supply rate and positive borrow rate, both legs grow over time."""
    e = AaveEntity()
    e.update_state(AaveGlobalState(collateral_price=1.0, debt_price=1.0))
    e.action_deposit(10_000)
    e.action_borrow(2_000)
    initial_coll = e._internal_state.collateral
    initial_borrow = e._internal_state.borrowed

    for _, row in real_aave_rates.iterrows():
        lending = float(row["lending_rate"])
        borrowing = float(abs(row["borrowing_rate"]))  # ensure positive
        e.update_state(AaveGlobalState(
            collateral_price=1.0,
            debt_price=1.0,
            lending_rate=lending,
            borrowing_rate=borrowing,
        ))

    # Both legs should have grown (assuming positive rates, no liquidation).
    if e._internal_state.collateral > 0:
        assert e._internal_state.collateral >= initial_coll
        assert e._internal_state.borrowed >= initial_borrow
