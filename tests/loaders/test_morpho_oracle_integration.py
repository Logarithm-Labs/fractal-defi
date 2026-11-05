"""Live check of :class:`fractal.loaders.onchain.MorphoOraclePriceLoader`.

Opt-in: past-block ``eth_call`` needs an archive node, so the test runs
only when ``ETH_ARCHIVE_RPC_URL`` points at one and skips otherwise.
"""
import os

import pandas as pd
import pytest

from fractal.loaders import MorphoOraclePriceLoader
from fractal.loaders.morpho import get_market_info

MARKET_ID = "0x1e9d614631a7df0ec07fb05b2c8cb2491575fd1a63a33bf187a6afb295a4fc64"  # PT-reUSD-10DEC2026 / USDC


@pytest.mark.integration
def test_live_daily_oracle_reads_track_the_api_state_price(tmp_path, monkeypatch):
    rpc_url = os.environ.get("ETH_ARCHIVE_RPC_URL")
    if not rpc_url:
        pytest.skip("ETH_ARCHIVE_RPC_URL is not set (historical eth_call needs an archive node)")
    monkeypatch.setenv("DATA_PATH", str(tmp_path))
    end = pd.Timestamp.now(tz="UTC").floor("1D")
    prices = MorphoOraclePriceLoader(rpc_url, MARKET_ID, end - pd.Timedelta("2D"), end).read(with_run=True)
    assert len(prices) == 3 and (prices["price"] > 0).all()
    # the API's state price is the live one; a PT oracle moves well under 2 % in a day
    assert prices["price"].iloc[-1] == pytest.approx(get_market_info(MARKET_ID, 1).oracle_price, rel=0.02)
