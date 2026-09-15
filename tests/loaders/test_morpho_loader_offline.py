"""Offline tests for the Morpho loaders against a fake GraphQL client:
``marketById`` queries, descending points, the dropped live point,
APY → per-bar conversion, optional columns, NaN policy, oracle read."""
import math
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from fractal.loaders import LendingHistory, LoaderType, MorphoMarketLoader
from fractal.loaders._dt import SECONDS_PER_YEAR
from fractal.loaders.morpho import MORPHO_GRAPHQL_URL, MorphoLoaderException, get_market_info, read_linear_discount

MARKET_ID = "0x1e9d614631a7df0ec07fb05b2c8cb2491575fd1a63a33bf187a6afb295a4fc64"  # PT-reUSD-10DEC2026 / USDC
START = datetime(2026, 9, 1, tzinfo=timezone.utc)
END = START + timedelta(hours=5)


class _FakeMorphoHttp:
    def __init__(self, gap_at: int = -1, errors=None, gap_everywhere: bool = False):
        self.gap_at = gap_at
        self.errors = errors
        self.gap_everywhere = gap_everywhere
        self.calls = []

    def _points(self, value, hours, allow_gap=True):
        pts = [{"x": int((START + timedelta(hours=h)).timestamp()), "y": value} for h in range(hours + 1)]
        if self.gap_at >= 0 and allow_gap:
            pts.pop(self.gap_at)
        pts.append({"x": int((START + timedelta(hours=hours)).timestamp()) + 1234, "y": value})  # live point
        return list(reversed(pts))  # API returns newest first

    def post(self, url, json=None, timeout=None, headers=None):
        assert url == MORPHO_GRAPHQL_URL
        self.calls.append(json)
        if self.errors:
            return {"errors": self.errors}
        if "MarketInfo" in json["query"]:
            market = {
                "marketId": MARKET_ID, "lltv": "915000000000000000", "irmAddress": "0xirm",
                "oracleAddress": "0xoracle", "oracle": {"address": "0xoracle", "type": "MorphoChainlinkOracleV2"},
                "loanAsset": {"address": "0xUSDC", "symbol": "USDC", "decimals": 6},
                "collateralAsset": {"address": "0xPT", "symbol": "PT-reUSD-10DEC2026", "decimals": 6},
                "state": {"price": "974400000000000000000000000000000000", "fee": 0, "borrowApy": 0.09,
                          "supplyApy": 0.08, "utilization": 0.82, "rateAtTarget": 31709791},
            }
            return {"data": {"marketById": market}}
        hours = int((END - START).total_seconds() // 3600)
        market = {
            "historicalState": {
                "borrowApy": self._points(0.09, hours),
                "supplyApy": self._points(0.08, hours, self.gap_everywhere),
                "utilization": self._points(0.82, hours, self.gap_everywhere),
                "rateAtTarget": self._points(0.07, hours, self.gap_everywhere),
            },
            "collateralAsset": {"historicalPriceUsd": self._points(0.974, hours, self.gap_everywhere)},
        }
        return {"data": {"marketById": market}}


@pytest.mark.core
def test_market_info_parses_lltv_assets_and_scaled_price():
    info = get_market_info(MARKET_ID, 1, http=_FakeMorphoHttp())
    assert info.lltv == pytest.approx(0.915)
    assert info.loan_symbol == "USDC" and info.collateral_decimals == 6
    assert info.price_scale == pytest.approx(1e36)
    assert info.oracle_price == pytest.approx(0.9744)
    assert info.oracle_type == "MorphoChainlinkOracleV2" and info.borrow_apy == 0.09


@pytest.mark.core
def test_history_is_ascending_aligned_and_converted_to_per_bar_rates():
    http = _FakeMorphoHttp()
    loader = MorphoMarketLoader(MARKET_ID, 1, START, END, http=http)
    history = loader.read(with_run=True)
    assert isinstance(history, LendingHistory)
    assert len(history) == 6  # live unaligned point dropped
    assert history.index.is_monotonic_increasing and history.index[0] == pd.Timestamp(START)
    assert history["borrowing_rate"].iloc[0] == pytest.approx(math.log1p(0.09) * 3600 / SECONDS_PER_YEAR)
    assert (history["lending_rate"] == 0.0).all()  # Morpho Blue collateral earns nothing
    assert history["supply_apy"].iloc[0] == pytest.approx(0.08)  # the supplier-side rate stays available
    assert list(history.columns) == ["lending_rate", "borrowing_rate", "utilization", "borrow_apy", "supply_apy",
                                     "rate_at_target"]
    assert history["borrow_apy"].iloc[0] == 0.09 and history["utilization"].iloc[0] == 0.82
    variables = http.calls[-1]["variables"]
    assert variables["id"] == MARKET_ID and variables["cid"] == 1
    assert variables["o"] == {"startTimestamp": int(START.timestamp()), "endTimestamp": int(END.timestamp()),
                              "interval": "HOUR"}
    assert "marketById(marketId: $id, chainId: $cid)" in http.calls[-1]["query"]


@pytest.mark.core
def test_linear_compounding_and_resolution_scale_like_aave():
    loader = MorphoMarketLoader(MARKET_ID, 1, START, END, resolution=24, compounding="linear", http=_FakeMorphoHttp())
    history = loader.read(with_run=True)
    assert history["borrowing_rate"].iloc[0] == pytest.approx(0.09 / 365)
    continuous = MorphoMarketLoader(MARKET_ID, 1, START, END, http=_FakeMorphoHttp())
    assert continuous._cache_key() != loader._cache_key()


@pytest.mark.core
def test_gap_in_one_series_raises_on_nan():
    loader = MorphoMarketLoader(MARKET_ID, 1, START, END, http=_FakeMorphoHttp(gap_at=2))
    with pytest.raises(ValueError, match="NaN in required column"):
        loader.read(with_run=True)


@pytest.mark.core
def test_missing_bar_in_every_series_raises_on_grid_gap():
    loader = MorphoMarketLoader(MARKET_ID, 1, START, END, http=_FakeMorphoHttp(gap_at=2, gap_everywhere=True))
    with pytest.raises(ValueError, match="gap"):
        loader.read(with_run=True)


@pytest.mark.core
def test_graphql_errors_and_validation_raise():
    with pytest.raises(MorphoLoaderException, match="GraphQL errors"):
        MorphoMarketLoader(MARKET_ID, 1, START, END, http=_FakeMorphoHttp(errors=[{"message": "boom"}])).extract()
    with pytest.raises(ValueError):
        MorphoMarketLoader("0x1234", 1, START, END)
    with pytest.raises(ValueError):
        MorphoMarketLoader(MARKET_ID, 1, START, END, interval="MINUTE")
    with pytest.raises(ValueError):
        MorphoMarketLoader(MARKET_ID, 1, START, END, compounding="daily")
    with pytest.raises(ValueError):
        MorphoMarketLoader(MARKET_ID, 1, END, START)


@pytest.mark.core
def test_collateral_price_series_and_cache_round_trip(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_PATH", str(tmp_path))
    loader = MorphoMarketLoader(MARKET_ID, 1, START, END, http=_FakeMorphoHttp(), loader_type=LoaderType.JSON)
    first = loader.read(with_run=True)
    price = loader.read_collateral_price()
    assert len(price) == 6 and price["price"].iloc[0] == pytest.approx(0.974)
    cached = MorphoMarketLoader(MARKET_ID, 1, START, END, http=_FakeMorphoHttp(), loader_type=LoaderType.JSON)
    pd.testing.assert_frame_equal(first, cached.read(with_run=False))


@pytest.mark.core
def test_read_linear_discount_decodes_the_feed():
    class _Rpc:
        def post(self, url, json=None, timeout=None, headers=None):
            assert json["method"] == "eth_call" and json["params"][0]["data"] == "0x61d5a1f7"
            return {"jsonrpc": "2.0", "id": 1, "result": "0x" + f"{int(0.06 * 1e18):064x}"}

    assert read_linear_discount("http://rpc", "0x" + "cd" * 20, http=_Rpc()) == pytest.approx(0.06)
