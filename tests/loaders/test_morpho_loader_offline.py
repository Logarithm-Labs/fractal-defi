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
RATE_AT_TARGET_APY = 0.07
# The API reports ``rateAtTarget`` as a per-second WAD BigInt (verified live).
RATE_AT_TARGET_WAD = int(math.log1p(RATE_AT_TARGET_APY) / SECONDS_PER_YEAR * 1e18)
SUPPLY_USD, BORROW_USD = 12_000_000.0, 9_800_000.0


class _FakeMorphoHttp:
    def __init__(self, gap_at: int = -1, errors=None, gap_everywhere: bool = False, usd_sizes: bool = True):
        self.gap_at = gap_at
        self.errors = errors
        self.gap_everywhere = gap_everywhere
        self.usd_sizes = usd_sizes
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
        hist = {
            "borrowApy": self._points(0.09, hours),
            "supplyApy": self._points(0.08, hours, self.gap_everywhere),
            "utilization": self._points(0.82, hours, self.gap_everywhere),
            "rateAtTarget": self._points(RATE_AT_TARGET_WAD, hours, self.gap_everywhere),
        }
        if self.usd_sizes:
            hist.update({
                "supplyAssetsUsd": self._points(SUPPLY_USD, hours, self.gap_everywhere),
                "borrowAssetsUsd": self._points(BORROW_USD, hours, self.gap_everywhere),
                "liquidityAssetsUsd": self._points(SUPPLY_USD - BORROW_USD, hours, self.gap_everywhere),
                "netSupplyApy": self._points(0.095, hours, self.gap_everywhere),
            })
        market = {
            "historicalState": hist,
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
                                     "rate_at_target", "supply_rate", "supply_assets_usd", "borrow_assets_usd",
                                     "liquidity_assets_usd", "net_supply_apy"]
    assert history["borrow_apy"].iloc[0] == 0.09 and history["utilization"].iloc[0] == 0.82
    # per-second WAD ``rateAtTarget`` is converted to the annual APY the column documents
    assert history["rate_at_target"].iloc[0] == pytest.approx(RATE_AT_TARGET_APY)
    # supplier-side per-bar rate, the sibling of ``borrowing_rate``
    assert history["supply_rate"].iloc[0] == pytest.approx(math.log1p(0.08) * 3600 / SECONDS_PER_YEAR)
    # USD sizes: the withdrawable amount the exit gate is measured against
    assert history["supply_assets_usd"].iloc[0] == pytest.approx(SUPPLY_USD)
    assert history["borrow_assets_usd"].iloc[0] == pytest.approx(BORROW_USD)
    assert history["liquidity_assets_usd"].iloc[0] == pytest.approx(SUPPLY_USD - BORROW_USD)
    # reward-inclusive APY: rewards = net − base = 1.5 pp in the fixture
    assert history["net_supply_apy"].iloc[0] - history["supply_apy"].iloc[0] == pytest.approx(0.015)
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
    assert history["supply_rate"].iloc[0] == pytest.approx(0.08 / 365)
    continuous = MorphoMarketLoader(MARKET_ID, 1, START, END, http=_FakeMorphoHttp())
    assert continuous._cache_key() != loader._cache_key()
    # the cache key carries the column-set tag, so a pre-extension CSV cannot be mis-read
    assert continuous._cache_key().endswith("v3rewards")


@pytest.mark.core
def test_absent_usd_mirrors_do_not_disturb_the_rate_spine():
    """A chain/field the API has not backfilled must not add rows or NaN the rates."""
    loader = MorphoMarketLoader(MARKET_ID, 1, START, END, http=_FakeMorphoHttp(usd_sizes=False))
    history = loader.read(with_run=True)
    assert len(history) == 6
    assert history["borrowing_rate"].notna().all() and history["utilization"].notna().all()
    assert history["liquidity_assets_usd"].isna().all()
    assert history["net_supply_apy"].isna().all()


@pytest.mark.core
def test_sparse_reward_series_preserves_utc_rate_spine(monkeypatch):
    """Missing rewards remain NaN; rewards never forward-fill or create new bars."""
    http = _FakeMorphoHttp()
    original = http.post

    def post(url, **kwargs):
        payload = original(url, **kwargs)
        history = payload["data"]["marketById"]["historicalState"]
        history["netSupplyApy"] = [
            {"x": int((START + timedelta(hours=2)).timestamp()), "y": 0.12},
            {"x": int((START + timedelta(minutes=30)).timestamp()), "y": 0.15},
        ]
        return payload

    monkeypatch.setattr(http, "post", post)
    history = MorphoMarketLoader(MARKET_ID, 1, START, END, http=http).read(with_run=True)
    assert history.index.equals(pd.date_range(START, END, freq="h"))
    assert history["net_supply_apy"].notna().sum() == 1
    assert history["net_supply_apy"].iloc[2] == pytest.approx(0.12)
    assert history["supply_rate"].notna().all()


@pytest.mark.core
def test_usd_size_series_never_creates_extra_bars():
    """Sizes are left-joined onto the rate spine, so a sparser size series cannot add rows."""
    http = _FakeMorphoHttp()
    original = http.post

    def post(url, json=None, timeout=None, headers=None):
        payload = original(url, json=json, timeout=timeout, headers=headers)
        hist = payload.get("data", {}).get("marketById", {}).get("historicalState")
        if hist and "supplyAssetsUsd" in hist:
            hist["supplyAssetsUsd"] = hist["supplyAssetsUsd"][:3]  # sparse: 3 of 6 bars
        return payload

    http.post = post
    history = MorphoMarketLoader(MARKET_ID, 1, START, END, http=http).read(with_run=True)
    assert len(history) == 6
    assert history["supply_assets_usd"].notna().sum() < 6
    assert history["borrow_assets_usd"].notna().all()


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
            assert json["method"] == "eth_call" and json["params"][0]["data"] == "0x598e5451"
            return {"jsonrpc": "2.0", "id": 1, "result": "0x" + f"{int(0.06 * 1e18):064x}"}

    assert read_linear_discount("http://rpc", "0x" + "cd" * 20, http=_Rpc()) == pytest.approx(0.06)
