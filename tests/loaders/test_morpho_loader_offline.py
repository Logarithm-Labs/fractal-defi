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
    assert continuous._cache_key().endswith("v4marketstate")


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


class _MarketTotalsHttp(_FakeMorphoHttp):
    def __init__(self, series, loan_asset=None):
        super().__init__()
        self.series = series
        self.loan_asset = {"decimals": 6} if loan_asset is None else loan_asset

    def post(self, url, json=None, timeout=None, headers=None):
        payload = super().post(url, json=json, timeout=timeout, headers=headers)
        market = payload["data"]["marketById"]
        market["loanAsset"] = self.loan_asset
        market["historicalState"].update(self.series)
        return payload


def _at(hour, value, minutes=0):
    return {"x": int((START + timedelta(hours=hour, minutes=minutes)).timestamp()), "y": value}


class _EmptyMarketTotalsHttp(_MarketTotalsHttp):
    def post(self, url, json=None, timeout=None, headers=None):
        payload = super().post(url, json=json, timeout=timeout, headers=headers)
        payload["data"]["marketById"]["collateralAsset"]["historicalPriceUsd"] = []
        return payload


@pytest.mark.core
def test_market_totals_are_opt_in_and_default_columns_are_unchanged():
    default = MorphoMarketLoader(MARKET_ID, 1, START, END, http=_FakeMorphoHttp())
    with_totals = MorphoMarketLoader(
        MARKET_ID, 1, START, END, include_market_totals=True,
        http=_MarketTotalsHttp({}),
    )
    default_history = default.read(with_run=True)
    assert list(default_history.columns) == [
        "lending_rate", "borrowing_rate", "utilization", "borrow_apy", "supply_apy",
        "rate_at_target", "supply_rate", "supply_assets_usd", "borrow_assets_usd",
        "liquidity_assets_usd", "net_supply_apy",
    ]
    assert default._cache_key() == MorphoMarketLoader(
        MARKET_ID, 1, START, END, http=_FakeMorphoHttp(),
    )._cache_key()
    assert default._cache_key() != with_totals._cache_key()


@pytest.mark.core
def test_market_totals_scale_raw_values_and_preserve_the_rate_spine():
    def point(hour, value):
        return [{"x": int((START + timedelta(hours=hour)).timestamp()), "y": value}]

    series = {
        "supplyAssets": point(0, "123456789"),
        "supplyShares": point(1, "124000000000000"),
        "borrowAssets": point(2, "98765432"),
        "borrowShares": [
            {"x": int((START + timedelta(hours=3)).timestamp()), "y": "99000000000000"},
            {"x": int((START + timedelta(minutes=30)).timestamp()), "y": "1000000000000"},
        ],
    }
    http = _MarketTotalsHttp(series)
    history = MorphoMarketLoader(
        MARKET_ID, 1, START, END, include_market_totals=True, http=http,
    ).read(with_run=True)
    assert history.index.equals(pd.date_range(START, END, freq="h"))
    assert list(history.columns[-4:]) == ["supply_assets", "supply_shares", "borrow_assets", "borrow_shares"]
    assert history["supply_assets"].iloc[0] == pytest.approx(123.456789)
    assert history["supply_shares"].iloc[1] == pytest.approx(124.0)
    assert history["borrow_assets"].iloc[2] == pytest.approx(98.765432)
    assert history["borrow_shares"].iloc[3] == pytest.approx(99.0)
    total_columns = ["supply_assets", "supply_shares", "borrow_assets", "borrow_shares"]
    assert history[total_columns].notna().sum().tolist() == [1, 1, 1, 1]
    query = http.calls[-1]["query"]
    assert all(field in query for field in ("supplyAssets", "supplyShares", "borrowAssets", "borrowShares"))
    assert "loanAsset { decimals" in query


@pytest.mark.core
def test_empty_market_totals_return_the_opt_in_schema():
    history = MorphoMarketLoader(
        MARKET_ID, 1, START, END, include_market_totals=True,
        http=_EmptyMarketTotalsHttp({
            "borrowApy": [], "supplyApy": [], "utilization": [], "rateAtTarget": [],
            "supplyAssets": [], "supplyShares": [], "borrowAssets": [], "borrowShares": [],
        }),
    ).read(with_run=True)
    assert history.empty
    assert list(history.columns) == [
        "lending_rate", "borrowing_rate", "supply_assets", "supply_shares", "borrow_assets", "borrow_shares",
    ]


@pytest.mark.core
def test_default_query_does_not_request_market_totals():
    http = _FakeMorphoHttp()
    MorphoMarketLoader(MARKET_ID, 1, START, END, http=http).read(with_run=True)
    query = http.calls[-1]["query"]
    assert not any(field in query for field in ("supplyAssets(", "supplyShares", "borrowAssets(", "borrowShares"))
    assert "supplyAssetsUsd" in query and "borrowAssetsUsd" in query


@pytest.mark.core
def test_market_totals_parse_every_bigint_encoding_exactly():
    big_shares = 123_456_789_012_345_678_901_234  # > 2**53: a float round trip of the raw value would lose digits
    series = {
        "supplyAssets": [_at(0, 123_456_789), _at(1, "0x75bcd15"), _at(2, "1.23456789e8"), _at(3, None),
                         _at(4, " 123456789 ")],
        "supplyShares": [_at(0, str(big_shares)), _at(1, big_shares)],
        "borrowAssets": [{"x": None, "y": "1"}, _at(0, "-5000000")],
    }
    history = MorphoMarketLoader(
        MARKET_ID, 1, START, END, include_market_totals=True, http=_MarketTotalsHttp(series),
    ).read(with_run=True)
    assert history["supply_assets"].iloc[:3].tolist() == pytest.approx([123.456789] * 3, rel=1e-15)
    assert math.isnan(history["supply_assets"].iloc[3])  # a null point is NaN, not a crash
    assert history["supply_assets"].iloc[4] == pytest.approx(123.456789, rel=1e-15)
    assert history["supply_shares"].iloc[0] == big_shares / 10**12  # correctly rounded int / int
    assert history["supply_shares"].iloc[1] == big_shares / 10**12
    assert history["borrow_assets"].iloc[0] == pytest.approx(-5.0)  # a point without ``x`` is skipped
    assert len(history) == 6


@pytest.mark.core
def test_market_totals_reject_malformed_bigints_and_missing_decimals():
    with pytest.raises(MorphoLoaderException, match="supply_assets"):
        MorphoMarketLoader(
            MARKET_ID, 1, START, END, include_market_totals=True,
            http=_MarketTotalsHttp({"supplyAssets": [_at(0, "not-a-number")]}),
        ).read(with_run=True)
    with pytest.raises(MorphoLoaderException, match="supply_shares"):
        MorphoMarketLoader(
            MARKET_ID, 1, START, END, include_market_totals=True,
            http=_MarketTotalsHttp({"supplyShares": [_at(0, True)]}),
        ).read(with_run=True)
    with pytest.raises(MorphoLoaderException, match="decimals missing"):
        MorphoMarketLoader(
            MARKET_ID, 1, START, END, include_market_totals=True,
            http=_MarketTotalsHttp({}, loan_asset={"symbol": "USDC"}),
        ).read(with_run=True)
    # the default loader never needs the loan decimals
    MorphoMarketLoader(MARKET_ID, 1, START, END, http=_MarketTotalsHttp({}, loan_asset={})).read(with_run=True)


@pytest.mark.core
def test_share_price_is_one_at_the_virtual_share_offset():
    # Morpho Blue mints 1e6 shares per asset unit at creation: 1e6 USDC (6 dp) ↔ 1e18 raw shares.
    series = {"supplyAssets": [_at(0, 10**12)], "supplyShares": [_at(0, 10**18)]}
    history = MorphoMarketLoader(
        MARKET_ID, 1, START, END, include_market_totals=True, http=_MarketTotalsHttp(series),
    ).read(with_run=True)
    assert history["supply_assets"].iloc[0] == pytest.approx(1_000_000.0)
    assert history["supply_assets"].iloc[0] / history["supply_shares"].iloc[0] == pytest.approx(1.0)


@pytest.mark.core
def test_loan_price_is_left_joined_onto_the_bars():
    prices = [_at(h, 0.999 + h / 10_000) for h in range(3)] + [_at(1, 1.5, minutes=30)]
    http = _MarketTotalsHttp({}, loan_asset={"decimals": 6, "historicalPriceUsd": prices})
    loader = MorphoMarketLoader(MARKET_ID, 1, START, END, http=http)
    history = loader.read(with_run=True)
    assert len(history) == 6  # an off-grid price point never adds a bar
    assert "loan_price_usd" not in history.columns  # LendingHistory columns are unchanged
    price = loader.read_loan_price()
    assert price["price"].tolist() == pytest.approx([0.999, 0.9991, 0.9992])
    assert price.index[0] == pd.Timestamp(START)
    assert "historicalPriceUsd" in http.calls[-1]["query"].split("loanAsset", 1)[1].split("}", 1)[0]
    # markets without a loan price (the base fake) read an empty series
    no_price = MorphoMarketLoader(MARKET_ID, 1, START, END, http=_FakeMorphoHttp())
    no_price.read(with_run=True)
    assert no_price.read_loan_price().empty


@pytest.mark.core
def test_market_totals_cache_is_isolated_from_the_default_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_PATH", str(tmp_path))
    series = {"supplyAssets": [_at(h, str(10**12 + h)) for h in range(6)],
              "supplyShares": [_at(h, str(10**18)) for h in range(6)]}
    MorphoMarketLoader(MARKET_ID, 1, START, END, http=_FakeMorphoHttp()).read(with_run=True)
    totals = MorphoMarketLoader(MARKET_ID, 1, START, END, include_market_totals=True,
                                http=_MarketTotalsHttp(series))
    with pytest.raises(FileNotFoundError):  # the default CSV is never read back as a totals history
        totals.read(with_run=False)
    first = totals.read(with_run=True)
    cached = MorphoMarketLoader(MARKET_ID, 1, START, END, include_market_totals=True,
                                http=_MarketTotalsHttp(series)).read(with_run=False)
    pd.testing.assert_frame_equal(first, cached)
    assert cached["supply_assets"].iloc[5] == pytest.approx(1_000_000.000005, rel=1e-15)


@pytest.mark.core
def test_read_linear_discount_decodes_the_feed():
    class _Rpc:
        def post(self, url, json=None, timeout=None, headers=None):
            assert json["method"] == "eth_call" and json["params"][0]["data"] == "0x598e5451"
            return {"jsonrpc": "2.0", "id": 1, "result": "0x" + f"{int(0.06 * 1e18):064x}"}

    assert read_linear_discount("http://rpc", "0x" + "cd" * 20, http=_Rpc()) == pytest.approx(0.06)
