"""Offline tests for :class:`MorphoMarketsLoader` against a fake GraphQL client:
chain enumeration, cursor-style paging, the client-side filters, the
server-side ``listed`` clause, cache round-trip and malformed payloads; and the
point-in-time :func:`market_sizes_as_of` snapshot rules."""
import math
import re
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from fractal.loaders import LoaderType, MorphoMarketListing, MorphoMarketsLoader, market_sizes_as_of
from fractal.loaders.morpho import MORPHO_GRAPHQL_URL, MorphoLoaderException
from fractal.loaders.morpho_markets import discover_chains

WAD = int(1e18)


def _market(market_id: str, symbol: str, lltv: float = 0.86, listed: bool = True,
            oracle: str = "ChainlinkOracleV2", liquidity: float = 1_000.0) -> dict:
    return {
        "marketId": market_id,
        "lltv": str(int(lltv * WAD)),
        "listed": listed,
        "creationTimestamp": 1725966209,
        "loanAsset": {"address": "0x" + symbol.lower().encode().hex().ljust(40, "0"), "symbol": symbol, "decimals": 6},
        "collateralAsset": {"address": "0xcoll", "symbol": "PT-" + symbol, "decimals": 18},
        "oracle": {"type": oracle},
        "state": {"supplyAssetsUsd": 10 * liquidity, "borrowAssetsUsd": 9 * liquidity,
                  "liquidityAssetsUsd": liquidity},
    }


MARKETS = {
    1: [
        _market("0x" + "a" * 64, "USDC", lltv=0.86, liquidity=5_000.0),
        _market("0x" + "b" * 64, "USDT", lltv=0.915, liquidity=3_000.0),
        _market("0x" + "c" * 64, "USDC", lltv=0.945, liquidity=9_000.0),
        _market("0x" + "d" * 64, "DAI", lltv=0.86, liquidity=7_000.0),
        _market("0x" + "e" * 64, "USDC", lltv=0.86, listed=False, liquidity=8_000.0),
    ],
    8453: [_market("0x" + "f" * 64, "USDC.e", lltv=0.86), _market("0x" + "1" * 64, "USDT", lltv=0.86)],
}


class _FakeMarketsHttp:
    def __init__(self, markets=None, chains=(1, 8453), malformed: str | None = None,
                 pages: dict[int, list[dict]] | None = None, ignore_skip: bool = False):
        self.markets = MARKETS if markets is None else markets
        self.chains = chains
        self.malformed = malformed
        self.pages = pages
        self.ignore_skip = ignore_skip
        self.market_calls = 0
        self.calls = []

    def post(self, url, json=None, timeout=None, headers=None):
        assert url == MORPHO_GRAPHQL_URL
        self.calls.append(json)
        if self.malformed == "chains":
            return {"data": {}}
        if "Chains" in json["query"]:
            return {"data": {"chains": [{"id": c, "network": f"chain-{c}"} for c in self.chains]}}
        if self.malformed == "markets":
            return {"data": {}}
        if self.malformed == "errors":
            return {"errors": [{"message": "boom"}]}
        variables = json["variables"]
        chain_id = variables["where"]["chainId_in"][0]
        items = self.markets.get(chain_id, [])
        skip, first = variables["skip"], variables["first"]
        self.market_calls += 1
        if self.ignore_skip:
            if self.market_calls > 2:
                raise AssertionError("loader did not stop after the server repeated a page")
            skip = 0
        page = self.pages.get(skip, []) if self.pages is not None else items[skip:skip + first]
        return {"data": {"markets": {"items": page}}}


@pytest.mark.core
def test_chain_enumeration_is_sorted_and_typed():
    assert discover_chains(_FakeMarketsHttp(chains=(8453, 1))) == [(1, "chain-1"), (8453, "chain-8453")]


@pytest.mark.core
def test_discovers_every_chain_and_pages_through_the_result():
    http = _FakeMarketsHttp()
    listings = MorphoMarketsLoader(loan_symbols=("USDC", "USDT"), page_size=2, http=http).read(with_run=True)
    # chain 1: 5 markets → 3 pages; chain 8453: 2 markets → 2 pages (last page short)
    market_calls = [call for call in http.calls if "Markets" in call["query"]]
    assert len(market_calls) == 5
    assert {call["variables"]["where"]["chainId_in"][0] for call in market_calls} == {1, 8453}
    assert all("orderBy: UniqueKey" in call["query"] for call in market_calls)
    assert all(isinstance(listing, MorphoMarketListing) for listing in listings)
    assert [listing.chain_id for listing in listings] == [1, 1, 1, 8453]


@pytest.mark.core
def test_shifting_pages_are_deduplicated_without_losing_markets():
    markets = MARKETS[1][:4]
    pages = {0: markets[:2], 2: markets[1:3], 4: markets[2:4], 6: []}
    listings = MorphoMarketsLoader(
        chain_ids=[1], loan_symbols=("USDC", "USDT", "DAI"), page_size=2,
        http=_FakeMarketsHttp(pages=pages),
    ).read(with_run=True)
    market_ids = [listing.market_id for listing in listings]
    assert len(market_ids) == len(set(market_ids)) == len(markets)
    assert set(market_ids) == {market["marketId"] for market in markets}


@pytest.mark.core
def test_repeated_page_stops_the_chain_sweep():
    http = _FakeMarketsHttp(ignore_skip=True)
    listings = MorphoMarketsLoader(chain_ids=[1], page_size=2, http=http).read(with_run=True)
    assert http.market_calls == 2
    assert len(listings) == 2


@pytest.mark.core
def test_exact_loan_symbol_filter_excludes_bridged_and_other_assets():
    listings = MorphoMarketsLoader(http=_FakeMarketsHttp()).read(with_run=True)
    assert {listing.loan_symbol for listing in listings} == {"USDC", "USDT"}  # no DAI, no USDC.e


@pytest.mark.core
def test_lltv_cap_filters_client_side_and_the_flag_stays_on_the_request():
    http = _FakeMarketsHttp()
    listings = MorphoMarketsLoader(lltv_max=0.915, http=http).read(with_run=True)
    assert all(listing.lltv <= 0.915 for listing in listings)
    assert all(call["variables"]["where"]["listed"] is True for call in http.calls if "Markets" in call["query"])


@pytest.mark.core
def test_listed_flag_excludes_unlisted_markets():
    listings = MorphoMarketsLoader(http=_FakeMarketsHttp()).read(with_run=True)
    assert all(listing.listed for listing in listings)
    everything = MorphoMarketsLoader(listed=False, http=_FakeMarketsHttp()).read(with_run=True)
    assert any(not listing.listed for listing in everything)


@pytest.mark.core
def test_max_markets_caps_the_sweep_independently_of_page_size():
    listings = MorphoMarketsLoader(max_markets=2, http=_FakeMarketsHttp()).read(with_run=True)
    assert len(listings) == 2


@pytest.mark.core
@pytest.mark.parametrize("page_size", [1, 100])
def test_max_markets_counts_kept_markets_across_chains(page_size):
    """The cap counts markets that pass the loan-asset filter, so filtered-out raw items never use it up."""
    loader = MorphoMarketsLoader(loan_symbols=("USDT",), max_markets=2, page_size=page_size, http=_FakeMarketsHttp())
    listings = loader.read(with_run=True)
    assert [(listing.chain_id, listing.loan_symbol) for listing in listings] == [(1, "USDT"), (8453, "USDT")]


@pytest.mark.core
def test_sizes_and_oracle_type_are_carried_and_sorted_by_liquidity():
    listings = MorphoMarketsLoader(http=_FakeMarketsHttp()).read(with_run=True)
    assert all(listing.oracle_type == "ChainlinkOracleV2" for listing in listings)
    assert all(listing.supply_assets_usd - listing.borrow_assets_usd == pytest.approx(listing.liquidity_assets_usd)
               for listing in listings)
    chain1 = [listing.liquidity_assets_usd for listing in listings if listing.chain_id == 1]
    assert chain1 == sorted(chain1, reverse=True)


@pytest.mark.core
@pytest.mark.parametrize("malformed,match", [("chains", "'chains' missing"), ("markets", "'markets' missing"),
                                             ("errors", "GraphQL errors")])
def test_malformed_payloads_raise(malformed, match):
    with pytest.raises(MorphoLoaderException, match=match):
        MorphoMarketsLoader(http=_FakeMarketsHttp(malformed=malformed)).read(with_run=True)


@pytest.mark.core
@pytest.mark.parametrize("kwargs", [{"page_size": 0}, {"max_markets": 0}, {"lltv_max": 1.5}, {"loan_symbols": ()}])
def test_invalid_arguments_rejected(kwargs):
    with pytest.raises(ValueError):
        MorphoMarketsLoader(**kwargs)


@pytest.mark.core
def test_pickle_cache_round_trip(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_PATH", str(tmp_path))
    loader = MorphoMarketsLoader(chain_ids=[1], http=_FakeMarketsHttp())
    first = loader.read(with_run=True)
    cached = MorphoMarketsLoader(chain_ids=[1], http=_FakeMarketsHttp()).read(with_run=False)
    assert first == cached and first
    assert loader.loader_type is LoaderType.PICKLE
    assert loader._cache_key() == "schema-v3-chains-1-listed-True-loan-USDC_USDT-lltv-all-max-all"


@pytest.mark.core
def test_cache_key_includes_max_markets():
    full = MorphoMarketsLoader(chain_ids=[1], http=_FakeMarketsHttp())
    limited = MorphoMarketsLoader(chain_ids=[1], max_markets=2, http=_FakeMarketsHttp())
    assert full._cache_key() != limited._cache_key()


@pytest.mark.core
def test_empty_result_reads_back_as_an_empty_list():
    listing = MorphoMarketsLoader(loan_symbols=("WETH",), http=_FakeMarketsHttp()).read(with_run=True)
    assert listing == []


@pytest.mark.core
def test_created_markets_carry_a_creation_timestamp():
    listings = MorphoMarketsLoader(http=_FakeMarketsHttp()).read(with_run=True)
    assert all(listing.creation_timestamp == 1725966209 for listing in listings)
    assert datetime.fromtimestamp(listings[0].creation_timestamp, tz=timezone.utc).year == 2024


AS_OF = datetime(2026, 6, 15, 13, 0, tzinfo=timezone.utc)
_ALIAS_RE = re.compile(r'(m\d+): marketById\(marketId: "(0x[0-9a-f]{64})", chainId: (\d+)\)')


def _hourly(start: datetime, hours: int, supply, borrow=None) -> dict:
    """``hours`` hourly points from ``start``, newest first like the API; values from callables of the hour."""
    xs = [int((start + timedelta(hours=h)).timestamp()) for h in range(hours)]
    sup = [{"x": x, "y": supply(h)} for h, x in enumerate(xs)]
    bor = [{"x": x, "y": (borrow or supply)(h)} for h, x in enumerate(xs)]
    return {"supplyAssetsUsd": sup[::-1], "borrowAssetsUsd": bor[::-1]}


class _FakeHistoryHttp:
    """Answers aliased ``marketById`` queries with every stored point (window filtering is the helper's job)."""

    def __init__(self, histories: dict[str, dict | None]):
        self.histories = histories
        self.calls = []

    def post(self, url, json=None, timeout=None, headers=None):
        assert url == MORPHO_GRAPHQL_URL
        self.calls.append(json)
        data = {}
        for alias, market_id, _chain in _ALIAS_RE.findall(json["query"]):
            history = self.histories.get(market_id)
            data[alias] = None if history is None else {"historicalState": history}
        return {"data": data}


@pytest.mark.core
def test_as_of_at_or_before_never_uses_points_after_the_instant():
    market = "0x" + "a" * 64
    start = AS_OF - timedelta(hours=3)
    http = _FakeHistoryHttp({market: _hourly(start, 6, lambda h: 100.0 + h, lambda h: 50.0 + h)})
    frame = market_sizes_as_of([(1, market)], AS_OF, http=http)
    assert list(frame.columns) == ["chain_id", "market_id", "time", "supply_assets_usd", "borrow_assets_usd"]
    row = frame.iloc[0]
    assert row["time"] == pd.Timestamp(AS_OF)  # hours +1 and +2 exist but are after the as-of instant
    assert row["supply_assets_usd"] == 103.0 and row["borrow_assets_usd"] == 53.0
    assert http.calls[0]["variables"]["o"] == {
        "startTimestamp": int((AS_OF - timedelta(days=1)).timestamp()),
        "endTimestamp": int(AS_OF.timestamp()), "interval": "HOUR",
    }


@pytest.mark.core
def test_as_of_end_of_day_takes_the_closing_point_of_the_utc_day():
    market = "0x" + "b" * 64
    day = AS_OF.replace(hour=0)
    http = _FakeHistoryHttp({market: _hourly(day - timedelta(hours=2), 30, float)})
    frame = market_sizes_as_of([(1, market)], AS_OF, rule="end_of_day", http=http)
    assert frame["time"].iloc[0] == pd.Timestamp(day + timedelta(days=1))  # the next midnight, after AS_OF
    assert frame["supply_assets_usd"].iloc[0] == 26.0
    variables = http.calls[0]["variables"]["o"]
    assert variables["startTimestamp"] == int(day.timestamp())
    assert variables["endTimestamp"] == int((day + timedelta(days=1)).timestamp())


@pytest.mark.core
def test_as_of_snapshot_is_one_point_and_skips_null_supply():
    market = "0x" + "c" * 64
    history = _hourly(AS_OF - timedelta(hours=2), 3, lambda h: 10.0 + h)
    history["supplyAssetsUsd"][0]["y"] = None  # newest supply point is null → the previous hour is the snapshot
    history["borrowAssetsUsd"] = [p for p in history["borrowAssetsUsd"] if p["x"] != int(AS_OF.timestamp()) - 3600]
    frame = market_sizes_as_of([(1, market)], AS_OF, http=_FakeHistoryHttp({market: history}))
    assert frame["time"].iloc[0] == pd.Timestamp(AS_OF - timedelta(hours=1))
    assert frame["supply_assets_usd"].iloc[0] == 11.0
    assert math.isnan(frame["borrow_assets_usd"].iloc[0])  # borrow is never taken from a different point


@pytest.mark.core
def test_as_of_batches_keep_input_order_and_mark_markets_without_history():
    ids = ["0x" + c * 64 for c in "fed"]
    histories = {
        ids[0]: _hourly(AS_OF - timedelta(hours=1), 2, lambda h: 1.0),
        ids[1]: {"supplyAssetsUsd": [], "borrowAssetsUsd": []},  # not created yet at AS_OF
        ids[2]: _hourly(AS_OF - timedelta(hours=1), 2, lambda h: 3.0),
    }
    http = _FakeHistoryHttp(histories)
    listing = MorphoMarketsLoader(http=_FakeMarketsHttp()).read(with_run=True)[0]
    histories[listing.market_id] = _hourly(AS_OF, 1, lambda h: 7.0)
    frame = market_sizes_as_of([(8453, ids[0]), (1, ids[1].upper().replace("0X", "0x")), (1, ids[2]), listing],
                               AS_OF, batch=2, http=http)
    assert len(http.calls) == 2
    assert frame["market_id"].tolist() == [ids[0], ids[1], ids[2], listing.market_id]
    assert frame["chain_id"].tolist() == [8453, 1, 1, listing.chain_id]
    assert frame["supply_assets_usd"].iloc[[0, 2, 3]].tolist() == [1.0, 3.0, 7.0]
    assert math.isnan(frame["supply_assets_usd"].iloc[1]) and pd.isna(frame["time"].iloc[1])
    assert str(frame["time"].dt.tz) == "UTC"


@pytest.mark.core
def test_as_of_validation_and_unknown_markets():
    market = "0x" + "a" * 64
    http = _FakeHistoryHttp({})
    assert market_sizes_as_of([], AS_OF, http=http).empty and not http.calls
    with pytest.raises(MorphoLoaderException, match="not found"):
        market_sizes_as_of([(1, market)], AS_OF, http=http)
    with pytest.raises(ValueError, match="market_id"):
        market_sizes_as_of([(1, 'x" } evil: chains { id')], AS_OF, http=http)
    with pytest.raises(ValueError, match="rule"):
        market_sizes_as_of([(1, market)], AS_OF, rule="nearest", http=http)
    with pytest.raises(ValueError, match="batch"):
        market_sizes_as_of([(1, market)], AS_OF, batch=0, http=http)
    with pytest.raises(ValueError, match="max_staleness"):
        market_sizes_as_of([(1, market)], AS_OF, max_staleness=timedelta(0), http=http)
    assert len(http.calls) == 1  # validation fails before any request
