"""Offline tests for :class:`MorphoMarketsLoader` against a fake GraphQL client:
chain enumeration, cursor-style paging, the client-side filters, the
server-side ``listed`` clause, cache round-trip and malformed payloads."""
from datetime import datetime, timezone

import pytest

from fractal.loaders import LoaderType, MorphoMarketListing, MorphoMarketsLoader
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
    def __init__(self, markets=None, chains=(1, 8453), malformed: str | None = None):
        self.markets = MARKETS if markets is None else markets
        self.chains = chains
        self.malformed = malformed
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
        return {"data": {"markets": {"items": items[skip:skip + first]}}}


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
    assert all(isinstance(listing, MorphoMarketListing) for listing in listings)
    assert [listing.chain_id for listing in listings] == [1, 1, 1, 8453]


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
def test_max_markets_caps_the_sweep():
    listings = MorphoMarketsLoader(max_markets=2, page_size=1, http=_FakeMarketsHttp()).read(with_run=True)
    assert len(listings) <= 2


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
    assert loader._cache_key() == "chains-1-listed-True-loan-USDC_USDT-lltv-all"


@pytest.mark.core
def test_empty_result_reads_back_as_an_empty_list():
    listing = MorphoMarketsLoader(loan_symbols=("WETH",), http=_FakeMarketsHttp()).read(with_run=True)
    assert listing == []


@pytest.mark.core
def test_created_markets_carry_a_creation_timestamp():
    listings = MorphoMarketsLoader(http=_FakeMarketsHttp()).read(with_run=True)
    assert all(listing.creation_timestamp == 1725966209 for listing in listings)
    assert datetime.fromtimestamp(listings[0].creation_timestamp, tz=timezone.utc).year == 2024
