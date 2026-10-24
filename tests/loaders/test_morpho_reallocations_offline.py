"""Offline tests for the Morpho vault-reallocation events loader."""
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

import fractal.loaders as loaders_package
from fractal.loaders import MorphoMarketLoader, MorphoMarketsLoader, MorphoReallocationsLoader
from fractal.loaders.morpho import MORPHO_GRAPHQL_URL
from fractal.loaders.morpho import MorphoMarketLoader as ModuleMorphoMarketLoader
from fractal.loaders.morpho_markets import MorphoMarketsLoader as ModuleMorphoMarketsLoader
from fractal.loaders.morpho_reallocations import MorphoReallocationsLoader as ModuleMorphoReallocationsLoader

pytestmark = pytest.mark.core

UTC = timezone.utc
CHAIN_ID = 999
START = datetime(2026, 1, 1, tzinfo=UTC)
END = START + timedelta(hours=1)
MARKET_ID = "0x" + "ab" * 32
VAULT = "0x" + "cd" * 20
ALLOCATOR = "0x" + "11" * 20
OTHER_CALLER = "0x" + "22" * 20


def _event(
    event_id: str,
    timestamp: int,
    *,
    event_type: str = "ReallocateSupply",
    assets: str = "1234567",
    shares: str = "7654321",
    caller: str = ALLOCATOR,
    tx_hash: str | None = None,
    log_index: int = 0,
) -> dict:
    return {
        "id": event_id,
        "timestamp": timestamp,
        "type": event_type,
        "assets": assets,
        "shares": shares,
        "caller": caller.upper(),
        "hash": tx_hash or "0x" + event_id[-1] * 64,
        "logIndex": log_index,
        "market": {"marketId": MARKET_ID.upper(), "loanAsset": {"decimals": 6}},
        "vault": {"address": VAULT.upper(), "chain": {"id": str(CHAIN_ID)}},
    }


class _FakeReallocationsHttp:
    def __init__(self, pages: dict[int, list[dict]], total: int | None = None, repeat_first: bool = False):
        self.pages = pages
        self.total = total
        self.repeat_first = repeat_first
        self.calls: list[dict] = []

    def post(self, url, json=None, timeout=None, headers=None):
        assert url == MORPHO_GRAPHQL_URL
        self.calls.append(json)
        skip = json["variables"]["skip"]
        page = self.pages.get(0 if self.repeat_first else skip, [])
        info = {"count": len(page)}
        if self.total is not None:
            info["countTotal"] = self.total
        return {"data": {"vaultReallocates": {"items": page, "pageInfo": info}}}


@pytest.fixture(autouse=True)
def _isolated_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_PATH", str(tmp_path))


def _loader(http=None, **kwargs) -> MorphoReallocationsLoader:
    return MorphoReallocationsLoader(
        CHAIN_ID,
        START,
        END,
        public_allocators={CHAIN_ID: ALLOCATOR},
        http=http or _FakeReallocationsHttp({0: []}, total=0),
        **kwargs,
    )


def test_empty_page_transforms_to_typed_empty_frame():
    loader = _loader()
    frame = loader.read(with_run=True)
    assert frame.empty
    assert list(frame.columns) == [
        "timestamp",
        "chain_id",
        "market_id",
        "vault_address",
        "event_type",
        "assets",
        "shares",
        "transaction_hash",
        "is_public_allocator",
    ]
    assert frame["timestamp"].dtype == "int64"
    assert frame["chain_id"].dtype == "int64"
    assert frame["assets"].dtype == "float64"
    assert frame["is_public_allocator"].dtype == bool


def test_single_event_locks_types_units_and_public_allocator_flag():
    timestamp = int(START.timestamp())
    http = _FakeReallocationsHttp({0: [_event("a", timestamp)]}, total=1)

    frame = _loader(http=http).read(with_run=True)

    assert frame["timestamp"].tolist() == [timestamp]
    assert frame["chain_id"].tolist() == [CHAIN_ID]
    assert frame["market_id"].tolist() == [MARKET_ID]
    assert frame["vault_address"].tolist() == [VAULT]
    assert frame["event_type"].tolist() == ["ReallocateSupply"]
    assert frame["assets"].tolist() == pytest.approx([1.234567])
    assert frame["shares"].tolist() == [7_654_321]
    assert frame["shares"].map(type).eq(int).all()
    assert frame["is_public_allocator"].tolist() == [True]
    assert frame["transaction_hash"].str.islower().all()
    assert "assets shares caller hash logIndex" in http.calls[0]["query"]


def test_withdraw_assets_are_negative_and_non_allocator_is_false():
    loader = _loader()
    loader._data = pd.DataFrame([
        _event(
            "b",
            int(START.timestamp()),
            event_type="ReallocateWithdraw",
            assets="2500000",
            shares="8000000",
            caller=OTHER_CALLER,
        )
    ])

    loader.transform()

    assert loader._data["assets"].tolist() == [-2.5]
    assert loader._data["shares"].tolist() == [8_000_000]
    assert loader._data["is_public_allocator"].tolist() == [False]


def test_two_pages_deduplicate_an_overlapping_event():
    timestamp = int(START.timestamp())
    first = _event("a", timestamp, log_index=0)
    overlap = _event("b", timestamp + 1, log_index=1)
    last = _event("c", timestamp + 2, log_index=2)
    http = _FakeReallocationsHttp({0: [first, overlap], 2: [overlap, last]}, total=3)

    frame = _loader(http=http, page_size=2).read(with_run=True)

    assert len(http.calls) == 2
    assert frame["timestamp"].tolist() == [timestamp, timestamp + 1, timestamp + 2]
    assert len(frame["transaction_hash"].unique()) == 3
    assert all(call["variables"]["where"]["chainId_in"] == [CHAIN_ID] for call in http.calls)


def test_repeated_page_stops_pagination():
    timestamp = int(START.timestamp())
    page = [_event("a", timestamp), _event("b", timestamp + 1)]
    http = _FakeReallocationsHttp({0: page}, total=None, repeat_first=True)

    frame = _loader(http=http, page_size=2).read(with_run=True)

    assert len(http.calls) == 2
    assert len(frame) == 2


def test_cache_key_is_deterministic_and_covers_inputs():
    base = _loader(page_size=50)
    same = _loader(page_size=50)
    assert base._cache_key() == same._cache_key()
    assert base._cache_key() != MorphoReallocationsLoader(
        CHAIN_ID + 1, START, END, public_allocators={CHAIN_ID: ALLOCATOR}, page_size=50
    )._cache_key()
    assert base._cache_key() != MorphoReallocationsLoader(
        CHAIN_ID, START + timedelta(seconds=1), END, public_allocators={CHAIN_ID: ALLOCATOR}, page_size=50
    )._cache_key()
    assert base._cache_key() != _loader(page_size=25)._cache_key()
    assert base._cache_key() != MorphoReallocationsLoader(
        CHAIN_ID, START, END, public_allocators={CHAIN_ID: OTHER_CALLER}, page_size=50
    )._cache_key()


def test_csv_cache_round_trip_preserves_the_tidy_frame():
    timestamp = int(START.timestamp())
    fresh = _loader(
        http=_FakeReallocationsHttp({0: [_event("a", timestamp)]}, total=1)
    ).read(with_run=True)
    cached = _loader().read(with_run=False)
    pd.testing.assert_frame_equal(fresh, cached)


def test_high_risk_package_export_edit_preserves_existing_morpho_exports():
    assert MorphoMarketLoader is ModuleMorphoMarketLoader
    assert MorphoMarketsLoader is ModuleMorphoMarketsLoader
    assert MorphoReallocationsLoader is ModuleMorphoReallocationsLoader
    assert loaders_package.MorphoMarketLoader is ModuleMorphoMarketLoader
    assert loaders_package.MorphoMarketsLoader is ModuleMorphoMarketsLoader


@pytest.mark.integration
def test_morpho_reallocations_live_smoke(request, monkeypatch, tmp_path):
    if request.config.option.markexpr == "core":
        pytest.skip("live smoke is excluded from the offline core suite")
    monkeypatch.setenv("DATA_PATH", str(tmp_path))
    end = datetime.now(UTC)
    frame = MorphoReallocationsLoader(1, end - timedelta(minutes=10), end, page_size=100).read(with_run=True)
    assert isinstance(frame, pd.DataFrame)
    assert list(frame.columns) == list(MorphoReallocationsLoader.COLUMNS)
