"""Offline tests for the Morpho vault-reallocation events loader."""
import warnings
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

import fractal.loaders as loaders_package
from fractal.loaders import MorphoMarketLoader, MorphoMarketsLoader, MorphoReallocationsLoader
from fractal.loaders.morpho import MORPHO_GRAPHQL_URL, MorphoLoaderException
from fractal.loaders.morpho import MorphoMarketLoader as ModuleMorphoMarketLoader
from fractal.loaders.morpho_markets import MorphoMarketsLoader as ModuleMorphoMarketsLoader
from fractal.loaders.morpho_reallocations import MorphoReallocationsLoader as ModuleMorphoReallocationsLoader

pytestmark = pytest.mark.core

UTC = timezone.utc
CHAIN_ID = 999
START = datetime(2026, 1, 1, tzinfo=UTC)
END = START + timedelta(hours=1)
T0 = int(START.timestamp())
MARKET_ID = "0x" + "ab" * 32
VAULT = "0x" + "cd" * 20
ALLOCATOR = "0x" + "11" * 20
OTHER_CALLER = "0x" + "22" * 20
COLUMNS = [
    "timestamp",
    "chain_id",
    "market_id",
    "vault_address",
    "event_type",
    "assets",
    "shares",
    "caller",
    "transaction_hash",
    "log_index",
    "is_public_allocator",
]


def _event(
    event_id: str,
    timestamp: int,
    *,
    event_type: str = "ReallocateSupply",
    assets: int | str = "1234567",
    shares: int | str = "7654321",
    caller: str = ALLOCATOR,
    tx_hash: str | None = None,
    log_index: int = 0,
    decimals: int | None = 6,
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
        "market": {"marketId": MARKET_ID.upper(), "loanAsset": {"decimals": decimals}},
        "vault": {"address": VAULT.upper(), "chain": {"id": CHAIN_ID}},
    }


class _FakeReallocationsHttp:
    """Serves fixed pages keyed by ``skip``, ignoring the timestamp filter."""

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


class _EventStoreHttp:
    """Answers each query from an event list by its inclusive timestamp range, ``skip`` and ``first``.

    With ``short_pages`` a page that is not the last one comes back one event
    short and that event never appears on any page, as the live API does.
    """

    def __init__(self, events: list[dict], short_pages: bool = False):
        self.events = sorted(events, key=lambda event: event["timestamp"])
        self.short_pages = short_pages
        self.windows: list[tuple[int, int, int]] = []

    def post(self, url, json=None, timeout=None, headers=None):
        assert url == MORPHO_GRAPHQL_URL
        variables = json["variables"]
        where = variables["where"]
        lower, upper = where["timestamp_gte"], where["timestamp_lte"]
        self.windows.append((lower, upper, variables["skip"]))
        matching = [event for event in self.events if lower <= event["timestamp"] <= upper]
        skip, first = variables["skip"], variables["first"]
        more_follow = skip + first < len(matching)
        page = matching[skip:skip + first - (1 if self.short_pages and more_follow else 0)]
        info = {"count": len(page), "countTotal": len(matching)}
        return {"data": {"vaultReallocates": {"items": page, "pageInfo": info}}}


@pytest.fixture(autouse=True)
def _isolated_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_PATH", str(tmp_path))


def _loader(http=None, start=START, end=END, **kwargs) -> MorphoReallocationsLoader:
    return MorphoReallocationsLoader(
        CHAIN_ID,
        start,
        end,
        public_allocators={CHAIN_ID: ALLOCATOR},
        http=http or _FakeReallocationsHttp({0: []}, total=0),
        **kwargs,
    )


def test_empty_page_transforms_to_typed_empty_frame():
    loader = _loader()
    frame = loader.read(with_run=True)
    assert frame.empty
    assert list(frame.columns) == COLUMNS
    assert list(MorphoReallocationsLoader.COLUMNS) == COLUMNS
    assert frame["timestamp"].dtype == "int64"
    assert frame["chain_id"].dtype == "int64"
    assert frame["log_index"].dtype == "int64"
    assert frame["assets"].dtype == "float64"
    assert frame["is_public_allocator"].dtype == bool


def test_single_event_locks_types_units_and_public_allocator_flag():
    http = _FakeReallocationsHttp({0: [_event("a", T0, log_index=7)]}, total=1)

    frame = _loader(http=http).read(with_run=True)

    assert frame["timestamp"].tolist() == [T0]
    assert frame["chain_id"].tolist() == [CHAIN_ID]
    assert frame["market_id"].tolist() == [MARKET_ID]
    assert frame["vault_address"].tolist() == [VAULT]
    assert frame["event_type"].tolist() == ["ReallocateSupply"]
    assert frame["assets"].tolist() == pytest.approx([1.234567])
    assert frame["shares"].tolist() == [7_654_321]
    assert frame["shares"].map(type).eq(int).all()
    assert frame["caller"].tolist() == [ALLOCATOR]
    assert frame["log_index"].tolist() == [7]
    assert frame["is_public_allocator"].tolist() == [True]
    assert frame["transaction_hash"].str.islower().all()
    assert "assets shares caller hash logIndex" in http.calls[0]["query"]


def test_withdraw_assets_are_negative_and_non_allocator_is_false():
    loader = _loader()
    loader._data = pd.DataFrame([
        _event(
            "b",
            T0,
            event_type="ReallocateWithdraw",
            assets="2500000",
            shares="8000000",
            caller=OTHER_CALLER,
        )
    ])

    loader.transform()

    assert loader._data["assets"].tolist() == [-2.5]
    assert loader._data["shares"].tolist() == [8_000_000]
    assert loader._data["caller"].tolist() == [OTHER_CALLER]
    assert loader._data["is_public_allocator"].tolist() == [False]


def test_allocator_on_another_chain_is_not_public_for_this_chain():
    loader = MorphoReallocationsLoader(
        CHAIN_ID, START, END, public_allocators={CHAIN_ID + 1: ALLOCATOR},
        http=_FakeReallocationsHttp({0: [_event("a", T0)]}, total=1),
    )
    assert loader.read(with_run=True)["is_public_allocator"].tolist() == [False]


def test_json_number_and_string_amounts_scale_exactly():
    big_shares = 10**30 + 1
    events = [
        _event("a", T0, assets=1_132_797, shares=1_031_999_657_071, log_index=0),
        _event("b", T0 + 1, assets="123456789012345678901234567", shares=str(big_shares),
               decimals=18, log_index=1),
    ]
    frame = _loader(http=_FakeReallocationsHttp({0: events}, total=2)).read(with_run=True)

    assert frame["assets"].tolist() == [1.132797, 123456789.01234567]
    assert frame["shares"].tolist() == [1_031_999_657_071, big_shares]


def test_non_integer_amount_is_rejected():
    loader = _loader(http=_FakeReallocationsHttp({0: [_event("a", T0, assets="1.5")]}, total=1))
    with pytest.raises(MorphoLoaderException, match="integer amount"):
        loader.run()


def test_unsupported_type_and_missing_decimals_raise():
    loader = _loader(http=_FakeReallocationsHttp({0: [_event("a", T0, event_type="Deposit")]}, total=1))
    with pytest.raises(MorphoLoaderException, match="unsupported reallocation type"):
        loader.run()
    loader = _loader(http=_FakeReallocationsHttp({0: [_event("a", T0, decimals=None)]}, total=1))
    with pytest.raises(MorphoLoaderException, match="decimals missing"):
        loader.run()


def test_missing_block_raises():
    class _NoBlock:
        def post(self, url, json=None, timeout=None, headers=None):
            return {"data": {}}

    with pytest.raises(MorphoLoaderException, match="'vaultReallocates' missing"):
        _loader(http=_NoBlock()).run()


@pytest.mark.parametrize(("kwargs", "message"), [
    ({"page_size": 0}, "page_size"),
    ({"page_size": 1001}, "page_size"),
    ({"start": END, "end": START}, "precedes"),
])
def test_invalid_arguments_raise(kwargs, message):
    with pytest.raises(ValueError, match=message):
        _loader(**kwargs)


def test_identical_events_in_one_transaction_stay_distinct_by_log_index():
    twin = {"tx_hash": "0x" + "e" * 64, "assets": "100", "shares": "100"}
    events = [_event("1", T0, log_index=5, **twin), _event("2", T0, log_index=3, **twin)]

    frame = _loader(http=_FakeReallocationsHttp({0: events}, total=2)).read(with_run=True)

    assert frame["log_index"].tolist() == [3, 5]
    assert frame.drop(columns="log_index").duplicated().iloc[1]


def test_two_pages_in_one_second_deduplicate_an_overlapping_event():
    first = _event("a", T0, log_index=0)
    overlap = _event("b", T0, log_index=1)
    last = _event("c", T0, log_index=2)
    http = _FakeReallocationsHttp({0: [first, overlap], 2: [overlap, last]}, total=3)

    frame = _loader(http=http, end=START + timedelta(seconds=1), page_size=2).read(with_run=True)

    assert len(http.calls) == 2
    assert frame["log_index"].tolist() == [0, 1, 2]
    assert len(frame["transaction_hash"].unique()) == 3
    assert all(call["variables"]["where"]["chainId_in"] == [CHAIN_ID] for call in http.calls)


def test_repeated_page_stops_pagination_and_warns_when_short_of_count_total():
    page = [_event("a", T0), _event("b", T0, log_index=1)]
    http = _FakeReallocationsHttp({0: page}, total=5, repeat_first=True)

    with pytest.warns(UserWarning, match="2 distinct events of countTotal 5"):
        frame = _loader(http=http, end=START + timedelta(seconds=1), page_size=2).read(with_run=True)

    assert len(http.calls) == 2
    assert len(frame) == 2


def test_repeated_page_without_count_total_stops_silently():
    page = [_event("a", T0), _event("b", T0 + 1)]
    http = _FakeReallocationsHttp({0: page}, total=None, repeat_first=True)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        frame = _loader(http=http, page_size=2).read(with_run=True)

    assert len(http.calls) == 2
    assert len(frame) == 2


def test_window_is_walked_in_inclusive_six_hour_slices_without_gaps():
    end = START + timedelta(hours=13)
    t_end = int(end.timestamp())
    six = 6 * 3600
    events = [
        _event("a", T0 - 1, log_index=0),
        _event("b", T0, log_index=1),
        _event("c", T0 + six - 1, log_index=2),
        _event("d", T0 + six, log_index=3),
        _event("e", t_end - 1, log_index=4),
        _event("f", t_end, log_index=5),
    ]
    http = _EventStoreHttp(events)

    frame = _loader(http=http, end=end).read(with_run=True)

    assert [window[:2] for window in http.windows] == [
        (T0, T0 + six - 1), (T0 + six, T0 + 2 * six - 1), (T0 + 2 * six, t_end - 1),
    ]
    assert frame["timestamp"].tolist() == [T0, T0 + six - 1, T0 + six, t_end - 1]


def test_fractional_bounds_round_up_to_whole_seconds_in_query_and_cache_key():
    http = _EventStoreHttp([_event("a", T0, log_index=0), _event("b", T0 + 10, log_index=1)])
    loader = _loader(http=http, start=START + timedelta(milliseconds=500), end=START + timedelta(seconds=10.5))

    frame = loader.read(with_run=True)

    assert http.windows == [(T0 + 1, T0 + 10, 0)]
    assert frame["timestamp"].tolist() == [T0 + 10]
    assert loader._cache_key() == _loader(start=START + timedelta(seconds=1),
                                          end=START + timedelta(seconds=11))._cache_key()
    assert loader._cache_key() != _loader(start=START, end=START + timedelta(seconds=11))._cache_key()


def test_slice_above_one_page_is_bisected_in_time_without_skip():
    events = [_event(f"e{i}", T0 + 600 * i, log_index=i) for i in range(6)]
    http = _EventStoreHttp(events)

    frame = _loader(http=http, page_size=2).read(with_run=True)

    assert frame["log_index"].tolist() == list(range(6))
    first_pages = [(lower, upper) for lower, upper, skip in http.windows if skip == 0]
    middle = (T0 + T0 + 3599) // 2
    assert first_pages[:2] == [(T0, T0 + 3599), (T0, middle)]
    assert (middle + 1, T0 + 3599) in first_pages
    assert all(skip == 0 for _, _, skip in http.windows)


def test_short_skip_pages_do_not_drop_events_across_seconds():
    events = [_event(f"e{i}", T0 + 60 * i, log_index=i) for i in range(7)]
    http = _EventStoreHttp(events, short_pages=True)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        frame = _loader(http=http, page_size=3).read(with_run=True)

    assert frame["log_index"].tolist() == list(range(7))
    assert all(skip == 0 for _, _, skip in http.windows)


def test_short_page_within_one_second_is_not_terminal():
    events = [_event(f"e{i}", T0, log_index=i) for i in range(5)]
    http = _EventStoreHttp(events, short_pages=True)

    with pytest.warns(UserWarning, match="3 distinct events of countTotal 5"):
        frame = _loader(http=http, end=START + timedelta(seconds=1), page_size=2).read(with_run=True)

    assert [skip for _, _, skip in http.windows] == [0, 2, 4]
    assert frame["log_index"].tolist() == [0, 2, 4]


def test_single_second_above_skip_cap_raises(monkeypatch):
    monkeypatch.setattr("fractal.loaders.morpho_reallocations._SKIP_CAP", 2)
    events = [_event(f"e{i}", T0, log_index=i) for i in range(3)]

    with pytest.raises(MorphoLoaderException, match="cannot paginate below the API skip cap"):
        _loader(http=_EventStoreHttp(events), end=START + timedelta(seconds=1)).run()


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
    events = [
        _event("a", T0, shares=str(10**30 + 1), log_index=2),
        _event("b", T0 + 1, event_type="ReallocateWithdraw", caller=OTHER_CALLER, log_index=0),
    ]
    fresh = _loader(http=_FakeReallocationsHttp({0: events}, total=2)).read(with_run=True)
    cached = _loader().read(with_run=False)
    pd.testing.assert_frame_equal(fresh, cached)
    assert cached["shares"].tolist()[0] == 10**30 + 1


def test_empty_csv_cache_round_trip_keeps_columns_and_dtypes():
    fresh = _loader().read(with_run=True)
    cached = _loader().read(with_run=False)
    pd.testing.assert_frame_equal(fresh, cached)


def test_high_risk_package_export_edit_preserves_existing_morpho_exports():
    assert MorphoMarketLoader is ModuleMorphoMarketLoader
    assert MorphoMarketsLoader is ModuleMorphoMarketsLoader
    assert MorphoReallocationsLoader is ModuleMorphoReallocationsLoader
    assert loaders_package.MorphoMarketLoader is ModuleMorphoMarketLoader
    assert loaders_package.MorphoMarketsLoader is ModuleMorphoMarketsLoader
