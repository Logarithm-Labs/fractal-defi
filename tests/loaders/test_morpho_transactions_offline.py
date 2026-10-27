"""Offline tests for the Morpho Blue market-transactions loader (synthetic payloads only)."""
import warnings
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

import fractal.loaders as loaders_package
from fractal.loaders import MorphoMarketLoader, MorphoMarketsLoader, MorphoMarketTransactionsLoader
from fractal.loaders.morpho import MORPHO_GRAPHQL_URL, MorphoLoaderException
from fractal.loaders.morpho import MorphoMarketLoader as ModuleMorphoMarketLoader
from fractal.loaders.morpho_markets import MorphoMarketsLoader as ModuleMorphoMarketsLoader
from fractal.loaders.morpho_transactions import MARKET_TRANSACTION_TYPES
from fractal.loaders.morpho_transactions import MorphoMarketTransactionsLoader as ModuleTransactionsLoader

pytestmark = pytest.mark.core

UTC = timezone.utc
CHAIN_ID = 999
START = datetime(2026, 1, 1, tzinfo=UTC)
END = START + timedelta(hours=1)
ONE_SECOND = START + timedelta(seconds=1)
T0 = int(START.timestamp())
DAY = 24 * 3600
MARKET_A = "0x" + "ab" * 32
MARKET_B = "0x" + "cd" * 32
USER = "0x" + "12" * 20
LIQUIDATOR = "0x" + "34" * 20
COLUMNS = [
    "timestamp",
    "chain_id",
    "block_number",
    "tx_index",
    "log_index",
    "transaction_hash",
    "market_id",
    "event_type",
    "user",
    "asset_side",
    "assets",
    "shares",
    "seized_assets",
    "bad_debt_assets",
    "bad_debt_shares",
    "liquidator",
    "loan_symbol",
    "collateral_symbol",
]


def _event(
    tag: str,
    timestamp: int,
    *,
    event_type: str = "Borrow",
    data: dict | None = None,
    log_index: int = 0,
    block: int | str = 100,
    loan_decimals: int | None = 6,
    collateral_decimals: int | None = 18,
    market_id: str = MARKET_A,
    tx_hash: str | None = None,
) -> dict:
    if data is None:
        data = {"__typename": "MarketTransactionTransferData", "assets": "2500000", "shares": "2500000000000"}
    return {
        "txHash": tx_hash or "0x" + tag[-1] * 64,
        "timestamp": timestamp,
        "blockNumber": block,
        "txIndex": 4,
        "logIndex": log_index,
        "type": event_type,
        "chain": {"id": CHAIN_ID},
        "user": {"address": USER.upper()},
        "market": {
            "marketId": market_id.upper(),
            "loanAsset": {"symbol": "LOAN", "decimals": loan_decimals},
            "collateralAsset": {"symbol": "COLL", "decimals": collateral_decimals},
        },
        "data": data,
    }


def _collateral(tag: str, timestamp: int, assets="3000000000000000000", **kwargs) -> dict:
    data = {"__typename": "MarketTransactionCollateralTransferData", "assets": assets}
    return _event(tag, timestamp, event_type=kwargs.pop("event_type", "SupplyCollateral"), data=data, **kwargs)


def _liquidation(tag: str, timestamp: int, **kwargs) -> dict:
    data = {
        "__typename": "MarketTransactionLiquidationData",
        "liquidator": LIQUIDATOR.upper(),
        "repaidAssets": "1500000",
        "repaidShares": "1499999000000",
        "seizedAssets": "500000000000000000",
        "badDebtAssets": 250_000,
        "badDebtShares": 249_000_000_000,
    }
    return _event(tag, timestamp, event_type="Liquidation", data=data, **kwargs)


class _FakeHttp:
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
        return {"data": {"marketTransactions": {"items": page, "pageInfo": info}}}


class _EventStoreHttp:
    """Answers each query from an event list by its inclusive timestamp range, ``skip`` and ``first``."""

    def __init__(self, events: list[dict]):
        self.events = sorted(events, key=lambda event: event["timestamp"])
        self.windows: list[tuple[int, int, int]] = []

    def post(self, url, json=None, timeout=None, headers=None):
        assert url == MORPHO_GRAPHQL_URL
        variables = json["variables"]
        where = variables["where"]
        lower, upper = where["timestamp_gte"], where["timestamp_lte"]
        self.windows.append((lower, upper, variables["skip"]))
        matching = [event for event in self.events if lower <= event["timestamp"] <= upper]
        page = matching[variables["skip"]:variables["skip"] + variables["first"]]
        info = {"count": len(page), "countTotal": len(matching)}
        return {"data": {"marketTransactions": {"items": page, "pageInfo": info}}}


@pytest.fixture(autouse=True)
def _isolated_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_PATH", str(tmp_path))


def _loader(http=None, start=START, end=END, **kwargs) -> MorphoMarketTransactionsLoader:
    return MorphoMarketTransactionsLoader(
        CHAIN_ID, start, end, http=http or _FakeHttp({0: []}, total=0), **kwargs
    )


def _frame(events: list[dict], **kwargs) -> pd.DataFrame:
    return _loader(http=_FakeHttp({0: events}, total=len(events)), **kwargs).read(with_run=True)


def test_empty_page_transforms_to_typed_empty_frame():
    frame = _loader().read(with_run=True)
    assert frame.empty
    assert list(frame.columns) == COLUMNS
    assert list(MorphoMarketTransactionsLoader.COLUMNS) == COLUMNS
    for column in ("timestamp", "chain_id", "block_number", "tx_index", "log_index"):
        assert frame[column].dtype == "int64"
    for column in ("assets", "seized_assets", "bad_debt_assets"):
        assert frame[column].dtype == "float64"


def test_loan_event_uses_loan_decimals_and_keeps_raw_shares():
    http = _FakeHttp({0: [_event("a", T0, log_index=7, block="123456789")]}, total=1)
    frame = _loader(http=http).read(with_run=True)

    row = frame.iloc[0]
    assert row["event_type"] == "Borrow"
    assert row["asset_side"] == "loan"
    assert row["assets"] == pytest.approx(2.5)
    assert row["shares"] == 2_500_000_000_000
    assert isinstance(row["shares"], int)
    assert (row["seized_assets"], row["bad_debt_assets"], row["bad_debt_shares"], row["liquidator"]) == (
        0.0, 0.0, 0, "",
    )
    assert row["block_number"] == 123_456_789
    assert (row["tx_index"], row["log_index"], row["chain_id"]) == (4, 7, CHAIN_ID)
    assert row["market_id"] == MARKET_A
    assert row["user"] == USER
    assert row["transaction_hash"] == "0x" + "a" * 64
    assert (row["loan_symbol"], row["collateral_symbol"]) == ("LOAN", "COLL")


def test_collateral_event_uses_collateral_decimals_and_has_no_shares():
    frame = _frame([_collateral("a", T0, event_type="WithdrawCollateral")])

    row = frame.iloc[0]
    assert row["event_type"] == "WithdrawCollateral"
    assert row["asset_side"] == "collateral"
    assert row["assets"] == pytest.approx(3.0)
    assert row["shares"] == 0


def test_liquidation_carries_repaid_seized_and_bad_debt_amounts():
    frame = _frame([_liquidation("a", T0)])

    row = frame.iloc[0]
    assert row["event_type"] == "Liquidation"
    assert row["asset_side"] == "loan"
    assert row["assets"] == pytest.approx(1.5)
    assert row["shares"] == 1_499_999_000_000
    assert row["seized_assets"] == pytest.approx(0.5)
    assert row["bad_debt_assets"] == pytest.approx(0.25)
    assert row["bad_debt_shares"] == 249_000_000_000
    assert row["liquidator"] == LIQUIDATOR


def test_query_selects_one_fragment_per_event_family():
    http = _FakeHttp({0: []}, total=0)
    _loader(http=http).run()
    query = http.calls[0]["query"]
    assert "... on MarketTransactionTransferData { assets shares }" in query
    assert "... on MarketTransactionCollateralTransferData { assets }" in query
    assert "... on MarketTransactionLiquidationData" in query
    for field in ("liquidator", "repaidAssets", "repaidShares", "seizedAssets", "badDebtAssets", "badDebtShares"):
        assert field in query


@pytest.mark.parametrize("event", [
    _event("a", T0, event_type="SupplyCollateral", data={"__typename": "MarketTransactionTransferData"}),
    _event("a", T0, event_type="Liquidation", data={"__typename": "MarketTransactionTransferData"}),
    _event("a", T0, event_type="Repay", data={}),
])
def test_event_without_its_family_amounts_raises_instead_of_dropping_them(event):
    with pytest.raises(MorphoLoaderException, match="event data lacks"):
        _loader(http=_FakeHttp({0: [event]}, total=1)).run()


@pytest.mark.parametrize(("event", "message"), [
    (_event("a", T0, event_type="Deposit"), "unsupported market transaction type"),
    (_event("a", T0, loan_decimals=None), "loan-asset decimals missing"),
    (_collateral("a", T0, collateral_decimals=None), "collateral-asset decimals missing"),
    (_liquidation("a", T0, collateral_decimals=None), "collateral-asset decimals missing"),
    (_event("a", T0, data={"assets": "1.5", "shares": "1"}), "integer amount"),
])
def test_malformed_events_raise(event, message):
    with pytest.raises(MorphoLoaderException, match=message):
        _loader(http=_FakeHttp({0: [event]}, total=1)).run()


def test_json_number_and_string_amounts_scale_exactly():
    big_shares = 10**30 + 1
    events = [
        _event("a", T0, data={"assets": 1_132_797, "shares": 1_031_999_657_071}, log_index=0),
        _event("b", T0 + 1, data={"assets": "123456789012345678901234567", "shares": str(big_shares)},
               loan_decimals=18, log_index=1),
    ]
    frame = _frame(events)
    assert frame["assets"].tolist() == [1.132797, 123456789.01234567]
    assert frame["shares"].tolist() == [1_031_999_657_071, big_shares]


def test_missing_block_raises():
    class _NoBlock:
        def post(self, url, json=None, timeout=None, headers=None):
            return {"data": {}}

    with pytest.raises(MorphoLoaderException, match="'marketTransactions' missing"):
        _loader(http=_NoBlock()).run()


def test_filters_are_normalised_into_the_where_clause():
    http = _FakeHttp({0: []}, total=0)
    _loader(
        http=http,
        market_ids=[MARKET_B.upper().replace("0X", "0x"), MARKET_A, MARKET_B],
        types=["Liquidation", "Borrow", "Borrow"],
    ).run()

    where = http.calls[0]["variables"]["where"]
    assert where["chainId_in"] == [CHAIN_ID]
    assert where["marketUniqueKey_in"] == [MARKET_A, MARKET_B]
    assert where["type_in"] == ["Borrow", "Liquidation"]


def test_no_filters_query_every_market_and_type():
    http = _FakeHttp({0: []}, total=0)
    _loader(http=http).run()
    where = http.calls[0]["variables"]["where"]
    assert "marketUniqueKey_in" not in where
    assert "type_in" not in where
    assert set(MARKET_TRANSACTION_TYPES) == {
        "Supply", "Withdraw", "Borrow", "Repay", "SupplyCollateral", "WithdrawCollateral", "Liquidation",
    }


def test_single_string_filters_are_accepted():
    http = _FakeHttp({0: []}, total=0)
    _loader(http=http, market_ids=MARKET_A, types="Repay").run()
    where = http.calls[0]["variables"]["where"]
    assert where["marketUniqueKey_in"] == [MARKET_A]
    assert where["type_in"] == ["Repay"]


@pytest.mark.parametrize(("kwargs", "message"), [
    ({"page_size": 0}, "page_size"),
    ({"page_size": 1001}, "page_size"),
    ({"start": END, "end": START}, "precedes"),
    ({"types": ["Deposit"]}, "unknown market transaction type"),
    ({"types": []}, "at least one market transaction type"),
    ({"market_ids": ["0x1234"]}, "market_id must match"),
    ({"market_ids": []}, "at least one market"),
])
def test_invalid_arguments_raise(kwargs, message):
    with pytest.raises(ValueError, match=message):
        _loader(**kwargs)


def test_rows_sort_by_time_block_and_log_index_and_deduplicate_by_hash_and_log():
    tx = "0x" + "e" * 64
    events = [
        _event("1", T0 + 5, block=11, log_index=1),
        _liquidation("2", T0, block=10, log_index=9, tx_hash=tx),
        _collateral("3", T0, block=10, log_index=2, tx_hash=tx),
        _collateral("4", T0, block=10, log_index=2, tx_hash=tx.upper().replace("0X", "0x")),
    ]
    frame = _frame(events)

    assert frame["event_type"].tolist() == ["SupplyCollateral", "Liquidation", "Borrow"]
    assert frame["log_index"].tolist() == [2, 9, 1]


def test_two_pages_deduplicate_an_overlapping_event():
    first = _event("a", T0, log_index=0)
    overlap = _event("b", T0 + 1, log_index=1)
    last = _event("c", T0 + 2, log_index=2)
    http = _FakeHttp({0: [first, overlap], 2: [overlap, last]}, total=3)

    frame = _loader(http=http, page_size=2, end=ONE_SECOND).read(with_run=True)

    assert len(http.calls) == 2
    assert [call["variables"]["skip"] for call in http.calls] == [0, 2]
    assert frame["timestamp"].tolist() == [T0, T0 + 1, T0 + 2]


def test_repeated_page_stops_pagination_and_warns_when_short_of_count_total():
    page = [_event("a", T0), _event("b", T0 + 1)]
    http = _FakeHttp({0: page}, total=5, repeat_first=True)

    with pytest.warns(UserWarning, match="2 distinct events of countTotal 5"):
        frame = _loader(http=http, page_size=2, end=ONE_SECOND).read(with_run=True)

    assert len(http.calls) == 2
    assert len(frame) == 2


def test_repeated_page_without_count_total_stops_silently():
    page = [_event("a", T0), _event("b", T0 + 1)]
    http = _FakeHttp({0: page}, total=None, repeat_first=True)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        frame = _loader(http=http, page_size=2).read(with_run=True)

    assert len(http.calls) == 2
    assert len(frame) == 2


def test_window_is_walked_in_inclusive_one_day_slices_without_gaps():
    end = START + timedelta(days=2, hours=3)
    t_end = int(end.timestamp())
    events = [
        _event("a", T0 - 1, log_index=0),
        _event("b", T0, log_index=1),
        _event("c", T0 + DAY - 1, log_index=2),
        _event("d", T0 + DAY, log_index=3),
        _event("e", t_end - 1, log_index=4),
        _event("f", t_end, log_index=5),
    ]
    http = _EventStoreHttp(events)

    frame = _loader(http=http, end=end).read(with_run=True)

    assert [window[:2] for window in http.windows] == [
        (T0, T0 + DAY - 1), (T0 + DAY, T0 + 2 * DAY - 1), (T0 + 2 * DAY, t_end - 1),
    ]
    assert frame["timestamp"].tolist() == [T0, T0 + DAY - 1, T0 + DAY, t_end - 1]


def test_fractional_bounds_round_up_to_whole_seconds_in_query_and_cache_key():
    http = _EventStoreHttp([_event("a", T0, log_index=0), _event("b", T0 + 10, log_index=1)])
    loader = _loader(http=http, start=START + timedelta(milliseconds=500), end=START + timedelta(seconds=10.5))

    frame = loader.read(with_run=True)

    assert http.windows == [(T0 + 1, T0 + 10, 0)]
    assert frame["timestamp"].tolist() == [T0 + 10]
    assert loader._cache_key() == _loader(start=START + timedelta(seconds=1),
                                          end=START + timedelta(seconds=11))._cache_key()


class _LossySkipStoreHttp(_EventStoreHttp):
    """Event store whose ``skip > 0`` pages lose their first event, like the live endpoint."""

    def post(self, url, json=None, timeout=None, headers=None):
        payload = super().post(url, json=json, timeout=timeout, headers=headers)
        if json["variables"]["skip"] > 0:
            block = payload["data"]["marketTransactions"]
            block["items"] = block["items"][1:]
            block["pageInfo"]["count"] = len(block["items"])
        return payload


def test_slice_denser_than_a_page_is_bisected_instead_of_skip_paged():
    events = [_event(f"e{i}", T0 + 600 * i, log_index=i) for i in range(6)]
    http = _EventStoreHttp(events)

    frame = _loader(http=http, page_size=2).read(with_run=True)

    assert frame["log_index"].tolist() == list(range(6))
    assert {skip for _, _, skip in http.windows} == {0}
    middle = (T0 + T0 + 3599) // 2
    assert http.windows[:3] == [(T0, T0 + 3599, 0), (T0, middle, 0), (T0, (T0 + middle) // 2, 0)]


def test_bisection_recovers_every_event_from_a_lossy_skip_endpoint():
    events = [_event(f"e{i}", T0 + 7 * i, log_index=i) for i in range(9)]
    http = _LossySkipStoreHttp(events)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        frame = _loader(http=http, page_size=2).read(with_run=True)

    assert frame["log_index"].tolist() == list(range(9))


def test_single_second_denser_than_a_page_is_skip_paged():
    events = [_event(f"e{i}", T0, log_index=i) for i in range(3)]
    http = _EventStoreHttp(events)

    frame = _loader(http=http, page_size=2, end=ONE_SECOND).read(with_run=True)

    assert http.windows == [(T0, T0, 0), (T0, T0, 2)]
    assert frame["log_index"].tolist() == [0, 1, 2]


def test_short_page_does_not_stop_paging_before_count_total():
    events = [_event("a", T0, log_index=0), _event("b", T0, log_index=1), _event("c", T0, log_index=2)]
    http = _FakeHttp({0: events[:1], 2: events[1:]}, total=3)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        frame = _loader(http=http, page_size=2, end=ONE_SECOND).read(with_run=True)

    assert [call["variables"]["skip"] for call in http.calls] == [0, 2]
    assert frame["log_index"].tolist() == [0, 1, 2]


def test_single_second_above_skip_cap_raises(monkeypatch):
    monkeypatch.setattr("fractal.loaders.morpho_transactions._SKIP_CAP", 2)
    events = [_event(f"e{i}", T0, log_index=i) for i in range(3)]

    with pytest.raises(MorphoLoaderException, match="cannot paginate below the API skip cap"):
        _loader(http=_EventStoreHttp(events), page_size=2, end=ONE_SECOND).run()


def test_cache_key_is_deterministic_and_covers_inputs():
    base = _loader(market_ids=[MARKET_A, MARKET_B], types=["Borrow", "Repay"])
    assert base._cache_key() == _loader(market_ids=[MARKET_B, MARKET_A], types=["Repay", "Borrow"])._cache_key()
    variants = [
        MorphoMarketTransactionsLoader(CHAIN_ID + 1, START, END, market_ids=[MARKET_A, MARKET_B],
                                       types=["Borrow", "Repay"]),
        _loader(start=START + timedelta(seconds=1), market_ids=[MARKET_A, MARKET_B], types=["Borrow", "Repay"]),
        _loader(market_ids=[MARKET_A], types=["Borrow", "Repay"]),
        _loader(market_ids=[MARKET_A, MARKET_B], types=["Borrow"]),
        _loader(types=["Borrow", "Repay"]),
        _loader(market_ids=[MARKET_A, MARKET_B]),
        _loader(market_ids=[MARKET_A, MARKET_B], types=["Borrow", "Repay"], page_size=50),
    ]
    keys = {base._cache_key(), *(variant._cache_key() for variant in variants)}
    assert len(keys) == len(variants) + 1


def test_csv_cache_round_trip_preserves_every_event_family():
    events = [
        _event("a", T0, data={"assets": "1", "shares": str(10**30 + 1)}, log_index=0),
        _collateral("b", T0 + 1, log_index=1),
        _liquidation("c", T0 + 2, log_index=2),
    ]
    fresh = _frame(events)
    cached = _loader().read(with_run=False)

    pd.testing.assert_frame_equal(fresh, cached)
    assert cached["shares"].tolist()[0] == 10**30 + 1
    assert cached["liquidator"].tolist() == ["", "", LIQUIDATOR]


def test_empty_csv_cache_round_trip_keeps_columns_and_dtypes():
    fresh = _loader().read(with_run=True)
    cached = _loader().read(with_run=False)
    pd.testing.assert_frame_equal(fresh, cached)


def test_package_export_is_additive():
    assert MorphoMarketTransactionsLoader is ModuleTransactionsLoader
    assert loaders_package.MorphoMarketLoader is ModuleMorphoMarketLoader
    assert loaders_package.MorphoMarketsLoader is ModuleMorphoMarketsLoader
    assert MorphoMarketLoader is ModuleMorphoMarketLoader
    assert MorphoMarketsLoader is ModuleMorphoMarketsLoader
