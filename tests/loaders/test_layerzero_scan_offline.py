"""Offline tests for the LayerZero Scan OFT message loader."""

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from fractal.loaders import LayerZeroScanLoader
from fractal.loaders._http import LoaderHttpError
from fractal.loaders.bridges.layerzero_scan import LAYERZERO_SCAN_API

SOURCE_EID = 30101
DESTINATION_EID = 30184
OFT_ADDRESS = "0x" + "ab" * 20
START = datetime(2026, 1, 1, tzinfo=timezone.utc)
END = START + timedelta(hours=1)


def _payload(amount_sd: int, *, compose: bytes = b"") -> str:
    send_to = bytes(12) + bytes.fromhex("cd" * 20)
    return "0x" + (send_to + amount_sd.to_bytes(8, "big") + compose).hex()


def _message(
    index: int,
    *,
    destination_timestamp: int | None = None,
    status: str = "DELIVERED",
    payload: str | None = None,
    source_timestamp: int | None = None,
    destination_eid: int = DESTINATION_EID,
    sender: str = OFT_ADDRESS,
) -> dict:
    if source_timestamp is None:
        source_timestamp = int(START.timestamp()) + index * 60
    destination_tx = {} if destination_timestamp is None else {"blockTimestamp": destination_timestamp}
    source_tx = {"txHash": f"0x{index:064x}", "blockTimestamp": source_timestamp}
    if payload is not None:
        source_tx["payload"] = payload
    return {
        "guid": f"0x{index:064x}",
        "pathway": {
            "srcEid": SOURCE_EID,
            "dstEid": destination_eid,
            "sender": {"address": sender.upper()},
            "nonce": index,
        },
        "source": {"tx": source_tx},
        "destination": {"tx": destination_tx},
        "status": {"name": status},
    }


class _FakeHttp:
    def __init__(self, pages: dict[str | None, dict | Exception]) -> None:
        self.pages = pages
        self.calls: list[tuple[str, dict]] = []

    def get(self, url, params=None, timeout=None):
        params = dict(params or {})
        self.calls.append((url, params))
        response = self.pages[params.get("nextToken")]
        if isinstance(response, Exception):
            raise response
        return response


def _loader(**kwargs) -> LayerZeroScanLoader:
    return LayerZeroScanLoader(SOURCE_EID, DESTINATION_EID, OFT_ADDRESS, START, END, **kwargs)


@pytest.mark.core
def test_transform_empty_response_has_stable_columns():
    loader = _loader()
    loader._data = []
    loader.transform()
    assert loader._data.empty
    assert list(loader._data.columns) == [
        "guid",
        "src_eid",
        "dst_eid",
        "src_tx",
        "src_ts",
        "dst_ts",
        "status",
        "amount",
        "latency_s",
    ]


@pytest.mark.core
def test_transform_parses_one_message_and_latency_seconds():
    destination_timestamp = int(START.timestamp()) + 95
    loader = _loader()
    loader._data = [_message(1, destination_timestamp=destination_timestamp)]
    loader.transform()
    row = loader._data.iloc[0]
    assert row["src_eid"] == SOURCE_EID
    assert row["dst_eid"] == DESTINATION_EID
    assert row["src_tx"] == f"0x{1:064x}"
    assert row["src_ts"] == int(START.timestamp()) + 60
    assert row["dst_ts"] == destination_timestamp
    assert row["status"] == "DELIVERED"
    assert row["latency_s"] == 35.0


@pytest.mark.core
def test_continuation_404_stops_pagination():
    http = _FakeHttp({
        None: {"data": [_message(0)], "nextToken": "next-page"},
        "next-page": LoaderHttpError("HTTP 404 for continuation"),
    })
    loader = _loader(http=http)
    loader.extract()
    loader.transform()
    assert len(loader._data) == 1
    assert [params.get("nextToken") for _, params in http.calls] == [None, "next-page"]
    assert all(url == f"{LAYERZERO_SCAN_API}/messages/oapp/{SOURCE_EID}/{OFT_ADDRESS}" for url, _ in http.calls)


@pytest.mark.core
def test_missing_destination_timestamp_has_nan_latency():
    loader = _loader()
    loader._data = [_message(2)]
    loader.transform()
    assert pd.isna(loader._data.loc[0, "dst_ts"])
    assert pd.isna(loader._data.loc[0, "latency_s"])


@pytest.mark.core
def test_max_messages_uses_deterministic_even_time_subsample():
    messages = [_message(index) for index in range(6)]
    first = _loader(max_messages=3)
    first._data = list(reversed(messages))
    first.transform()
    second = _loader(max_messages=3)
    second._data = messages
    second.transform()
    expected = [f"0x{index:064x}" for index in (0, 2, 4)]
    assert first._data["guid"].tolist() == expected
    assert second._data["guid"].tolist() == expected


@pytest.mark.core
def test_cache_key_is_deterministic_and_covers_inputs():
    base = _loader(max_messages=5)._cache_key()
    assert _loader(max_messages=5)._cache_key() == base
    different_source = LayerZeroScanLoader(
        SOURCE_EID + 1, DESTINATION_EID, OFT_ADDRESS, START, END, max_messages=5
    )
    different_destination = LayerZeroScanLoader(
        SOURCE_EID, DESTINATION_EID + 1, OFT_ADDRESS, START, END, max_messages=5
    )
    different_address = LayerZeroScanLoader(
        SOURCE_EID, DESTINATION_EID, "0x" + "cd" * 20, START, END, max_messages=5
    )
    assert different_source._cache_key() != base
    assert different_destination._cache_key() != base
    assert different_address._cache_key() != base
    assert LayerZeroScanLoader(SOURCE_EID, DESTINATION_EID, OFT_ADDRESS, START + timedelta(seconds=1), END,
                               max_messages=5)._cache_key() != base
    assert _loader(max_messages=6)._cache_key() != base


@pytest.mark.core
def test_cache_round_trip_preserves_typed_columns(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_PATH", str(tmp_path))
    destination_timestamp = int(START.timestamp()) + 95
    messages = [_message(0), _message(1, destination_timestamp=destination_timestamp), _message(2)]
    http = _FakeHttp({None: {"data": list(reversed(messages)), "nextToken": ""}})
    first = _loader(http=http).read(with_run=True)
    cached = _loader(http=_FakeHttp({None: {"data": [], "nextToken": ""}})).read(with_run=False)
    pd.testing.assert_frame_equal(first, cached)
    assert first.dtypes["src_ts"] == "int64" and first.dtypes["dst_ts"] == "Int64"
    assert pd.isna(first.loc[0, "dst_ts"]) and first.loc[1, "dst_ts"] == destination_timestamp
    assert pd.isna(first.loc[0, "latency_s"]) and first.loc[1, "latency_s"] == 35.0


@pytest.mark.integration
def test_layerzero_scan_integration_smoke():
    # USDT0's OFT on Ethereum — public protocol constant from the LayerZero app registry.
    usdt0_ethereum = "0x6c96de32cea08842dcc4058c14d3aaad7fa41dee"
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=7)
    loader = LayerZeroScanLoader(30101, 30367, usdt0_ethereum, start, end, max_messages=5)
    frame = loader.read(with_run=True)
    assert list(frame.columns) == list(LayerZeroScanLoader._columns)
    if not frame.empty:  # the route may carry no traffic inside the window
        assert (frame["src_eid"] == 30101).all()
        assert (frame["dst_eid"] == 30367).all()
        assert frame["latency_s"].dropna().ge(0).all()


@pytest.mark.core
def test_latency_only_for_delivered_messages():
    destination_timestamp = int(START.timestamp()) + 300
    loader = _loader()
    loader._data = [_message(1, destination_timestamp=destination_timestamp, status="PAYLOAD_STORED")]
    loader.transform()
    row = loader._data.iloc[0]
    assert row["status"] == "PAYLOAD_STORED"
    assert row["dst_ts"] == destination_timestamp
    assert pd.isna(row["latency_s"])


@pytest.mark.core
def test_amount_decoded_from_oft_payload_with_shared_decimals():
    messages = [
        _message(0, payload=_payload(1_234_567)),
        _message(1, payload=_payload(2_500_000, compose=b"\x01" * 40)),
        _message(2, payload="0x" + "00" * 39),
        _message(3, payload="0xzz"),
        _message(4),
    ]
    loader = _loader()
    loader._data = messages
    loader.transform()
    amounts = loader._data["amount"].tolist()
    assert amounts[:2] == [1.234567, 2.5]
    assert all(pd.isna(value) for value in amounts[2:])
    scaled = _loader(shared_decimals=8)
    scaled._data = messages[:1]
    scaled.transform()
    assert scaled._data["amount"].tolist() == [0.01234567]
    assert _loader(shared_decimals=8)._cache_key() != _loader()._cache_key()


@pytest.mark.core
def test_transform_filters_route_sender_window_and_duplicates():
    start = int(START.timestamp())
    end = int(END.timestamp())
    keep_first = _message(0, source_timestamp=start)
    keep_last = _message(1, source_timestamp=end - 1)
    loader = _loader()
    loader._data = [
        keep_last,
        keep_first,
        dict(keep_first),  # duplicate GUID
        _message(2, source_timestamp=start - 1),
        _message(3, source_timestamp=end),
        _message(4, destination_eid=DESTINATION_EID + 1),
        _message(5, sender="0x" + "ef" * 20),
        {"guid": "0xmalformed"},
        _message(6, source_timestamp=None) | {"source": {"tx": {"txHash": "0x06"}}},
    ]
    loader.transform()
    assert loader._data["guid"].tolist() == [keep_first["guid"], keep_last["guid"]]
    assert loader._data["src_ts"].tolist() == [start, end - 1]


@pytest.mark.core
def test_iso_and_nan_timestamps_are_parsed():
    source_iso = "2026-01-01T00:05:00.000Z"
    message = _message(1, destination_timestamp=None)
    message["source"]["tx"]["blockTimestamp"] = source_iso
    message["destination"]["tx"]["blockTimestamp"] = float("nan")
    loader = _loader()
    loader._data = [message]
    loader.transform()
    assert loader._data.loc[0, "src_ts"] == int(START.timestamp()) + 300
    assert pd.isna(loader._data.loc[0, "dst_ts"])


@pytest.mark.core
def test_extract_follows_tokens_and_sends_utc_window():
    http = _FakeHttp({
        None: {"data": [_message(0)], "nextToken": "page-2"},
        "page-2": {"data": [_message(1)], "nextToken": "page-3"},
        "page-3": {"data": [_message(2)]},
    })
    loader = _loader(http=http)
    loader.extract()
    assert len(loader._data) == 3
    first_params = http.calls[0][1]
    assert first_params == {"limit": 100, "start": "2026-01-01T00:00:00.000Z", "end": "2026-01-01T01:00:00.000Z"}
    assert [params.get("nextToken") for _, params in http.calls] == [None, "page-2", "page-3"]


@pytest.mark.core
def test_repeated_continuation_token_raises():
    http = _FakeHttp({
        None: {"data": [], "nextToken": "loop"},
        "loop": {"data": [], "nextToken": "loop"},
    })
    with pytest.raises(ValueError, match="repeated continuation token"):
        _loader(http=http).extract()


@pytest.mark.core
def test_first_page_404_is_not_swallowed():
    http = _FakeHttp({None: LoaderHttpError("HTTP 404 for unknown application")})
    with pytest.raises(LoaderHttpError):
        _loader(http=http).extract()


@pytest.mark.core
@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"source_eid": -1}, "non-negative"),
        ({"oft_address": " "}, "oft_address"),
        ({"end_time": START}, "must be after"),
        ({"max_messages": 0}, "max_messages"),
        ({"shared_decimals": -1}, "shared_decimals"),
    ],
)
def test_constructor_validation(kwargs, message):
    arguments = {
        "source_eid": SOURCE_EID,
        "destination_eid": DESTINATION_EID,
        "oft_address": OFT_ADDRESS,
        "start_time": START,
        "end_time": END,
    }
    options = {key: kwargs.pop(key) for key in list(kwargs) if key in ("max_messages", "shared_decimals")}
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=message):
        LayerZeroScanLoader(**arguments, **options)
