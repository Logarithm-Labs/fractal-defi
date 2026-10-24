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


def _message(index: int, *, destination_timestamp: int | None = None) -> dict:
    source_timestamp = int(START.timestamp()) + index * 60
    destination_tx = {} if destination_timestamp is None else {"blockTimestamp": destination_timestamp}
    return {
        "guid": f"0x{index:064x}",
        "pathway": {
            "srcEid": SOURCE_EID,
            "dstEid": DESTINATION_EID,
            "sender": {"address": OFT_ADDRESS.upper()},
            "nonce": index,
        },
        "source": {
            "tx": {
                "txHash": f"0x{index:064x}",
                "blockTimestamp": source_timestamp,
            }
        },
        "destination": {"tx": destination_tx},
        "status": {"name": "DELIVERED"},
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
