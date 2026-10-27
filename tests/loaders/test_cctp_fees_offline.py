"""Offline tests for the Circle Iris V2 CCTP fee, allowance and message loaders."""

import pandas as pd
import pytest

from fractal.loaders import CCTP_DOMAINS, CctpFastAllowanceLoader, CctpFeeLoader, CctpMessagesLoader
from fractal.loaders._http import LoaderHttpError
from fractal.loaders.bridges.cctp import IRIS_API

SOURCE_DOMAIN = 0
DESTINATION_DOMAIN = 6


class _FakeHttp:
    def __init__(self, payload) -> None:
        self.payload = payload
        self.calls: list[tuple[str, dict]] = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, dict(params or {})))
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


@pytest.mark.core
def test_parses_fast_and_standard_fee_entries():
    http = _FakeHttp([
        {"finalityThreshold": 1000, "minimumFee": "5"},
        {"finalityThreshold": 2000, "minimumFee": "0"},
    ])
    loader = CctpFeeLoader(SOURCE_DOMAIN, DESTINATION_DOMAIN, http=http)
    loader.extract()
    loader.transform()
    assert loader._data.to_dict("records") == [
        {
            "src_domain": SOURCE_DOMAIN,
            "dst_domain": DESTINATION_DOMAIN,
            "mode": "fast",
            "minimum_fee_bps": 5.0,
            "finality_threshold": 1000,
        },
        {
            "src_domain": SOURCE_DOMAIN,
            "dst_domain": DESTINATION_DOMAIN,
            "mode": "standard",
            "minimum_fee_bps": 0.0,
            "finality_threshold": 2000,
        },
    ]
    assert http.calls == [(f"{IRIS_API}/v2/burn/USDC/fees/{SOURCE_DOMAIN}/{DESTINATION_DOMAIN}", {})]


@pytest.mark.core
def test_unknown_route_404_returns_empty_frame():
    loader = CctpFeeLoader(
        SOURCE_DOMAIN,
        999,
        http=_FakeHttp(LoaderHttpError("HTTP 404 for unknown route")),
    )
    loader.extract()
    loader.transform()
    assert loader._data.empty
    assert list(loader._data.columns) == [
        "src_domain",
        "dst_domain",
        "mode",
        "minimum_fee_bps",
        "finality_threshold",
    ]


@pytest.mark.core
def test_data_envelope_and_non_fast_standard_entries():
    loader = CctpFeeLoader(SOURCE_DOMAIN, DESTINATION_DOMAIN)
    loader._data = {"data": [
        {"finalityThreshold": 1500, "minimumFee": 9},
        {"finalityThreshold": 2000, "minimumFee": 1.5},
    ]}
    loader.transform()
    assert loader._data["mode"].tolist() == ["standard"]
    assert loader._data["minimum_fee_bps"].tolist() == [1.5]


@pytest.mark.core
def test_domain_map_matches_documented_reference_chains():
    assert CCTP_DOMAINS == {
        "ethereum": 0,
        "optimism": 2,
        "arbitrum": 3,
        "base": 6,
        "polygon": 7,
        "unichain": 10,
        "monad": 15,
        "hyperevm": 19,
        "arc": 26,
        "plasma": 33,
    }


@pytest.mark.core
def test_cache_key_is_deterministic_and_covers_route():
    base = CctpFeeLoader(SOURCE_DOMAIN, DESTINATION_DOMAIN)._cache_key()
    assert CctpFeeLoader(SOURCE_DOMAIN, DESTINATION_DOMAIN)._cache_key() == base
    assert CctpFeeLoader(SOURCE_DOMAIN + 1, DESTINATION_DOMAIN)._cache_key() != base
    assert CctpFeeLoader(SOURCE_DOMAIN, DESTINATION_DOMAIN + 1)._cache_key() != base


@pytest.mark.core
def test_cache_round_trip_preserves_fees(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_PATH", str(tmp_path))
    fresh = CctpFeeLoader(
        SOURCE_DOMAIN,
        DESTINATION_DOMAIN,
        http=_FakeHttp([
            {"finalityThreshold": 1000, "minimumFee": "5"},
            {"finalityThreshold": 2000, "minimumFee": "0"},
        ]),
    )
    first = fresh.read(with_run=True)
    cached = CctpFeeLoader(SOURCE_DOMAIN, DESTINATION_DOMAIN, http=_FakeHttp([])).read(with_run=False)
    pd.testing.assert_frame_equal(first, cached)
    assert first["minimum_fee_bps"].tolist() == [5.0, 0.0]


@pytest.mark.core
def test_rows_are_sorted_by_threshold_and_non_dict_rows_skipped():
    loader = CctpFeeLoader(SOURCE_DOMAIN, DESTINATION_DOMAIN)
    loader._data = [
        {"finalityThreshold": "2000", "minimumFee": "0"},
        "unexpected",
        {"finalityThreshold": 1000, "minimumFee": 1.3},
    ]
    loader.transform()
    assert loader._data["mode"].tolist() == ["fast", "standard"]
    assert loader._data["minimum_fee_bps"].tolist() == [1.3, 0.0]
    assert loader._data["finality_threshold"].dtype == "int64"


@pytest.mark.core
@pytest.mark.parametrize(
    "payload",
    [
        [{"finalityThreshold": 1000}],
        [{"finalityThreshold": "fast", "minimumFee": 1}],
        {"data": {"finalityThreshold": 1000}},
    ],
)
def test_malformed_fee_payload_raises(payload):
    loader = CctpFeeLoader(SOURCE_DOMAIN, DESTINATION_DOMAIN)
    loader._data = payload
    with pytest.raises(ValueError):
        loader.transform()


@pytest.mark.core
def test_non_404_http_error_propagates():
    loader = CctpFeeLoader(
        SOURCE_DOMAIN,
        DESTINATION_DOMAIN,
        http=_FakeHttp(LoaderHttpError("HTTP 500 for upstream")),
    )
    with pytest.raises(LoaderHttpError):
        loader.extract()


@pytest.mark.core
def test_negative_domain_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        CctpFeeLoader(-1, DESTINATION_DOMAIN)


SOURCE_TX = "0x" + "AB" * 32
NONCE = "0x" + "12" * 32


def _iris_message(**overrides) -> dict:
    message = {
        "message": "0x" + "ff" * 64,
        "attestation": "0x" + "ee" * 65,
        "eventNonce": NONCE,
        "cctpVersion": 2,
        "status": "complete",
        "delayReason": None,
        "decodedMessage": {
            "sourceDomain": str(SOURCE_DOMAIN),
            "destinationDomain": str(DESTINATION_DOMAIN),
            "nonce": NONCE,
            "sender": "0x" + "AA" * 32,
            "recipient": "0x" + "BB" * 32,
            "minFinalityThreshold": "1000",
            "finalityThresholdExecuted": "1000",
            "decodedMessageBody": {
                "burnToken": "0x" + "CC" * 20,
                "mintRecipient": "0x" + "DD" * 20,
                "amount": "2500000",
                "messageSender": "0x" + "EE" * 20,
                "maxFee": "0x1f4",
                "feeExecuted": "250",
            },
        },
    }
    message.update(overrides)
    return message


@pytest.mark.core
def test_fast_allowance_parses_snapshot():
    http = _FakeHttp({"allowance": "1250000.5", "lastUpdated": "2026-01-01T00:00:00Z"})
    loader = CctpFastAllowanceLoader(http=http)
    loader.extract()
    loader.transform()
    assert loader._data["allowance_usdc"].tolist() == [1250000.5]
    assert loader._data.loc[0, "last_updated"] == pd.Timestamp("2026-01-01", tz="UTC")
    assert http.calls == [(f"{IRIS_API}/v2/fastBurn/USDC/allowance", {})]


@pytest.mark.core
@pytest.mark.parametrize("payload", [None, [], {"lastUpdated": "2026-01-01T00:00:00Z"}, {"allowance": "n/a"}])
def test_fast_allowance_malformed_payload_raises(payload):
    loader = CctpFastAllowanceLoader()
    loader._data = payload
    with pytest.raises(ValueError, match="allowance"):
        loader.transform()


@pytest.mark.core
def test_fast_allowance_cache_round_trip(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_PATH", str(tmp_path))
    first = CctpFastAllowanceLoader(http=_FakeHttp({"allowance": 42.0})).read(with_run=True)
    cached = CctpFastAllowanceLoader(http=_FakeHttp(LoaderHttpError("offline"))).read(with_run=False)
    pd.testing.assert_frame_equal(first, cached)
    assert pd.isna(first.loc[0, "last_updated"])


@pytest.mark.core
def test_messages_parse_metadata_without_attestation():
    http = _FakeHttp({"messages": [_iris_message()]})
    loader = CctpMessagesLoader(SOURCE_DOMAIN, SOURCE_TX, http=http)
    loader.extract()
    loader.transform()
    assert http.calls == [(f"{IRIS_API}/v2/messages/{SOURCE_DOMAIN}", {"transactionHash": SOURCE_TX.lower()})]
    row = loader._data.iloc[0]
    assert "attestation" not in loader._data.columns and "message" not in loader._data.columns
    assert row["src_domain"] == SOURCE_DOMAIN and row["dst_domain"] == DESTINATION_DOMAIN
    assert row["tx_hash"] == SOURCE_TX.lower()
    assert row["nonce"] == NONCE
    assert row["status"] == "complete"
    assert row["amount_raw"] == 2_500_000
    assert row["max_fee_raw"] == 500 and row["fee_executed_raw"] == 250
    assert row["min_finality_threshold"] == 1000 and row["finality_threshold_executed"] == 1000
    assert row["mint_recipient"] == "0x" + "dd" * 20
    assert loader._data["amount_raw"].dtype == "Int64"


@pytest.mark.core
def test_messages_unassigned_nonce_and_pending_fields_are_missing():
    pending = _iris_message(status="pending_confirmations", eventNonce=None)
    pending["decodedMessage"] = dict(pending["decodedMessage"], nonce="0x" + "0" * 64, finalityThresholdExecuted=None)
    decimal_nonce = _iris_message(eventNonce="7", decodedMessage=None)
    loader = CctpMessagesLoader(SOURCE_DOMAIN, SOURCE_TX)
    loader._data = {"messages": [pending, decimal_nonce, "junk"]}
    loader.transform()
    assert len(loader._data) == 2
    assert loader._data.loc[0, "nonce"] is None
    assert pd.isna(loader._data.loc[0, "finality_threshold_executed"])
    assert loader._data.loc[1, "nonce"] == "0x" + format(7, "064x")
    assert loader._data.loc[1, "src_domain"] == SOURCE_DOMAIN
    assert pd.isna(loader._data.loc[1, "dst_domain"]) and pd.isna(loader._data.loc[1, "amount_raw"])


@pytest.mark.core
def test_messages_unknown_transaction_is_empty_and_other_errors_propagate():
    unknown = CctpMessagesLoader(SOURCE_DOMAIN, SOURCE_TX, http=_FakeHttp(LoaderHttpError("HTTP 404 not found")))
    unknown.extract()
    unknown.transform()
    assert unknown._data.empty
    assert list(unknown._data.columns) == list(CctpMessagesLoader._columns)
    failing = CctpMessagesLoader(SOURCE_DOMAIN, SOURCE_TX, http=_FakeHttp(LoaderHttpError("HTTP 503 busy")))
    with pytest.raises(LoaderHttpError):
        failing.extract()


@pytest.mark.core
def test_messages_malformed_integer_raises():
    bad = _iris_message()
    bad["decodedMessage"] = dict(bad["decodedMessage"], destinationDomain="base")
    loader = CctpMessagesLoader(SOURCE_DOMAIN, SOURCE_TX)
    loader._data = {"messages": [bad]}
    with pytest.raises(ValueError, match="destinationDomain"):
        loader.transform()


@pytest.mark.core
@pytest.mark.parametrize("tx_hash", ["", "0x12/34", "../escape"])
def test_messages_rejects_unsafe_transaction_hash(tx_hash):
    with pytest.raises(ValueError, match="transaction_hash"):
        CctpMessagesLoader(SOURCE_DOMAIN, tx_hash)


@pytest.mark.core
def test_messages_cache_key_and_round_trip(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_PATH", str(tmp_path))
    signature = "5VERv8NMvzbJMEkV8xnrLkEaWRtSz9CosKDYjCJjBRnb"  # non-EVM signatures keep their case
    assert CctpMessagesLoader(5, signature).transaction_hash == signature
    assert CctpMessagesLoader(SOURCE_DOMAIN, SOURCE_TX)._cache_key() != CctpMessagesLoader(
        SOURCE_DOMAIN + 1, SOURCE_TX
    )._cache_key()
    pending = _iris_message(status="pending_confirmations", eventNonce=None)
    pending["decodedMessage"] = dict(pending["decodedMessage"], nonce=None)
    http = _FakeHttp({"messages": [_iris_message(), pending]})
    first = CctpMessagesLoader(SOURCE_DOMAIN, SOURCE_TX, http=http).read(with_run=True)
    cached = CctpMessagesLoader(SOURCE_DOMAIN, SOURCE_TX, http=_FakeHttp({"messages": []})).read(with_run=False)
    pd.testing.assert_frame_equal(first, cached)
    assert cached.loc[0, "nonce"] == NONCE and cached.loc[1, "nonce"] is None


@pytest.mark.integration
def test_cctp_fee_integration_smoke():
    frame = CctpFeeLoader(CCTP_DOMAINS["ethereum"], CCTP_DOMAINS["base"]).read(with_run=True)
    assert {"fast", "standard"}.issubset(set(frame["mode"]))
    assert (frame["minimum_fee_bps"] >= 0).all()


@pytest.mark.integration
def test_cctp_fast_allowance_integration_smoke():
    frame = CctpFastAllowanceLoader().read(with_run=True)
    assert len(frame) == 1
    assert frame.loc[0, "allowance_usdc"] >= 0
