"""Offline tests for :mod:`fractal.loaders.onchain.cctp_transfers`: V2 event
decoding and the burn -> mint join on two synthetic chains (fake JSON-RPC)."""
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from fractal.loaders.bridges.cctp import CctpMessagesLoader
from fractal.loaders.bridges.decoders import TOPIC_CCTP_V2_DEPOSIT_FOR_BURN as TOPIC_DEPOSIT_FOR_BURN
from fractal.loaders.bridges.decoders import TOPIC_CCTP_V2_MESSAGE_RECEIVED as TOPIC_MESSAGE_RECEIVED
from fractal.loaders.bridges.decoders import TOPIC_CCTP_V2_MINT_AND_WITHDRAW as TOPIC_MINT_AND_WITHDRAW
from fractal.loaders.bridges.decoders import (
    decode_cctp_v2_burn_message_body,
    decode_cctp_v2_deposit_for_burn,
    decode_cctp_v2_message_received,
)
from fractal.loaders.onchain.cctp_transfers import (
    MESSAGE_TRANSMITTER_V2,
    TOKEN_MESSENGER_V2,
    CctpTransfersLoader,
    assign_iris_nonces,
    cctp_mode,
)
from fractal.loaders.rpc import encode_address, event_topic

T0 = 1_700_000_000
E6 = 10 ** 6
USDC = "0x" + "c0" * 20
R1, R2, R4 = ("0x" + c * 20 for c in ("a1", "a2", "a4"))
D1, D2 = "0x" + "d1" * 20, "0x" + "d2" * 20


def _w(value: int) -> bytes:
    return value.to_bytes(32, "big")


def _addr_word(addr: str) -> bytes:
    return bytes.fromhex(encode_address(addr))


def _dyn(blob: bytes) -> bytes:
    return _w(len(blob)) + blob + b"\0" * (-len(blob) % 32)


def _t(value: int) -> str:
    return "0x" + format(value, "064x")


def _log(address, topics, data: bytes, block, index, tx):
    return {"address": address, "topics": topics, "data": "0x" + data.hex(), "blockNumber": hex(block),
            "logIndex": hex(index), "transactionHash": tx}


def burn_log(block, index, tx, recipient, depositor, amount, max_fee, min_finality, dst=6):
    data = (_w(amount) + _addr_word(recipient) + _w(dst) + _w(0) + _w(0) + _w(max_fee) + _w(7 * 32)
            + _dyn(b"\x01\x02"))
    topics = ["0x" + encode_address(USDC), "0x" + encode_address(depositor), _t(min_finality)]
    return _log(TOKEN_MESSENGER_V2, [TOPIC_DEPOSIT_FOR_BURN, *topics], data, block, index, tx)


def body(recipient, amount, sender, max_fee, fee_executed):
    return (b"\0\0\0\x01" + _addr_word(USDC) + _addr_word(recipient) + _w(amount) + _addr_word(sender)
            + _w(max_fee) + _w(fee_executed) + _w(0))


def received_log(block, index, tx, nonce, msg_body, src=0, finality=1000):
    data = _w(src) + _addr_word(TOKEN_MESSENGER_V2) + _w(3 * 32) + _dyn(msg_body)
    topics = [TOPIC_MESSAGE_RECEIVED, "0x" + encode_address("0x" + "ee" * 20), _t(nonce), _t(finality)]
    return _log(MESSAGE_TRANSMITTER_V2, topics, data, block, index, tx)


def mint_log(block, index, tx, recipient, amount, fee):
    topics = [TOPIC_MINT_AND_WITHDRAW, "0x" + encode_address(recipient), "0x" + encode_address(USDC)]
    return _log(TOKEN_MESSENGER_V2, topics, _w(amount) + _w(fee), block, index, tx)


class _Chain:
    def __init__(self, block_time, n_blocks, logs=(), gas=None):
        self.block_time, self.n_blocks, self.logs, self.gas = block_time, n_blocks, list(logs), gas or {}
        self.queries = []

    @staticmethod
    def _topics_match(log, wanted):
        return all(w is None or (log["topics"][i] in w if isinstance(w, list) else log["topics"][i] == w)
                   for i, w in enumerate(wanted))

    def post(self, url, json=None, timeout=None, headers=None):
        method, params = json["method"], json["params"]
        if method == "eth_blockNumber":
            result = hex(self.n_blocks - 1)
        elif method == "eth_getBlockByNumber":
            result = {"timestamp": hex(T0 + self.block_time * int(params[0], 16))}
        elif method == "eth_getLogs":
            q = params[0]
            self.queries.append(q)
            lo, hi = int(q["fromBlock"], 16), int(q["toBlock"], 16)
            result = [lg for lg in self.logs if lo <= int(lg["blockNumber"], 16) <= hi
                      and lg["address"] == q["address"] and self._topics_match(lg, q["topics"])]
        elif method == "eth_getTransactionReceipt":
            result = {"gasUsed": hex(100_000), "effectiveGasPrice": hex(self.gas.get(params[0], 1)),
                      "blockNumber": "0x1", "logs": [lg for lg in self.logs if lg["transactionHash"] == params[0]]}
        else:
            raise AssertionError(method)
        return {"jsonrpc": "2.0", "id": json["id"], "result": result}


# src: 12 s blocks; dst: 2 s blocks
SRC_LOGS = [
    burn_log(100, 3, "0xb1", R1, D1, 1_000 * E6, 1 * E6, 1000),       # Fast, joined by nonce
    burn_log(120, 0, "0xb2", R2, D2, 500 * E6, 0, 2000),              # Standard, joined by fingerprint
    burn_log(130, 0, "0xb3", R1, D1, 7 * E6, 0, 2000, dst=3),          # another destination: ignored
    burn_log(140, 1, "0xb4", R4, D1, 9 * E6, 0, 1000),                # never minted: unmatched
]
DST_LOGS = [
    # decoy: same fingerprint as b1, received earlier than its nonce-matched mint
    received_log(650, 2, "0xm0", 0xD, body(R1, 1_000 * E6, D1, 1 * E6, 0)),
    mint_log(700, 0, "0xm1", R1, 999_600_000, 400_000),
    received_log(700, 1, "0xm1", 0x1, body(R1, 1_000 * E6, D1, 1 * E6, 500_000)),
    received_log(710, 0, "0xm2e", 0x2E, body(R2, 500 * E6, D2, 0, 0)),             # before b2's burn
    received_log(900, 0, "0xm2", 0x2, body(R2, 500 * E6, D2, 0, 0), finality=2000),
    received_log(905, 0, "0xmx", 0x9, body(R2, 500 * E6, D2, 0, 0), src=3),        # other source domain
]
START = datetime.fromtimestamp(T0 + 1_000, tz=timezone.utc)
END = datetime.fromtimestamp(T0 + 2_000, tz=timezone.utc)


@pytest.fixture
def offline_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_PATH", str(tmp_path))
    return tmp_path


def _loader(nonces=None, with_gas=False, src_logs=None, **kwargs):
    gas = {"0xb1": 10 ** 9, "0xb2": 10 ** 9, "0xb4": 10 ** 9, "0xm1": 10 ** 7, "0xm2": 10 ** 7, "0xm0": 10 ** 7}
    return CctpTransfersLoader("https://src.example", 0, "https://dst.example", 6, START, END, nonces=nonces,
                               with_gas=with_gas, **kwargs,
                               src_http=_Chain(12, 1_000, SRC_LOGS if src_logs is None else src_logs, gas),
                               dst_http=_Chain(2, 5_000, DST_LOGS, gas))


@pytest.mark.core
def test_fixture_logs_match_the_shared_v2_decoders():
    burn = decode_cctp_v2_deposit_for_burn(SRC_LOGS[0]["topics"], SRC_LOGS[0]["data"])
    assert (burn["amount"], burn["max_fee"], burn["min_finality"], burn["dst_domain"]) == (1_000 * E6, E6, 1000, 6)
    assert burn["depositor"] == D1 and burn["burn_token"] == USDC and burn["hook_data"] == "0x0102"
    assert burn["mint_recipient"] == "0x" + encode_address(R1)
    rec = decode_cctp_v2_message_received(DST_LOGS[2]["topics"], DST_LOGS[2]["data"])
    assert rec["nonce"] == _t(1) and rec["src_domain"] == 0 and rec["finality_executed"] == 1000
    assert rec["body"]["fee_executed"] == 500_000 and rec["body"]["message_sender"] == D1
    with pytest.raises(ValueError, match="at least 228"):
        decode_cctp_v2_burn_message_body(b"\0" * 100)
    assert [cctp_mode(m) for m in (500, 1000, 1500, 2000)] == ["fast", "fast", "other", "standard"]
    signature = "DepositForBurn(address,uint256,address,bytes32,uint32,bytes32,bytes32,uint256,uint32,bytes)"
    assert event_topic(signature) == TOPIC_DEPOSIT_FOR_BURN


@pytest.mark.core
def test_join_prefers_supplied_nonces_then_fingerprints(offline_cache):
    df = _loader(nonces={("0xB1", 3): "0x1"}).read(with_run=True)
    assert df["burn_tx"].tolist() == ["0xb1", "0xb2", "0xb4"]
    assert df["join"].tolist() == ["nonce", "fingerprint", "unmatched"]
    assert df["mint_tx"].tolist() == ["0xm1", "0xm2", None]  # not the decoy, not the pre-burn receipt
    assert df["mode"].tolist() == ["fast", "standard", "fast"]
    assert df["fee"].iloc[:2].tolist() == pytest.approx([0.4, 0.0])  # MintAndWithdraw fee, else feeExecuted
    assert df["amount"].tolist() == pytest.approx([1_000.0, 500.0, 9.0])
    assert df.loc[0, "amount_received"] == pytest.approx(999.6) and np.isnan(df.loc[1, "amount_received"])
    assert df["latency_s"].iloc[:2].tolist() == [200, 360]
    assert df.loc[0, "burn_time"] == pd.Timestamp(T0 + 1_200, unit="s", tz="UTC")
    assert pd.isna(df.loc[2, "mint_time"]) and df.loc[2, "nonce"] is None
    assert df.loc[1, "nonce"] == _t(2) and df.loc[1, "finality_executed"] == 2000

    cached = CctpTransfersLoader("https://src.example", 0, "https://dst.example", 6, START, END,
                                 nonces={("0xB1", 3): "0x1"}).read()
    pd.testing.assert_frame_equal(cached, df, check_dtype=False)


@pytest.mark.core
def test_without_nonces_the_first_receipt_after_the_burn_wins(offline_cache):
    df = _loader().read(with_run=True)
    assert df["mint_tx"].tolist() == ["0xm0", "0xm2", None]
    assert df["join"].tolist() == ["fingerprint", "fingerprint", "unmatched"]
    assert _loader()._cache_key() != _loader(nonces={("0xb1", 3): "0x1"})._cache_key()  # pylint: disable=protected-access


@pytest.mark.core
def test_gas_per_leg_in_native_units(offline_cache):
    df = _loader(nonces={("0xb1", 3): "0x1"}, with_gas=True).read(with_run=True)
    assert df["burn_gas_native"].tolist() == pytest.approx([1e-4, 1e-4, 1e-4])
    assert df["mint_gas_native"].iloc[:2].tolist() == pytest.approx([1e-6, 1e-6])
    assert np.isnan(df.loc[2, "mint_gas_native"])


@pytest.mark.core
def test_no_burns_in_window(offline_cache):
    df = _loader(src_logs=[]).read(with_run=True)
    assert df.empty and list(df.columns) == CctpTransfersLoader.COLUMNS
    with pytest.raises(ValueError, match="end_time"):
        CctpTransfersLoader("https://s.example", 0, "https://d.example", 6, END, START)


class _Iris:
    """Fake Iris ``/v2/messages`` endpoint: one canned message list per transaction."""

    def __init__(self, by_tx):
        self.by_tx = by_tx

    def get(self, url, params=None, **_):
        return {"messages": self.by_tx.get(params["transactionHash"], [])}


def _iris(dst, nonce, amount):
    decoded = {"sourceDomain": "0", "destinationDomain": str(dst), "nonce": None if nonce is None else _t(nonce),
               "decodedMessageBody": {"amount": None if amount is None else str(amount)}}
    return {"status": "complete" if nonce else "pending_confirmations", "cctpVersion": 2, "decodedMessage": decoded}


def _iris_frame(by_tx):
    http = _Iris(by_tx)
    return pd.concat([CctpMessagesLoader(0, tx, http=http).read(with_run=True) for tx in by_tx], ignore_index=True)


@pytest.mark.core
def test_iris_messages_are_assigned_to_burns_in_log_order(offline_cache):
    tx = "0x" + "ab" * 32
    burns = [{"tx": tx, "log_index": 9, "amount": 5, "dst_domain": 6},
             {"tx": tx, "log_index": 2, "amount": 5, "dst_domain": 6},
             {"tx": tx, "log_index": 4, "amount": 7, "dst_domain": 3},
             {"tx": tx, "log_index": 6, "amount": 8, "dst_domain": 6}]
    frame = _iris_frame({tx: [_iris(3, 0x30, 7), _iris(6, 0x51, 5), _iris(6, None, 8), _iris(6, 0x52, 5),
                              _iris(6, 0x80, None)]})
    assert frame["amount_raw"].dtype == "Int64" and frame["amount_raw"].isna().any()
    got = assign_iris_nonces(burns, frame)
    # same amount and destination: first message to the lowest log index; the pending message (no nonce) is
    # skipped, so the burn of 8 takes the message with an unknown amount
    assert got == {(tx, 2): _t(0x51), (tx, 4): _t(0x30), (tx, 9): _t(0x52), (tx, 6): _t(0x80)}
    assert not assign_iris_nonces(burns, [])
    assert not assign_iris_nonces(burns[:1], [{"tx_hash": tx.upper().replace("0X", "0x"), "nonce": "0x1",
                                           "amount_raw": 6, "dst_domain": 6}])


@pytest.mark.core
def test_loader_takes_a_cctp_messages_frame(offline_cache):
    frame = _iris_frame({"0xb1": [_iris(6, 0x1, 1_000 * E6)], "0xb4": [_iris(6, 0x4, 9 * E6)]})
    df = _loader(iris_messages=frame).read(with_run=True)
    assert df["join"].tolist() == ["nonce", "fingerprint", "unmatched"]
    assert df["mint_tx"].tolist() == ["0xm1", "0xm2", None]  # the Iris nonce beats the decoy
    assert df.loc[2, "nonce"] is None  # assigned, but nothing received with that nonce yet
    assert _loader(iris_messages=frame)._cache_key() != _loader()._cache_key()  # pylint: disable=protected-access
    # an explicit nonce wins over the Iris assignment
    df = _loader(iris_messages=frame, nonces={("0xb1", 3): "0xd"}).read(with_run=True)
    assert df.loc[0, "mint_tx"] == "0xm0" and df.loc[0, "join"] == "nonce"


@pytest.mark.core
def test_targeted_lookup_filters_by_nonce_and_reads_mints_from_receipts(offline_cache):
    frame = _iris_frame({"0xb1": [_iris(6, 0x1, 1_000 * E6)], "0xb4": [_iris(6, 0x4, 9 * E6)]})
    loader = _loader(iris_messages=frame, targeted=True)
    df = loader.read(with_run=True)
    assert df["join"].tolist() == ["nonce", "unmatched", "unmatched"]  # b2 has no Iris nonce
    assert df.loc[0, "mint_tx"] == "0xm1" and df.loc[0, "amount_received"] == pytest.approx(999.6)
    assert df.loc[0, "fee"] == pytest.approx(0.4)
    queries = loader._dst._http.queries  # pylint: disable=protected-access
    assert queries and all(q["topics"] == [TOPIC_MESSAGE_RECEIVED, None, [_t(0x1), _t(0x4)]] for q in queries)
    assert _loader(iris_messages=frame)._cache_key() != loader._cache_key()  # pylint: disable=protected-access
    no_nonces = _loader(targeted=True)
    assert no_nonces.read(with_run=True)["join"].tolist() == ["unmatched"] * 3
    assert not no_nonces._dst._http.queries  # pylint: disable=protected-access
