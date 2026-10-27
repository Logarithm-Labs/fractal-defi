"""Offline tests for :mod:`fractal.loaders.onchain.oft_fees` (synthetic receipts
and transactions behind a fake JSON-RPC node)."""
import math

import pytest

from fractal.loaders import JsonRpcClient
from fractal.loaders.bridges.decoders import TOPIC_OFT_SENT, decode_oft_sent
from fractal.loaders.onchain.oft_fees import (
    OFT_FEE_COLUMNS,
    TOPIC_DVN_FEE_PAID,
    TOPIC_EXECUTOR_FEE_PAID,
    oft_fee_from_receipt,
    oft_send_fees,
)
from fractal.loaders.rpc import encode_address

OAPP = "0x" + "0f" * 20
OTHER_OAPP = "0x" + "99" * 20
SENDER = "0x" + "5e" * 20
SEND_LIB = "0x" + "11" * 20
G1, G2, G3 = ("0x" + format(i, "064x") for i in (0xA1, 0xA2, 0xA3))
FINNEY = 10 ** 15


def _w(value: int) -> str:
    return format(value, "064x")


def _log(index, address, topics, data):
    return {"logIndex": hex(index), "address": address, "topics": topics, "data": "0x" + data}


def oft_sent(index, guid, sent, received, oapp=OAPP, dst=30_110):
    return _log(index, oapp, [TOPIC_OFT_SENT, guid, "0x" + encode_address(SENDER)], _w(dst) + _w(sent) + _w(received))


def executor_fee(index, wei):
    return _log(index, SEND_LIB, [TOPIC_EXECUTOR_FEE_PAID], encode_address("0x" + "ec" * 20) + _w(wei))


def dvn_fee(index, fees):
    # (address[] required, address[] optional, uint256[] fees): heads, then three arrays
    required = _w(len(fees)) + "".join(encode_address("0x" + format(i + 1, "040x")) for i in range(len(fees)))
    optional = _w(0)
    head_len = 3 * 32
    off_req, off_opt = head_len, head_len + len(required) // 2
    off_fees = off_opt + len(optional) // 2
    data = _w(off_req) + _w(off_opt) + _w(off_fees) + required + optional + _w(len(fees)) + "".join(map(_w, fees))
    return _log(index, SEND_LIB, [TOPIC_DVN_FEE_PAID], data)


# Two sends of OAPP in one transaction, each preceded by its send-library fee events, plus another OApp's send.
BATCH = {"blockNumber": "0x64", "gasUsed": hex(300_000), "effectiveGasPrice": hex(10 ** 9), "logs": [
    oft_sent(9, G2, 5 * 10 ** 6, 5 * 10 ** 6),
    executor_fee(1, 100 * FINNEY // 100),
    dvn_fee(2, [30 * FINNEY // 100, 20 * FINNEY // 100]),
    oft_sent(3, G1, 1_000_000_123, 1_000_000_000),
    oft_sent(5, G3, 1, 1, oapp=OTHER_OAPP),
    executor_fee(7, 7 * FINNEY // 10),
]}
PLAIN = {"blockNumber": "0x65", "gasUsed": hex(100_000), "logs": [
    oft_sent(0, G1, 2 * 10 ** 6, 2 * 10 ** 6), oft_sent(1, G2, 3 * 10 ** 6, 2_999_000)]}
TXS = {"0xbatch": {"value": hex(5 * FINNEY), "gasPrice": hex(2 * 10 ** 9)},
       "0xplain": {"value": hex(4 * FINNEY), "gasPrice": hex(2 * 10 ** 9)}}


class _Node:
    def post(self, url, json=None, timeout=None, headers=None):
        method, params = json["method"], json["params"]
        result = {
            "eth_getTransactionReceipt": lambda: {"0xbatch": BATCH, "0xplain": PLAIN}[params[0]],
            "eth_getTransactionByHash": lambda: TXS[params[0]],
            "eth_getBlockByNumber": lambda: {"timestamp": hex(1_000 + int(params[0], 16))},
        }[method]()
        return {"jsonrpc": "2.0", "id": json["id"], "result": result}


@pytest.mark.core
def test_fixture_logs_match_the_shared_oft_sent_decoder():
    log = oft_sent(3, G1, 10, 9)
    sent = decode_oft_sent(log["topics"], log["data"])
    assert sent == {"guid": G1, "dst_eid": 30_110, "from_address": SENDER, "amount_sent_ld": 10,
                    "amount_received_ld": 9}
    with pytest.raises(ValueError, match="not a OFTSent"):
        decode_oft_sent(executor_fee(1, 5)["topics"], "0x")


@pytest.mark.core
def test_send_library_fee_events_are_attributed_per_message():
    first = oft_fee_from_receipt(BATCH, TXS["0xbatch"], G1, OAPP.upper().replace("0X", "0x"), 6)
    assert first["fee_method"] == "sendlib_fee_events"
    assert first["native_fee"] == pytest.approx(1.5e-3)  # executor 1.0e-3 + DVNs 0.3e-3 + 0.2e-3
    assert first["oft_fee"] == pytest.approx(1.23e-4) and first["amount_received"] == pytest.approx(1_000.0)
    assert first["n_oft_sent"] == 2 and first["dst_eid"] == 30_110  # the other OApp's send is not counted
    second = oft_fee_from_receipt(BATCH, TXS["0xbatch"], G2, OAPP, 6)
    assert second["native_fee"] == pytest.approx(7e-4) and second["oft_fee"] == 0.0


@pytest.mark.core
def test_msg_value_fallback_and_missing_message():
    fees = oft_fee_from_receipt(PLAIN, TXS["0xplain"], G2, OAPP, 6)
    assert fees["fee_method"] == "msg_value" and fees["native_fee"] == pytest.approx(2e-3)  # 4e-3 over two sends
    assert fees["oft_fee"] == pytest.approx(1e-3)
    with pytest.raises(ValueError, match="not in the receipt"):
        oft_fee_from_receipt(PLAIN, TXS["0xplain"], G3, OAPP, 6)


@pytest.mark.core
def test_oft_send_fees_with_gas_and_caller_price():
    rpc = JsonRpcClient("http://fake", http=_Node())
    frame = oft_send_fees(rpc, [("0xBATCH", G1), ("0xbatch", "0xa2"), ("0xplain", G1)], OAPP, 6,
                          native_price={1_100: 2_000.0}.get)
    assert list(frame.columns) == OFT_FEE_COLUMNS
    assert frame["guid"].tolist() == [G1, G2, G1]
    assert frame["gas_native"].tolist() == pytest.approx([3e-4, 3e-4, 2e-4])  # effective price, else tx gasPrice
    assert frame["gas_native_per_send"].tolist() == pytest.approx([1.5e-4, 1.5e-4, 1e-4])
    assert frame.loc[0, "native_fee_usd"] == pytest.approx(3.0)
    assert frame.loc[0, "gas_usd_per_send"] == pytest.approx(0.3)
    assert math.isnan(frame.loc[2, "native_fee_usd"])  # no price for block 101
    assert frame["time"].tolist() == [1_100, 1_100, 1_101]
