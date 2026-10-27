"""Realised cost of LayerZero OFT sends, read from the source transaction.

For each OFT send (source transaction hash + message GUID) the receipt
gives:

* the **LayerZero native fee**: the send library's ``ExecutorFeePaid`` and
  ``DVNFeePaid`` events emitted between the previous ``OFTSent`` of the
  same OApp and this message's ``OFTSent`` (``fee_method =
  "sendlib_fee_events"``; the ULN treasury fee, zero by default, is not
  evented). When the transaction has no such events, ``msg.value`` split
  over its ``OFTSent`` events is used instead (``"msg_value"``, an upper
  bound: LayerZero refunds excess value);
* the **OFT fee**: ``amountSentLD − amountReceivedLD`` of the ``OFTSent``
  event (dust removal and any OFT-level fee), in token units;
* the **gas** of the transaction (:func:`fractal.loaders.rpc.gas_cost_wei`),
  also split over the transaction's ``OFTSent`` events.

USD values are produced only from a caller-supplied
``native_price(timestamp)``. ``OFTSent`` is decoded locally; the bridge
decoders in ``fractal.loaders.bridges`` (``OFTReceived`` and others) live in
a separate change and can be consolidated with this one later.
"""
from collections.abc import Callable, Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from fractal.loaders.rpc import JsonRpcClient, decode_words, event_topic, gas_cost_wei, hex_to_int, topic_to_address

__all__ = ["OFT_FEE_COLUMNS", "decode_oft_sent", "oft_fee_from_receipt", "oft_send_fees"]

TOPIC_OFT_SENT = event_topic("OFTSent(bytes32,uint32,address,uint256,uint256)")
TOPIC_EXECUTOR_FEE_PAID = event_topic("ExecutorFeePaid(address,uint256)")
TOPIC_DVN_FEE_PAID = event_topic("DVNFeePaid(address[],address[],uint256[])")
_WEI = 10 ** 18

OFT_FEE_COLUMNS = [
    "tx", "guid", "block", "time", "dst_eid", "amount_sent", "amount_received", "oft_fee", "native_fee",
    "fee_method", "n_oft_sent", "gas_native", "gas_native_per_send", "native_fee_usd", "gas_usd_per_send",
]


def _topic0(log: Mapping[str, Any]) -> str:
    topics = log.get("topics") or []
    return str(topics[0]).lower() if topics else ""


def decode_oft_sent(log: Mapping[str, Any]) -> dict[str, Any]:
    """``OFTSent(guid idx, dstEid, fromAddress idx, amountSentLD, amountReceivedLD)`` (raw local units)."""
    if _topic0(log) != TOPIC_OFT_SENT or len(log["topics"]) < 3:
        raise ValueError("log is not an OFTSent event")
    dst_eid, sent, received = decode_words(log["data"], 3)
    return {"guid": "0x" + format(int(log["topics"][1], 16), "064x"), "dst_eid": dst_eid,
            "from_address": topic_to_address(log["topics"][2]), "amount_sent_ld": sent,
            "amount_received_ld": received}


def _dvn_fees_wei(data_hex: str) -> int:
    """Sum of the ``uint256[] fees`` (third argument) of a ``DVNFeePaid`` log."""
    data = bytes.fromhex(data_hex[2:] if data_hex.startswith("0x") else data_hex)
    offset = int.from_bytes(data[64:96], "big")
    count = int.from_bytes(data[offset: offset + 32], "big")
    start = offset + 32
    if start + 32 * count > len(data):
        raise ValueError("DVNFeePaid fees array runs past the end of data")
    return sum(int.from_bytes(data[start + 32 * i: start + 32 * (i + 1)], "big") for i in range(count))


def oft_fee_from_receipt(
    receipt: Mapping[str, Any],
    tx: Mapping[str, Any],
    guid: str,
    oapp: str,
    local_decimals: int,
) -> dict[str, Any]:
    """Native and OFT fees of message ``guid`` sent by ``oapp`` in one transaction.

    Returns ``native_fee`` (whole native tokens), ``oft_fee`` /
    ``amount_sent`` / ``amount_received`` (token units), ``dst_eid``,
    ``n_oft_sent`` (``OFTSent`` events of ``oapp`` in the transaction) and
    ``fee_method``. Raises :class:`ValueError` when the message is not in
    the receipt.
    """
    want_guid, want_oapp = "0x" + format(int(guid, 16), "064x"), oapp.lower()
    logs = sorted(receipt.get("logs") or [], key=lambda log: hex_to_int(log.get("logIndex")))
    n_sent, segment_wei, seen_fee, found, native_wei = 0, 0, False, None, None
    for log in logs:
        topic = _topic0(log)
        if topic == TOPIC_EXECUTOR_FEE_PAID:
            segment_wei += decode_words(log["data"], 2)[1]
            seen_fee = True
        elif topic == TOPIC_DVN_FEE_PAID:
            segment_wei += _dvn_fees_wei(log["data"])
            seen_fee = True
        elif topic == TOPIC_OFT_SENT and str(log.get("address", "")).lower() == want_oapp:
            n_sent += 1
            sent = decode_oft_sent(log)
            if sent["guid"] == want_guid:
                found = sent
                native_wei = segment_wei if seen_fee else None
            segment_wei, seen_fee = 0, False
    if found is None:
        raise ValueError(f"OFTSent {want_guid} of {want_oapp} not in the receipt")
    method = "sendlib_fee_events"
    if native_wei is None:
        native_wei, method = hex_to_int(tx.get("value")) / max(1, n_sent), "msg_value"
    scale = 10 ** int(local_decimals)
    return {
        "dst_eid": found["dst_eid"],
        "amount_sent": found["amount_sent_ld"] / scale,
        "amount_received": found["amount_received_ld"] / scale,
        "oft_fee": (found["amount_sent_ld"] - found["amount_received_ld"]) / scale,
        "native_fee": native_wei / _WEI,
        "fee_method": method,
        "n_oft_sent": n_sent,
    }


def oft_send_fees(
    rpc: JsonRpcClient,
    sends: Iterable[tuple[str, str]],
    oapp: str,
    local_decimals: int,
    *,
    native_price: Callable[[int], float | None] | None = None,
) -> pd.DataFrame:
    """Fees and gas of OFT sends, one row per ``(tx hash, guid)`` (see :data:`OFT_FEE_COLUMNS`).

    The transaction's gas is split evenly over its ``OFTSent`` events
    (``gas_native_per_send``); ``time`` is the block's UNIX timestamp and
    USD columns are ``NaN`` without a price.
    """
    rows, cache, times = [], {}, {}
    for tx_hash, guid in sends:
        tx_hash = tx_hash.lower()
        if tx_hash not in cache:
            cache[tx_hash] = (rpc.get_transaction_receipt(tx_hash), rpc.get_transaction(tx_hash))
        receipt, tx = cache[tx_hash]
        fees = oft_fee_from_receipt(receipt, tx, guid, oapp, local_decimals)
        block = hex_to_int(receipt.get("blockNumber"))
        if block not in times:
            times[block] = rpc.block_timestamp(block)
        gas = gas_cost_wei(receipt, gas_price=tx.get("gasPrice")) / _WEI
        per_send = gas / max(1, fees["n_oft_sent"])
        price = native_price(times[block]) if native_price is not None else None
        rows.append({
            "tx": tx_hash, "guid": "0x" + format(int(guid, 16), "064x"), "block": block, "time": times[block],
            **fees, "gas_native": gas, "gas_native_per_send": per_send,
            "native_fee_usd": fees["native_fee"] * price if price is not None else np.nan,
            "gas_usd_per_send": per_send * price if price is not None else np.nan,
        })
    return pd.DataFrame(rows, columns=OFT_FEE_COLUMNS)
