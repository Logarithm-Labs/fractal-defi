"""Pure decoders for bridge delivery events.

Each decoder takes a log's ``topics`` (``0x`` hex strings, ``topics[0]`` the
event signature hash) and ``data`` (``0x`` hex) and returns a dict of plain
Python values: integers in the token's base units (arbitrary precision),
addresses as lower-case ``0x`` hex. Nothing here reads a chain; pair the
output with the log's transaction hash and index at the call site.

Covered events: Across V3 ``FilledRelay``, deBridge DLN ``FulfilledOrder``
(documented and deployed variants), LayerZero OFT ``OFTReceived`` and
``OFTSent``, CCTP V1 ``MintAndWithdraw``, CCTP V2 ``DepositForBurn``,
``MessageReceived`` (with the ``BurnMessageV2`` body) and ``MintAndWithdraw``,
and Circle Gateway ``AttestationUsed``.
"""

from collections.abc import Iterator, Sequence
from typing import Any

from fractal.loaders.rpc import event_topic, topic_to_address, topic_to_int

_WORD = 32

SIG_OFT_RECEIVED = "OFTReceived(bytes32,uint32,address,uint256)"
SIG_OFT_SENT = "OFTSent(bytes32,uint32,address,uint256,uint256)"
SIG_CCTP_V2_DEPOSIT_FOR_BURN = (
    "DepositForBurn(address,uint256,address,bytes32,uint32,bytes32,bytes32,uint256,uint32,bytes)"
)
SIG_CCTP_V2_MESSAGE_RECEIVED = "MessageReceived(address,uint32,bytes32,bytes32,uint32,bytes)"
SIG_CCTP_V2_MINT_AND_WITHDRAW = "MintAndWithdraw(address,uint256,address,uint256)"
SIG_CCTP_V1_MINT_AND_WITHDRAW = "MintAndWithdraw(address,uint256,address)"
SIG_ACROSS_FILLED_RELAY = (
    "FilledRelay(bytes32,bytes32,uint256,uint256,uint256,uint256,uint256,uint32,uint32,bytes32,bytes32,bytes32,"
    "bytes32,bytes32,(bytes32,bytes32,uint256,uint8))"
)
DLN_ORDER = "(uint64,bytes,uint256,bytes,uint256,uint256,bytes,uint256,bytes,bytes,bytes,bytes,bytes,bytes)"
SIG_DLN_FULFILLED_ORDER = f"FulfilledOrder({DLN_ORDER},bytes32,address,address)"
# The deployed DlnDestination emits an extra uint256 (the filled amount) after orderId.
SIG_DLN_FULFILLED_ORDER_V2 = f"FulfilledOrder({DLN_ORDER},bytes32,uint256,address,address)"
SIG_GATEWAY_ATTESTATION_USED = "AttestationUsed(address,address,bytes32,uint32,bytes32,bytes32,uint256)"

TOPIC_OFT_RECEIVED = event_topic(SIG_OFT_RECEIVED)
TOPIC_OFT_SENT = event_topic(SIG_OFT_SENT)
TOPIC_CCTP_V2_DEPOSIT_FOR_BURN = event_topic(SIG_CCTP_V2_DEPOSIT_FOR_BURN)
TOPIC_CCTP_V2_MESSAGE_RECEIVED = event_topic(SIG_CCTP_V2_MESSAGE_RECEIVED)
TOPIC_CCTP_V2_MINT_AND_WITHDRAW = event_topic(SIG_CCTP_V2_MINT_AND_WITHDRAW)
CCTP_V2_BURN_BODY_MIN_BYTES = 228  # BurnMessageV2 up to and including expirationBlock
TOPIC_CCTP_V1_MINT_AND_WITHDRAW = event_topic(SIG_CCTP_V1_MINT_AND_WITHDRAW)
TOPIC_ACROSS_FILLED_RELAY = event_topic(SIG_ACROSS_FILLED_RELAY)
TOPIC_DLN_FULFILLED_ORDER = event_topic(SIG_DLN_FULFILLED_ORDER)
TOPIC_DLN_FULFILLED_ORDER_V2 = event_topic(SIG_DLN_FULFILLED_ORDER_V2)
TOPIC_GATEWAY_ATTESTATION_USED = event_topic(SIG_GATEWAY_ATTESTATION_USED)


# --------------------------------------------------------------- ABI decoding
def _split_types(inner: str) -> list[str]:
    """Split a comma-separated ABI type list at depth zero."""
    parts, depth, start = [], 0, 0
    for index, char in enumerate(inner):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(inner[start:index])
            start = index + 1
    if inner:
        parts.append(inner[start:])
    return parts


def _components(abi_type: str) -> list[str] | None:
    return _split_types(abi_type[1:-1]) if abi_type.startswith("(") else None


def _is_dynamic(abi_type: str) -> bool:
    components = _components(abi_type)
    if components is not None:
        return any(_is_dynamic(component) for component in components)
    return abi_type == "bytes"


def _head_size(abi_type: str) -> int:
    components = _components(abi_type)
    if components is not None and not _is_dynamic(abi_type):
        return sum(_head_size(component) for component in components)
    return _WORD


def _word(data: bytes, offset: int) -> bytes:
    if offset < 0 or offset + _WORD > len(data):
        raise ValueError(f"ABI data too short: need {offset + _WORD} bytes, have {len(data)}")
    return data[offset:offset + _WORD]


def _uint(data: bytes, offset: int) -> int:
    return int.from_bytes(_word(data, offset), "big")


def _decode_static(abi_type: str, data: bytes, offset: int) -> Any:
    components = _components(abi_type)
    if components is not None:
        return _decode_sequence(components, data, offset)
    word = _word(data, offset)
    if abi_type == "bytes32":
        return word
    if abi_type == "address":
        return "0x" + word[-20:].hex()
    if abi_type.startswith("uint"):
        return int.from_bytes(word, "big")
    raise ValueError(f"unsupported ABI type {abi_type!r}")


def _decode_dynamic(abi_type: str, data: bytes, offset: int) -> Any:
    components = _components(abi_type)
    if components is not None:
        return _decode_sequence(components, data, offset)
    if abi_type != "bytes":
        raise ValueError(f"unsupported ABI type {abi_type!r}")
    length = _uint(data, offset)
    end = offset + _WORD + length
    if end > len(data):
        raise ValueError("ABI bytes value runs past the end of data")
    return data[offset + _WORD:end]


def _iter_sequence(types: Sequence[str], data: bytes, base: int) -> Iterator[Any]:
    cursor = base
    for abi_type in types:
        if _is_dynamic(abi_type):
            yield _decode_dynamic(abi_type, data, base + _uint(data, cursor))
        else:
            yield _decode_static(abi_type, data, cursor)
        cursor += _head_size(abi_type)


def _decode_sequence(types: Sequence[str], data: bytes, base: int) -> tuple:
    """Decode a head/tail-encoded sequence whose head starts at absolute offset ``base``.

    Offsets of dynamic members are relative to ``base``.
    """
    return tuple(_iter_sequence(types, data, base))


def abi_decode(types: Sequence[str], data: str | bytes) -> tuple:
    """Decode ABI-encoded ``data`` for ``types`` (``uintN``, ``address``, ``bytes32``, ``bytes``, tuples)."""
    raw = data if isinstance(data, bytes) else bytes.fromhex(data[2:] if data.startswith("0x") else data)
    return _decode_sequence(list(types), raw, 0)


# --------------------------------------------------------------- helpers
def _address32(word: bytes) -> str:
    """Address from a bytes32 word; a non-EVM identifier keeps all 32 bytes."""
    return "0x" + (word[-20:].hex() if not any(word[:12]) else word.hex())


def _topics(topics: Sequence[str], expected: Sequence[str], count: int, event: str) -> list[str]:
    normalized = [str(topic).lower() for topic in topics]
    if len(normalized) < count or normalized[0] not in expected:
        raise ValueError(f"log is not a {event} event")
    return normalized


# --------------------------------------------------------------- decoders
def decode_oft_received(topics: Sequence[str], data: str) -> dict[str, Any]:
    """LayerZero OFT ``OFTReceived``: GUID, source EID, recipient and amount received (local decimals)."""
    topics = _topics(topics, (TOPIC_OFT_RECEIVED,), 3, "OFTReceived")
    src_eid, amount = abi_decode(["uint32", "uint256"], data)
    return {
        "guid": topics[1],
        "src_eid": src_eid,
        "to": topic_to_address(topics[2]),
        "amount_received": amount,
    }


def decode_cctp_v1_mint_and_withdraw(topics: Sequence[str], data: str) -> dict[str, Any]:
    """CCTP V1 ``MintAndWithdraw``: recipient, minted token and amount (V1 charges no protocol fee)."""
    topics = _topics(topics, (TOPIC_CCTP_V1_MINT_AND_WITHDRAW,), 3, "CCTP V1 MintAndWithdraw")
    (amount,) = abi_decode(["uint256"], data)
    return {
        "mint_recipient": topic_to_address(topics[1]),
        "mint_token": topic_to_address(topics[2]),
        "amount": amount,
        "fee_collected": 0,
    }


def decode_across_filled_relay(topics: Sequence[str], data: str) -> dict[str, Any]:
    """Across V3 ``FilledRelay`` (bytes32 addresses).

    ``delivered`` is the updated output amount when the relay was sped up,
    else ``output_amount``; ``recipient`` follows the same rule.
    """
    topics = _topics(topics, (TOPIC_ACROSS_FILLED_RELAY,), 4, "Across FilledRelay")
    (
        input_token,
        output_token,
        input_amount,
        output_amount,
        repayment_chain_id,
        _fill_deadline,
        _exclusivity_deadline,
        _exclusive_relayer,
        depositor,
        recipient,
        _message_hash,
        execution_info,
    ) = abi_decode(
        [
            "bytes32", "bytes32", "uint256", "uint256", "uint256", "uint32", "uint32", "bytes32", "bytes32",
            "bytes32", "bytes32", "(bytes32,bytes32,uint256,uint8)",
        ],
        data,
    )  # fmt: skip
    updated_recipient, _updated_message_hash, updated_output, fill_type = execution_info
    return {
        "origin_chain_id": topic_to_int(topics[1]),
        "deposit_id": topic_to_int(topics[2]),
        "relayer": _address32(bytes.fromhex(topics[3][2:])),
        "input_token": _address32(input_token),
        "output_token": _address32(output_token),
        "input_amount": input_amount,
        "output_amount": output_amount,
        "delivered": updated_output if updated_output > 0 else output_amount,
        "repayment_chain_id": repayment_chain_id,
        "depositor": _address32(depositor),
        "recipient": _address32(updated_recipient if any(updated_recipient) else recipient),
        "fill_type": fill_type,
    }


def decode_dln_fulfilled_order(topics: Sequence[str], data: str) -> dict[str, Any]:
    """deBridge DLN ``FulfilledOrder`` (documented and deployed signatures).

    ``filled_amount`` is the deployed event's extra amount field, or the
    order's ``take_amount`` for the documented signature. Token and receiver
    fields are raw bytes (chain-specific encodings) as ``0x`` hex.
    """
    topics = _topics(topics, (TOPIC_DLN_FULFILLED_ORDER, TOPIC_DLN_FULFILLED_ORDER_V2), 1, "DLN FulfilledOrder")
    if topics[0] == TOPIC_DLN_FULFILLED_ORDER_V2:
        order, order_id, filled, taker, _unlock_authority = abi_decode(
            [DLN_ORDER, "bytes32", "uint256", "address", "address"], data
        )
    else:
        order, order_id, taker, _unlock_authority = abi_decode([DLN_ORDER, "bytes32", "address", "address"], data)
        filled = order[7]
    return {
        "maker_order_nonce": order[0],
        "give_chain_id": order[2],
        "give_token": "0x" + order[3].hex(),
        "give_amount": order[4],
        "take_chain_id": order[5],
        "take_token": "0x" + order[6].hex(),
        "take_amount": order[7],
        "filled_amount": filled,
        "receiver": "0x" + order[8].hex(),
        "order_id": "0x" + order_id.hex(),
        "taker": taker,
    }


def decode_gateway_attestation_used(topics: Sequence[str], data: str) -> dict[str, Any]:
    """Circle Gateway minter ``AttestationUsed``: token, recipient, source domain and value minted."""
    topics = _topics(topics, (TOPIC_GATEWAY_ATTESTATION_USED,), 3, "Gateway AttestationUsed")
    src_domain, source_depositor, source_signer, value = abi_decode(["uint32", "bytes32", "bytes32", "uint256"], data)
    return {
        "token": topic_to_address(topics[1]),
        "recipient": topic_to_address(topics[2]),
        "transfer_spec_hash": topics[3] if len(topics) > 3 else None,
        "src_domain": src_domain,
        "source_depositor": _address32(source_depositor),
        "source_signer": _address32(source_signer),
        "amount": value,
    }


def decode_oft_sent(topics: Sequence[str], data: str) -> dict[str, Any]:
    """LayerZero OFT ``OFTSent``: GUID, destination EID, sender and the amounts sent/received (local decimals).

    ``amount_sent_ld - amount_received_ld`` is the OFT-level fee (dust removal and any fee the OFT charges).
    """
    topics = _topics(topics, (TOPIC_OFT_SENT,), 3, "OFTSent")
    dst_eid, sent, received = abi_decode(["uint32", "uint256", "uint256"], data)
    return {
        "guid": topics[1],
        "dst_eid": dst_eid,
        "from_address": topic_to_address(topics[2]),
        "amount_sent_ld": sent,
        "amount_received_ld": received,
    }


def decode_cctp_v2_deposit_for_burn(topics: Sequence[str], data: str) -> dict[str, Any]:
    """CCTP V2 ``DepositForBurn`` (TokenMessengerV2).

    ``mint_recipient`` / ``destination_*`` stay full ``bytes32`` hex (non-EVM
    destinations); ``min_finality`` is ``minFinalityThreshold`` (<= 1000 Fast,
    >= 2000 Standard).
    """
    topics = _topics(topics, (TOPIC_CCTP_V2_DEPOSIT_FOR_BURN,), 4, "CCTP V2 DepositForBurn")
    amount, recipient, dst_domain, dst_messenger, dst_caller, max_fee, hook = abi_decode(
        ["uint256", "bytes32", "uint32", "bytes32", "bytes32", "uint256", "bytes"], data
    )
    return {
        "burn_token": topic_to_address(topics[1]),
        "depositor": topic_to_address(topics[2]),
        "min_finality": topic_to_int(topics[3]),
        "amount": amount,
        "mint_recipient": "0x" + recipient.hex(),
        "dst_domain": dst_domain,
        "destination_token_messenger": "0x" + dst_messenger.hex(),
        "destination_caller": "0x" + dst_caller.hex(),
        "max_fee": max_fee,
        "hook_data": "0x" + hook.hex(),
    }


def decode_cctp_v2_burn_message_body(body: bytes) -> dict[str, Any]:
    """``BurnMessageV2`` body: a 4-byte version, then packed 32-byte burnToken, mintRecipient,
    amount, messageSender, maxFee, feeExecuted, expirationBlock, then hookData."""
    if len(body) < CCTP_V2_BURN_BODY_MIN_BYTES:
        raise ValueError(f"burn message body has {len(body)} bytes, expected at least {CCTP_V2_BURN_BODY_MIN_BYTES}")
    return {
        "version": int.from_bytes(body[:4], "big"),
        "burn_token": "0x" + body[16:36].hex(),
        "mint_recipient": "0x" + body[36:68].hex(),
        "amount": _uint(body, 68),
        "message_sender": "0x" + body[112:132].hex(),
        "max_fee": _uint(body, 132),
        "fee_executed": _uint(body, 164),
        "expiration_block": _uint(body, 196),
        "hook_data": "0x" + body[CCTP_V2_BURN_BODY_MIN_BYTES:].hex(),
    }


def decode_cctp_v2_message_received(topics: Sequence[str], data: str) -> dict[str, Any]:
    """CCTP V2 ``MessageReceived`` (MessageTransmitterV2).

    ``body`` is the decoded burn message, or ``None`` when the message body is
    not a ``BurnMessageV2`` (too short).
    """
    topics = _topics(topics, (TOPIC_CCTP_V2_MESSAGE_RECEIVED,), 4, "CCTP V2 MessageReceived")
    src_domain, sender, body = abi_decode(["uint32", "bytes32", "bytes"], data)
    return {
        "caller": topic_to_address(topics[1]),
        "nonce": topics[2],
        "finality_executed": topic_to_int(topics[3]),
        "src_domain": src_domain,
        "sender": "0x" + sender.hex(),
        "body": decode_cctp_v2_burn_message_body(body) if len(body) >= CCTP_V2_BURN_BODY_MIN_BYTES else None,
    }


def decode_cctp_v2_mint_and_withdraw(topics: Sequence[str], data: str) -> dict[str, Any]:
    """CCTP V2 ``MintAndWithdraw``: recipient, minted token, amount minted to the recipient and fee collected."""
    topics = _topics(topics, (TOPIC_CCTP_V2_MINT_AND_WITHDRAW,), 3, "CCTP V2 MintAndWithdraw")
    amount, fee = abi_decode(["uint256", "uint256"], data)
    return {
        "mint_recipient": topic_to_address(topics[1]),
        "mint_token": topic_to_address(topics[2]),
        "amount": amount,
        "fee_collected": fee,
    }
