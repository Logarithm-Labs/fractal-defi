"""Offline tests for the pure bridge-event decoders (synthetic logs)."""

import pytest

from fractal.loaders.bridges import decoders
from fractal.loaders.bridges.decoders import (
    DLN_ORDER,
    TOPIC_ACROSS_FILLED_RELAY,
    TOPIC_CCTP_V1_MINT_AND_WITHDRAW,
    TOPIC_DLN_FULFILLED_ORDER,
    TOPIC_DLN_FULFILLED_ORDER_V2,
    TOPIC_GATEWAY_ATTESTATION_USED,
    TOPIC_OFT_RECEIVED,
    abi_decode,
    decode_across_filled_relay,
    decode_cctp_v1_mint_and_withdraw,
    decode_dln_fulfilled_order,
    decode_gateway_attestation_used,
    decode_oft_received,
)

ADDRESS_A = "0x" + "a1" * 20
ADDRESS_B = "0x" + "b2" * 20
ADDRESS_C = "0x" + "c3" * 20
GUID = "0x" + "4d" * 32


# --------------------------------------------------------------- tiny ABI encoder (test side)
def _split(inner: str) -> list[str]:
    parts, depth, start = [], 0, 0
    for index, char in enumerate(inner):
        depth += {"(": 1, ")": -1}.get(char, 0)
        if char == "," and depth == 0:
            parts.append(inner[start:index])
            start = index + 1
    return [*parts, inner[start:]] if inner else parts


def _dynamic(abi_type: str) -> bool:
    if abi_type.startswith("("):
        return any(_dynamic(item) for item in _split(abi_type[1:-1]))
    return abi_type == "bytes"


def _encode_one(abi_type: str, value) -> bytes:
    if abi_type.startswith("("):
        return _encode(_split(abi_type[1:-1]), value)
    if abi_type == "bytes":
        padded = value + b"\0" * (-len(value) % 32)
        return len(value).to_bytes(32, "big") + padded
    if abi_type == "bytes32":
        return value
    if abi_type == "address":
        return bytes(12) + bytes.fromhex(value[2:])
    return int(value).to_bytes(32, "big")


def _encode(types, values) -> bytes:
    heads, tails = [], []
    head_size = sum(32 if _dynamic(t) else len(_encode_one(t, v)) for t, v in zip(types, values, strict=True))
    for abi_type, value in zip(types, values, strict=True):
        encoded = _encode_one(abi_type, value)
        if _dynamic(abi_type):
            heads.append((head_size + sum(len(tail) for tail in tails)).to_bytes(32, "big"))
            tails.append(encoded)
        else:
            heads.append(encoded)
    return b"".join(heads + tails)


def _hex(types, values) -> str:
    return "0x" + _encode(types, values).hex()


def _topic_address(address: str) -> str:
    return "0x" + "0" * 24 + address[2:]


def _word(value: int) -> str:
    return "0x" + format(value, "064x")


def _b32(address: str) -> bytes:
    return bytes(12) + bytes.fromhex(address[2:])


# --------------------------------------------------------------- tests
@pytest.mark.core
def test_abi_decode_static_dynamic_and_nested_tuple():
    types = ["uint32", "bytes", "(uint64,bytes,bytes32)", "address"]
    values = [7, b"\x01\x02\x03", (2**63, b"x" * 40, b"\x11" * 32), ADDRESS_A]
    assert abi_decode(types, _hex(types, values)) == tuple(values)


@pytest.mark.core
def test_abi_decode_rejects_truncated_data():
    data = _hex(["uint256", "uint256"], [1, 2])
    with pytest.raises(ValueError, match="too short"):
        abi_decode(["uint256", "uint256"], data[:-2])
    with pytest.raises(ValueError, match="unsupported"):
        abi_decode(["int256"], data)


@pytest.mark.core
def test_decode_oft_received():
    topics = [TOPIC_OFT_RECEIVED, GUID, _topic_address(ADDRESS_A)]
    row = decode_oft_received(topics, _hex(["uint32", "uint256"], [30101, 1_500_000]))
    assert row == {"guid": GUID, "src_eid": 30101, "to": ADDRESS_A, "amount_received": 1_500_000}


@pytest.mark.core
def test_decode_rejects_wrong_event_and_missing_topics():
    data = _hex(["uint32", "uint256"], [1, 2])
    with pytest.raises(ValueError, match="OFTReceived"):
        decode_oft_received([TOPIC_CCTP_V1_MINT_AND_WITHDRAW, GUID, _topic_address(ADDRESS_A)], data)
    with pytest.raises(ValueError, match="OFTReceived"):
        decode_oft_received([TOPIC_OFT_RECEIVED, GUID], data)


@pytest.mark.core
def test_decode_cctp_v1_mint_and_withdraw():
    topics = [TOPIC_CCTP_V1_MINT_AND_WITHDRAW.upper().replace("0X", "0x"), _topic_address(ADDRESS_A),
              _topic_address(ADDRESS_B)]
    row = decode_cctp_v1_mint_and_withdraw(topics, _hex(["uint256"], [25_000_000]))
    assert row == {"mint_recipient": ADDRESS_A, "mint_token": ADDRESS_B, "amount": 25_000_000, "fee_collected": 0}


_ACROSS_TYPES = [
    "bytes32", "bytes32", "uint256", "uint256", "uint256", "uint32", "uint32", "bytes32", "bytes32",
    "bytes32", "bytes32", "(bytes32,bytes32,uint256,uint8)",
]  # fmt: skip


def _across_data(updated_recipient: bytes, updated_output: int) -> str:
    non_evm_recipient = b"\x09" * 32
    return _hex(
        _ACROSS_TYPES,
        [_b32(ADDRESS_A), _b32(ADDRESS_B), 10**18, 999_000, 8453, 1, 2, bytes(32), _b32(ADDRESS_C),
         non_evm_recipient, bytes(32), (updated_recipient, bytes(32), updated_output, 1)],
    )  # fmt: skip


@pytest.mark.core
def test_decode_across_filled_relay_plain_and_sped_up():
    topics = [TOPIC_ACROSS_FILLED_RELAY, _word(56), _word(12345), "0x" + bytes(_b32(ADDRESS_C)).hex()]
    plain = decode_across_filled_relay(topics, _across_data(bytes(32), 0))
    assert plain["origin_chain_id"] == 56 and plain["deposit_id"] == 12345
    assert plain["relayer"] == ADDRESS_C
    assert plain["input_token"] == ADDRESS_A and plain["output_token"] == ADDRESS_B
    assert plain["input_amount"] == 10**18 and plain["output_amount"] == 999_000
    assert plain["delivered"] == 999_000
    assert plain["recipient"] == "0x" + "09" * 32  # non-EVM recipient keeps 32 bytes
    assert plain["repayment_chain_id"] == 8453 and plain["fill_type"] == 1
    sped_up = decode_across_filled_relay(topics, _across_data(_b32(ADDRESS_B), 998_000))
    assert sped_up["delivered"] == 998_000 and sped_up["recipient"] == ADDRESS_B


def _order(take_amount: int) -> tuple:
    return (
        42,  # maker order nonce
        bytes.fromhex(ADDRESS_A[2:]),
        1,  # give chain
        bytes.fromhex(ADDRESS_B[2:]),
        1_000_000_000,
        8453,  # take chain
        bytes.fromhex(ADDRESS_C[2:]),
        take_amount,
        bytes.fromhex(ADDRESS_A[2:]),
        b"",
        b"",
        b"",
        b"",
        b"\x01\x02",
    )


@pytest.mark.core
def test_decode_dln_fulfilled_order_documented_and_deployed():
    order_id = b"\x77" * 32
    documented = decode_dln_fulfilled_order(
        [TOPIC_DLN_FULFILLED_ORDER],
        _hex([DLN_ORDER, "bytes32", "address", "address"], [_order(999_500), order_id, ADDRESS_B, ADDRESS_C]),
    )
    assert documented["give_chain_id"] == 1 and documented["take_chain_id"] == 8453
    assert documented["give_token"] == ADDRESS_B and documented["take_token"] == ADDRESS_C
    assert documented["give_amount"] == 1_000_000_000 and documented["take_amount"] == 999_500
    assert documented["filled_amount"] == 999_500
    assert documented["receiver"] == ADDRESS_A
    assert documented["order_id"] == "0x" + "77" * 32 and documented["taker"] == ADDRESS_B
    assert documented["maker_order_nonce"] == 42
    deployed = decode_dln_fulfilled_order(
        [TOPIC_DLN_FULFILLED_ORDER_V2],
        _hex([DLN_ORDER, "bytes32", "uint256", "address", "address"],
             [_order(999_500), order_id, 999_400, ADDRESS_B, ADDRESS_C]),
    )
    assert deployed["take_amount"] == 999_500 and deployed["filled_amount"] == 999_400
    assert deployed["taker"] == ADDRESS_B


@pytest.mark.core
def test_decode_gateway_attestation_used():
    spec_hash = "0x" + "5e" * 32
    topics = [TOPIC_GATEWAY_ATTESTATION_USED, _topic_address(ADDRESS_A), _topic_address(ADDRESS_B), spec_hash]
    data = _hex(["uint32", "bytes32", "bytes32", "uint256"], [6, _b32(ADDRESS_C), b"\x08" * 32, 3_000_000])
    row = decode_gateway_attestation_used(topics, data)
    assert row == {
        "token": ADDRESS_A,
        "recipient": ADDRESS_B,
        "transfer_spec_hash": spec_hash,
        "src_domain": 6,
        "source_depositor": ADDRESS_C,
        "source_signer": "0x" + "08" * 32,
        "amount": 3_000_000,
    }


@pytest.mark.core
def test_event_topics_are_distinct_keccak_hashes():
    topics = [getattr(decoders, name) for name in dir(decoders) if name.startswith("TOPIC_")]
    assert len(topics) == len(set(topics)) == 6
    assert all(topic.startswith("0x") and len(topic) == 66 for topic in topics)
    # Keccak-256 of the ERC-20 Transfer signature, a well-known public constant.
    assert decoders.event_topic("Transfer(address,address,uint256)") == (
        "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"
    )
