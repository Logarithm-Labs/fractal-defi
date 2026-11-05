"""Offline tests for :mod:`fractal.loaders.rpc`: keccak / ABI helpers and the
client's call, block and adaptive ``eth_getLogs`` paths."""
import pytest

from fractal.loaders import JsonRpcClient, RpcCallError
from fractal.loaders._http import LoaderHttpError
from fractal.loaders.rpc import (
    as_signed,
    decode_address,
    decode_words,
    encode_address,
    eth_call,
    event_topic,
    function_selector,
    keccak256,
    topic_to_address,
    topic_to_int,
)

POOL = "0xd0b53d9277642d899df5c87a3966a349a798f224"


class _FakeNode:
    """Serves blocks, one static call and eth_getLogs limited to ``max_span`` blocks."""

    def __init__(self, max_span: int = 10 ** 9, reject_as_http: bool = False, logs=None):
        self.max_span, self.reject_as_http, self.logs = max_span, reject_as_http, logs or []
        self.calls = []

    def post(self, url, json=None, timeout=None, headers=None):
        self.calls.append(json)
        method, params = json["method"], json["params"]
        if method == "eth_blockNumber":
            return {"jsonrpc": "2.0", "id": json["id"], "result": hex(20_000)}
        if method == "eth_getBlockByNumber":
            return {"jsonrpc": "2.0", "id": json["id"],
                    "result": {"number": params[0], "timestamp": hex(1_700_000_000)}}
        if method == "eth_call":
            if params[0]["data"].startswith("0xdeadbeef"):
                return {"jsonrpc": "2.0", "id": json["id"], "error": {"code": 3, "message": "execution reverted"}}
            words = format(57, "064x") + format(2 ** 256 - 1, "064x")
            return {"jsonrpc": "2.0", "id": json["id"], "result": "0x" + words}
        if method == "eth_getLogs":
            lo, hi = int(params[0]["fromBlock"], 16), int(params[0]["toBlock"], 16)
            if hi - lo + 1 > self.max_span:
                if self.reject_as_http:
                    raise LoaderHttpError("HTTP 413: range too large")
                return {"jsonrpc": "2.0", "id": json["id"], "error": {"message": "range too large"}}
            matched = [log for log in self.logs if lo <= log["block"] <= hi]
            return {"jsonrpc": "2.0", "id": json["id"], "result": matched}
        raise AssertionError(method)


@pytest.mark.core
def test_keccak_matches_known_ethereum_digests():
    assert keccak256(b"").hex() == "c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470"
    assert event_topic("Transfer(address,address,uint256)") == (
        "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef")
    assert event_topic("Swap(address,address,int256,int256,uint160,uint128,int24)") == (
        "0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67")
    assert function_selector("transfer(address,uint256)") == "0xa9059cbb"
    assert function_selector("readState(address)") == "0x794052f3"  # verified against a mainnet Pendle market
    assert keccak256(b"x" * 300).hex() == keccak256(bytes(b"x" * 300)).hex()  # multi-block absorb is stable


@pytest.mark.core
def test_abi_helpers_round_trip():
    word = int(encode_address(POOL), 16)
    assert decode_address(word) == POOL
    assert topic_to_address("0x" + encode_address(POOL)) == POOL
    assert as_signed(2 ** 256 - 1) == -1 and as_signed(5) == 5 and as_signed(2 ** 23, bits=24) == -(2 ** 23)
    assert topic_to_int("0x" + format(2 ** 256 - 3, "064x"), signed=True) == -3
    assert decode_words("0x" + format(1, "064x") + format(2, "064x"), 2) == [1, 2]
    with pytest.raises(RpcCallError, match="expected 3 return words"):
        decode_words("0x" + format(1, "064x"), 3)


@pytest.mark.core
def test_client_calls_blocks_and_errors():
    node = _FakeNode()
    rpc = JsonRpcClient("http://fake", http=node)
    assert rpc.block_number() == 20_000
    assert rpc.block_timestamp(15) == 1_700_000_000 and node.calls[-1]["params"] == ["0xf", False]
    words = decode_words(rpc.eth_call(POOL, "0x12345678", block=100), 2)
    assert words[0] == 57 and as_signed(words[1]) == -1
    assert node.calls[-1]["params"][1] == "0x64"
    with pytest.raises(RpcCallError, match="execution reverted"):
        rpc.eth_call(POOL, "0xdeadbeef")
    assert [c["id"] for c in node.calls] == list(range(1, len(node.calls) + 1))
    assert eth_call("http://fake", POOL, "0x12345678", http=node) == rpc.eth_call(POOL, "0x12345678")


@pytest.mark.core
@pytest.mark.parametrize("reject_as_http", [True, False])
def test_iter_logs_halves_on_rejection_and_recovers(reject_as_http):
    logs = [{"block": b, "data": "0x"} for b in (10, 6_000, 19_999)]
    node = _FakeNode(max_span=5_000, reject_as_http=reject_as_http, logs=logs)
    rpc = JsonRpcClient("http://fake", http=node)
    out = list(rpc.iter_logs(POOL, [event_topic("Transfer(address,address,uint256)")], 1, chunk_size=10_000))
    assert [log["block"] for log in out] == [10, 6_000, 19_999]  # every block covered exactly once
    spans = [int(c["params"][0]["toBlock"], 16) - int(c["params"][0]["fromBlock"], 16) + 1
             for c in node.calls if c["method"] == "eth_getLogs"]
    assert spans[0] == 10_000 and spans[1] == 5_000 and max(spans[2:]) == 10_000
    assert node.calls[0]["method"] == "eth_blockNumber"  # to_block resolved from the node


@pytest.mark.core
def test_iter_logs_reraises_below_the_minimum_chunk():
    node = _FakeNode(max_span=100, reject_as_http=True)
    rpc = JsonRpcClient("http://fake", http=node)
    with pytest.raises(LoaderHttpError):
        list(rpc.iter_logs(POOL, None, 1, 20_000, chunk_size=10_000, min_chunk=500))
