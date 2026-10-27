"""Offline tests for :class:`fractal.loaders.onchain.log_reader.LogReader` and the
RPC error classification it relies on (fake node and explorer; synthetic keys,
addresses and logs; no network)."""
import logging

import pytest

from fractal.loaders import EtherscanClient
from fractal.loaders._http import LoaderHttpError
from fractal.loaders.onchain import LogReader
from fractal.loaders.rpc import JsonRpcClient, is_range_limit_error, is_splittable_error, is_transient_error

ADDR = "0x" + "ab" * 20
ADDR2 = "0x" + "ac" * 20
T0, T1, T1B = "0x" + "01" * 32, "0x" + "02" * 32, "0x" + "03" * 32
RPC_KEY = "synthetic0rpc0key0000000000000004"
ES_KEY = "SYNTHETICETHERSCANKEY000000000005"


def _log(block, index, address=ADDR, topics=(T0,)):
    return {"address": address, "topics": list(topics), "data": "0x", "blockNumber": hex(block),
            "logIndex": hex(index), "transactionHash": "0x" + format(block * 100 + index, "064x")}


LOGS = [_log(5, 0), _log(40, 1), _log(40, 0), _log(77, 2), _log(150, 0)]


def _matches(log, address, topics):
    addresses = [address] if isinstance(address, str) else address
    if log["address"] not in addresses:
        return False
    for pos, want in enumerate(topics or []):
        have = log["topics"][pos] if pos < len(log["topics"]) else None
        if want is not None and have not in ([want] if isinstance(want, str) else want):
            return False
    return True


class _Node:
    """Fake JSON-RPC transport: ``fail(lo, hi, n)`` returns an error to raise or answer, else logs."""

    def __init__(self, fail=None, logs=None):
        self.fail, self.logs, self.ranges = fail, LOGS if logs is None else logs, []

    def post(self, url, json=None, timeout=None, headers=None):
        q = json["params"][0]
        lo, hi = int(q["fromBlock"], 16), int(q["toBlock"], 16)
        self.ranges.append((lo, hi))
        error = self.fail(lo, hi, len(self.ranges)) if self.fail else None
        if isinstance(error, Exception):
            raise error
        if error is not None:
            return {"jsonrpc": "2.0", "id": json["id"], "error": error}
        rows = [lg for lg in self.logs if lo <= int(lg["blockNumber"], 16) <= hi
                and _matches(lg, q.get("address"), q.get("topics"))]
        return {"jsonrpc": "2.0", "id": json["id"], "result": rows}


class _Explorer:
    """Fake Etherscan V2 transport over the same synthetic logs."""

    def __init__(self, logs=None, error=None):
        self.logs, self.error, self.queries = LOGS if logs is None else logs, error, []

    def get(self, url, params=None, timeout=None):
        self.queries.append(dict(params))
        if self.error:
            return {"status": "0", "message": "NOTOK", "result": self.error}
        lo, hi = int(params["fromBlock"]), int(params["toBlock"])
        topics = [params.get(f"topic{i}") for i in range(4)]
        rows = [{**lg, "timeStamp": "0x1"} for lg in self.logs if lo <= int(lg["blockNumber"], 16) <= hi
                and _matches(lg, params["address"], topics)]
        return {"status": "1" if rows else "0", "message": "OK" if rows else "No records found", "result": rows}


def _es(explorer):
    return EtherscanClient(api_key=ES_KEY, http=explorer, min_interval=0.0)


def _reader(node=None, explorer=None, **kwargs):
    rpc = JsonRpcClient("https://node.example", http=node) if node is not None else None
    slept = []
    reader = LogReader(rpc, _es(explorer) if explorer is not None else None, sleep=slept.append, **kwargs)
    return reader, slept


def _keys(logs):
    return [(int(lg["blockNumber"], 16), int(lg["logIndex"], 16)) for lg in logs]


ALL = [(5, 0), (40, 0), (40, 1), (77, 2), (150, 0)]
TOO_MANY = {"code": -32005, "message": "query returned more than 10000 results"}


@pytest.mark.core
@pytest.mark.parametrize(("text", "transient", "range_limit", "splittable"), [
    ("RPC eth_getLogs failed: {'code': -32005, 'message': 'query returned more than 10000 results'}",
     False, True, True),
    ("Log response size exceeded. You can make eth_getLogs requests with up to a 2K block range",
     False, True, True),
    ("HTTP 400 for https://node.example params=None: block range is too wide", False, True, True),
    ("HTTP 413 for https://node.example params=None: payload too large", False, False, True),
    ("Read timed out. (read timeout=30)", False, False, True),
    ("HTTP 503 for https://node.example params=None: service temporarily unavailable", True, False, False),
    ("HTTP 429 for https://node.example params=None: slow down", True, False, False),
    ("RPC eth_getLogs failed: {'code': -32603, 'message': 'internal error'}", True, False, False),
    ("('Connection aborted.', RemoteDisconnected('Remote end closed connection without response'))",
     True, False, False),
    ("RPC eth_getLogs failed: {'code': -32602, 'message': 'invalid argument 0: hex string has odd length'}",
     False, False, False),
])
def test_error_classification(text, transient, range_limit, splittable):
    exc = RuntimeError(text)
    assert (is_transient_error(exc), is_range_limit_error(exc), is_splittable_error(exc)) == (
        transient, range_limit, splittable)


@pytest.mark.core
def test_rpc_chunks_and_splits_rejected_ranges():
    node = _Node(fail=lambda lo, hi, n: TOO_MANY if hi - lo + 1 > 40 else None)
    reader, slept = _reader(node, chunk_size=100)
    assert _keys(reader.get_logs(ADDR, [T0], 0, 199)) == ALL
    assert node.ranges[:4] == [(0, 99), (0, 49), (0, 24), (25, 49)]  # halved, not retried as-is
    assert all(hi - lo + 1 <= 100 for lo, hi in node.ranges) and not slept
    assert reader.calls["rpc"] == len(node.ranges) and not reader.calls["etherscan"]
    assert not reader.get_logs(ADDR, [T0], 10, 9)


@pytest.mark.core
def test_transient_errors_are_retried_with_backoff():
    node = _Node(fail=lambda lo, hi, n: LoaderHttpError("HTTP 503 for https://node.example: busy") if n < 3 else None)
    reader, slept = _reader(node)
    assert _keys(reader.get_logs(ADDR, [T0], 0, 199)) == ALL
    assert slept == [3.0, 6.0] and node.ranges == [(0, 199)] * 3


@pytest.mark.core
def test_unresolved_rpc_failures_fall_back_to_etherscan():
    # a range error at the smallest chunk
    reader, _ = _reader(_Node(fail=lambda *_: TOO_MANY), _Explorer(), min_chunk=60, chunk_size=100)
    assert _keys(reader.get_logs(ADDR, [T0], 0, 199)) == ALL
    assert reader.calls["etherscan"] == 4  # [0, 49], [50, 99], [100, 149], [150, 199]
    # transient errors that outlast the retries
    explorer = _Explorer()
    reader, slept = _reader(_Node(fail=lambda *_: {"code": -32603, "message": "internal error"}), explorer,
                            transient_retries=2)
    assert _keys(reader.get_logs(ADDR, [T0], 0, 199)) == ALL
    assert slept == [3.0] and len(explorer.queries) == 1
    # an error that is neither: no fallback
    bad = {"code": -32602, "message": "invalid argument"}
    reader, _ = _reader(_Node(fail=lambda *_: bad), _Explorer())
    with pytest.raises(Exception, match="invalid argument"):
        reader.get_logs(ADDR, [T0], 0, 199)
    # no Etherscan client: the RPC error propagates
    reader, _ = _reader(_Node(fail=lambda *_: TOO_MANY), min_chunk=500)
    with pytest.raises(Exception, match="-32005"):
        reader.get_logs(ADDR, [T0], 0, 199)


@pytest.mark.core
def test_or_filters_expand_into_etherscan_queries():
    logs = [_log(10, 0, topics=(T0, T1)), _log(11, 0, topics=(T0, T1B)), _log(12, 0, address=ADDR2, topics=(T0, T1)),
            _log(13, 0, topics=(T0, "0x" + "09" * 32))]
    explorer = _Explorer(logs)
    reader, _ = _reader(explorer=explorer)
    got = reader.get_logs([ADDR, ADDR2], [T0, [T1, T1B]], 0, 99)
    assert _keys(got) == [(10, 0), (11, 0), (12, 0)]
    assert len(explorer.queries) == 4
    assert {(q["address"], q["topic1"], q["topic0_1_opr"]) for q in explorer.queries} == {
        (a, t, "and") for a in (ADDR, ADDR2) for t in (T1, T1B)}
    # the node reads the same filter in one request
    node = _Node(logs=logs)
    assert _keys(_reader(node)[0].get_logs([ADDR, ADDR2], [T0, [T1, T1B]], 0, 99)) == [(10, 0), (11, 0), (12, 0)]
    assert len(node.ranges) == 1


@pytest.mark.core
def test_prefer_etherscan_falls_back_to_the_node():
    explorer = _Explorer()
    reader, _ = _reader(_Node(), explorer, prefer="etherscan")
    assert _keys(reader.get_logs(ADDR, [T0], 0, 199)) == ALL and reader.calls == {"etherscan": 1}
    reader, _ = _reader(_Node(), _Explorer(error="Query Timeout occured"), prefer="etherscan")
    assert _keys(reader.get_logs(ADDR, [T0], 0, 199)) == ALL and reader.calls["rpc"] == 1
    with pytest.raises(ValueError, match="prefer"):
        LogReader("https://node.example", prefer="explorer")
    with pytest.raises(ValueError, match="needs"):
        LogReader()


@pytest.mark.core
def test_keys_stay_out_of_errors_and_fallback_warnings(caplog):
    def refuse(lo, hi, n):
        return LoaderHttpError(f"HTTP 503 for https://node.example/v3/{RPC_KEY} params=None: busy")

    rpc = JsonRpcClient(f"https://node.example/v3/{RPC_KEY}", http=_Node(fail=refuse))
    explorer = _Explorer(error=f"Invalid API Key {ES_KEY}")
    reader = LogReader(rpc, _es(explorer), transient_retries=1, sleep=lambda _: None)
    with caplog.at_level(logging.DEBUG), pytest.raises(Exception) as caught:
        reader.get_logs(ADDR, [T0], 0, 9)
    warnings = [r.getMessage() for r in caplog.records if r.name == "fractal.loaders.onchain.log_reader"]
    assert warnings and "falling back to etherscan" in warnings[0]
    for secret in (RPC_KEY, ES_KEY):
        assert secret not in str(caught.value) and secret not in caplog.text and secret not in repr(reader)
        assert all(secret not in message for message in warnings)
