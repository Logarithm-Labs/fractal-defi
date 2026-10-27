"""Offline tests for :class:`fractal.loaders.etherscan.EtherscanClient` with a
fake transport (synthetic key, addresses and logs; no network)."""
import pytest

from fractal.loaders import EtherscanClient, EtherscanError, RpcCallError
from fractal.loaders.etherscan import ETHERSCAN_V2_URL

KEY = "SYNTHETICETHERSCANKEY000000000003"
ADDR = "0x" + "ab" * 20
TOPIC = "0x" + "cd" * 32


class _FakeExplorer:
    """Serves queued payloads, or a callable of the query, and records every query."""

    def __init__(self, *payloads):
        self.payloads, self.queries = list(payloads), []

    def get(self, url, params=None, timeout=None):
        assert url == ETHERSCAN_V2_URL
        self.queries.append(dict(params))
        nxt = self.payloads.pop(0)
        return nxt(params) if callable(nxt) else nxt


def _client(*payloads, **kwargs):
    fake = _FakeExplorer(*payloads)
    return EtherscanClient(api_key=KEY, chain_id=8453, http=fake, min_interval=0.0, **kwargs), fake


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    slept = []
    monkeypatch.setattr("fractal.loaders.etherscan.time.sleep", slept.append)
    return slept


@pytest.mark.core
def test_key_resolution(monkeypatch):
    monkeypatch.delenv("ETHERSCAN_API_KEY", raising=False)
    with pytest.raises(ValueError, match="ETHERSCAN_API_KEY"):
        EtherscanClient()
    monkeypatch.setenv("ETHERSCAN_API_KEY", "env-key-synthetic")
    fake = _FakeExplorer({"status": "1", "message": "OK", "result": "123"})
    assert EtherscanClient(http=fake, min_interval=0.0).block_by_time(1) == 123
    assert fake.queries[0]["apikey"] == "env-key-synthetic"


@pytest.mark.core
def test_block_by_time_sends_chain_and_closest():
    es, fake = _client({"status": "1", "message": "OK", "result": "19000000"})
    assert es.block_by_time(1_700_000_000.9, closest="after") == 19_000_000
    q = fake.queries[0]
    assert (q["chainid"], q["module"], q["action"], q["timestamp"], q["closest"]) == (
        8453, "block", "getblocknobytime", 1_700_000_000, "after")
    with pytest.raises(ValueError, match="closest"):
        es.block_by_time(1, closest="nearest")


@pytest.mark.core
def test_rate_limit_body_is_retried_then_errors_are_raised(_no_sleep):
    limited = {"status": "0", "message": "NOTOK", "result": "Max calls per sec rate limit reached (3/sec)"}
    es, _ = _client(limited, limited, {"status": "1", "message": "OK", "result": "7"})
    assert es.block_by_time(1) == 7
    assert _no_sleep == [1.0, 2.0]

    es, _ = _client(*[limited] * 3, rate_limit_retries=2)
    with pytest.raises(EtherscanError, match="rate limit"):
        es.block_by_time(1)

    es, _ = _client({"status": "0", "message": "NOTOK", "result": "Invalid API Key"})
    with pytest.raises(EtherscanError, match="Invalid API Key") as caught:
        es.block_by_time(1)
    assert KEY not in str(caught.value)


@pytest.mark.core
def test_throttle_spaces_calls(monkeypatch, _no_sleep):
    clock = iter([100.0, 100.0, 100.1, 100.35])
    monkeypatch.setattr("fractal.loaders.etherscan.time.monotonic", lambda: next(clock))
    ok = {"status": "1", "message": "OK", "result": "1"}
    es, _ = _client(ok, ok)
    es.min_interval = 0.35
    es.block_by_time(1)
    es.block_by_time(2)
    assert _no_sleep == [pytest.approx(0.25)]


def _log(block, index, topics=(TOPIC,)):
    return {"address": ADDR.upper().replace("0X", "0x"), "topics": list(topics), "data": "0x" + "00" * 32,
            "blockNumber": hex(block), "logIndex": "0x" if index == 0 else hex(index),
            "transactionHash": "0x" + "EF" * 32, "timeStamp": hex(1_700_000_000 + block)}


@pytest.mark.core
def test_get_logs_pages_splits_past_the_window_and_normalises():
    def page(params):
        lo, hi, p = params["fromBlock"], params["toBlock"], params["page"]
        rows = [_log(b, i) for b in range(lo, hi + 1) for i in range(3)]  # 3 logs per block
        chunk = rows[(p - 1) * 1_000: p * 1_000]
        return {"status": "1" if chunk else "0", "message": "OK" if chunk else "No records found",
                "result": chunk}

    # 5 000 blocks x 3 logs = 15 000 rows > the 10 000-row window: one split
    es, fake = _client(*[page] * 40)
    logs = es.get_logs(ADDR, 0, 4_999, topic0=TOPIC, topics={2: "0x" + "00" * 32})
    assert len(logs) == 15_000
    blocks = [int(log["blockNumber"], 16) for log in logs]
    assert blocks == sorted(blocks) and blocks[0] == 0 and blocks[-1] == 4_999
    first = logs[0]
    assert first["logIndex"] == "0x0" and first["address"] == ADDR
    assert first["transactionHash"] == "0x" + "ef" * 32 and first["timeStamp"] == 1_700_000_000
    ranges = {(q["fromBlock"], q["toBlock"]) for q in fake.queries}
    assert (0, 2_499) in ranges and (2_500, 4_999) in ranges
    q = fake.queries[0]
    assert q["topic0"] == TOPIC and q["topic2"] == "0x" + "00" * 32 and q["topic0_2_opr"] == "and"
    with pytest.raises(ValueError, match="topic positions"):
        es.get_logs(ADDR, 0, 1, topics={4: TOPIC})


@pytest.mark.core
def test_get_logs_raises_when_one_block_exceeds_the_window():
    full = {"status": "1", "message": "OK", "result": [_log(9, i) for i in range(1_000)]}
    es, _ = _client(*[full] * 10)
    with pytest.raises(EtherscanError, match="block 9"):
        es.get_logs(ADDR, 9, 9)


@pytest.mark.core
def test_token_transfers_and_empty_answers():
    rows = [{"hash": "0x1", "value": "5", "tokenDecimal": "6"}]
    es, fake = _client({"status": "1", "message": "OK", "result": rows},
                       {"status": "0", "message": "No transactions found", "result": []})
    assert es.token_transfers(ADDR, 10, 20, contract=ADDR) == rows
    assert es.token_transfers(ADDR, 10, 20) == []
    assert fake.queries[0]["contractaddress"] == ADDR and fake.queries[0]["startblock"] == 10
    assert "contractaddress" not in fake.queries[1]


@pytest.mark.core
def test_proxy_eth_call_and_errors():
    es, fake = _client({"jsonrpc": "2.0", "id": 1, "result": "0x" + "00" * 31 + "2a"},
                       {"jsonrpc": "2.0", "id": 1, "error": {"code": -32000, "message": "execution reverted"}},
                       {"status": "0", "message": "NOTOK", "result": "Max rate limit reached"},
                       {"jsonrpc": "2.0", "id": 1, "result": "0x10"})
    assert int(es.eth_call(ADDR, "0x35ea6a75", block=123), 16) == 42
    assert fake.queries[0]["tag"] == "0x7b"
    with pytest.raises(RpcCallError, match="execution reverted"):
        es.eth_call(ADDR, "0x")
    assert es.block_number() == 16  # proxy throttling answer retried
