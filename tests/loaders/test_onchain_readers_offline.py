"""Offline tests for the Aave V3 reserve-rate reader and the debt-token flow
loader (synthetic chain, reserve states and logs; fake JSON-RPC transport)."""
import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from fractal.loaders import AaveV3ReserveRatesLoader, DebtTokenFlowsLoader, LoaderType
from fractal.loaders.onchain.aave_reserve_rates import RAY, decode_reserve_data, ray_apr_to_apy
from fractal.loaders.onchain.debt_token_flows import BURN_TOPIC, MINT_TOPIC, TRANSFER_TOPIC, ZERO_TOPIC
from fractal.loaders.rpc import encode_address

POOL = "0x" + "11" * 20
ASSET = "0x" + "22" * 20
DEBT = "0x" + "33" * 20
ALICE = "0x" + "a1" * 20
BOB = "0x" + "b0" * 20
T0 = 1_700_000_000  # block 0 time; one block every 12 s
YEAR = 365 * 86_400


@pytest.fixture
def offline_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_PATH", str(tmp_path))
    return tmp_path


def _reserve_words(block: int) -> str:
    """15-word reserve state whose rates depend on the block (2 % + 1bp/block supply, 2x borrow)."""
    supply = (2 * RAY // 100) + block * RAY // 1_000_000
    words = [0, RAY, supply, RAY, 2 * supply, 0, T0 + 12 * block, 7,
             int("aa" * 20, 16), int("bb" * 20, 16), int(DEBT[2:], 16), int("cc" * 20, 16), 0, 0, 0]
    return "0x" + "".join(format(w, "064x") for w in words)


class _Node:
    def __init__(self, n_blocks=2_000, logs=()):
        self.n_blocks, self.logs, self.calls = n_blocks, list(logs), []

    def post(self, url, json=None, timeout=None, headers=None):
        method, params = json["method"], json["params"]
        self.calls.append((method, params))
        if method == "eth_blockNumber":
            result = hex(self.n_blocks - 1)
        elif method == "eth_getBlockByNumber":
            result = {"timestamp": hex(T0 + 12 * int(params[0], 16))}
        elif method == "eth_call":
            assert params[0]["to"] == POOL
            assert params[0]["data"] == "0x35ea6a75" + encode_address(ASSET)
            result = _reserve_words(int(params[1], 16))
        elif method == "eth_getLogs":
            result = [lg for lg in self.logs if self._match(lg, params[0])]
        else:
            raise AssertionError(method)
        return {"jsonrpc": "2.0", "id": json["id"], "result": result}

    @staticmethod
    def _match(log, query):
        lo, hi = int(query["fromBlock"], 16), int(query["toBlock"], 16)
        if not lo <= int(log["blockNumber"], 16) <= hi or query["address"] != DEBT:
            return False
        for want, got in zip(query["topics"], log["topics"], strict=False):
            if want is not None and got not in (want if isinstance(want, list) else [want]):
                return False
        return len(query["topics"]) <= len(log["topics"])


# ------------------------------------------------------------ reserve rates
@pytest.mark.core
def test_decode_reserve_data_layout():
    reserve = decode_reserve_data(_reserve_words(100))
    assert reserve.liquidity_rate == 2 * RAY // 100 + 100 * RAY // 1_000_000
    assert reserve.variable_borrow_rate == 2 * reserve.liquidity_rate
    assert reserve.last_update_timestamp == T0 + 1_200
    assert reserve.a_token == "0x" + "aa" * 20
    assert reserve.variable_debt_token == DEBT
    assert reserve.interest_rate_strategy == "0x" + "cc" * 20
    assert reserve.borrow_apy == pytest.approx(math.expm1(2 * 0.0201))
    assert ray_apr_to_apy(RAY // 10) == pytest.approx(math.expm1(0.1))


@pytest.mark.core
def test_reserve_rates_loader_reads_last_block_at_or_before_each_bar(offline_cache):
    node = _Node()
    start = datetime.fromtimestamp(T0 + 600 + 5, tz=timezone.utc)  # between blocks 50 and 51
    loader = AaveV3ReserveRatesLoader(f"https://rpc.example/v3/{'k' * 32}", POOL, ASSET,
                                      start_time=start, end_time=start + pd.Timedelta(hours=2),
                                      step_seconds=1_800, http=node)
    history = loader.read(with_run=True)
    blocks = [50 + 150 * i for i in range(5)]  # 30 min = 150 blocks
    assert loader._data["block"].tolist() == blocks  # pylint: disable=protected-access
    supply_apr = np.array([0.02 + b / 1e6 for b in blocks])
    assert history["lending_rate"].to_numpy() == pytest.approx(supply_apr * 1_800 / YEAR)
    assert history["borrowing_rate"].to_numpy() == pytest.approx(2 * supply_apr * 1_800 / YEAR)
    assert history["borrow_apy"].to_numpy() == pytest.approx(np.expm1(2 * supply_apr))
    assert history.index[0] == pd.Timestamp(T0 + 605, unit="s", tz="UTC")
    assert "k" * 32 not in loader._cache_key()  # pylint: disable=protected-access

    cached = AaveV3ReserveRatesLoader(f"https://rpc.example/v3/{'k' * 32}", POOL, ASSET, start_time=start,
                                      end_time=start + pd.Timedelta(hours=2), step_seconds=1_800,
                                      http=_Node(n_blocks=0)).read()
    pd.testing.assert_frame_equal(cached, history)


@pytest.mark.core
def test_reserve_rates_loader_drops_bars_past_the_chain_tip(offline_cache):
    node = _Node(n_blocks=301)  # tip at T0 + 3600
    start = datetime.fromtimestamp(T0, tz=timezone.utc)
    loader = AaveV3ReserveRatesLoader("https://rpc.example", POOL, ASSET, start_time=start,
                                      end_time=start + pd.Timedelta(hours=3), step_seconds=1_800,
                                      http=node, loader_type=LoaderType.JSON)
    history = loader.read(with_run=True)
    assert len(history) == 3  # T0, +30 min, +60 min (= tip)
    with pytest.raises(ValueError, match="step_seconds"):
        AaveV3ReserveRatesLoader("https://rpc.example", POOL, ASSET, start, start, step_seconds=0)


# ------------------------------------------------------------- debt flows
def _topic(addr):
    return "0x" + encode_address(addr)


def _log(block, index, topics, *words, tx="0x" + "0e" * 32):
    return {"address": DEBT, "blockNumber": hex(block), "logIndex": hex(index), "transactionHash": tx,
            "topics": topics, "data": "0x" + "".join(format(w, "064x") for w in words)}


# Alice borrows 100 (2 accrued interest folded in), repays 1 while owing 3 interest (net mint 2),
# repays 50 with 1 interest (burn 49). Bob borrows 10. Values in 6-decimal raw units.
E6 = 10 ** 6
LOGS = [
    _log(10, 0, [TRANSFER_TOPIC, ZERO_TOPIC, _topic(ALICE)], 102 * E6),
    _log(10, 1, [MINT_TOPIC, _topic(ALICE), _topic(ALICE)], 102 * E6, 2 * E6, RAY),
    _log(20, 4, [TRANSFER_TOPIC, ZERO_TOPIC, _topic(ALICE)], 2 * E6),
    _log(20, 5, [MINT_TOPIC, _topic(ALICE), _topic(ALICE)], 2 * E6, 3 * E6, RAY),
    _log(30, 0, [TRANSFER_TOPIC, _topic(ALICE), ZERO_TOPIC], 49 * E6),
    _log(30, 1, [BURN_TOPIC, _topic(ALICE), _topic(BOB)], 49 * E6, 1 * E6, RAY),
    _log(25, 2, [TRANSFER_TOPIC, ZERO_TOPIC, _topic(BOB)], 10 * E6),
    _log(25, 3, [MINT_TOPIC, _topic(BOB), _topic(BOB)], 10 * E6, 0, RAY),
    _log(26, 0, [TRANSFER_TOPIC, ZERO_TOPIC, ZERO_TOPIC], 5 * E6),  # zero-to-zero: ignored
]


@pytest.mark.core
def test_debt_flows_from_zero_address_transfers(offline_cache):
    loader = DebtTokenFlowsLoader("https://rpc.example", DEBT, decimals=6, start_block=0, end_block=100,
                                  http=_Node(logs=LOGS))
    flows = loader.read(with_run=True)
    assert flows["block"].tolist() == [10, 20, 25, 30]
    assert flows["amount"].tolist() == pytest.approx([102.0, 2.0, 10.0, -49.0])
    assert flows["user"].tolist() == [ALICE, ALICE, BOB, ALICE]
    assert flows.index[0] == pd.Timestamp(T0 + 120, unit="s", tz="UTC")
    assert "interest" not in flows.columns
    cached = DebtTokenFlowsLoader("https://rpc.example", DEBT, 6, 0, 100, http=_Node()).read()
    pd.testing.assert_frame_equal(cached, flows)


@pytest.mark.core
def test_debt_flows_split_interest_recovers_principal(offline_cache):
    loader = DebtTokenFlowsLoader("https://rpc.example", DEBT, decimals=6, start_block=0, end_block=100,
                                  users=[ALICE.upper().replace("0X", "0x")], split_interest=True,
                                  http=_Node(logs=LOGS))
    flows = loader.read(with_run=True)
    assert flows["user"].unique().tolist() == [ALICE]
    assert flows["amount"].tolist() == pytest.approx([102.0, 2.0, -49.0])
    assert flows["interest"].tolist() == pytest.approx([2.0, 3.0, 1.0])
    assert flows["principal"].tolist() == pytest.approx([100.0, -1.0, -50.0])  # borrow, repay, repay


@pytest.mark.core
def test_debt_flows_user_filter_goes_to_the_node_and_cache_key(offline_cache):
    node = _Node(logs=LOGS)
    loader = DebtTokenFlowsLoader("https://rpc.example", DEBT, 6, 0, 100, users=[BOB], http=node)
    flows = loader.read(with_run=True)
    assert flows["user"].tolist() == [BOB]
    topics = [params[0]["topics"] for method, params in node.calls if method == "eth_getLogs"]
    assert [TRANSFER_TOPIC, ZERO_TOPIC, [_topic(BOB)]] in topics
    assert [TRANSFER_TOPIC, [_topic(BOB)], ZERO_TOPIC] in topics
    keys = {DebtTokenFlowsLoader("https://rpc.example", DEBT, 6, 0, 100, users=u)._cache_key()  # pylint: disable=protected-access
            for u in (None, [BOB], [ALICE], [ALICE, BOB], [BOB, ALICE])}
    assert len(keys) == 4  # user order does not matter, the set does


@pytest.mark.core
def test_debt_flows_empty_range(offline_cache):
    loader = DebtTokenFlowsLoader("https://rpc.example", DEBT, 6, 500, 600, split_interest=True,
                                  http=_Node(logs=LOGS))
    flows = loader.read(with_run=True)
    assert flows.empty and list(flows.columns) == ["block", "log_index", "tx", "user", "amount", "interest",
                                                   "principal"]
