"""Offline tests for :func:`gas_cost_wei`, :meth:`JsonRpcClient.transaction_gas_cost`
and :func:`fractal.loaders.onchain.gas.transaction_gas_costs` (synthetic receipts)."""
import math

import pytest

from fractal.loaders import JsonRpcClient, RpcCallError
from fractal.loaders.onchain.gas import GAS_COLUMNS, count_legs, transaction_gas_costs
from fractal.loaders.rpc import gas_cost_wei

PROTOCOL = "0x" + "70" * 20
SUPPLY = "0x" + "51" * 32
WITHDRAW = "0x" + "57" * 32
GWEI = 10 ** 9


def _leg(address=PROTOCOL, topic=SUPPLY):
    return {"address": address, "topics": [topic, "0x" + "00" * 32]}


RECEIPTS = {
    "0xa": {"blockNumber": "0xa", "gasUsed": hex(100_000), "effectiveGasPrice": hex(30 * GWEI),
            "logs": [_leg(), _leg(topic=WITHDRAW), _leg(address="0x" + "99" * 20)]},
    "0xb": {"blockNumber": "0xa", "gasUsed": hex(50_000), "logs": []},  # pre-1559: price from the tx
    "0xc": {"blockNumber": "0x14", "gasUsed": hex(200_000), "effectiveGasPrice": hex(10 ** 6),
            "l1Fee": hex(7 * 10 ** 12), "blobGasUsed": hex(131_072), "blobGasPrice": "0x1",
            "logs": [_leg(PROTOCOL.upper().replace("0X", "0x"), SUPPLY.upper().replace("0X", "0x"))]},
}
TXS = {"0xb": {"gasPrice": hex(20 * GWEI)}}


class _Node:
    def __init__(self):
        self.methods = []

    def post(self, url, json=None, timeout=None, headers=None):
        method, params = json["method"], json["params"]
        self.methods.append(method)
        result = {
            "eth_getTransactionReceipt": lambda: RECEIPTS.get(params[0]),
            "eth_getTransactionByHash": lambda: TXS.get(params[0]),
            "eth_getBlockByNumber": lambda: {"timestamp": hex(1_000 + int(params[0], 16))},
        }[method]()
        return {"jsonrpc": "2.0", "id": json["id"], "result": result}


@pytest.mark.core
def test_gas_cost_wei_components():
    assert gas_cost_wei(RECEIPTS["0xa"]) == 100_000 * 30 * GWEI
    assert gas_cost_wei(RECEIPTS["0xb"], gas_price=hex(20 * GWEI)) == 50_000 * 20 * GWEI
    assert gas_cost_wei(RECEIPTS["0xc"]) == 200_000 * 10 ** 6 + 7 * 10 ** 12 + 131_072
    with pytest.raises(RpcCallError, match="effectiveGasPrice"):
        gas_cost_wei(RECEIPTS["0xb"])


@pytest.mark.core
def test_client_transaction_gas_cost_falls_back_to_the_tx_price():
    node = _Node()
    rpc = JsonRpcClient("http://fake", http=node)
    assert rpc.transaction_gas_cost("0xa") == 100_000 * 30 * GWEI
    assert node.methods == ["eth_getTransactionReceipt"]  # no tx read when the receipt has the price
    assert rpc.transaction_gas_cost("0xb") == 50_000 * 20 * GWEI


@pytest.mark.core
def test_count_legs_matches_address_and_topic_case_insensitively():
    assert count_legs(RECEIPTS["0xa"], PROTOCOL, [SUPPLY, WITHDRAW]) == 2
    assert count_legs(RECEIPTS["0xa"], PROTOCOL, [SUPPLY]) == 1
    assert count_legs(RECEIPTS["0xc"], PROTOCOL, [SUPPLY]) == 1
    assert count_legs(RECEIPTS["0xb"], PROTOCOL, [SUPPLY]) == 0


@pytest.mark.core
def test_transaction_gas_costs_per_leg_and_caller_price():
    rpc = JsonRpcClient("http://fake", http=_Node())
    prices = {1_010: 2_000.0, 1_020: None}  # no price for block 20: USD stays NaN
    frame = transaction_gas_costs(rpc, ["0xA", "0xb", "0xa", "0xc"], leg_address=PROTOCOL,
                                  leg_topics=[SUPPLY, WITHDRAW], native_price=prices.get)
    assert list(frame.columns) == GAS_COLUMNS
    assert frame["tx"].tolist() == ["0xa", "0xb", "0xc"]
    assert frame["time"].tolist() == [1_010, 1_010, 1_020]
    assert frame["n_legs"].tolist() == [2, 0, 1]
    assert frame.loc[0, "gas_native"] == pytest.approx(0.003)
    assert frame.loc[0, "gas_native_per_leg"] == pytest.approx(0.0015)
    assert frame.loc[0, "gas_usd_per_leg"] == pytest.approx(3.0)
    assert math.isnan(frame.loc[1, "gas_native_per_leg"]) and frame.loc[1, "gas_usd"] == pytest.approx(2.0)
    assert math.isnan(frame.loc[2, "gas_usd"])


@pytest.mark.core
def test_transaction_gas_costs_without_filter_or_price():
    frame = transaction_gas_costs(JsonRpcClient("http://fake", http=_Node()), ["0xa"])
    assert frame.loc[0, "n_legs"] == 1 and frame.loc[0, "gas_native_per_leg"] == pytest.approx(0.003)
    assert math.isnan(frame.loc[0, "gas_usd"])
