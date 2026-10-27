"""Offline tests for :class:`fractal.loaders.onchain.dex_fee_tiers.OnchainDexFeeTierReader`
(synthetic factory, pools and balances behind a fake JSON-RPC node)."""
import math

import pytest

from fractal.loaders.onchain.dex_fee_tiers import OnchainDexFeeTierReader
from fractal.loaders.rpc import encode_address, function_selector

FACTORY = "0x" + "fa" * 20
TOK_A, TOK_B = "0x" + "0a" * 20, "0x" + "0b" * 20
UNI_100, UNI_500 = "0x" + "51" * 20, "0x" + "55" * 20
CURVE = "0x" + "c4" * 20
DECIMALS = {TOK_A: 6, TOK_B: 18}
BALANCES = {(TOK_A, UNI_100): 2_000_000 * 10 ** 6, (TOK_B, UNI_100): 1_000_000 * 10 ** 18,
            (TOK_A, UNI_500): 10 * 10 ** 6, (TOK_B, UNI_500): 10 * 10 ** 18,
            (TOK_A, CURVE): 5_000_000 * 10 ** 6, (TOK_B, CURVE): 5_000_000 * 10 ** 18}


def _word(value):
    return "0x" + format(value, "064x")


class _Node:
    def __init__(self):
        self.blocks = set()

    def post(self, url, json=None, timeout=None, headers=None):
        call, block = json["params"]
        self.blocks.add(block)
        to, data = call["to"], call["data"]
        sel, args = data[:10], data[10:]
        if to == FACTORY and sel == function_selector("getPool(address,address,uint24)"):
            assert args[:128] == encode_address(TOK_A) + encode_address(TOK_B)
            pool = {100: UNI_100, 500: UNI_500}.get(int(args[128:], 16), "0x" + "00" * 20)
            result = "0x" + encode_address(pool)
        elif sel == function_selector("liquidity()"):
            result = _word({UNI_100: 7 * 10 ** 12, UNI_500: 3}[to])
        elif sel == function_selector("fee()"):
            assert to == CURVE
            result = _word(4_000_000)  # 0.04 %
        elif sel == function_selector("decimals()"):
            result = _word(DECIMALS[to])
        elif sel == function_selector("balanceOf(address)"):
            result = _word(BALANCES[(to, "0x" + args[-40:])])
        else:
            raise AssertionError((to, sel))
        return {"jsonrpc": "2.0", "id": json["id"], "result": result}


def _reader(**kwargs):
    node = _Node()
    kwargs.setdefault("http", node)
    return OnchainDexFeeTierReader("https://rpc.example", TOK_A, TOK_B, uniswap_v3_factory=FACTORY,
                                   curve_pools=[CURVE], **kwargs), node


@pytest.mark.core
def test_reads_existing_tiers_and_curve_fee_cheapest_first():
    reader, node = _reader(token_prices_usd={TOK_A: 1.0, TOK_B: 0.5}, min_tvl_usd=1_000.0, block=123)
    frame = reader.read()
    assert list(frame.columns) == OnchainDexFeeTierReader.COLUMNS
    assert frame["pool"].tolist() == [UNI_100, CURVE, UNI_500]  # 0.05 % pool is below the TVL floor
    assert frame["fee_pct"].tolist() == pytest.approx([0.01, 0.04, 0.05])
    assert frame["eligible"].tolist() == [True, True, False]
    assert frame["tvl_usd"].tolist() == pytest.approx([2_500_000.0, 7_500_000.0, 15.0])
    assert frame["project"].tolist() == ["uniswap-v3", "curve-dex", "uniswap-v3"]
    assert frame.loc[0, "in_range_liquidity"] == 7e12 and math.isnan(frame.loc[1, "in_range_liquidity"])
    assert set(frame["fee_source"]) == {"onchain"} and (frame["fee_resolution_pct"] == 0.0).all()
    assert node.blocks == {"0x7b"}
    assert reader.cheapest()["pool"] == UNI_100


@pytest.mark.core
def test_without_prices_tvl_is_nan_and_the_floor_is_not_applied():
    reader, _ = _reader()
    frame = reader.read()
    assert frame["tvl_usd"].isna().all() and frame["eligible"].all()
    floored, _ = _reader(min_tvl_usd=1.0)
    assert floored.cheapest() is None  # unknown TVL never passes a floor


@pytest.mark.core
def test_curve_only_and_empty():
    reader = OnchainDexFeeTierReader("https://rpc.example", TOK_A, TOK_B, http=_Node())
    assert reader.read().empty and reader.cheapest() is None
    reader = OnchainDexFeeTierReader("https://rpc.example", TOK_A, TOK_B, curve_pools=[CURVE], http=_Node())
    assert reader.read()["fee_pct"].tolist() == pytest.approx([0.04])
