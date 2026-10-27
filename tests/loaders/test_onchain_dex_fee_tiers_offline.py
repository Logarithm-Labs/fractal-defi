"""Offline tests for :class:`fractal.loaders.onchain.dex_fee_tiers.OnchainDexFeeTierReader`
(synthetic factory, pools and balances behind a fake JSON-RPC node)."""
import math

import numpy as np
import pandas as pd
import pytest

from fractal.loaders.onchain.dex_fee_tiers import OnchainDexFeeTierReader, cheapest_fee_tier, merge_fee_tiers
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


LLAMA_COLUMNS = ["pool", "project", "symbol", "pool_meta", "fee_pct", "fee_resolution_pct", "fee_source", "tvl_usd",
                 "eligible"]


def _llama(*rows):
    """A ``DefiLlamaDexFeeTierLoader.read()``-shaped frame (synthetic pools)."""
    frame = pd.DataFrame([dict(zip(["pool", "project", "fee_pct", "tvl_usd"], r, strict=True)) for r in rows])
    frame["symbol"], frame["pool_meta"], frame["fee_resolution_pct"], frame["fee_source"] = "A-B", "", 0.0, "pool_meta"
    frame["eligible"] = frame["fee_pct"].notna() & (frame["tvl_usd"] >= 1_000.0)
    return frame[LLAMA_COLUMNS]


LLAMA = _llama(
    ("llama-uni", "uniswap-v3", 0.005, 9e6),       # covered on chain: dropped
    ("llama-curve", "curve-dex", 0.001, 9e6),      # covered on chain: dropped
    ("llama-pancake", "pancakeswap-amm-v3", 0.01, 9e6),
    ("llama-fluid", "fluid-dex", np.nan, 9e7),      # dynamic fee: never cheapest
    ("llama-aero", "aerodrome-slipstream", 0.01, 5e3),
)


@pytest.mark.core
def test_covered_projects():
    assert _reader()[0].covered_projects == {"uniswap-v3", "curve-dex"}
    assert OnchainDexFeeTierReader("https://rpc.example", TOK_A, TOK_B, http=_Node()).covered_projects == set()
    curve_only = OnchainDexFeeTierReader("https://rpc.example", TOK_A, TOK_B, curve_pools=[CURVE], http=_Node())
    assert curve_only.covered_projects == {"curve-dex"}


@pytest.mark.core
def test_merge_drops_defillama_rows_of_projects_read_on_chain():
    reader, _ = _reader(token_prices_usd={TOK_A: 1.0, TOK_B: 0.5}, min_tvl_usd=1_000.0)
    onchain = reader.read()
    merged = merge_fee_tiers(LLAMA, onchain, covered_projects=reader.covered_projects)
    assert merged["pool"].tolist() == [UNI_100, CURVE, UNI_500, "llama-pancake", "llama-fluid", "llama-aero"]
    assert merged["fee_source"].tolist()[:3] == ["onchain"] * 3 and merged["in_range_liquidity"].iloc[3:].isna().all()
    # same fee as the on-chain 0.01 % pool, larger TVL: the DefiLlama pool wins the tie
    best = cheapest_fee_tier(LLAMA, onchain, covered_projects=reader.covered_projects)
    assert best["pool"].tolist() == ["llama-pancake"]
    # a stricter floor recomputes eligibility for both sources
    best = cheapest_fee_tier(LLAMA, onchain, covered_projects=reader.covered_projects, min_tvl_usd=1e7)
    assert best.empty
    # without pancake, the on-chain pool wins; the cheaper DefiLlama uniswap-v3 row stays dropped
    best = cheapest_fee_tier(LLAMA.drop(index=2), onchain, min_tvl_usd=2e6)
    assert best["pool"].tolist() == [UNI_100]


@pytest.mark.core
def test_a_project_read_on_chain_without_pools_still_hides_its_defillama_rows():
    reader = OnchainDexFeeTierReader("https://rpc.example", TOK_A, TOK_B, uniswap_v3_factory=FACTORY,
                                     fee_tiers=[3_000], http=_Node())
    onchain = reader.read()
    assert onchain.empty
    assert cheapest_fee_tier(LLAMA, onchain)["pool"].tolist() == ["llama-curve"]  # frame-only coverage
    best = cheapest_fee_tier(LLAMA, onchain, covered_projects=reader.covered_projects)
    assert best["pool"].tolist() == ["llama-curve"]  # Curve was not read on chain here
    best = cheapest_fee_tier(LLAMA, None, covered_projects={"uniswap-v3", "curve-dex"})
    assert best["pool"].tolist() == ["llama-pancake"]
    assert cheapest_fee_tier(None, None).empty


@pytest.mark.core
def test_cheapest_per_pair():
    reader, _ = _reader(token_prices_usd={TOK_A: 1.0, TOK_B: 0.5}, min_tvl_usd=1_000.0)
    onchain = reader.read().assign(chain="base", pair="A/B")
    llama = pd.concat([LLAMA.assign(chain="base", pair="A/B"),
                       _llama(("llama-c2", "fluid-dex", np.nan, 1e6)).assign(chain="base", pair="A/C"),
                       _llama(("llama-d1", "uniswap-v2", 0.3, 1e6), ("llama-d2", "curve-dex", 0.02, 2e6),
                              ("llama-d3", "balancer-v3", 0.02, 3e6)).assign(chain="arbitrum", pair="A/D")],
                      ignore_index=True)
    best = cheapest_fee_tier(llama, onchain, covered_projects={"uniswap-v3"}, by=["chain", "pair"])
    # A/C has no eligible pool; curve-dex is not covered here, so DefiLlama's Curve rows stay
    assert best[["chain", "pair", "pool"]].values.tolist() == [
        ["arbitrum", "A/D", "llama-d3"], ["base", "A/B", "llama-curve"]]
