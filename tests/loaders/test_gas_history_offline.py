"""Offline tests for :class:`fractal.loaders.onchain.GasPriceHistoryLoader` and
:meth:`JsonRpcClient.fee_history` (synthetic chain and fees, fake transport)."""
import numpy as np
import pandas as pd
import pytest

from fractal.loaders import GasPriceHistory, GasPriceHistoryLoader, JsonRpcClient, LoaderType, RpcCallError
from fractal.loaders.onchain.gas_history import percentile_column

GENESIS, BLOCK_TIME, TIP = pd.Timestamp("2024-01-01", tz="UTC"), 12, 3_000
T0 = int(GENESIS.timestamp())
RPC_KEY = "synthetic0rpc0key0000000000000007"
GWEI = 10 ** 9


def _time(block):
    return T0 + BLOCK_TIME * block


def _base_fee(block):
    return (10 + block % 7) * GWEI


def _ratio(block):
    return 0.0 if block % 5 == 0 else 0.5  # every fifth block is empty


def _reward(block, q):
    return 0 if _ratio(block) == 0 else int((1 + block % 3) * q / 50 * GWEI)


class _Node:
    """Fake node: blocks ``0..TIP`` every ``BLOCK_TIME`` s; records fee-history calls."""

    def __init__(self, base_fee=_base_fee):
        self.base_fee, self.fee_calls = base_fee, []

    def post(self, url, json=None, timeout=None, headers=None):
        method, params = json["method"], json["params"]
        if method == "eth_blockNumber":
            result = hex(TIP)
        elif method == "eth_getBlockByNumber":
            result = {"timestamp": hex(_time(int(params[0], 16)))}
        elif method == "eth_feeHistory":
            count, newest, qs = int(params[0], 16), int(params[1], 16), params[2]
            self.fee_calls.append((count, newest, tuple(qs)))
            blocks = range(newest - count + 1, newest + 1)
            result = {
                "oldestBlock": hex(blocks[0]),
                "baseFeePerGas": [hex(self.base_fee(b)) for b in blocks] + [hex(999 * GWEI)],  # + next block
                "gasUsedRatio": [_ratio(b) for b in blocks],
                "reward": [[hex(_reward(b, q)) for q in qs] for b in blocks],
            }
        else:
            raise AssertionError(method)
        return {"jsonrpc": "2.0", "id": json["id"], "result": result}


def _expected(newest_blocks, count, qs):
    blocks = [b for newest in newest_blocks for b in range(max(0, newest - count + 1), newest + 1)]
    full = [b for b in blocks if _ratio(b) > 0] or blocks
    base = np.mean([_base_fee(b) for b in blocks]) / GWEI
    tips = {q: float(np.median([_reward(b, q) for b in full])) / GWEI for q in qs}
    return base, tips, float(np.mean([_ratio(b) for b in blocks]))


def _loader(node=None, url="https://node.example", **kwargs):
    kwargs.setdefault("start_time", GENESIS + pd.Timedelta("1h"))
    kwargs.setdefault("end_time", GENESIS + pd.Timedelta("3h"))
    return GasPriceHistoryLoader(url, "testnet", http=node or _Node(), **kwargs)


@pytest.fixture(autouse=True)
def _cache_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_PATH", str(tmp_path))


@pytest.mark.core
def test_hourly_bars_sample_the_blocks_before_each_bar():
    node = _Node()
    gas = _loader(node).read(with_run=True)
    assert isinstance(gas, GasPriceHistory) and len(gas) == 3
    assert list(gas.index) == list(pd.date_range(GENESIS + pd.Timedelta("1h"), periods=3, freq="1h"))
    # 1h = 300 blocks: bar k ends exactly at block 300 k (no look-ahead past the bar time)
    assert list(gas["block"]) == [300, 600, 900]
    assert node.fee_calls == [(20, 300, (25.0, 50.0, 75.0)), (20, 600, (25.0, 50.0, 75.0)),
                              (20, 900, (25.0, 50.0, 75.0))]
    for newest, (_, row) in zip((300, 600, 900), gas.iterrows(), strict=True):
        base, tips, ratio = _expected([newest], 20, (25, 50, 75))
        assert row["base_fee_gwei"] == pytest.approx(base)  # the extra next-block entry is not averaged
        assert row["gas_used_ratio"] == pytest.approx(ratio)
        for q, tip in tips.items():
            assert row[percentile_column(q)] == pytest.approx(tip)
        assert row["priority_fee_gwei"] == pytest.approx(tips[50])
        assert row["gas_price_gwei"] == pytest.approx(base + tips[50])
        assert row["tx_cost_native"] == pytest.approx(21_000 * (base + tips[50]) / GWEI)
    assert "tx_cost_usd" not in gas.columns


@pytest.mark.core
def test_bar_time_between_blocks_uses_the_previous_block():
    start = GENESIS + pd.Timedelta(seconds=BLOCK_TIME * 100 + 5)
    gas = _loader(start_time=start, end_time=start).read(with_run=True)
    assert list(gas["block"]) == [100]


@pytest.mark.core
def test_empty_blocks_are_left_out_of_the_tip_median_unless_all_are_empty():
    gas = _loader(start_time=GENESIS + pd.Timedelta(seconds=BLOCK_TIME * 10), end_time=GENESIS
                  + pd.Timedelta(seconds=BLOCK_TIME * 10), blocks_per_sample=1).read(with_run=True)
    assert gas["block"].tolist() == [10] and gas["priority_fee_gwei"].tolist() == [0.0]  # only an empty block
    gas = _loader(start_time=GENESIS + pd.Timedelta(seconds=BLOCK_TIME * 11), end_time=GENESIS
                  + pd.Timedelta(seconds=BLOCK_TIME * 11), blocks_per_sample=2).read(with_run=True)
    assert gas["priority_fee_gwei"].tolist() == [pytest.approx(_reward(11, 50) / GWEI)]  # block 10 ignored


@pytest.mark.core
def test_samples_per_bar_pool_evenly_spaced_windows():
    node = _Node()
    gas = _loader(node, start_time=GENESIS + pd.Timedelta("2h"), end_time=GENESIS + pd.Timedelta("2h"),
                  samples_per_bar=4, blocks_per_sample=10, percentiles=(50,)).read(with_run=True)
    newest = [375, 450, 525, 600]  # (t - 1h, t] in quarters, 75 blocks apart
    assert [newest_block for _, newest_block, _ in node.fee_calls] == newest
    base, tips, _ = _expected(newest, 10, (50,))
    assert gas["base_fee_gwei"].iloc[0] == pytest.approx(base)
    assert gas["priority_fee_gwei"].iloc[0] == pytest.approx(tips[50])
    assert gas["block"].iloc[0] == 600


@pytest.mark.core
def test_bars_outside_the_chain_are_dropped_and_windows_clip_at_genesis():
    node = _Node()
    gas = _loader(node, start_time=GENESIS - pd.Timedelta("1h"), end_time=GENESIS + pd.Timedelta("12h"),
                  freq="1h").read(with_run=True)
    # the bar before genesis is dropped; TIP=3000 blocks = 10h, so bars after 10h are dropped
    assert gas.index[0] == GENESIS and gas.index[-1] == GENESIS + pd.Timedelta("10h")
    assert node.fee_calls[0][:2] == (1, 0)  # genesis: one block, not 20


@pytest.mark.core
def test_pre_london_blocks_keep_the_whole_price_in_the_tip():
    gas = _loader(_Node(base_fee=lambda b: 0)).read(with_run=True)
    assert (gas["base_fee_gwei"] == 0).all()
    assert (gas["gas_price_gwei"] == gas["priority_fee_gwei"]).all() and (gas["gas_price_gwei"] > 0).all()


@pytest.mark.core
def test_usd_comes_only_from_the_caller_price():
    seen = []

    def price(t):
        seen.append(t)
        return None if t == GENESIS + pd.Timedelta("2h") else 2_000.0

    gas = _loader(native_price=price, gas_units=100_000).read(with_run=True)
    assert seen == list(gas.index)
    assert gas["native_price_usd"].isna().tolist() == [False, True, False]
    expected = 100_000 * gas["gas_price_gwei"] / GWEI * 2_000.0
    assert gas["tx_cost_usd"].iloc[0] == pytest.approx(expected.iloc[0])
    assert np.isnan(gas["tx_cost_usd"].iloc[1])


@pytest.mark.core
@pytest.mark.parametrize("loader_type", [LoaderType.CSV, LoaderType.JSON, LoaderType.PICKLE])
def test_cache_round_trip_without_the_key_or_the_price(loader_type):
    fresh = _loader(url=f"https://node.example/v3/{RPC_KEY}", loader_type=loader_type).read(with_run=True)
    # a reader with no node at all, and a different price function, reads the cache
    cached = _loader(node=object(), url=f"https://node.example/v3/{RPC_KEY}", loader_type=loader_type,
                     native_price=lambda t: 1.0).read()
    plain = pd.DataFrame(cached)[list(fresh.columns)]
    pd.testing.assert_frame_equal(plain, pd.DataFrame(fresh), check_exact=False)
    assert (cached["native_price_usd"] == 1.0).all()
    loader = _loader(url=f"https://node.example/v3/{RPC_KEY}")
    assert RPC_KEY not in loader._cache_key() and RPC_KEY not in repr(loader)  # pylint: disable=protected-access


@pytest.mark.core
def test_empty_range_keeps_the_shape():
    later = GENESIS + pd.Timedelta("100D")
    gas = _loader(start_time=later, end_time=later, native_price=lambda t: 1.0).read(with_run=True)
    assert gas.empty
    assert {"base_fee_gwei", "priority_fee_gwei", "gas_price_gwei", percentile_column(25), "tx_cost_usd"} <= set(
        gas.columns)


@pytest.mark.core
@pytest.mark.parametrize(("kwargs", "match"), [
    ({"freq": "0h"}, "freq"),
    ({"blocks_per_sample": 0}, "blocks_per_sample"),
    ({"blocks_per_sample": 1025}, "blocks_per_sample"),
    ({"samples_per_bar": 0}, "samples_per_bar"),
    ({"percentiles": (50, 25)}, "increasing"),
    ({"percentiles": (50, 101)}, "increasing"),
    ({"cost_percentile": 90}, "cost_percentile"),
    ({"start_time": GENESIS + pd.Timedelta("2h"), "end_time": GENESIS}, "before start_time"),
])
def test_invalid_arguments(kwargs, match):
    with pytest.raises(ValueError, match=match):
        _loader(**kwargs)
    with pytest.raises(ValueError, match="chain"):
        GasPriceHistoryLoader("https://node.example", "../etc", GENESIS, GENESIS)


@pytest.mark.core
def test_fee_history_request_and_malformed_answer():
    node = _Node()
    rpc = JsonRpcClient("https://node.example", http=node)
    result = rpc.fee_history(3, 42, [10, 90])
    assert node.fee_calls == [(3, 42, (10.0, 90.0))]
    assert result["oldestBlock"] == hex(40) and len(result["baseFeePerGas"]) == 4

    class _Bad:
        def post(self, url, json=None, timeout=None, headers=None):
            return {"jsonrpc": "2.0", "id": json["id"], "result": None}

    with pytest.raises(RpcCallError, match="eth_feeHistory"):
        JsonRpcClient("https://node.example", http=_Bad()).fee_history(1, "latest")
