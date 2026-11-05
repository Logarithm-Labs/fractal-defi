"""Offline tests for :class:`fractal.loaders.onchain.MorphoOraclePriceLoader` and
:func:`fractal.loaders.onchain.calibrate_oracle_price` (synthetic chain, market and prices, fake transport)."""
import math

import numpy as np
import pandas as pd
import pytest

from fractal.loaders import LoaderType, MorphoOraclePriceLoader, PriceHistory, calibrate_oracle_price
from fractal.loaders.morpho import MORPHO_GRAPHQL_URL
from fractal.loaders.rpc import function_selector

GENESIS, BLOCK_TIME, BLOCKS_PER_DAY = pd.Timestamp("2026-01-01", tz="UTC"), 12, 7_200
TIP = 3 * BLOCKS_PER_DAY
T0 = int(GENESIS.timestamp())
MARKET_ID = "0x" + "ab" * 32
ORACLE = "0x" + "0c" * 20
RPC_KEY = "synthetic0rpc0key0000000000000011"
PRICE_SELECTOR = function_selector("price()")


def _time(block):
    return T0 + BLOCK_TIME * block


def _price(block):
    """Loan-token units per collateral unit: drifts up by 1e-7 per block from 0.95."""
    return 0.95 + block * 1e-7


def _raw_price(block, loan_decimals, collateral_decimals):
    scale = 36 + loan_decimals - collateral_decimals
    return 95 * 10 ** (scale - 2) + block * 10 ** (scale - 7)


class _Node:
    """Fake Morpho API + node: blocks ``0..TIP`` every ``BLOCK_TIME`` s, the oracle deployed at ``deployed``."""

    def __init__(self, loan_decimals=6, collateral_decimals=18, deployed=0):
        self.loan_decimals, self.collateral_decimals, self.deployed = loan_decimals, collateral_decimals, deployed
        self.api_calls, self.price_calls = 0, []

    def _market_info(self):
        self.api_calls += 1
        return {"data": {"marketById": {
            "marketId": MARKET_ID, "lltv": "915000000000000000", "irmAddress": "0x" + "1e" * 20,
            "oracle": {"address": ORACLE, "type": "MorphoChainlinkOracleV2"},
            "loanAsset": {"address": "0x" + "10" * 20, "symbol": "LOAN", "decimals": self.loan_decimals},
            "collateralAsset": {"address": "0x" + "20" * 20, "symbol": "PT-SYN",
                                "decimals": self.collateral_decimals},
            "state": None,
        }}}

    def post(self, url, json=None, timeout=None, headers=None):
        if url == MORPHO_GRAPHQL_URL:
            assert "MarketInfo" in json["query"] and json["variables"]["id"] == MARKET_ID
            return self._market_info()
        method, params = json["method"], json["params"]
        if method == "eth_blockNumber":
            result = hex(TIP)
        elif method == "eth_getBlockByNumber":
            result = {"timestamp": hex(_time(int(params[0], 16)))}
        elif method == "eth_call":
            call, block = params[0], int(params[1], 16)
            assert call == {"to": ORACLE, "data": PRICE_SELECTOR}
            self.price_calls.append(block)
            if block < self.deployed:
                result = "0x"  # no code at the address yet
            else:
                result = "0x" + f"{_raw_price(block, self.loan_decimals, self.collateral_decimals):064x}"
        else:
            raise AssertionError(method)
        return {"jsonrpc": "2.0", "id": json["id"], "result": result}


def _loader(node=None, url="https://node.example", **kwargs):
    kwargs.setdefault("start_time", GENESIS + pd.Timedelta("1D"))
    kwargs.setdefault("end_time", GENESIS + pd.Timedelta("3D"))
    return MorphoOraclePriceLoader(url, MARKET_ID, http=node or _Node(), **kwargs)


@pytest.fixture(autouse=True)
def _cache_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_PATH", str(tmp_path))


@pytest.mark.core
def test_daily_bars_read_the_oracle_at_the_last_block_before_each_bar():
    node = _Node()
    prices = _loader(node).read(with_run=True)
    assert isinstance(prices, PriceHistory) and len(prices) == 3
    assert list(prices.index) == list(pd.date_range(GENESIS + pd.Timedelta("1D"), periods=3, freq="1D"))
    # one day = 7 200 blocks: bar k ends exactly at block 7 200 k (no look-ahead past the bar time)
    assert node.price_calls == [7_200, 14_400, 21_600]
    assert prices["price"].tolist() == pytest.approx([_price(b) for b in (7_200, 14_400, 21_600)], rel=1e-15)
    assert node.api_calls == 1


@pytest.mark.core
@pytest.mark.parametrize(("loan_decimals", "collateral_decimals"), [(6, 18), (18, 18), (18, 6), (6, 6)])
def test_price_is_scaled_by_both_token_decimals(loan_decimals, collateral_decimals):
    node = _Node(loan_decimals=loan_decimals, collateral_decimals=collateral_decimals)
    prices = _loader(node, end_time=GENESIS + pd.Timedelta("1D")).read(with_run=True)
    assert prices["price"].tolist() == [pytest.approx(_price(BLOCKS_PER_DAY), rel=1e-15)]


@pytest.mark.core
def test_bar_time_between_blocks_uses_the_previous_block():
    start = GENESIS + pd.Timedelta(seconds=BLOCK_TIME * 100 + 5)
    node = _Node()
    _loader(node, start_time=start, end_time=start).read(with_run=True)
    assert node.price_calls == [100]


@pytest.mark.core
def test_bars_after_the_chain_tip_are_dropped():
    node = _Node()
    prices = _loader(node, end_time=GENESIS + pd.Timedelta("10D")).read(with_run=True)
    assert prices.index[-1] == GENESIS + pd.Timedelta("3D")  # TIP = 3 days of blocks
    assert node.price_calls == [7_200, 14_400, 21_600]


@pytest.mark.core
def test_an_oracle_without_code_at_the_block_raises():
    with pytest.raises(ValueError, match="no data at block 7200"):
        _loader(_Node(deployed=10_000)).read(with_run=True)


@pytest.mark.core
@pytest.mark.parametrize("loader_type", [LoaderType.CSV, LoaderType.JSON, LoaderType.PICKLE])
def test_cache_round_trip_without_the_node_or_the_key(loader_type):
    url = f"https://node.example/v3/{RPC_KEY}"
    fresh = _loader(url=url, loader_type=loader_type).read(with_run=True)
    cached = _loader(node=object(), url=url, loader_type=loader_type).read()  # no API, no node
    pd.testing.assert_frame_equal(pd.DataFrame(cached), pd.DataFrame(fresh), check_exact=False)
    loader = _loader(url=url)
    assert RPC_KEY not in loader._cache_key() and RPC_KEY not in repr(loader)  # pylint: disable=protected-access


@pytest.mark.core
def test_cache_key_is_deterministic_and_covers_every_input():
    key = _loader()._cache_key()  # pylint: disable=protected-access
    assert _loader(url="https://other.example")._cache_key() == key  # pylint: disable=protected-access
    changed = [
        _loader(start_time=GENESIS + pd.Timedelta("2D")),
        _loader(end_time=GENESIS + pd.Timedelta("2D")),
        _loader(freq="1h"),
        _loader(chain_id=8453),
        MorphoOraclePriceLoader("https://node.example", "0x" + "cd" * 32, GENESIS + pd.Timedelta("1D"),
                                GENESIS + pd.Timedelta("3D"), http=_Node()),
    ]
    keys = [loader._cache_key() for loader in changed]  # pylint: disable=protected-access
    assert key not in keys and len(set(keys)) == len(keys)


@pytest.mark.core
def test_empty_range_keeps_the_shape():
    later = GENESIS + pd.Timedelta("100D")
    prices = _loader(start_time=later, end_time=later).read(with_run=True)
    assert isinstance(prices, PriceHistory) and prices.empty and list(prices.columns) == ["price"]


@pytest.mark.core
@pytest.mark.parametrize(("kwargs", "match"), [
    ({"freq": "0h"}, "freq"),
    ({"start_time": GENESIS + pd.Timedelta("2D"), "end_time": GENESIS}, "before start_time"),
])
def test_invalid_arguments(kwargs, match):
    with pytest.raises(ValueError, match=match):
        _loader(**kwargs)
    with pytest.raises(ValueError, match="market_id"):
        MorphoOraclePriceLoader("https://node.example", "0x1234", GENESIS, GENESIS)


# ------------------------------------------------------------------ calibration
HOURS = pd.date_range(GENESIS, periods=49, freq="1h")  # two days, both ends included
MODEL = pd.Series(np.linspace(0.96, 0.97, len(HOURS)), index=HOURS, name="price")


@pytest.mark.core
def test_calibration_with_constant_kappa_scales_the_whole_model():
    days = HOURS[::24]
    oracle = MODEL.loc[days] * 0.87
    calibrated = calibrate_oracle_price(MODEL, oracle)
    pd.testing.assert_series_equal(calibrated, MODEL * 0.87, rtol=1e-14)


@pytest.mark.core
def test_calibration_interpolates_log_kappa_linearly_between_reads():
    days = HOURS[[0, 24]]
    oracle = MODEL.loc[days] * np.exp([0.0, 0.01])
    calibrated = calibrate_oracle_price(MODEL, oracle)
    assert calibrated.iloc[12] / MODEL.iloc[12] == pytest.approx(math.exp(0.005), rel=1e-12)  # midday
    assert calibrated.iloc[24] == pytest.approx(oracle.iloc[1], rel=1e-14)  # exact at a read


@pytest.mark.core
def test_calibration_holds_the_edge_kappa_outside_the_reads():
    oracle = MODEL.loc[HOURS[[12, 24]]] * np.array([0.99, 1.01])
    kappa = calibrate_oracle_price(MODEL, oracle) / MODEL
    assert kappa.iloc[:12].tolist() == pytest.approx([0.99] * 12, rel=1e-14)
    assert kappa.iloc[25:].tolist() == pytest.approx([1.01] * 24, rel=1e-14)


@pytest.mark.core
def test_calibration_rejects_reads_off_the_model_grid():
    oracle = pd.Series([0.95], index=[GENESIS + pd.Timedelta(minutes=30)])
    with pytest.raises(ValueError, match="not on the model grid"):
        calibrate_oracle_price(MODEL, oracle)


@pytest.mark.core
@pytest.mark.parametrize("bad", [0.0, -0.9, float("nan")])
def test_calibration_rejects_non_positive_or_missing_prices(bad):
    oracle = pd.Series([bad], index=[GENESIS])
    with pytest.raises(ValueError, match="positive"):
        calibrate_oracle_price(MODEL, oracle)
    with pytest.raises(ValueError, match="at least one"):
        calibrate_oracle_price(MODEL, oracle.iloc[:0])
