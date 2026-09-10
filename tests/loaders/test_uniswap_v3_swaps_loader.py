"""Tests for the on-chain Uniswap V3 swap-events loader."""
import os

import pytest

from fractal.loaders import LoaderType, UniswapV3SwapsLoader
from fractal.loaders._http import LoaderHttpError

BASE_WETH_USDC_500 = "0xd0b53d9277642d899df5c87a3966a349a798f224"


class _RangeLimitedHttp:
    """Fake HttpClient: rejects eth_getLogs ranges wider than max_span.

    ``reject_as_http`` picks the failure mode — a JSON-RPC error body
    (RpcLoaderException path) or a transport-level LoaderHttpError.
    """

    def __init__(self, max_span: int, reject_as_http: bool):
        self.max_span = max_span
        self.reject_as_http = reject_as_http
        self.spans = []  # requested eth_getLogs spans, in order

    def post(self, url, json=None, timeout=None, headers=None):
        method = json["method"]
        if method == "eth_blockNumber":
            return {"jsonrpc": "2.0", "id": 1, "result": hex(20_000)}
        assert method == "eth_getLogs"
        params = json["params"][0]
        span = int(params["toBlock"], 16) - int(params["fromBlock"], 16) + 1
        self.spans.append(span)
        if span > self.max_span:
            if self.reject_as_http:
                raise LoaderHttpError("HTTP 413 for eth_getLogs: range too large")
            return {"jsonrpc": "2.0", "id": 1, "error": {"message": "range too large"}}
        return {"jsonrpc": "2.0", "id": 1, "result": []}


@pytest.mark.core
@pytest.mark.parametrize("reject_as_http", [True, False])
def test_chunk_narrows_on_range_rejection_and_recovers(reject_as_http):
    """Both rejection flavors (HTTP-level and JSON-RPC-level) must halve
    the chunk and retry the same range; after a success the chunk grows
    back toward the configured size instead of staying shrunken."""
    http = _RangeLimitedHttp(max_span=5_000, reject_as_http=reject_as_http)
    loader = UniswapV3SwapsLoader(
        rpc_url="http://fake", pool=BASE_WETH_USDC_500,
        token0_decimals=18, token1_decimals=6,
        start_block=1, end_block=20_000, chunk_size=10_000, http=http,
    )
    loader.extract()
    assert http.spans[0] == 10_000       # first attempt at full chunk
    assert http.spans[1] == 5_000        # halved and retried the same range
    assert max(http.spans[2:]) == 10_000  # grew back after success
    assert loader._data.empty            # no logs returned


@pytest.mark.core
def test_chunk_floor_reraises():
    """When even the minimum chunk is rejected, the error propagates."""
    http = _RangeLimitedHttp(max_span=100, reject_as_http=True)
    loader = UniswapV3SwapsLoader(
        rpc_url="http://fake", pool=BASE_WETH_USDC_500,
        token0_decimals=18, token1_decimals=6,
        start_block=1, end_block=20_000, chunk_size=10_000, http=http,
    )
    with pytest.raises(LoaderHttpError):
        loader.extract()


@pytest.mark.core
def test_decode_log_signed_fields():
    """int256 amounts and int24 tick decode with two's complement."""
    # amount0 = -2 (token0 leaves the pool), amount1 = 1000, sqrtP = 2**96
    # (price 1.0 raw), liquidity = 7, tick = -5
    words = [
        (-2) % (1 << 256),
        1000,
        1 << 96,
        7,
        (-5) % (1 << 256),
    ]
    log = {
        "data": "0x" + "".join(hex(w)[2:].rjust(64, "0") for w in words),
        "blockNumber": hex(123),
        "logIndex": hex(4),
    }
    row = UniswapV3SwapsLoader._decode_log(log)
    assert row == {
        "amount0": -2, "amount1": 1000, "sqrt_price_x96": 1 << 96,
        "liquidity": 7, "tick": -5, "block": 123, "log_index": 4,
    }


@pytest.mark.integration
@pytest.mark.slow
def test_swaps_loader_real_range():
    """Small real range on Base WETH/USDC 0.05% — the pool trades every block."""
    rpc_url = os.getenv("BASE_RPC_URL", "https://mainnet.base.org")
    loader = UniswapV3SwapsLoader(
        rpc_url=rpc_url,
        pool=BASE_WETH_USDC_500,
        token0_decimals=18,
        token1_decimals=6,
        start_block=48_138_800,
        end_block=48_139_800,  # ~33 minutes of Base blocks
        loader_type=LoaderType.CSV,
    )
    swaps = loader.read(with_run=True)
    assert len(swaps) > 0
    assert set(swaps.columns) == {"amount0", "amount1", "liquidity", "tick",
                                  "price", "block", "log_index"}
    # one leg enters the pool, the other leaves
    assert ((swaps["amount0"] > 0) ^ (swaps["amount1"] > 0)).all()
    assert (swaps["liquidity"] > 0).all()
    # human price of WETH in USDC — sane range
    assert swaps["price"].between(100, 100_000).all()
    assert swaps.index.is_monotonic_increasing
