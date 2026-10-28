"""Offline tests for :meth:`JsonRpcClient.block_at_timestamp` and
:func:`fractal.loaders.rpc.hex_to_int` (synthetic chain, fake transport)."""
import pytest

from fractal.loaders import JsonRpcClient
from fractal.loaders.rpc import hex_to_int


class _Chain:
    """Fake node over a list of block times; counts header reads."""

    def __init__(self, times):
        self.times = list(times)
        self.header_reads = 0

    def post(self, url, json=None, timeout=None, headers=None):
        method, params = json["method"], json["params"]
        if method == "eth_blockNumber":
            result = hex(len(self.times) - 1)
        elif method == "eth_getBlockByNumber":
            self.header_reads += 1
            result = {"timestamp": hex(self.times[int(params[0], 16)])}
        else:
            raise AssertionError(method)
        return {"jsonrpc": "2.0", "id": json["id"], "result": result}


def _irregular_times(n=5_000, t0=1_600_000_000):
    """Strictly non-decreasing times with gaps, repeats and a long stall."""
    times, t = [], t0
    for i in range(n):
        t += (0 if i % 7 == 0 else 12) + (3_600 if i == 2_500 else 0)
        times.append(t)
    return times


def _brute(times, ts, closest):
    if closest == "after":
        return next(i for i, t in enumerate(times) if t >= ts)
    return max(i for i, t in enumerate(times) if t <= ts)


@pytest.mark.core
@pytest.mark.parametrize("closest", ["after", "before"])
def test_block_at_timestamp_matches_brute_force(closest):
    times = _irregular_times()
    rpc = JsonRpcClient("http://fake", http=_Chain(times))
    probes = [times[0], times[1] - 1, times[2_499] + 1, times[2_500] - 1, times[2_500], times[-1],
              times[700], times[700] + 5, times[4_321] - 3]
    for ts in probes:
        assert rpc.block_at_timestamp(ts, closest=closest) == _brute(times, ts, closest), ts


@pytest.mark.core
def test_block_at_timestamp_converges_fast_on_a_regular_chain():
    times = [1_700_000_000 + 12 * i for i in range(2_000_000)]
    node = _Chain(times)
    rpc = JsonRpcClient("http://fake", http=node)
    assert rpc.block_at_timestamp(times[1_234_567] - 5) == 1_234_567
    assert node.header_reads <= 8  # interpolation, not ~21 bisection steps


@pytest.mark.core
def test_block_at_timestamp_bounds_and_errors():
    times = [100, 112, 124, 136]
    rpc = JsonRpcClient("http://fake", http=_Chain(times))
    assert rpc.block_at_timestamp(50) == 0
    assert rpc.block_at_timestamp(500, closest="before") == 3
    assert rpc.block_at_timestamp(124, closest="before", low=1, high=2) == 2
    with pytest.raises(ValueError, match="no block at or after"):
        rpc.block_at_timestamp(137)
    with pytest.raises(ValueError, match="no block at or before"):
        rpc.block_at_timestamp(99, closest="before")
    with pytest.raises(ValueError, match="closest"):
        rpc.block_at_timestamp(120, closest="nearest")


@pytest.mark.core
def test_block_at_timestamp_reuses_cached_headers():
    times = [1_700_000_000 + 12 * i for i in range(10_000)]
    node = _Chain(times)
    rpc = JsonRpcClient("http://fake", http=node)
    first = rpc.block_at_timestamp(times[4_321], closest="before")
    reads = node.header_reads
    assert rpc.block_at_timestamp(times[4_321], closest="before") == first == 4_321
    assert node.header_reads == reads  # the second search only reads the tip number


@pytest.mark.core
def test_hex_to_int_accepts_provider_encodings():
    assert [hex_to_int(v) for v in (7, "0x", None, "", "0x1f", "31")] == [7, 0, 0, 0, 31, 31]
