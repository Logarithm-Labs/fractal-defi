"""Offline coverage for the DefiLlama DEX fee-tier loader (synthetic payloads)."""
import math

import pandas as pd
import pytest

from fractal.loaders import DefiLlamaDexFeeTierLoader
from fractal.loaders._http import LoaderHttpError
from fractal.loaders.base_loader import LoaderType
from fractal.loaders.defillama import parse_pool_fee_pct

pytestmark = pytest.mark.core

TOKEN_A = "0x" + "a" * 40
TOKEN_B = "0x" + "b" * 40
TOKEN_C = "0x" + "c" * 40


class _FakeHttp:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def get(self, url, params=None, timeout=None):
        del timeout
        self.calls.append((url, params))
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


def _pool(pool, project, meta, tvl, tokens=(TOKEN_A, TOKEN_B), chain="Ethereum"):
    return {
        "pool": pool,
        "chain": chain,
        "project": project,
        "symbol": "AAA-BBB",
        "poolMeta": meta,
        "tvlUsd": tvl,
        "underlyingTokens": list(tokens),
    }


def _loader(payload, tmp_path, monkeypatch, **kwargs):
    monkeypatch.setenv("DATA_PATH", str(tmp_path))
    loader = DefiLlamaDexFeeTierLoader([TOKEN_B.upper().replace("0X", "0x"), TOKEN_A], "ethereum", **kwargs)
    loader._http = _FakeHttp(payload)
    return loader


@pytest.mark.parametrize(
    ("meta", "fee", "resolution"),
    [
        ("0.05%", 0.05, 0.005),
        ("0.3 %", 0.3, 0.05),
        ("1%", 1.0, 0.5),
        ("0.00%", 0.0, 0.005),
        ("0.01% fee | CL range 0.09%", 0.01, 0.005),
        ("Fee: 0.04%", 0.04, 0.005),
        ("5 bps", 0.05, 0.005),
        ("1bp", 0.01, 0.005),
    ],
)
def test_parse_pool_fee_pct_known_shapes(meta, fee, resolution):
    parsed = parse_pool_fee_pct(meta)
    assert parsed == (pytest.approx(fee), pytest.approx(resolution))


@pytest.mark.parametrize(
    "meta",
    [
        None,
        "",
        "   ",
        float("nan"),
        42,
        ["0.05%"],
        {"fee": "0.05%"},
        "Dynamic fee (hook)",
        "dynamic 0.05%",
        "CL - 1",
        "Univ4 Kyber FairFlow",
        "80% AAA 20% BBB",
        "0.05% fee / 0.3% fee",
        "0.05% and 5 bps",
        "150%",
        "1" * 400 + "%",
    ],
)
def test_parse_pool_fee_pct_unknown_or_ambiguous_shapes_yield_nan(meta):
    fee, resolution = parse_pool_fee_pct(meta)
    assert math.isnan(fee) and math.isnan(resolution)


def test_matches_exact_token_set_on_chain_and_ranks_cheapest_first(tmp_path, monkeypatch):
    payload = {
        "status": "success",
        "data": [
            _pool("p-30", "uniswap-v3", "0.3%", 5e6),
            _pool("p-5-small", "uniswap-v3", "0.05%", 2e5),
            _pool("p-5-big", "pancakeswap-amm-v3", "0.05%", 9e5),
            _pool("p-1-thin", "sushiswap-v3", "0.01%", 5e4),
            _pool("p-dyn", "uniswap-v4", "Dynamic fee (hook)", 8e6),
            _pool("p-v2", "uniswap-v2", None, 3e6),
            _pool("p-other-chain", "uniswap-v3", "0.01%", 9e9, chain="Base"),
            _pool("p-three-coin", "curve-dex", "0.01%", 9e9, tokens=(TOKEN_A, TOKEN_B, TOKEN_C)),
            _pool("p-wrong-pair", "uniswap-v3", "0.01%", 9e9, tokens=(TOKEN_A, TOKEN_C)),
            _pool("p-not-dex", "aave-v3", "0.01%", 9e9),
            "not a pool",
            {"chain": "Ethereum", "underlyingTokens": None},
        ],
    }
    frame = _loader(payload, tmp_path, monkeypatch).read(with_run=True)
    assert frame["pool"].tolist() == ["p-5-big", "p-5-small", "p-30", "p-v2", "p-1-thin", "p-dyn"]
    assert frame["eligible"].tolist() == [True, True, True, True, False, False]
    v2 = frame.set_index("pool").loc["p-v2"]
    assert v2["fee_pct"] == pytest.approx(0.30)
    assert v2["fee_source"] == "protocol_constant"
    dyn = frame.set_index("pool").loc["p-dyn"]
    assert math.isnan(dyn["fee_pct"]) and dyn["fee_source"] == "unknown"
    assert frame.set_index("pool").loc["p-30", "fee_source"] == "pool_meta"


def test_cheapest_returns_first_eligible_or_none(tmp_path, monkeypatch):
    payload = {"data": [_pool("p-30", "uniswap-v3", "0.3%", 5e6), _pool("p-1", "uniswap-v3", "0.01%", 10.0)]}
    best = _loader(payload, tmp_path, monkeypatch).cheapest(with_run=True)
    assert best is not None and best["pool"] == "p-30"
    payload = {"data": [_pool("p-dyn", "uniswap-v4", "Dynamic fee (hook)", 5e6)]}
    assert _loader(payload, tmp_path, monkeypatch).cheapest(with_run=True) is None


def test_projects_min_tvl_and_fixed_fees_are_configurable(tmp_path, monkeypatch):
    payload = {"data": [_pool("p-v2", "uniswap-v2", None, 3e6), _pool("p-5", "uniswap-v3", "0.05%", 1e3)]}
    loader = _loader(payload, tmp_path, monkeypatch, projects={"uniswap-v2"}, fixed_fee_pct={})
    frame = loader.read(with_run=True)
    assert frame["pool"].tolist() == ["p-v2"]
    assert not frame["eligible"].any()
    frame = _loader(payload, tmp_path, monkeypatch, min_tvl_usd=0.0).read(with_run=True)
    assert frame["pool"].tolist() == ["p-5", "p-v2"]
    assert frame["eligible"].all()


def test_bad_tvl_values_are_not_eligible(tmp_path, monkeypatch):
    payload = {"data": [_pool("p-bad", "uniswap-v3", "0.01%", "n/a"), _pool("p-none", "uniswap-v3", "0.05%", None)]}
    frame = _loader(payload, tmp_path, monkeypatch, min_tvl_usd=0.0).read(with_run=True)
    bad = frame.set_index("pool").loc["p-bad"]
    assert math.isnan(bad["tvl_usd"]) and not bad["eligible"]
    assert frame.set_index("pool").loc["p-none", "tvl_usd"] == 0.0


@pytest.mark.parametrize("loader_type", [LoaderType.CSV, LoaderType.JSON, LoaderType.PICKLE])
def test_cached_read_matches_fresh_read(loader_type, tmp_path, monkeypatch):
    payload = {
        "data": [
            _pool("p-30", "uniswap-v3", "0.3%", 5e6),
            _pool("p-v2", "uniswap-v2", None, 3e6),
            _pool("p-dyn", "uniswap-v4", "Dynamic fee (hook)", 8e6),
        ]
    }
    fresh = _loader(payload, tmp_path, monkeypatch, loader_type=loader_type).read(with_run=True)
    cached = _loader(LoaderHttpError("network must not be used"), tmp_path, monkeypatch, loader_type=loader_type)
    pd.testing.assert_frame_equal(fresh, cached.read())


def test_empty_match_and_payload_errors(tmp_path, monkeypatch):
    frame = _loader({"data": []}, tmp_path, monkeypatch).read(with_run=True)
    assert frame.empty
    assert frame.dtypes["fee_pct"] == "float64"
    assert _loader({"data": []}, tmp_path, monkeypatch).cheapest(with_run=True) is None
    with pytest.raises(ValueError, match="no data array"):
        _loader({"status": "error"}, tmp_path, monkeypatch).read(with_run=True)
    with pytest.raises(LoaderHttpError):
        _loader(LoaderHttpError("synthetic"), tmp_path, monkeypatch).read(with_run=True)


def test_request_hits_the_keyless_yields_pools_endpoint(tmp_path, monkeypatch):
    loader = _loader({"data": []}, tmp_path, monkeypatch)
    loader.read(with_run=True)
    assert loader._http.calls == [("https://yields.llama.fi/pools", None)]


def test_cache_key_covers_chain_and_token_set_only():
    base = DefiLlamaDexFeeTierLoader([TOKEN_A, TOKEN_B], "Ethereum")
    assert base._cache_key() == DefiLlamaDexFeeTierLoader([TOKEN_B.upper(), TOKEN_A], "ethereum")._cache_key()
    assert base._cache_key() != DefiLlamaDexFeeTierLoader([TOKEN_A, TOKEN_C], "Ethereum")._cache_key()
    assert base._cache_key() != DefiLlamaDexFeeTierLoader([TOKEN_A, TOKEN_B], "Base")._cache_key()
    # Read-time options do not change what is fetched, so they share the cache.
    assert base._cache_key() == DefiLlamaDexFeeTierLoader([TOKEN_A, TOKEN_B], "Ethereum", min_tvl_usd=0)._cache_key()


def test_constructor_validation():
    with pytest.raises(ValueError, match="two distinct"):
        DefiLlamaDexFeeTierLoader([TOKEN_A, TOKEN_A.upper()], "Ethereum")
    with pytest.raises(ValueError, match="chain"):
        DefiLlamaDexFeeTierLoader([TOKEN_A, TOKEN_B], "")
    with pytest.raises(ValueError, match="min_tvl_usd"):
        DefiLlamaDexFeeTierLoader([TOKEN_A, TOKEN_B], "Ethereum", min_tvl_usd=float("nan"))
