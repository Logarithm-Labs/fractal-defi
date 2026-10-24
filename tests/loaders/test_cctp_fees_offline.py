"""Offline tests for the Circle Iris V2 CCTP fee loader."""

import pandas as pd
import pytest

from fractal.loaders import CCTP_DOMAINS, CctpFeeLoader
from fractal.loaders._http import LoaderHttpError
from fractal.loaders.bridges.cctp import IRIS_API

SOURCE_DOMAIN = 0
DESTINATION_DOMAIN = 6


class _FakeHttp:
    def __init__(self, payload) -> None:
        self.payload = payload
        self.calls: list[tuple[str, dict]] = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, dict(params or {})))
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


@pytest.mark.core
def test_parses_fast_and_standard_fee_entries():
    http = _FakeHttp([
        {"finalityThreshold": 1000, "minimumFee": "5"},
        {"finalityThreshold": 2000, "minimumFee": "0"},
    ])
    loader = CctpFeeLoader(SOURCE_DOMAIN, DESTINATION_DOMAIN, http=http)
    loader.extract()
    loader.transform()
    assert loader._data.to_dict("records") == [
        {
            "src_domain": SOURCE_DOMAIN,
            "dst_domain": DESTINATION_DOMAIN,
            "mode": "fast",
            "minimum_fee_bps": 5.0,
            "finality_threshold": 1000,
        },
        {
            "src_domain": SOURCE_DOMAIN,
            "dst_domain": DESTINATION_DOMAIN,
            "mode": "standard",
            "minimum_fee_bps": 0.0,
            "finality_threshold": 2000,
        },
    ]
    assert http.calls == [(f"{IRIS_API}/v2/burn/USDC/fees/{SOURCE_DOMAIN}/{DESTINATION_DOMAIN}", {})]


@pytest.mark.core
def test_unknown_route_404_returns_empty_frame():
    loader = CctpFeeLoader(
        SOURCE_DOMAIN,
        999,
        http=_FakeHttp(LoaderHttpError("HTTP 404 for unknown route")),
    )
    loader.extract()
    loader.transform()
    assert loader._data.empty
    assert list(loader._data.columns) == [
        "src_domain",
        "dst_domain",
        "mode",
        "minimum_fee_bps",
        "finality_threshold",
    ]


@pytest.mark.core
def test_data_envelope_and_non_fast_standard_entries():
    loader = CctpFeeLoader(SOURCE_DOMAIN, DESTINATION_DOMAIN)
    loader._data = {"data": [
        {"finalityThreshold": 1500, "minimumFee": 9},
        {"finalityThreshold": 2000, "minimumFee": 1.5},
    ]}
    loader.transform()
    assert loader._data["mode"].tolist() == ["standard"]
    assert loader._data["minimum_fee_bps"].tolist() == [1.5]


@pytest.mark.core
def test_domain_map_matches_documented_reference_chains():
    assert CCTP_DOMAINS == {
        "ethereum": 0,
        "optimism": 2,
        "arbitrum": 3,
        "base": 6,
        "polygon": 7,
        "unichain": 10,
        "monad": 15,
        "hyperevm": 19,
        "arc": 26,
        "plasma": 33,
    }


@pytest.mark.core
def test_cache_key_is_deterministic_and_covers_route():
    base = CctpFeeLoader(SOURCE_DOMAIN, DESTINATION_DOMAIN)._cache_key()
    assert CctpFeeLoader(SOURCE_DOMAIN, DESTINATION_DOMAIN)._cache_key() == base
    assert CctpFeeLoader(SOURCE_DOMAIN + 1, DESTINATION_DOMAIN)._cache_key() != base
    assert CctpFeeLoader(SOURCE_DOMAIN, DESTINATION_DOMAIN + 1)._cache_key() != base


@pytest.mark.core
def test_cache_round_trip_preserves_fees(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_PATH", str(tmp_path))
    fresh = CctpFeeLoader(
        SOURCE_DOMAIN,
        DESTINATION_DOMAIN,
        http=_FakeHttp([
            {"finalityThreshold": 1000, "minimumFee": "5"},
            {"finalityThreshold": 2000, "minimumFee": "0"},
        ]),
    )
    first = fresh.read(with_run=True)
    cached = CctpFeeLoader(SOURCE_DOMAIN, DESTINATION_DOMAIN, http=_FakeHttp([])).read(with_run=False)
    pd.testing.assert_frame_equal(first, cached)
    assert first["minimum_fee_bps"].tolist() == [5.0, 0.0]


@pytest.mark.integration
def test_cctp_fee_integration_smoke():
    frame = CctpFeeLoader(CCTP_DOMAINS["ethereum"], CCTP_DOMAINS["base"]).read(with_run=True)
    assert {"fast", "standard"}.issubset(set(frame["mode"]))
    assert (frame["minimum_fee_bps"] >= 0).all()
