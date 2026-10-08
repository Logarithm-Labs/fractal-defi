"""Offline tests for the DefiLlama loaders (issue #57) — no network.

The shared ``HttpClient`` is replaced with canned payloads so the full
``read(with_run=True)`` lifecycle (extract → transform → CSV cache →
typed struct) runs entirely offline.
"""
import logging
import time
import traceback
import warnings
from datetime import datetime, timezone

import pandas as pd
import pytest

from fractal.loaders._http import HttpClient, HttpConfig, LoaderHttpError
from fractal.loaders.defillama import DefiLlamaDEXLoader, DefiLlamaPoolLoader, DefiLlamaTVLLoader, DefiLlamaYieldsLoader
from fractal.loaders.defillama.defillama import _parse_chart
from fractal.loaders.structs import DEXHistory, RateHistory, TVLHistory

pytestmark = pytest.mark.core

UTC = timezone.utc
DAY = 86400
T0 = 1704067200  # 2024-01-01 00:00:00 UTC


def _http_with(payloads):
    """Fake ``HttpClient.get`` serving a canned path→payload map."""

    class FakeHttp:
        def __init__(self):
            self.calls = []

        def get(self, url, params=None, timeout=None):
            self.calls.append((url, params))
            for path, payload in payloads.items():
                if url.endswith(path):
                    if isinstance(payload, Exception):
                        raise payload
                    return payload
            raise AssertionError(f"unexpected URL {url}")

    return FakeHttp()


@pytest.fixture
def offline_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_PATH", str(tmp_path))
    return tmp_path


def _chart(epochs, values):
    return [[e, v] for e, v in zip(epochs, values, strict=False)]


# ------------------------------------------------------------- chart parser
@pytest.mark.core
def test_parse_chart_supports_pair_and_dict_shapes_and_skips_missing():
    # pairs (summary charts) and dict points (/protocol chainTvls) both parse
    epochs, values = _parse_chart([[2, 1.0], [3, None]])
    assert epochs.tolist() == [2] and values.tolist() == [1.0]
    epochs, values = _parse_chart([{"date": 2, "totalLiquidityUSD": 5.0},
                                   {"date": 3, "value": 6.0}])
    assert epochs.tolist() == [2, 3] and values.tolist() == [5.0, 6.0]


@pytest.mark.core
def test_parse_chart_rejects_unknown_shapes():
    with pytest.raises(ValueError):
        _parse_chart([{"weird": 1}])
    with pytest.raises(ValueError):
        _parse_chart([42])


# ------------------------------------------------------------------- TVL
def _tvl_payload():
    tvl_a = [{"date": T0, "totalLiquidityUSD": 100.0},
             {"date": T0 + DAY, "totalLiquidityUSD": 120.0}]
    tvl_b = [{"date": T0, "totalLiquidityUSD": 1.0},
             {"date": T0 + DAY, "totalLiquidityUSD": 2.0}]
    return {
        "name": "proto", "chains": ["A", "B"],
        "chainTvls": {"A": {"tvl": tvl_a}, "A-staking": {"tvl": [{"date": T0, "totalLiquidityUSD": 999.0}]},
                      "B": {"tvl": tvl_b}},
    }


def test_tvl_loader_aggregates_chains_and_round_trips(offline_cache, monkeypatch):
    loader = DefiLlamaTVLLoader("proto")
    loader._http = _http_with({"/protocol/proto": _tvl_payload()})
    history = loader.read(with_run=True)
    assert isinstance(history, TVLHistory)
    assert history["tvl"].tolist() == pytest.approx([101.0, 122.0])
    assert history.index[0] == pd.Timestamp(T0, unit="s", tz="UTC")
    assert history.index.name == "time"
    # CSV cache round trip returns the same series
    fresh = DefiLlamaTVLLoader("proto")
    again = fresh.read()
    assert again["tvl"].tolist() == pytest.approx([101.0, 122.0])


def test_tvl_loader_single_chain_and_window(offline_cache):
    loader = DefiLlamaTVLLoader("proto", chain="A",
                                start_time=datetime(2024, 1, 2, tzinfo=UTC))
    loader._http = _http_with({"/protocol/proto": _tvl_payload()})
    history = loader.read(with_run=True)
    assert history["tvl"].tolist() == pytest.approx([120.0])
    assert history.index[0] == pd.Timestamp(T0 + DAY, unit="s", tz="UTC")


def test_tvl_loader_unknown_chain_raises(offline_cache):
    loader = DefiLlamaTVLLoader("proto", chain="Nope")
    loader._http = _http_with({"/protocol/proto": _tvl_payload()})
    with pytest.raises(ValueError, match="Nope"):
        loader.read(with_run=True)


def test_tvl_loader_empty_payload_returns_empty_history(offline_cache):
    loader = DefiLlamaTVLLoader("proto")
    loader._http = _http_with({"/protocol/proto": {"chains": [], "chainTvls": {}}})
    history = loader.read(with_run=True)
    assert isinstance(history, TVLHistory) and len(history) == 0


# ------------------------------------------------------------------- DEX
def _dex_payloads():
    dex = {"totalDataChart": _chart([T0, T0 + DAY, T0 + 2 * DAY], [10.0, 20.0, 30.0])}
    fees = {"totalDataChart": _chart([T0 + DAY, T0 + 2 * DAY], [0.1, 0.2])}
    return {"/summary/dexs/proto": dex, "/summary/fees/proto": fees}


def test_dex_loader_joins_volume_and_fees_by_day(offline_cache):
    loader = DefiLlamaDEXLoader("proto")
    loader._http = _http_with(_dex_payloads())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        history = loader.read(with_run=True)
    assert isinstance(history, DEXHistory)
    # day 0 dropped (fees missing) → warning
    assert any("dropped 1 day" in str(w.message) for w in caught)
    assert history["volume"].tolist() == pytest.approx([20.0, 30.0])
    assert history["fees"].tolist() == pytest.approx([0.1, 0.2])
    assert history.index[0] == pd.Timestamp(T0 + DAY, unit="s", tz="UTC")


def test_dex_loader_params_flow_to_fees_endpoint(offline_cache):
    loader = DefiLlamaDEXLoader("proto", fees_data_type="dailyRevenue")
    loader._http = _http_with(_dex_payloads())
    loader.read(with_run=True)
    # both requests were issued; fees request carried the dataType param
    urls, params = loader._http.calls[-1]
    assert url_has_path(urls, "/summary/fees/proto")
    assert params == {"dataType": "dailyRevenue"}


def url_has_path(url: str, path: str) -> bool:
    return url.endswith(path)


# --------------------------------------------------------------- Pro key
def test_pro_loader_requires_api_key(monkeypatch):
    monkeypatch.delenv("DEFILLAMA_API_KEY", raising=False)
    with pytest.raises(ValueError, match="Pro API key"):
        DefiLlamaYieldsLoader("pool-1")
    with pytest.raises(ValueError, match="Pro API key"):
        DefiLlamaPoolLoader("pool-1")


def test_pro_loader_resolves_env_key_and_keeps_it_out_of_cache_key(monkeypatch, offline_cache):
    monkeypatch.setenv("DEFILLAMA_API_KEY", "sekrit-key")
    loader = DefiLlamaYieldsLoader("pool-1")
    assert loader._api_key == "sekrit-key"
    assert "sekrit-key" not in loader._cache_key()
    # URL base carries the key, but cache files do not
    assert loader._base_url.endswith("/sekrit-key")


def _yields_payload():
    return {
        "status": "success",
        "data": [
            # DefiLlama APY fields are annual percent: 3.5 means 3.5 %
            {"timestamp": "2024-01-01T00:00:00.000Z", "apy": 3.5, "apyBase": 3.0, "tvlUsd": 1000.0},
            {"timestamp": "2024-01-02T00:00:00.000Z", "apy": None, "apyBase": 4.0, "tvlUsd": 1100.0},
            {"timestamp": "2024-01-03T00:00:00.000Z", "apy": 5.0, "tvlUsd": 1200.0},
        ],
    }


def _hourly(apy_percent):
    """Per-hour geometric fraction equivalent to an annual percent APY."""
    return (1 + apy_percent / 100) ** (1 / (365 * 24)) - 1


def test_yields_loader_parses_apy_and_prefers_total(offline_cache, monkeypatch):
    monkeypatch.setenv("DEFILLAMA_API_KEY", "k")
    loader = DefiLlamaYieldsLoader("pool-1")
    loader._http = _http_with({"/yields/chart/pool-1": _yields_payload()})
    history = loader.read(with_run=True)
    assert isinstance(history, RateHistory)
    assert history.index.name == "time"
    # daily percent APY → hourly grid, per-step geometric fraction
    assert len(history) == 49
    assert history.index[0] == pd.Timestamp("2024-01-01 00:00", tz="UTC")
    assert history.index[-1] == pd.Timestamp("2024-01-03 00:00", tz="UTC")
    assert (history.index.to_series().diff().dropna() == pd.Timedelta(hours=1)).all()
    expected = [_hourly(3.5)] * 24 + [_hourly(4.0)] * 24 + [_hourly(5.0)]
    assert history["rate"].tolist() == pytest.approx(expected, rel=1e-12)


def test_pool_loader_parses_tvl(offline_cache, monkeypatch):
    monkeypatch.setenv("DEFILLAMA_API_KEY", "k")
    loader = DefiLlamaPoolLoader("pool-1")
    loader._http = _http_with({"/yields/chart/pool-1": _yields_payload()})
    history = loader.read(with_run=True)
    assert isinstance(history, TVLHistory)
    assert history["tvl"].tolist() == pytest.approx([1000.0, 1100.0, 1200.0])


def test_pro_loader_redacts_key_from_transport_errors(monkeypatch, offline_cache):
    monkeypatch.setenv("DEFILLAMA_API_KEY", "sekrit-key")
    loader = DefiLlamaPoolLoader("pool-1")
    loader._http = _http_with({"/yields/chart/pool-1":
                               RuntimeError("connection refused for /sekrit-key/yields/chart/pool-1")})
    with pytest.raises(Exception) as caught:
        loader.read(with_run=True)
    assert "sekrit-key" not in str(caught.value)
    assert "<redacted>" in str(caught.value)


def test_pro_loader_cache_round_trip(offline_cache, monkeypatch):
    monkeypatch.setenv("DEFILLAMA_API_KEY", "k")
    loader = DefiLlamaPoolLoader("pool-1")
    loader._http = _http_with({"/yields/chart/pool-1": _yields_payload()})
    loader.read(with_run=True)
    fresh = DefiLlamaPoolLoader("pool-1")
    again = fresh.read()
    assert again["tvl"].tolist() == pytest.approx([1000.0, 1100.0, 1200.0])


# ------------------------------------------------- review regression fixes
@pytest.mark.core
def test_tvl_loader_drops_days_where_a_chain_reports_null(offline_cache):
    """A per-chain ``null`` must drop the day, not be added as 0.0.

    Summing A=100 with a missing B published 100 as the protocol total with no
    NaN the ``read`` guard could see, silently understating TVL.
    """
    payload = {
        "chains": ["A", "B"],
        "chainTvls": {
            "A": {"tvl": [
                {"date": T0, "totalLiquidityUSD": 100.0},
                {"date": T0 + DAY, "totalLiquidityUSD": 300.0},
            ]},
            "B": {"tvl": [
                {"date": T0, "totalLiquidityUSD": 100.0},
                {"date": T0 + DAY, "totalLiquidityUSD": None},
            ]},
        },
    }
    loader = DefiLlamaTVLLoader("proto")
    loader._http = _http_with({"/protocol/proto": payload})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        history = loader.read(with_run=True)
    assert any("reported no TVL" in str(w.message) for w in caught)
    # only the fully-reported day survives; 300 must never be published as the total
    assert history["tvl"].tolist() == pytest.approx([200.0])
    assert history.index[0] == pd.Timestamp(T0, unit="s", tz="UTC")
    assert 300.0 not in history["tvl"].tolist()


@pytest.mark.core
def test_tvl_loader_normalizes_a_utc_day_reported_at_different_times(offline_cache):
    """The same UTC day from two chains is one row, not two partial rows."""
    payload = {
        "chains": ["A", "B"],
        "chainTvls": {
            "A": {"tvl": [{"date": T0, "totalLiquidityUSD": 100.0}]},
            "B": {"tvl": [{"date": T0 + 60, "totalLiquidityUSD": 200.0}]},
        },
    }
    loader = DefiLlamaTVLLoader("proto")
    loader._http = _http_with({"/protocol/proto": payload})
    history = loader.read(with_run=True)
    assert len(history) == 1
    assert history["tvl"].tolist() == pytest.approx([300.0])
    assert history.index[0] == pd.Timestamp(T0, unit="s", tz="UTC")


@pytest.mark.core
def test_dex_cache_key_includes_fees_data_type():
    """Otherwise a dailyRevenue run is served from the default run's cache."""
    default = DefiLlamaDEXLoader("proto")
    revenue = DefiLlamaDEXLoader("proto", fees_data_type="dailyRevenue")
    assert default._cache_key() != revenue._cache_key()


@pytest.mark.core
def test_dex_loader_returns_empty_history_when_a_chart_is_empty(offline_cache):
    """An empty (or null-only) fees chart is valid data, not a KeyError."""
    for fee_chart in ([], [[T0, None]]):
        loader = DefiLlamaDEXLoader("proto")
        loader._http = _http_with({
            "/summary/dexs/proto": {"totalDataChart": _chart([T0, T0 + DAY], [10.0, 20.0])},
            "/summary/fees/proto": {"totalDataChart": fee_chart},
        })
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            history = loader.read(with_run=True)
        assert isinstance(history, DEXHistory) and len(history) == 0


@pytest.mark.core
def test_dex_loader_joins_a_day_reported_at_different_times(offline_cache):
    """Volume at 00:00 and fees at 01:00 of the same UTC day must join."""
    loader = DefiLlamaDEXLoader("proto")
    loader._http = _http_with({
        "/summary/dexs/proto": {"totalDataChart": _chart([T0], [10.0])},
        "/summary/fees/proto": {"totalDataChart": _chart([T0 + 3600], [0.5])},
    })
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        history = loader.read(with_run=True)
    assert not any("dropped" in str(w.message) for w in caught)
    assert history["volume"].tolist() == pytest.approx([10.0])
    assert history["fees"].tolist() == pytest.approx([0.5])


@pytest.mark.core
def test_pro_loader_redaction_covers_cause_and_traceback(monkeypatch, offline_cache):
    """``from exc`` kept the key-bearing original reachable via ``__cause__``."""
    monkeypatch.setenv("DEFILLAMA_API_KEY", "sekrit-key")
    loader = DefiLlamaPoolLoader("pool-1")
    loader._http = _http_with({"/yields/chart/pool-1":
                               RuntimeError("connection refused for /sekrit-key/yields/chart/pool-1")})
    with pytest.raises(Exception) as caught:
        loader.read(with_run=True)
    formatted = ''.join(traceback.format_exception(caught.value))
    assert "sekrit-key" not in str(caught.value)
    assert "sekrit-key" not in formatted
    assert caught.value.__cause__ is None


@pytest.mark.core
def test_pro_series_is_sorted_and_deduplicated(offline_cache, monkeypatch):
    """Out-of-order and duplicated API points must not reach the typed frame."""
    monkeypatch.setenv("DEFILLAMA_API_KEY", "k")
    payload = {
        "status": "success",
        "data": [
            {"timestamp": "2024-01-02T00:00:00.000Z", "apy": 12.0},
            {"timestamp": "2024-01-01T00:00:00.000Z", "apy": 10.0},
            {"timestamp": "2024-01-01T00:00:00.000Z", "apy": 11.0},
        ],
    }
    loader = DefiLlamaYieldsLoader("pool-1")
    loader._http = _http_with({"/yields/chart/pool-1": payload})
    history = loader.read(with_run=True)
    assert history.index.is_monotonic_increasing
    assert history.index.is_unique
    assert history["rate"].tolist() == pytest.approx([_hourly(11.0)] * 24 + [_hourly(12.0)], rel=1e-12)


# ---------------------------------------------- per-chain intraday collapse
@pytest.mark.core
def test_tvl_loader_same_chain_intraday_points_do_not_double_count(offline_cache):
    """Two points from one chain on one UTC day are snapshots, not addends."""
    payload = {
        "chains": ["A"],
        "chainTvls": {"A": {"tvl": [
            {"date": T0, "totalLiquidityUSD": 110.0},
            {"date": T0 + 13 * 3600, "totalLiquidityUSD": 111.0},
        ]}},
    }
    loader = DefiLlamaTVLLoader("proto")
    loader._http = _http_with({"/protocol/proto": payload})
    history = loader.read(with_run=True)
    assert history["tvl"].tolist() == pytest.approx([111.0])
    assert history.index[0] == pd.Timestamp(T0, unit="s", tz="UTC")


@pytest.mark.core
def test_tvl_loader_sums_per_chain_last_points_across_chains(offline_cache):
    """Each chain contributes its last point of the day, then chains are summed."""
    payload = {
        "chains": ["A", "B"],
        "chainTvls": {
            # out of order on purpose: the latest epoch wins, not the last listed
            "A": {"tvl": [
                {"date": T0 + 20 * 3600, "totalLiquidityUSD": 105.0},
                {"date": T0, "totalLiquidityUSD": 100.0},
                {"date": T0 + DAY, "totalLiquidityUSD": 130.0},
            ]},
            "B": {"tvl": [
                {"date": T0 + 60, "totalLiquidityUSD": 7.0},
                {"date": T0 + DAY, "totalLiquidityUSD": 9.0},
            ]},
        },
    }
    loader = DefiLlamaTVLLoader("proto")
    loader._http = _http_with({"/protocol/proto": payload})
    history = loader.read(with_run=True)
    assert history["tvl"].tolist() == pytest.approx([112.0, 139.0])


@pytest.mark.core
def test_tvl_loader_day_incomplete_when_chain_last_point_is_null(offline_cache):
    """A chain whose last point of the day is null leaves that day incomplete."""
    payload = {
        "chains": ["A", "B"],
        "chainTvls": {
            "A": {"tvl": [{"date": T0, "totalLiquidityUSD": 100.0}]},
            "B": {"tvl": [
                {"date": T0, "totalLiquidityUSD": 5.0},
                {"date": T0 + 3600, "totalLiquidityUSD": None},
            ]},
        },
    }
    loader = DefiLlamaTVLLoader("proto")
    loader._http = _http_with({"/protocol/proto": payload})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        history = loader.read(with_run=True)
    assert any("reported no TVL" in str(w.message) for w in caught)
    assert len(history) == 0


@pytest.mark.core
def test_tvl_loader_earlier_null_is_superseded_by_later_point(offline_cache):
    """Only the chain's last point of the day decides completeness."""
    payload = {
        "chains": ["A"],
        "chainTvls": {"A": {"tvl": [
            {"date": T0, "totalLiquidityUSD": None},
            {"date": T0 + 3600, "totalLiquidityUSD": 50.0},
        ]}},
    }
    loader = DefiLlamaTVLLoader("proto")
    loader._http = _http_with({"/protocol/proto": payload})
    history = loader.read(with_run=True)
    assert history["tvl"].tolist() == pytest.approx([50.0])


# ------------------------------------------------ yields unit / hourly grid
@pytest.mark.core
def test_yields_hourly_rate_compounds_back_to_the_annual_apy(offline_cache, monkeypatch):
    """A year of per-step ``amount *= 1 + rate`` must reproduce the quoted APY."""
    monkeypatch.setenv("DEFILLAMA_API_KEY", "fake-test-key")
    payload = {"data": [
        {"timestamp": "2024-01-01T00:00:00.000Z", "apy": 3.5},
        {"timestamp": "2024-01-02T00:00:00.000Z", "apy": 3.5},
    ]}
    loader = DefiLlamaYieldsLoader("pool-1")
    loader._http = _http_with({"/yields/chart/pool-1": payload})
    rate = loader.read(with_run=True)["rate"].iloc[0]
    assert (1 + rate) ** (365 * 24) == pytest.approx(1.035, rel=1e-12)
    assert rate < 0.035 / (365 * 24)  # geometric, not the linear APR split


@pytest.mark.core
def test_yields_window_applies_inclusively_after_hourly_grid(offline_cache, monkeypatch):
    monkeypatch.setenv("DEFILLAMA_API_KEY", "fake-test-key")
    loader = DefiLlamaYieldsLoader("pool-1",
                                   start_time=datetime(2024, 1, 1, 12, tzinfo=UTC),
                                   end_time=datetime(2024, 1, 2, 6, tzinfo=UTC))
    loader._http = _http_with({"/yields/chart/pool-1": _yields_payload()})
    history = loader.read(with_run=True)
    assert history.index[0] == pd.Timestamp("2024-01-01 12:00", tz="UTC")
    assert history.index[-1] == pd.Timestamp("2024-01-02 06:00", tz="UTC")
    assert len(history) == 19
    assert history["rate"].tolist() == pytest.approx([_hourly(3.5)] * 12 + [_hourly(4.0)] * 7, rel=1e-12)


@pytest.mark.core
def test_yields_cache_key_is_versioned_for_the_rate_unit(monkeypatch):
    """Caches written with percent-unit daily rates must not be reused."""
    monkeypatch.setenv("DEFILLAMA_API_KEY", "fake-test-key")
    key = DefiLlamaYieldsLoader("pool-1")._cache_key()
    assert key.startswith("yields-v2-pool-1-")
    assert "fake-test-key" not in key


@pytest.mark.core
def test_yields_cache_round_trip_keeps_hourly_rates(offline_cache, monkeypatch):
    monkeypatch.setenv("DEFILLAMA_API_KEY", "fake-test-key")
    loader = DefiLlamaYieldsLoader("pool-1")
    loader._http = _http_with({"/yields/chart/pool-1": _yields_payload()})
    first = loader.read(with_run=True)
    again = DefiLlamaYieldsLoader("pool-1").read()
    assert again.index.equals(first.index)
    assert again["rate"].tolist() == pytest.approx(first["rate"].tolist(), rel=1e-12)


# ---------------------------------------- Pro key leak via transport/logging
FAKE_KEY = "fake-pro-key-0000"


def _closed_port_loader(monkeypatch):
    """Pool loader aimed at a closed local port with a fast, small retry budget."""
    monkeypatch.setattr("fractal.loaders.defillama.defillama.PRO_BASE_URL", "http://127.0.0.1:9")
    loader = DefiLlamaPoolLoader("pool-1", api_key=FAKE_KEY)
    loader._http = HttpClient(HttpConfig(timeout=0.5, max_retries=2, backoff_factor=0.0))
    return loader


@pytest.mark.core
def test_pro_transport_error_has_no_key_bearing_context(monkeypatch, offline_cache):
    """``raise ... from None`` inside ``except`` still sets ``__context__``."""
    loader = _closed_port_loader(monkeypatch)
    started = time.monotonic()
    with pytest.raises(LoaderHttpError) as caught:
        loader.read(with_run=True)
    assert time.monotonic() - started < 2.0
    err = caught.value
    assert "<redacted>" in str(err)
    assert FAKE_KEY not in str(err)
    assert err.__cause__ is None
    assert err.__context__ is None
    assert FAKE_KEY not in "".join(traceback.format_exception(err))


@pytest.mark.core
def test_pro_transport_retry_warnings_do_not_log_the_key(monkeypatch, offline_cache, caplog):
    """urllib3 logs each retry with the request path, which embeds the key."""
    loader = _closed_port_loader(monkeypatch)
    filters_before = list(logging.getLogger("urllib3.connectionpool").filters)
    with (caplog.at_level(logging.WARNING, logger="urllib3.connectionpool"),
          pytest.raises(LoaderHttpError)):
        loader.read(with_run=True)
    records = [r for r in caplog.records if r.name == "urllib3.connectionpool"]
    assert records, "expected urllib3 retry warnings to be captured"
    for record in records:
        assert FAKE_KEY not in record.getMessage()
    assert FAKE_KEY not in caplog.text
    # the redaction filter is removed once the request finishes (other
    # libraries, e.g. MLflow, may install their own filters on this logger)
    assert logging.getLogger("urllib3.connectionpool").filters == filters_before
