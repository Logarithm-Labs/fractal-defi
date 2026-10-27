"""Offline tests for ``FredSeriesLoader`` and ``HttpClient.get_text`` — no network.

``HttpClient.get_text`` is replaced with canned, synthetic CSV bodies so the
full ``read(with_run=True)`` lifecycle (extract → transform → cache → typed
struct) runs offline. The rate values are made up.
"""
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest
import requests

from fractal.loaders import FredSeriesLoader
from fractal.loaders._http import HttpClient, LoaderHttpError
from fractal.loaders.fred import FREDGRAPH_URL, parse_fredgraph_csv
from fractal.loaders.structs import RateHistory

pytestmark = pytest.mark.core

UTC = timezone.utc

# Synthetic week: Wed 2024-01-03 .. Wed 2024-01-10, weekend absent,
# Mon 2024-01-08 published as missing (empty field).
CSV_BODY = (
    "observation_date,DGS3MO\n"
    "2024-01-03,5.00\n"
    "2024-01-04,5.10\n"
    "2024-01-05,5.20\n"
    "2024-01-08,\n"
    "2024-01-09,5.40\n"
    "2024-01-10,5.50\n"
)


def _ts(text):
    return pd.Timestamp(text, tz="UTC")


class FakeHttp:
    def __init__(self, body):
        self.body = body
        self.calls = []

    def get_text(self, url, params=None, timeout=None):
        self.calls.append((url, params))
        if isinstance(self.body, Exception):
            raise self.body
        return self.body


@pytest.fixture
def offline_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_PATH", str(tmp_path))
    return tmp_path


def _loader(body=CSV_BODY, **kwargs):
    loader = FredSeriesLoader(**kwargs)
    loader._http = FakeHttp(body)
    return loader


# ------------------------------------------------------------------ parser
def test_parse_current_header_skips_empty_values_and_keeps_units():
    frame = parse_fredgraph_csv(CSV_BODY, "DGS3MO")
    days = pd.to_datetime(frame["date"], unit="s", utc=True)
    assert [d.strftime("%Y-%m-%d") for d in days] == [
        "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-09", "2024-01-10"]
    assert frame["value"].tolist() == [5.00, 5.10, 5.20, 5.40, 5.50]  # still percent
    assert int(frame["date"].iloc[0]) == int(_ts("2024-01-03").timestamp())  # UTC midnight


def test_parse_legacy_header_dot_missing_bom_and_unsorted_rows():
    body = "﻿DATE,DFF\r\n2024-01-02,5.33\r\n2024-01-01,.\r\n2023-12-31,5.31\r\n"
    frame = parse_fredgraph_csv(body, "DFF")
    assert frame["value"].tolist() == [5.31, 5.33]
    assert frame["date"].is_monotonic_increasing


def test_parse_header_only_is_empty():
    frame = parse_fredgraph_csv("observation_date,DGS3MO\n", "DGS3MO")
    assert frame.empty and list(frame.columns) == ["date", "value"]


@pytest.mark.parametrize("body", [
    "",
    "<!DOCTYPE html>\n<html><title>Error</title></html>\n",
    "observation_date,DGS10\n2024-01-03,4.00\n",  # another series
    "observation_date,DGS3MO\n2024-01-03,abc\n",
    "observation_date,DGS3MO\n03/01/2024,5.00\n",
    "observation_date,DGS3MO\n2024-01-03,5.00,7\n",
    "observation_date,DGS3MO\n2024-01-03,nan\n",
    "observation_date,DGS3MO\n2024-01-03,5.00\n2024-01-03,5.10\n",
])
def test_parse_rejects_bodies_that_are_not_a_clean_fred_series(body):
    with pytest.raises(ValueError):
        parse_fredgraph_csv(body, "DGS3MO")


# ------------------------------------------------------------------ loader
def test_read_converts_percent_and_indexes_by_next_day_availability(offline_cache):
    loader = _loader()
    history = loader.read(with_run=True)
    assert isinstance(history, RateHistory)
    assert history.index.tz is not None and str(history.index.tz) == "UTC"
    assert list(history.index) == [_ts("2024-01-04"), _ts("2024-01-05"), _ts("2024-01-06"),
                                   _ts("2024-01-10"), _ts("2024-01-11")]
    np.testing.assert_allclose(history["rate"].to_numpy(), [0.050, 0.051, 0.052, 0.054, 0.055])
    url, params = loader._http.calls[0]
    assert url == FREDGRAPH_URL and params == {"id": "DGS3MO"}


def test_asof_lookup_never_sees_the_same_day_value(offline_cache):
    history = _loader().read(with_run=True)
    query = pd.DatetimeIndex([_ts("2024-01-05 12:00"), _ts("2024-01-07 00:00"),
                              _ts("2024-01-09 23:59"), _ts("2024-01-10 00:00")])
    asof = history["rate"].reindex(query, method="ffill").to_numpy()
    # Fri noon: Thursday's value; Sunday: Friday's; Tue: still Friday's (Mon missing);
    # Wed 00:00: Tuesday's value becomes known.
    np.testing.assert_allclose(asof, [0.051, 0.052, 0.052, 0.054])


def test_lag_zero_and_raw_units(offline_cache):
    history = _loader(availability_lag_days=0, percent_to_decimal=False).read(with_run=True)
    assert history.index[0] == _ts("2024-01-03")
    assert history["rate"].iloc[0] == pytest.approx(5.0)


def test_custom_lag_shifts_by_calendar_days(offline_cache):
    history = _loader(availability_lag_days=3).read(with_run=True)
    assert history.index[2] == _ts("2024-01-08")  # Friday value, 3 calendar days later


def test_window_keeps_value_known_at_start_and_bounds_end_by_availability(offline_cache):
    loader = _loader(start_time=datetime(2024, 1, 8, 12), end_time=datetime(2024, 1, 10, 6, tzinfo=UTC))
    history = loader.read(with_run=True)
    # 2024-01-06 (Friday's value) is the last one known before Monday noon and is kept;
    # earlier rows are dropped; Wednesday's value (known on the 11th) is past end_time.
    assert list(history.index) == [_ts("2024-01-06"), _ts("2024-01-10")]
    np.testing.assert_allclose(history["rate"].to_numpy(), [0.052, 0.054])
    _, params = loader._http.calls[0]
    # cosd = start - 1 day lag - 14 days padding; coed = end - 1 day lag.
    assert params == {"id": "DGS3MO", "cosd": "2023-12-24", "coed": "2024-01-09"}


def test_window_start_exactly_on_availability_does_not_add_an_extra_row(offline_cache):
    history = _loader(start_time=datetime(2024, 1, 5)).read(with_run=True)
    # The row known exactly at start_time is the anchor; the 2024-01-04 row is not kept.
    assert list(history.index) == [_ts("2024-01-05"), _ts("2024-01-06"), _ts("2024-01-10"), _ts("2024-01-11")]


def test_window_before_any_data_has_no_anchor(offline_cache):
    history = _loader(start_time=datetime(2023, 1, 1), end_time=datetime(2024, 1, 4)).read(with_run=True)
    assert list(history.index) == [_ts("2024-01-04")]


def test_cache_round_trip_and_key_tracks_units_and_lag(offline_cache):
    loader = _loader(start_time=datetime(2024, 1, 1), end_time=datetime(2024, 2, 1))
    fresh = loader.read(with_run=True)
    cached = FredSeriesLoader(start_time=datetime(2024, 1, 1), end_time=datetime(2024, 2, 1)).read()
    pd.testing.assert_frame_equal(pd.DataFrame(fresh), pd.DataFrame(cached), check_freq=False)
    keys = {
        FredSeriesLoader()._cache_key(),
        FredSeriesLoader(availability_lag_days=0)._cache_key(),
        FredSeriesLoader(percent_to_decimal=False)._cache_key(),
        FredSeriesLoader(series_id="DFF")._cache_key(),
    }
    assert len(keys) == 4


def test_header_only_response_reads_as_empty_history(offline_cache):
    history = _loader(body="observation_date,DGS3MO\n").read(with_run=True)
    assert isinstance(history, RateHistory) and history.empty
    assert FredSeriesLoader().read().empty  # empty cache file round trips


def test_http_error_propagates(offline_cache):
    loader = _loader(body=LoaderHttpError("HTTP 404 for fredgraph"))
    with pytest.raises(LoaderHttpError):
        loader.read(with_run=True)


@pytest.mark.parametrize("kwargs, exc", [
    ({"series_id": ""}, ValueError),
    ({"series_id": "../DGS3MO"}, ValueError),
    ({"series_id": "DGS 3MO"}, ValueError),
    ({"availability_lag_days": -1}, ValueError),
    ({"availability_lag_days": 1.5}, TypeError),
    ({"availability_lag_days": True}, TypeError),
    ({"start_time": datetime(2024, 2, 1), "end_time": datetime(2024, 1, 1)}, ValueError),
])
def test_constructor_validation(kwargs, exc):
    with pytest.raises(exc):
        FredSeriesLoader(**kwargs)


# ------------------------------------------------------- HttpClient.get_text
def _response(status, body, url="https://example.invalid/data.csv"):
    resp = requests.Response()
    resp.status_code = status
    resp._content = body.encode()
    resp.encoding = "utf-8"
    resp.url = url
    resp.reason = "Not Found" if status == 404 else "OK"
    return resp


def test_get_text_returns_body_unparsed_and_forwards_params(monkeypatch):
    client = HttpClient()
    seen = {}

    def fake_get(url, params=None, timeout=None):
        seen.update(url=url, params=params, timeout=timeout)
        return _response(200, '{"looks": "like json"}')

    monkeypatch.setattr(client.session, "get", fake_get)
    text = client.get_text("https://example.invalid/data.csv", params={"id": "X"}, timeout=3.0)
    assert text == '{"looks": "like json"}' and isinstance(text, str)
    assert seen == {"url": "https://example.invalid/data.csv", "params": {"id": "X"}, "timeout": 3.0}


def test_get_text_raises_loader_error_with_body_snippet(monkeypatch):
    client = HttpClient()
    monkeypatch.setattr(client.session, "get",
                        lambda url, params=None, timeout=None: _response(404, "<html>missing series</html>"))
    with pytest.raises(LoaderHttpError, match=r"HTTP 404 .*missing series"):
        client.get_text("https://example.invalid/data.csv")


def test_get_still_rejects_non_json(monkeypatch):
    client = HttpClient()
    monkeypatch.setattr(client.session, "get",
                        lambda url, params=None, timeout=None: _response(200, "a,b\n1,2\n"))
    with pytest.raises(LoaderHttpError, match="Non-JSON"):
        client.get("https://example.invalid/data.csv")
