"""Offline tests for the Boros loaders against a fake client: market
config parsing (rate floor from ticks), 200-candle paging, settlement
join at exact timestamps, underlying fill, maturity term."""
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from fractal.loaders import BorosMarketHistory, BorosMarketLoader
from fractal.loaders._dt import annualise_funding
from fractal.loaders.boros import BOROS_API, BorosLoaderException, bar_seconds, get_market_info

MARKET_ID = 130  # BINANCE-BTCUSDT-25SEP2026
MATURITY = datetime(2026, 9, 25, tzinfo=timezone.utc)
START = datetime(2026, 9, 1, tzinfo=timezone.utc)
EIGHT_HOURS = 8 * 3600


class _FakeBorosHttp:
    def __init__(self, ohlcv_cap: int = 200):
        self.ohlcv_cap = ohlcv_cap
        self.calls = []

    def get(self, url, params=None, timeout=None):
        params = dict(params or {})
        self.calls.append((url, params))
        if url == f"{BOROS_API}/markets":
            if params["isMatured"] == "false":
                return {"results": [{
                    "marketId": MARKET_ID,
                    "imData": {"symbol": "BINANCE-BTCUSDT-25SEP2026", "maturity": int(MATURITY.timestamp()),
                               "tickStep": 1, "iTickThresh": 1166},
                    "config": {"kIM": "645161290322580645", "kMM": "322580645161290322", "tThresh": 864000,
                               "takerFee": "500000000000000"},
                    "extConfig": {"paymentPeriod": EIGHT_HOURS, "settleFeeRate": "1000000000000000"},
                    "metadata": {"fundingRateSymbol": "BTCUSDT", "maxLeverage": 1.55},
                }]}
            return {"results": []}
        if url == f"{BOROS_API}/markets/ohlcv":
            start, end = params["startTimestamp"], params["endTimestamp"]
            rows, ts = [], start
            while ts <= end and len(rows) < self.ohlcv_cap:
                rows.append({"ts": ts, "o": 0.05, "h": 0.06, "l": 0.04, "c": 0.055, "v": 3.0})
                ts += 3600
            return {"results": rows}
        if url == f"{BOROS_API}/markets/historical-underlying-apr":
            start, end = params["startTimestamp"], params["endTimestamp"]
            return {"results": [{"periodStartTimestamp": ts, "underlyingApr": 0.11}
                                for ts in range(start, end + 1, 3600)]}
        raise AssertionError(url)

    def post(self, url, json=None, timeout=None, headers=None):
        self.calls.append((url, json))
        assert url == f"{BOROS_API}/funding-rate/settlement-summary"
        start, end = json["fromTimestamp"], json["toTimestamp"]
        first = start - start % EIGHT_HOURS + EIGHT_HOURS
        return {"results": [{"marketId": MARKET_ID, "periodTimestamp": ts, "settlementApr": 0.0001 * 1095,
                             "totalNotionalSize": 40.0} for ts in range(first, end + 1, EIGHT_HOURS)]}


@pytest.mark.core
def test_market_info_parses_margin_params_and_rate_floor():
    info = get_market_info(MARKET_ID, http=_FakeBorosHttp())
    assert info.maturity == MATURITY and info.payment_period_seconds == EIGHT_HOURS
    assert info.k_im == pytest.approx(1 / 1.55) and info.mm_to_im_ratio == pytest.approx(0.5)
    assert info.taker_fee_rate == pytest.approx(0.0005) and info.settle_fee_rate == pytest.approx(0.001)
    assert info.rate_floor == pytest.approx(1.00005 ** 1166 - 1, rel=1e-9)  # ≈ 6 %
    assert 0.059 < info.rate_floor < 0.061
    assert info.time_threshold_seconds == 864000 and info.max_leverage == 1.55
    with pytest.raises(BorosLoaderException):
        get_market_info(999, http=_FakeBorosHttp())


@pytest.mark.core
def test_candles_page_in_200_bar_windows_and_join_settlements(monkeypatch):
    monkeypatch.setattr("fractal.loaders.boros._time.sleep", lambda s: None)
    http = _FakeBorosHttp()
    end = START + timedelta(hours=450)
    loader = BorosMarketLoader(MARKET_ID, START, end, maturity=MATURITY, underlying=("BTC", "Binance"), http=http)
    history = loader.read(with_run=True)
    assert isinstance(history, BorosMarketHistory)
    assert len(history) == 451
    ohlcv_calls = [p for u, p in http.calls if u.endswith("/ohlcv")]
    assert len(ohlcv_calls) == 3
    assert ohlcv_calls[1]["startTimestamp"] - ohlcv_calls[0]["startTimestamp"] == 200 * bar_seconds("1h")
    settled = history["settlement_apr"].dropna()
    assert len(settled) == len([t for t in history.index if t.timestamp() % EIGHT_HOURS == 0 and t > START])
    assert settled.iloc[0] == pytest.approx(annualise_funding(0.0001, EIGHT_HOURS))
    assert history["oi"].dropna().iloc[0] == 40.0
    # underlying fills the non-settlement bars, settlement wins where both exist
    assert history["underlying_apr"].notna().all()
    assert history.loc[settled.index, "underlying_apr"].iloc[0] == pytest.approx(settled.iloc[0])
    assert history["seconds_to_expiry"].iloc[0] == (MATURITY - START).total_seconds()
    assert history["mark_apr_close"].iloc[0] == 0.055 and history["volume"].iloc[0] == 3.0


@pytest.mark.core
def test_without_settlements_columns_stay_nan_and_cache_keys_differ(monkeypatch):
    monkeypatch.setattr("fractal.loaders.boros._time.sleep", lambda s: None)
    end = START + timedelta(hours=3)
    plain = BorosMarketLoader(MARKET_ID, START, end, maturity=MATURITY, include_settlements=False,
                              http=_FakeBorosHttp())
    history = plain.read(with_run=True)
    assert history["settlement_apr"].isna().all() and history["underlying_apr"].isna().all()
    full = BorosMarketLoader(MARKET_ID, START, end, maturity=MATURITY, http=_FakeBorosHttp())
    later = BorosMarketLoader(MARKET_ID, START, end, maturity=MATURITY + timedelta(days=30), http=_FakeBorosHttp())
    assert len({plain._cache_key(), full._cache_key(), later._cache_key()}) == 3


@pytest.mark.core
def test_missing_close_raises_and_validation(monkeypatch):
    monkeypatch.setattr("fractal.loaders.boros._time.sleep", lambda s: None)
    loader = BorosMarketLoader(MARKET_ID, START, START + timedelta(hours=2), maturity=MATURITY,
                               include_settlements=False, http=_FakeBorosHttp())
    loader.extract()
    loader._candles[1]["c"] = None
    with pytest.raises(ValueError, match="mark_apr_close"):
        loader.transform()
    with pytest.raises(ValueError):
        BorosMarketLoader(MARKET_ID, START, START, maturity=MATURITY, time_frame="2h")
    with pytest.raises(ValueError):
        BorosMarketLoader(MARKET_ID, START + timedelta(hours=1), START, maturity=MATURITY)


@pytest.mark.core
def test_cache_round_trip(monkeypatch, tmp_path):
    monkeypatch.setattr("fractal.loaders.boros._time.sleep", lambda s: None)
    monkeypatch.setenv("DATA_PATH", str(tmp_path))
    end = START + timedelta(hours=10)
    first = BorosMarketLoader(MARKET_ID, START, end, maturity=MATURITY, http=_FakeBorosHttp()).read(with_run=True)
    second = BorosMarketLoader(MARKET_ID, START, end, maturity=MATURITY, http=_FakeBorosHttp()).read(with_run=False)
    pd.testing.assert_frame_equal(first, second)
