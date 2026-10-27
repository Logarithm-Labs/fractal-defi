"""FRED (Federal Reserve Economic Data) series loader.

:class:`FredSeriesLoader` reads one daily FRED series through the public,
keyless ``fredgraph.csv`` download (``https://fred.stlouisfed.org``). The
default series is ``DGS3MO``, the 3-month Treasury constant-maturity yield,
a common risk-free benchmark for the "outside option" of a DeFi strategy.

Two conventions matter for backtests:

* **Units.** Rate series such as ``DGS3MO`` are published in **percent per
  annum** (``4.35`` means 4.35 %). With ``percent_to_decimal=True`` (the
  default) the loader divides by 100 so ``0.0435`` comes out. The value is
  only rescaled: its quoting convention is kept (``DGS3MO`` is an annual
  yield on an investment basis, not an APY and not a per-bar rate), so
  convert it explicitly before comparing it with a compounded APY. Pass
  ``percent_to_decimal=False`` for series that are not in percent.
* **Availability.** A daily observation dated ``d`` is not known at the
  start of ``d``. Each row is indexed by the time it becomes usable:
  00:00 UTC of ``d + availability_lag_days`` (default 1, so the value
  for ``d`` is first known on ``d + 1``). The lag is in calendar days, so
  a Friday value is usable from Saturday 00:00 UTC and holds over the
  weekend. ``availability_lag_days=0`` indexes rows by their observation
  date, which looks ahead by up to one day.

Missing observations (FRED writes an empty field, or ``.`` in older files,
for holidays) are dropped rather than filled. To get the value known at an
arbitrary time ``t``, take the last row at or before ``t``, e.g.
``history["rate"].reindex(times, method="ffill")``. So that this lookup is
defined at ``start_time`` too, the window also keeps the last row that
became available at or before ``start_time``.
"""
import re
from datetime import datetime

import numpy as np
import pandas as pd

from fractal.loaders._dt import to_seconds, to_utc
from fractal.loaders._http import HttpClient
from fractal.loaders.base_loader import Loader, LoaderType
from fractal.loaders.structs import RateHistory

FREDGRAPH_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
DEFAULT_SERIES_ID = "DGS3MO"
SECONDS_PER_DAY = 86_400
# Calendar days fetched before ``start_time`` so the value already known at
# ``start_time`` is found across weekends and multi-day holiday gaps.
_ASOF_PADDING_DAYS = 14
_SERIES_ID_RE = re.compile(r"^[A-Za-z0-9_.]+$")
_DATE_HEADERS = ("observation_date", "date")
_MISSING = ("", ".")


def parse_fredgraph_csv(text: str, series_id: str) -> pd.DataFrame:
    """Parse a ``fredgraph.csv`` body into ``date`` (epoch seconds, UTC midnight) and ``value``.

    The header must be a date column (``observation_date``, or ``DATE`` in
    older files) followed by ``series_id``; anything else, such as an HTML
    error page served with status 200, raises ``ValueError``. Missing values
    (empty or ``.``) are skipped. Values are returned exactly as published,
    in the series' own units. Duplicate dates raise ``ValueError``.
    """
    lines = [line.strip() for line in str(text).lstrip("﻿").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"FRED {series_id}: empty response")
    header = [cell.strip().lower() for cell in lines[0].split(",")]
    if len(header) < 2 or header[0] not in _DATE_HEADERS or series_id.lower() not in header[1:]:
        raise ValueError(f"FRED {series_id}: unexpected CSV header {lines[0][:120]!r}")
    col = header.index(series_id.lower())
    dates: list[int] = []
    values: list[float] = []
    for line in lines[1:]:
        cells = [cell.strip() for cell in line.split(",")]
        if len(cells) != len(header):
            raise ValueError(f"FRED {series_id}: malformed CSV row {line[:120]!r}")
        if cells[col] in _MISSING:
            continue
        try:
            day = datetime.strptime(cells[0], "%Y-%m-%d")
            value = float(cells[col])
        except ValueError as exc:
            raise ValueError(f"FRED {series_id}: malformed CSV row {line[:120]!r}") from exc
        if not np.isfinite(value):
            raise ValueError(f"FRED {series_id}: non-finite value in row {line[:120]!r}")
        dates.append(to_seconds(day))
        values.append(value)
    frame = pd.DataFrame({"date": np.asarray(dates, dtype=np.int64),
                          "value": np.asarray(values, dtype=float)})
    if frame["date"].duplicated().any():
        raise ValueError(f"FRED {series_id}: duplicate observation dates")
    return frame.sort_values("date", kind="stable").reset_index(drop=True)


class FredSeriesLoader(Loader):
    """One daily FRED series as a :class:`~fractal.loaders.structs.RateHistory`.

    Keyless: it reads the public ``fredgraph.csv`` download, so no API key or
    environment variable is needed. The ``rate`` column is the published value
    divided by 100 when ``percent_to_decimal`` is true, indexed by the time
    each value becomes known (see the module docstring for both conventions).

    Args:
        series_id: FRED series identifier, e.g. ``"DGS3MO"`` or ``"DFF"``.
        start_time: Optional window start (naive datetimes are read as UTC).
            The last row available at or before it is kept as well.
        end_time: Optional window end, inclusive, applied to availability time.
        availability_lag_days: Whole calendar days between the observation
            date and the time the value is treated as known; ``0`` disables it.
        percent_to_decimal: Divide published values by 100.
        loader_type: Cache format.
        base_url: Override of the ``fredgraph.csv`` URL (tests, mirrors).
    """

    def __init__(
        self,
        series_id: str = DEFAULT_SERIES_ID,
        start_time: pd.Timestamp | None = None,
        end_time: pd.Timestamp | None = None,
        availability_lag_days: int = 1,
        percent_to_decimal: bool = True,
        loader_type: LoaderType = LoaderType.CSV,
        base_url: str = FREDGRAPH_URL,
    ) -> None:
        super().__init__(loader_type=loader_type)
        if not isinstance(series_id, str) or not _SERIES_ID_RE.match(series_id):
            raise ValueError(f"invalid FRED series id {series_id!r}")
        if isinstance(availability_lag_days, bool) or not isinstance(availability_lag_days, (int, np.integer)):
            raise TypeError("availability_lag_days must be an int")
        if availability_lag_days < 0:
            raise ValueError("availability_lag_days must be >= 0")
        self.series_id: str = series_id
        self.start_time: datetime | None = to_utc(start_time)
        self.end_time: datetime | None = to_utc(end_time)
        if self.start_time is not None and self.end_time is not None and self.start_time > self.end_time:
            raise ValueError("start_time must not be after end_time")
        self.availability_lag_days: int = int(availability_lag_days)
        self.percent_to_decimal: bool = bool(percent_to_decimal)
        self._url: str = base_url
        self._http = HttpClient()

    def _cache_key(self) -> str:
        s = to_seconds(self.start_time) if self.start_time is not None else "open"
        e = to_seconds(self.end_time) if self.end_time is not None else "now"
        units = "decimal" if self.percent_to_decimal else "raw"
        return f"{self.series_id}-lag{self.availability_lag_days}-{units}-{s}-{e}"

    def _request_params(self) -> dict[str, str]:
        """``fredgraph.csv`` query: series id plus an observation-date range."""
        params = {"id": self.series_id}
        lag = pd.Timedelta(days=self.availability_lag_days)
        if self.start_time is not None:
            first = pd.Timestamp(self.start_time) - lag - pd.Timedelta(days=_ASOF_PADDING_DAYS)
            params["cosd"] = first.strftime("%Y-%m-%d")
        if self.end_time is not None:
            params["coed"] = (pd.Timestamp(self.end_time) - lag).strftime("%Y-%m-%d")
        return params

    def extract(self) -> None:
        text = self._http.get_text(self._url, params=self._request_params())
        self._data = parse_fredgraph_csv(text, self.series_id)

    def transform(self) -> None:
        cols = ["time", "rate"]
        if self._data is None or self._data.empty:
            self._data = pd.DataFrame(columns=cols)
            return
        times = self._data["date"].to_numpy(dtype=np.int64) + self.availability_lag_days * SECONDS_PER_DAY
        rates = self._data["value"].to_numpy(dtype=float)
        if self.percent_to_decimal:
            rates = rates / 100.0
        keep = np.ones(len(times), dtype=bool)
        if self.end_time is not None:
            keep &= times <= to_seconds(self.end_time)
        if self.start_time is not None:
            start = to_seconds(self.start_time)
            known = np.flatnonzero(keep & (times <= start))
            keep &= times >= start
            if known.size:
                keep[known[-1]] = True  # value already known at start_time
        self._data = pd.DataFrame({"time": times[keep], "rate": rates[keep]}).reset_index(drop=True)

    def read(self, with_run: bool = False) -> RateHistory:
        if with_run:
            self.run()
        else:
            self._read(self._cache_key())
        if self._data is None or self._data.empty:
            return RateHistory(rates=[], time=[])
        return RateHistory(
            rates=self._data["rate"].astype(float).values,
            time=self._utc_index("time"),
        )
