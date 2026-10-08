"""DefiLlama loaders (issue #57).

Free endpoints (no auth, ``api.llama.fi``):

* :class:`DefiLlamaTVLLoader` — historical TVL for a protocol slug,
  aggregated across its chains (or a single chain) into a daily series.
* :class:`DefiLlamaDEXLoader` — daily DEX volume plus daily fees for a
  protocol, joined by UTC day.

Pro v2 endpoints (``pro-api.llama.fi/<KEY>/...``, optional):

* :class:`DefiLlamaYieldsLoader` — daily APY history for a yield pool.
* :class:`DefiLlamaPoolLoader` — daily TVL history for a yield pool.

The yield-pool endpoints are **not** available on the unauthenticated free
API (verified 2026-10-08), so they require a DefiLlama Pro key passed as a
constructor argument or the ``DEFILLAMA_API_KEY`` environment variable.
The key never appears in cache paths, reprs, logs, error messages, or
formatted tracebacks — a Pro request failure re-raises a redacted error
detached from the key-bearing original.
"""
import os
import warnings
from typing import Any

import numpy as np
import pandas as pd

from fractal.loaders._dt import to_seconds, to_utc
from fractal.loaders._http import HttpClient, LoaderHttpError
from fractal.loaders.base_loader import Loader, LoaderType
from fractal.loaders.structs import DEXHistory, RateHistory, TVLHistory

DEFAULT_BASE_URL = "https://api.llama.fi"
PRO_BASE_URL = "https://pro-api.llama.fi"
DEFAULT_FEES_DATA_TYPE = "dailyFees"
SECONDS_PER_DAY = 86_400


def _utc_day(epoch: int) -> int:
    """Floor an epoch-seconds timestamp to the start of its UTC day."""
    return int(epoch) - int(epoch) % SECONDS_PER_DAY


def _parse_chart(points: list[Any], keep_missing: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Normalize a DefiLlama history chart to ``(epoch_seconds, values)``.

    Two response shapes exist in the wild:

    * pairs: ``[[unix_seconds, value], ...]`` (``/summary/*`` charts);
    * dict points: ``[{"date": unix_seconds, "totalLiquidityUSD": value}, ...]``
      (``/protocol/<slug>`` ``chainTvls``).

    Points with missing values are skipped; timestamps must be integer
    epoch seconds (DefiLlama contract), anything else raises ``ValueError``.
    With ``keep_missing=True`` an explicit ``null`` is kept as ``NaN`` so a
    caller can tell "this source did not report that day" from "this source
    never covered that day".
    """
    epochs: list[int] = []
    values: list[float] = []
    for point in points:
        if isinstance(point, dict):
            if "date" not in point:
                raise ValueError(f"DefiLlama chart point has an unexpected shape: {point!r}")
            epoch = point["date"]
            value = point.get("totalLiquidityUSD", point.get("value"))
        elif isinstance(point, (list, tuple)) and len(point) == 2:
            epoch, value = point[0], point[1]
        else:
            raise ValueError(f"DefiLlama chart point has an unexpected shape: {point!r}")
        if epoch is None or (value is None and not keep_missing):
            continue
        epochs.append(int(epoch))
        values.append(np.nan if value is None else float(value))
    return np.asarray(epochs, dtype=np.int64), np.asarray(values, dtype=float)


class DefiLlamaBaseLoader(Loader):
    """Shared plumbing: HTTP client, time window, cache key."""

    def __init__(
        self,
        loader_type: LoaderType = LoaderType.CSV,
        base_url: str = DEFAULT_BASE_URL,
        start_time: pd.Timestamp | None = None,
        end_time: pd.Timestamp | None = None,
    ) -> None:
        super().__init__(loader_type=loader_type)
        self._base_url: str = base_url.rstrip("/")
        self.start_time: pd.Timestamp | None = to_utc(start_time)
        self.end_time: pd.Timestamp | None = to_utc(end_time)
        self._http = HttpClient()

    def _get_json(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """GET ``{base}{path}``; loader errors carry no credentials."""
        url = f"{self._base_url}{path}"
        try:
            return self._http.get(url, params=params)
        except LoaderHttpError:
            raise
        except Exception as exc:  # transport errors may embed the URL/key
            raise LoaderHttpError(f"DefiLlama request failed: {exc}") from exc

    def _cache_key(self) -> str:
        s = to_seconds(self.start_time) if self.start_time is not None else "open"
        e = to_seconds(self.end_time) if self.end_time is not None else "now"
        return f"{self._cache_subject()}-{s}-{e}"

    def _cache_subject(self) -> str:
        raise NotImplementedError

    def _window_mask(self, times: pd.DatetimeIndex) -> np.ndarray:
        """Boolean mask of ``times`` inside the optional requested window."""
        mask = np.ones(len(times), dtype=bool)
        if self.start_time is not None:
            mask &= times >= self.start_time
        if self.end_time is not None:
            mask &= times <= self.end_time
        return mask


class DefiLlamaTVLLoader(DefiLlamaBaseLoader):
    """Daily historical TVL for a protocol slug, in USD.

    ``chain=None`` sums the TVL of every chain the protocol reports
    (the response's ``chains`` list); pass an exact chain name (e.g.
    ``"Ethereum"``) to restrict the series to one chain.
    """

    def __init__(
        self,
        protocol: str,
        chain: str | None = None,
        loader_type: LoaderType = LoaderType.CSV,
        base_url: str = DEFAULT_BASE_URL,
        start_time: pd.Timestamp | None = None,
        end_time: pd.Timestamp | None = None,
    ) -> None:
        super().__init__(loader_type=loader_type, base_url=base_url,
                         start_time=start_time, end_time=end_time)
        self.protocol: str = protocol
        self.chain: str | None = chain

    def _cache_subject(self) -> str:
        return f"tvl-{self.protocol}-{self.chain or 'all'}"

    def extract(self) -> None:
        payload = self._get_json(f"/protocol/{self.protocol}")
        if not isinstance(payload, dict):
            raise ValueError(f"DefiLlama /protocol/{self.protocol}: expected an object")
        chain_tvls = payload.get("chainTvls")
        if chain_tvls is None or not isinstance(chain_tvls, dict):
            raise ValueError(f"DefiLlama /protocol/{self.protocol}: no chainTvls data")
        if not chain_tvls:
            # Empty periods are valid: return a well-shaped empty frame.
            self._data = pd.DataFrame(columns=["time", "tvl"])
            return
        if self.chain is not None:
            keys = [self.chain]
            if self.chain not in chain_tvls:
                raise ValueError(
                    f"DefiLlama /protocol/{self.protocol}: chain {self.chain!r} not in "
                    f"chainTvls {sorted(chain_tvls)!r}"
                )
        else:
            keys = [name for name in payload.get("chains", []) if name in chain_tvls]
            if not keys:
                raise ValueError(
                    f"DefiLlama /protocol/{self.protocol}: 'chains' is empty; "
                    "pass an explicit ``chain`` argument"
                )
        frames: dict[int, float] = {}
        complete: dict[int, bool] = {}
        for name in keys:
            epochs, values = _parse_chart(chain_tvls[name].get("tvl") or [], keep_missing=True)
            # TVL points are snapshots, not flows: several points from one chain
            # on one UTC day must collapse to that day's latest snapshot before
            # the cross-chain sum, otherwise intraday updates double count.
            order = np.argsort(epochs, kind="stable")
            per_day: dict[int, float] = {}
            for epoch, value in zip(epochs[order].tolist(), values[order].tolist(), strict=False):
                per_day[_utc_day(epoch)] = value
            for day, value in per_day.items():
                missing = bool(np.isnan(value))
                complete[day] = complete.get(day, True) and not missing
                if not missing:
                    frames[day] = frames.get(day, 0.0) + value
        # Only whole days are published: summing a day where a selected chain
        # reported nothing would silently understate the protocol total, which
        # ``read`` cannot detect because the sum itself is a valid float.
        days = [day for day in sorted(complete) if complete[day]]
        if len(days) != len(complete):
            warnings.warn(
                f"DefiLlama /protocol/{self.protocol}: dropped "
                f"{len(complete) - len(days)} day(s) where at least one selected chain "
                "reported no TVL, rather than understating the total.",
                UserWarning, stacklevel=3,
            )
        self._data = pd.DataFrame({"time": days, "tvl": [frames[day] for day in days]})

    def transform(self) -> None:
        if self._data is None or self._data.empty:
            self._data = pd.DataFrame(columns=["time", "tvl"])
            return
        df = self._data.astype({"time": np.int64, "tvl": float})
        times = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df[self._window_mask(pd.DatetimeIndex(times).tz_convert("UTC"))]
        self._data = df.reset_index(drop=True)

    def read(self, with_run: bool = False) -> TVLHistory:
        if with_run:
            self.run()
        else:
            self._read(self._cache_key())
        if self._data is None or self._data.empty:
            return TVLHistory(tvls=[], time=[])
        if self._data["tvl"].isna().any():
            raise ValueError(
                "DefiLlama TVL history has missing values; refusing to "
                "substitute zeros which would silently understate TVL."
            )
        return TVLHistory(
            tvls=self._data["tvl"].astype(float).values,
            time=self._utc_index("time"),
        )


class DefiLlamaDEXLoader(DefiLlamaBaseLoader):
    """Daily DEX volume + fees for a protocol slug, joined by UTC day.

    Volume comes from ``/summary/dexs/{protocol}`` and fees from
    ``/summary/fees/{protocol}`` (``dataType=dailyFees``). Only days
    present in **both** charts are returned; a warning is emitted when
    the join drops days so silent history shortening is visible.
    """

    def __init__(
        self,
        protocol: str,
        loader_type: LoaderType = LoaderType.CSV,
        base_url: str = DEFAULT_BASE_URL,
        fees_data_type: str = DEFAULT_FEES_DATA_TYPE,
        start_time: pd.Timestamp | None = None,
        end_time: pd.Timestamp | None = None,
    ) -> None:
        super().__init__(loader_type=loader_type, base_url=base_url,
                         start_time=start_time, end_time=end_time)
        self.protocol: str = protocol
        self.fees_data_type: str = fees_data_type

    def _cache_subject(self) -> str:
        # The fees column depends on ``fees_data_type``: without it in the key a
        # dailyRevenue run and a default run share one cache file and the second
        # read silently returns the first one's metric.
        return f"dex-{self.protocol}-{self.fees_data_type}"

    def extract(self) -> None:
        dex = self._get_json(f"/summary/dexs/{self.protocol}")
        fees = self._get_json(f"/summary/fees/{self.protocol}",
                              params={"dataType": self.fees_data_type})
        for name, payload in (("dexs", dex), ("fees", fees)):
            if not isinstance(payload, dict) or not isinstance(payload.get("totalDataChart"), list):
                raise ValueError(f"DefiLlama /summary/{name}/{self.protocol}: no totalDataChart")
        vol_epochs, vol_values = _parse_chart(dex["totalDataChart"])
        fee_epochs, fee_values = _parse_chart(fees["totalDataChart"])
        # Key on the UTC day, not the raw timestamp: two sources reporting the
        # same day at different times must join into one row.
        times = np.concatenate([vol_epochs, fee_epochs])
        self._data = pd.DataFrame({
            "time": times - times % SECONDS_PER_DAY,
            "value": np.concatenate([vol_values, fee_values]),
            "kind": ["volume"] * len(vol_epochs) + ["fees"] * len(fee_epochs),
        })

    def transform(self) -> None:
        cols = ["time", "volume", "fees"]
        if self._data is None or self._data.empty:
            self._data = pd.DataFrame(columns=cols)
            return
        wide = self._data.pivot_table(index="time", columns="kind",
                                      values="value", aggfunc="first")
        # A legitimately empty chart (new protocol, null-only fees) drops its
        # pivot column entirely; reindex so the join reports "nothing to join"
        # instead of raising KeyError.
        wide = wide.reindex(columns=["volume", "fees"])
        wide = wide.reset_index().astype({"time": np.int64})
        has_both = wide["volume"].notna() & wide["fees"].notna()
        if int((~has_both).sum()) > 0:
            warnings.warn(
                f"DefiLlama DEX join dropped {int((~has_both).sum())} day(s) where "
                f"volume or fees are missing for {self.protocol!r}.",
                UserWarning, stacklevel=3,
            )
        wide = wide[has_both]
        times = pd.to_datetime(wide["time"], unit="s", utc=True)
        mask = self._window_mask(pd.DatetimeIndex(times).tz_convert("UTC"))
        df = wide[mask]
        self._data = df[["time", "volume", "fees"]].reset_index(drop=True)

    def read(self, with_run: bool = False) -> DEXHistory:
        if with_run:
            self.run()
        else:
            self._read(self._cache_key())
        if self._data is None or self._data.empty:
            return DEXHistory(volumes=[], fees=[], time=[])
        if self._data[["volume", "fees"]].isna().any().any():
            raise ValueError(
                "DefiLlama DEX history has missing volume/fees after the day join; "
                "refusing to substitute zeros."
            )
        return DEXHistory(
            volumes=self._data["volume"].astype(float).values,
            fees=self._data["fees"].astype(float).values,
            time=self._utc_index("time"),
        )


class DefiLlamaProLoader(DefiLlamaBaseLoader):
    """Base for the Pro v2 yield endpoints.

    The API key is resolved from the ``api_key`` argument or the
    ``DEFILLAMA_API_KEY`` environment variable. It is embedded in the
    request path only, never in cache keys, and is scrubbed from any
    error message.
    """

    def __init__(
        self,
        api_key: str | None = None,
        loader_type: LoaderType = LoaderType.CSV,
        start_time: pd.Timestamp | None = None,
        end_time: pd.Timestamp | None = None,
    ) -> None:
        resolved = api_key or os.getenv("DEFILLAMA_API_KEY")
        if not resolved:
            raise ValueError(
                "DefiLlama yield endpoints require a Pro API key: pass api_key= "
                "or set the DEFILLAMA_API_KEY environment variable."
            )
        super().__init__(loader_type=loader_type, base_url=f"{PRO_BASE_URL}/{resolved}",
                         start_time=start_time, end_time=end_time)
        self._api_key: str = resolved

    def _get_json(self, path: str, params: dict[str, Any] | None = None) -> Any:
        try:
            return super()._get_json(path, params=params)
        except Exception as exc:
            # ``from None`` detaches the original exception, whose URL and
            # response body carry the key; otherwise the credential is still
            # reachable through ``__cause__`` and any formatted traceback.
            raise type(exc)(str(exc).replace(self._api_key, "<redacted>")) from None


class DefiLlamaYieldsLoader(DefiLlamaProLoader):
    """Hourly per-step yield rate for one DefiLlama yield pool (Pro API).

    Reads ``/yields/chart/{pool_id}``, whose daily ``apy`` (or ``apyBase``
    when ``apy`` is null) is an **annual percentage** (``3.5`` = 3.5 %).
    The daily series is forward-filled onto a 1h UTC grid and converted
    to the library's :class:`RateHistory` convention: ``rate`` is the
    per-hour fraction applied as ``amount *= 1 + rate`` each step, i.e.
    ``(1 + apy / 100) ** (1 / (365 * 24)) - 1`` (geometric, because APY
    already includes compounding). The requested window is applied to
    the hourly grid with inclusive bounds.
    """

    def __init__(
        self,
        pool_id: str,
        api_key: str | None = None,
        loader_type: LoaderType = LoaderType.CSV,
        start_time: pd.Timestamp | None = None,
        end_time: pd.Timestamp | None = None,
    ) -> None:
        super().__init__(api_key=api_key, loader_type=loader_type,
                         start_time=start_time, end_time=end_time)
        self.pool_id: str = pool_id

    def _cache_subject(self) -> str:
        # v2: rates are hourly per-step fractions; v1 caches held daily percent.
        return f"yields-v2-{self.pool_id}"

    def extract(self) -> None:
        payload = self._get_json(f"/yields/chart/{self.pool_id}")
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list):
            raise ValueError(f"DefiLlama /yields/chart/{self.pool_id}: no data array")
        rows = []
        for point in data:
            if not isinstance(point, dict) or point.get("timestamp") is None:
                continue
            rate = point.get("apy")
            if rate is None:
                rate = point.get("apyBase")
            if rate is None:
                continue
            epoch = int(pd.to_datetime(point["timestamp"], utc=True).timestamp())
            rows.append({"time": epoch, "rate": float(rate)})
        self._data = pd.DataFrame(rows, columns=["time", "rate"])

    def transform(self) -> None:
        if self._data is None or self._data.empty:
            self._data = pd.DataFrame(columns=["time", "rate"])
            return
        df = self._data.astype({"time": np.int64, "rate": float})
        # The Pro API does not guarantee ordering or uniqueness; a duplicated or
        # non-monotonic index breaks downstream time alignment.
        df = (df.sort_values("time", kind="stable")
              .drop_duplicates("time", keep="last"))
        apy = pd.Series(df["rate"].to_numpy(),
                        index=pd.DatetimeIndex(pd.to_datetime(df["time"], unit="s", utc=True)))
        # Daily annual-percent APY → 1h UTC grid (forward fill, like the Lido
        # loader) → per-hour geometric fraction.
        hourly = apy.resample("1h").last().ffill()
        rate = (1.0 + hourly / 100.0) ** (1.0 / (365 * 24)) - 1.0
        grid = pd.DatetimeIndex(rate.index).tz_convert("UTC")
        mask = self._window_mask(grid)
        # Resolution-agnostic epoch seconds (pandas 3 may not use ns units).
        epochs = (grid[mask] - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(seconds=1)
        self._data = pd.DataFrame({
            "time": np.asarray(epochs, dtype=np.int64),
            "rate": rate.to_numpy()[mask],
        })

    def read(self, with_run: bool = False) -> RateHistory:
        if with_run:
            self.run()
        else:
            self._read(self._cache_key())
        if self._data is None or self._data.empty:
            return RateHistory(rates=[], time=[])
        if self._data["rate"].isna().any():
            raise ValueError(
                "DefiLlama yield history has missing APY values; refusing to "
                "substitute zeros."
            )
        return RateHistory(
            rates=self._data["rate"].astype(float).values,
            time=self._utc_index("time"),
        )


class DefiLlamaPoolLoader(DefiLlamaProLoader):
    """Daily TVL history for one DefiLlama yield pool (Pro API).

    Reads ``/yields/chart/{pool_id}`` (``tvlUsd`` per day).
    """

    def __init__(
        self,
        pool_id: str,
        api_key: str | None = None,
        loader_type: LoaderType = LoaderType.CSV,
        start_time: pd.Timestamp | None = None,
        end_time: pd.Timestamp | None = None,
    ) -> None:
        super().__init__(api_key=api_key, loader_type=loader_type,
                         start_time=start_time, end_time=end_time)
        self.pool_id: str = pool_id

    def _cache_subject(self) -> str:
        return f"pool-{self.pool_id}"

    def extract(self) -> None:
        payload = self._get_json(f"/yields/chart/{self.pool_id}")
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list):
            raise ValueError(f"DefiLlama /yields/chart/{self.pool_id}: no data array")
        rows = []
        for point in data:
            if not isinstance(point, dict) or point.get("timestamp") is None:
                continue
            tvl = point.get("tvlUsd")
            if tvl is None:
                continue
            epoch = int(pd.to_datetime(point["timestamp"], utc=True).timestamp())
            rows.append({"time": epoch, "tvl": float(tvl)})
        self._data = pd.DataFrame(rows, columns=["time", "tvl"])

    def transform(self) -> None:
        if self._data is None or self._data.empty:
            self._data = pd.DataFrame(columns=["time", "tvl"])
            return
        df = self._data.astype({"time": np.int64, "tvl": float})
        times = pd.to_datetime(df["time"], unit="s", utc=True)
        mask = self._window_mask(pd.DatetimeIndex(times).tz_convert("UTC"))
        # Same normalization as the yield loader (see there).
        self._data = (df[mask]
                      .sort_values("time", kind="stable")
                      .drop_duplicates("time", keep="last")
                      .reset_index(drop=True))

    def read(self, with_run: bool = False) -> TVLHistory:
        if with_run:
            self.run()
        else:
            self._read(self._cache_key())
        if self._data is None or self._data.empty:
            return TVLHistory(tvls=[], time=[])
        if self._data["tvl"].isna().any():
            raise ValueError(
                "DefiLlama pool TVL history has missing values; refusing to "
                "substitute zeros."
            )
        return TVLHistory(
            tvls=self._data["tvl"].astype(float).values,
            time=self._utc_index("time"),
        )
