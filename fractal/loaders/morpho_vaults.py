"""MetaMorpho (V1) vault size history from the ``vaults`` query.

The loader lists the vaults of one chain, optionally restricted to vault or
underlying-asset addresses, and returns one row per vault and interval point
over an inclusive UTC ``[start_time, end_time]`` window from
``historicalState { totalAssets totalAssetsUsd }``. Morpho Vaults V2 are a
different API type and are not covered here.

Points arrive newest first; the newest one can be the live value rather than
an interval boundary and is dropped, as in
:class:`~fractal.loaders.morpho.MorphoMarketLoader`. ``total_assets`` is the
raw BigInt (a JSON number when small, a decimal string when large) parsed as
an exact integer and scaled by the underlying asset's decimals; it is ``NaN``
when the API omits the decimals or the point. ``total_assets_usd`` is the
API's USD notional for the same point.

The vault list is paged with ``skip`` in ``orderBy: Address`` order, which is
unique per chain. A known ``countTotal`` (not a short page) ends paging, and a
listing that yields fewer distinct vaults than ``countTotal`` emits a
``UserWarning``. Each vault carries its whole time series, so long ``HOUR``
windows can exceed the API's query-complexity limit; lower ``page_size`` or
narrow the window in that case.
"""
import hashlib
import re
import warnings
from collections.abc import Iterable, Mapping
from datetime import datetime
from typing import Any, ClassVar

import pandas as pd

from fractal.loaders._dt import to_seconds, to_utc, utcnow
from fractal.loaders._http import HttpClient
from fractal.loaders.base_loader import Loader, LoaderType
from fractal.loaders.morpho import MorphoLoaderException, graphql_query
from fractal.loaders.morpho_reallocations import _epoch_ceil, _raw_int

__all__ = ["MorphoVaultLoader"]

_PAGE_SIZE_MAX = 1000
_SKIP_CAP = 10_000
_CACHE_SCHEMA = "v1"
_INTERVAL_SECONDS = {"HOUR": 3600, "DAY": 86_400}
_ADDRESS = re.compile(r"^0x[0-9a-f]{40}$")

_VAULTS_QUERY = """
query VaultHistory($first: Int!, $skip: Int!, $where: VaultFilters, $o: TimeseriesOptions) {
  vaults(first: $first, skip: $skip, orderBy: Address, orderDirection: Asc, where: $where) {
    items {
      address symbol chain { id }
      asset { address symbol decimals }
      historicalState {
        totalAssets(options: $o) { x y }
        totalAssetsUsd(options: $o) { x y }
      }
    }
    pageInfo { count countTotal }
  }
}
"""


def _addresses(values: Iterable[str] | None, name: str) -> tuple[str, ...] | None:
    if values is None:
        return None
    if isinstance(values, str):
        raise TypeError(f"{name} must be an iterable of addresses, not a single string")
    out = sorted({str(value).lower() for value in values})
    bad = [value for value in out if not _ADDRESS.match(value)]
    if bad:
        raise ValueError(f"{name} holds invalid addresses: {bad}")
    if not out:
        raise ValueError(f"{name} must not be empty; pass None for no filter")
    return tuple(out)


def _series(points: Any) -> dict[int, Any]:
    if not points:
        return {}
    return {int(point["x"]): point.get("y") for point in points if point.get("x") is not None}


class MorphoVaultLoader(Loader):
    """Load MetaMorpho (V1) vault ``totalAssets`` history for one chain.

    Args:
        chain_id: EVM chain id.
        start_time: inclusive beginning of the UTC window.
        end_time: inclusive end of the UTC window; current UTC time when omitted.
        vault_addresses: optional vault addresses to keep (``address_in``).
        asset_addresses: optional underlying-asset addresses to keep
            (``assetAddress_in``).
        interval: API interval, ``"HOUR"`` or ``"DAY"``.
        page_size: vaults per GraphQL page, at most 1000.
        loader_type: on-disk cache format.
        http: injectable client for offline tests.
    """

    COLUMNS: ClassVar[tuple[str, ...]] = (
        "timestamp",
        "chain_id",
        "vault_address",
        "vault_symbol",
        "asset_address",
        "asset_symbol",
        "total_assets",
        "total_assets_usd",
    )

    def __init__(
        self,
        chain_id: int,
        start_time: datetime,
        end_time: datetime | None = None,
        *,
        vault_addresses: Iterable[str] | None = None,
        asset_addresses: Iterable[str] | None = None,
        interval: str = "DAY",
        page_size: int = 50,
        loader_type: LoaderType = LoaderType.CSV,
        http: HttpClient | None = None,
    ) -> None:
        super().__init__(loader_type=loader_type)
        if interval not in _INTERVAL_SECONDS:
            raise ValueError(f"interval must be one of {sorted(_INTERVAL_SECONDS)}, got {interval!r}")
        if not 0 < page_size <= _PAGE_SIZE_MAX:
            raise ValueError(f"page_size must be in [1, {_PAGE_SIZE_MAX}], got {page_size}")
        self.chain_id = int(chain_id)
        self.start_time = to_utc(start_time)
        self.end_time = to_utc(end_time) if end_time is not None else utcnow()
        if self.end_time < self.start_time:
            raise ValueError(f"end_time {self.end_time} precedes start_time {self.start_time}")
        self.vault_addresses = _addresses(vault_addresses, "vault_addresses")
        self.asset_addresses = _addresses(asset_addresses, "asset_addresses")
        self.interval = interval
        self.page_size = int(page_size)
        self._http = http or HttpClient()

    def _bounds(self) -> tuple[int, int]:
        return _epoch_ceil(self.start_time), int(to_seconds(self.end_time))

    def _cache_key(self) -> str:
        filters = f"vaults={','.join(self.vault_addresses or ())};assets={','.join(self.asset_addresses or ())}"
        filter_hash = hashlib.sha256(filters.encode()).hexdigest()[:12]
        lower, upper = self._bounds()
        return f"schema-{_CACHE_SCHEMA}-chain-{self.chain_id}-{self.interval}-{lower}-{upper}-filters-{filter_hash}"

    def _where(self) -> dict[str, Any]:
        where: dict[str, Any] = {"chainId_in": [self.chain_id]}
        if self.vault_addresses is not None:
            where["address_in"] = list(self.vault_addresses)
        if self.asset_addresses is not None:
            where["assetAddress_in"] = list(self.asset_addresses)
        return where

    def _page(self, skip: int) -> tuple[list[dict[str, Any]], int | None]:
        lower, upper = self._bounds()
        data = graphql_query(self._http, _VAULTS_QUERY, {
            "first": self.page_size,
            "skip": skip,
            "where": self._where(),
            "o": {"startTimestamp": lower, "endTimestamp": upper, "interval": self.interval},
        })
        block = data.get("vaults")
        if not isinstance(block, dict):
            raise MorphoLoaderException("Morpho GraphQL: 'vaults' missing from the response")
        items = block.get("items") or []
        if not isinstance(items, list):
            raise MorphoLoaderException("Morpho GraphQL: 'vaults.items' must be a list")
        total = (block.get("pageInfo") or {}).get("countTotal")
        return items, None if total is None else int(total)

    def extract(self) -> None:
        page, total = self._page(0)
        vaults: list[dict[str, Any]] = []
        seen: set[str] = set()
        skip = 0
        while page:
            new_vaults = [item for item in page if str(item.get("address") or "").lower() not in seen]
            if not new_vaults:
                break
            vaults.extend(new_vaults)
            seen.update(str(item.get("address") or "").lower() for item in page)
            skip += self.page_size
            # Pages can come back short of ``first`` while more vaults remain,
            # so a known ``countTotal`` decides when to stop.
            done = len(page) < self.page_size if total is None else skip >= total
            if done:
                break
            if skip > _SKIP_CAP:
                raise MorphoLoaderException(f"vault listing requires skip {skip} above the API cap; add filters")
            page, _ = self._page(skip)
        if total is not None and len(vaults) < total:
            warnings.warn(
                f"MorphoVaultLoader: chain {self.chain_id} listing returned {len(vaults)} distinct vaults "
                f"of countTotal {total}",
                stacklevel=2,
            )
        self._data = vaults

    def _rows(self, item: Mapping[str, Any]) -> list[dict[str, Any]]:
        asset = item.get("asset") or {}
        decimals = asset.get("decimals")
        history = item.get("historicalState") or {}
        raw = _series(history.get("totalAssets"))
        usd = _series(history.get("totalAssetsUsd"))
        chain_id = int((item.get("chain") or {}).get("id", self.chain_id))
        rows = []
        for timestamp in sorted(set(raw) | set(usd)):
            raw_value = raw.get(timestamp)
            if raw_value is None or decimals is None:
                total_assets = float("nan")
            else:
                total_assets = _raw_int(raw_value) / 10 ** int(decimals)
            usd_value = usd.get(timestamp)
            rows.append({
                "timestamp": timestamp,
                "chain_id": chain_id,
                "vault_address": str(item.get("address") or "").lower(),
                "vault_symbol": str(item.get("symbol") or ""),
                "asset_address": str(asset.get("address") or "").lower(),
                "asset_symbol": str(asset.get("symbol") or ""),
                "total_assets": total_assets,
                "total_assets_usd": float("nan") if usd_value is None else float(usd_value),
            })
        return rows

    def _normalise_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return pd.DataFrame({
                "timestamp": pd.Series(dtype="int64"),
                "chain_id": pd.Series(dtype="int64"),
                "vault_address": pd.Series(dtype="object"),
                "vault_symbol": pd.Series(dtype="object"),
                "asset_address": pd.Series(dtype="object"),
                "asset_symbol": pd.Series(dtype="object"),
                "total_assets": pd.Series(dtype="float64"),
                "total_assets_usd": pd.Series(dtype="float64"),
            })[list(self.COLUMNS)]
        frame = frame.copy()
        frame["timestamp"] = frame["timestamp"].astype("int64")
        frame["chain_id"] = frame["chain_id"].astype("int64")
        for column in ("total_assets", "total_assets_usd"):
            frame[column] = frame[column].astype("float64")
        for column in ("vault_address", "vault_symbol", "asset_address", "asset_symbol"):
            frame[column] = frame[column].fillna("").astype(str)
        return frame[list(self.COLUMNS)].reset_index(drop=True)

    def transform(self) -> None:
        items = self._data.to_dict("records") if isinstance(self._data, pd.DataFrame) else (self._data or [])
        frame = pd.DataFrame([row for item in items for row in self._rows(item)], columns=list(self.COLUMNS))
        if not frame.empty:
            lower, upper = self._bounds()
            step = _INTERVAL_SECONDS[self.interval]
            keep = (frame["timestamp"] % step == 0) & frame["timestamp"].between(lower, upper)
            frame = frame[keep]
            frame = frame.drop_duplicates(["chain_id", "vault_address", "timestamp"], keep="first")
            frame = frame.sort_values(["vault_address", "timestamp"], kind="stable")
        self._data = self._normalise_frame(frame)

    def read(self, with_run: bool = False) -> pd.DataFrame:
        """Return the tidy vault-history frame, running extraction when requested."""
        if with_run:
            self.run()
        else:
            self._read(self._cache_key())
        frame = self._data.drop(columns=["Unnamed: 0"], errors="ignore")
        self._data = self._normalise_frame(frame)
        return self._data.copy()
