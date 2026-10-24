"""LayerZero Scan loader for keyless OFT message metadata.

The loader queries the public LayerZero Scan v1 API. Timestamps are UTC
Unix seconds and the requested window is half-open: ``[start_time, end_time)``.
"""

from datetime import datetime
from typing import Any

import pandas as pd

from fractal.loaders._dt import to_seconds, to_utc
from fractal.loaders._http import HttpClient, LoaderHttpError
from fractal.loaders.base_loader import Loader, LoaderType

LAYERZERO_SCAN_API = "https://scan.layerzero-api.com/v1"
_CACHE_SCHEMA = "v1"


def _iso(timestamp: datetime) -> str:
    return timestamp.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _epoch(value: Any) -> int | None:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)) and not pd.isna(value):
        return int(value)
    text = str(value)
    try:
        return int(text)
    except ValueError:
        timestamp = pd.Timestamp(text)
        timestamp = timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")
        return int(timestamp.timestamp())


def _is_404(error: LoaderHttpError) -> bool:
    return "HTTP 404" in str(error)


class LayerZeroScanLoader(Loader):
    """OFT messages between two LayerZero endpoint IDs over a UTC window.

    Args:
        source_eid: LayerZero source endpoint ID.
        destination_eid: LayerZero destination endpoint ID.
        oft_address: Source-chain OFT application address used by LayerZero Scan.
        start_time: Inclusive start of the UTC window.
        end_time: Exclusive end of the UTC window.
        max_messages: Optional deterministic even time subsample size.
        loader_type: Cache format.
        http: Injectable HTTP client for offline tests.
    """

    _columns = (
        "guid",
        "src_eid",
        "dst_eid",
        "src_tx",
        "src_ts",
        "dst_ts",
        "status",
        "latency_s",
    )

    def __init__(
        self,
        source_eid: int,
        destination_eid: int,
        oft_address: str,
        start_time: datetime,
        end_time: datetime,
        *,
        max_messages: int | None = None,
        loader_type: LoaderType = LoaderType.CSV,
        http: HttpClient | None = None,
    ) -> None:
        super().__init__(loader_type=loader_type)
        if int(source_eid) < 0 or int(destination_eid) < 0:
            raise ValueError("source_eid and destination_eid must be non-negative")
        address = str(oft_address).strip().lower()
        if not address:
            raise ValueError("oft_address must not be empty")
        if max_messages is not None and max_messages <= 0:
            raise ValueError(f"max_messages must be > 0 or None, got {max_messages}")
        self.source_eid = int(source_eid)
        self.destination_eid = int(destination_eid)
        self.oft_address = address
        self.start_time = to_utc(start_time)
        self.end_time = to_utc(end_time)
        if self.start_time is None or self.end_time is None:
            raise ValueError("start_time and end_time are required")
        if self.end_time <= self.start_time:
            raise ValueError(f"end_time {self.end_time} must be after start_time {self.start_time}")
        self.max_messages = max_messages
        self._http = http or HttpClient()

    def _cache_key(self) -> str:
        limit = "all" if self.max_messages is None else str(self.max_messages)
        return (
            f"{_CACHE_SCHEMA}-{self.source_eid}-{self.destination_eid}-{self.oft_address}-"
            f"{to_seconds(self.start_time)}-{to_seconds(self.end_time)}-max-{limit}"
        )

    def extract(self) -> None:
        """Fetch every continuation page, treating a continuation 404 as the end."""
        url = f"{LAYERZERO_SCAN_API}/messages/oapp/{self.source_eid}/{self.oft_address}"
        token: str | None = None
        seen_tokens: set[str] = set()
        messages: list[dict[str, Any]] = []
        while True:
            params = {
                "limit": 100,
                "start": _iso(self.start_time),
                "end": _iso(self.end_time),
            }
            if token is not None:
                params["nextToken"] = token
            try:
                payload = self._http.get(url, params=params)
            except LoaderHttpError as exc:
                if token is not None and _is_404(exc):
                    break
                raise
            if not isinstance(payload, dict):
                raise ValueError("LayerZero Scan response must be an object")
            page = payload.get("data") or []
            if not isinstance(page, list):
                raise ValueError("LayerZero Scan response field 'data' must be a list")
            messages.extend(item for item in page if isinstance(item, dict))
            next_token = payload.get("nextToken")
            if not next_token:
                break
            token = str(next_token)
            if token in seen_tokens:
                raise ValueError(f"LayerZero Scan repeated continuation token {token!r}")
            seen_tokens.add(token)
        self._data = messages

    def _parse_message(self, message: dict[str, Any]) -> dict[str, Any] | None:
        pathway = message["pathway"]
        sender = pathway.get("sender")
        sender_address = sender.get("address") if isinstance(sender, dict) else sender
        source = message["source"]["tx"]
        source_timestamp = _epoch(source.get("blockTimestamp"))
        if source_timestamp is None:
            return None
        if (
            int(pathway["srcEid"]) != self.source_eid
            or int(pathway["dstEid"]) != self.destination_eid
            or str(sender_address).lower() != self.oft_address
            or not to_seconds(self.start_time) <= source_timestamp < to_seconds(self.end_time)
        ):
            return None
        destination = (message.get("destination") or {}).get("tx") or {}
        destination_timestamp = _epoch(destination.get("blockTimestamp"))
        status = message.get("status")
        status_name = status.get("name") if isinstance(status, dict) else status
        return {
            "guid": str(message["guid"]).lower(),
            "src_eid": self.source_eid,
            "dst_eid": self.destination_eid,
            "src_tx": str(source["txHash"]).lower(),
            "src_ts": source_timestamp,
            "dst_ts": destination_timestamp,
            "status": status_name,
            "latency_s": (
                float(destination_timestamp - source_timestamp) if destination_timestamp is not None else float("nan")
            ),
        }

    @classmethod
    def _frame(cls, rows: Any) -> pd.DataFrame:
        frame = pd.DataFrame(rows, columns=cls._columns)
        if frame.empty:
            return frame
        frame["src_eid"] = pd.to_numeric(frame["src_eid"], errors="raise").astype("int64")
        frame["dst_eid"] = pd.to_numeric(frame["dst_eid"], errors="raise").astype("int64")
        frame["src_ts"] = pd.to_numeric(frame["src_ts"], errors="raise").astype("int64")
        frame["dst_ts"] = pd.array(pd.to_numeric(frame["dst_ts"], errors="coerce"), dtype="Int64")
        frame["latency_s"] = pd.to_numeric(frame["latency_s"], errors="coerce").astype(float)
        return frame

    def transform(self) -> None:
        """Filter, parse, deduplicate and optionally subsample fetched messages."""
        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for message in self._data or []:
            try:
                row = self._parse_message(message)
            except (KeyError, TypeError, ValueError):
                continue
            if row is None or row["guid"] in seen:
                continue
            seen.add(row["guid"])
            rows.append(row)
        rows.sort(key=lambda row: (row["src_ts"], row["guid"]))
        if self.max_messages is not None and len(rows) > self.max_messages:
            step = len(rows) / self.max_messages
            rows = [rows[int(index * step)] for index in range(self.max_messages)]
        self._data = self._frame(rows)

    def read(self, with_run: bool = False) -> pd.DataFrame:
        """Return one row per matching OFT message."""
        if with_run:
            self.run()
        else:
            self._read(self._cache_key())
        return self._frame(self._data).copy()
