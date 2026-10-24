"""Circle Iris V2 published CCTP transfer fees.

``CCTP_DOMAINS`` is the documented chain-name to CCTP domain-ID map for
the chains covered by the bridge-cost reference collector.
"""

from typing import Any

import pandas as pd

from fractal.loaders._http import HttpClient, LoaderHttpError
from fractal.loaders.base_loader import Loader, LoaderType

IRIS_API = "https://iris-api.circle.com"
CCTP_DOMAINS = {
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
CCTP_DOMAIN_CHAINS = {domain: chain for chain, domain in CCTP_DOMAINS.items()}
_CACHE_SCHEMA = "v1"


def _is_404(error: LoaderHttpError) -> bool:
    return "HTTP 404" in str(error)


def _mode(finality_threshold: int) -> str | None:
    if finality_threshold <= 1000:
        return "fast"
    if finality_threshold >= 2000:
        return "standard"
    return None


class CctpFeeLoader(Loader):
    """Published Fast and Standard USDC transfer fees for one CCTP route.

    Args:
        source_domain: Circle CCTP source domain ID.
        destination_domain: Circle CCTP destination domain ID.
        loader_type: Cache format.
        http: Injectable HTTP client for offline tests.
    """

    _columns = (
        "src_domain",
        "dst_domain",
        "mode",
        "minimum_fee_bps",
        "finality_threshold",
    )

    def __init__(
        self,
        source_domain: int,
        destination_domain: int,
        *,
        loader_type: LoaderType = LoaderType.CSV,
        http: HttpClient | None = None,
    ) -> None:
        super().__init__(loader_type=loader_type)
        if int(source_domain) < 0 or int(destination_domain) < 0:
            raise ValueError("source_domain and destination_domain must be non-negative")
        self.source_domain = int(source_domain)
        self.destination_domain = int(destination_domain)
        self._http = http or HttpClient()

    def _cache_key(self) -> str:
        return f"{_CACHE_SCHEMA}-{self.source_domain}-{self.destination_domain}"

    def extract(self) -> None:
        """Fetch the current fee schedule; an unknown 404 route is empty."""
        url = f"{IRIS_API}/v2/burn/USDC/fees/{self.source_domain}/{self.destination_domain}"
        try:
            self._data = self._http.get(url)
        except LoaderHttpError as exc:
            if not _is_404(exc):
                raise
            self._data = []

    @classmethod
    def _frame(cls, rows: Any) -> pd.DataFrame:
        frame = pd.DataFrame(rows, columns=cls._columns)
        if frame.empty:
            return frame
        frame["src_domain"] = pd.to_numeric(frame["src_domain"], errors="raise").astype("int64")
        frame["dst_domain"] = pd.to_numeric(frame["dst_domain"], errors="raise").astype("int64")
        frame["minimum_fee_bps"] = pd.to_numeric(frame["minimum_fee_bps"], errors="raise").astype(float)
        frame["finality_threshold"] = pd.to_numeric(frame["finality_threshold"], errors="raise").astype("int64")
        return frame

    def transform(self) -> None:
        """Parse Iris fee entries into one row per supported transfer mode."""
        payload = self._data
        rows = payload if isinstance(payload, list) else (payload or {}).get("data", [])
        if not isinstance(rows, list):
            raise ValueError("Circle Iris fee response must be a list or an object with a 'data' list")
        parsed = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            try:
                finality_threshold = int(row["finalityThreshold"])
                mode = _mode(finality_threshold)
                minimum_fee_bps = float(row["minimumFee"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid Circle Iris fee entry: {row!r}") from exc
            if mode is not None:
                parsed.append({
                    "src_domain": self.source_domain,
                    "dst_domain": self.destination_domain,
                    "mode": mode,
                    "minimum_fee_bps": minimum_fee_bps,
                    "finality_threshold": finality_threshold,
                })
        parsed.sort(key=lambda row: row["finality_threshold"])
        self._data = self._frame(parsed)

    def read(self, with_run: bool = False) -> pd.DataFrame:
        """Return the route's published Fast and Standard fee schedule."""
        if with_run:
            self.run()
        else:
            self._read(self._cache_key())
        return self._frame(self._data).copy()
