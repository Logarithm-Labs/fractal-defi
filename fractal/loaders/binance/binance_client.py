"""Binance USDT-M futures REST wrapper used by the funding/kline loaders.

This is a thin adapter around the shared :class:`fractal.loaders._http.HttpClient`
so we don't have to duplicate retry/backoff logic per call site.
"""
from typing import Any, Dict, Optional

from fractal.loaders._http import HttpClient

FUTURES_SECTION = "futures"
SPOT_SECTION = "spot"

_BASES = {
    FUTURES_SECTION: "https://fapi.binance.com",
}


class BinanceHttp:
    """Issue GETs against Binance public REST endpoints."""

    def __init__(self, http: Optional[HttpClient] = None) -> None:
        self._http = http or HttpClient()

    def get(self, section: str, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        if section.lower() not in _BASES:
            raise ValueError(f"Unknown section {section} for Binance API")
        url = f"{_BASES[section.lower()]}{path}"
        return self._http.get(url, params=params)
