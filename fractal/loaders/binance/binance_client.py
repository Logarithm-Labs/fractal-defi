"""Binance public REST wrapper used by the funding/kline loaders.

Thin adapter around the shared :class:`fractal.loaders._http.HttpClient`
so retry/backoff logic isn't duplicated per call site. Two API
sections supported: ``futures`` (USDT-M perp endpoints under
``fapi.binance.com``) and ``spot`` (CEX spot endpoints under
``api.binance.com``). Loaders pick the section explicitly.
"""
from typing import Any, Dict, Optional

from fractal.loaders._http import HttpClient

FUTURES_SECTION = "futures"
SPOT_SECTION = "spot"

_BASES = {
    FUTURES_SECTION: "https://fapi.binance.com",
    SPOT_SECTION: "https://api.binance.com",
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
