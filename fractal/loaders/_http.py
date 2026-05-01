"""Shared HTTP client used by every loader.

A single ``requests.Session`` with retry/backoff and ``Retry-After``
support. Loaders MUST go through this module so transport behavior
(timeouts, retries, error messages) stays consistent.
"""
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry


@dataclass
class HttpConfig:
    timeout: float = 15.0
    max_retries: int = 5
    backoff_factor: float = 0.5
    status_forcelist: tuple = (408, 425, 429, 500, 502, 503, 504)


class LoaderHttpError(RuntimeError):
    """Wraps ``requests.HTTPError`` with the response body for easier debugging."""


class HttpClient:
    """A retrying ``requests.Session`` wrapper.

    Methods accept full URLs so per-protocol clients can carry their own
    base URL or override timeout per call.
    """

    def __init__(self, cfg: Optional[HttpConfig] = None) -> None:
        self.cfg = cfg or HttpConfig()
        self.session = requests.Session()

        retry = Retry(
            total=self.cfg.max_retries,
            read=self.cfg.max_retries,
            connect=self.cfg.max_retries,
            status=self.cfg.max_retries,
            status_forcelist=self.cfg.status_forcelist,
            allowed_methods=("GET", "POST"),
            backoff_factor=self.cfg.backoff_factor,
            raise_on_status=False,
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def _check(self, resp: requests.Response, url: str, params: Any) -> Any:
        try:
            resp.raise_for_status()
        except requests.HTTPError as exc:
            raise LoaderHttpError(
                f"HTTP {resp.status_code} for {url} params={params}: {resp.text[:300]}"
            ) from exc
        try:
            return resp.json()
        except ValueError as exc:
            raise LoaderHttpError(
                f"Non-JSON response from {url}: {resp.text[:300]}"
            ) from exc

    def get(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Any:
        resp = self.session.get(
            url,
            params=params or {},
            timeout=timeout or self.cfg.timeout,
        )
        return self._check(resp, url, params)

    def post(
        self,
        url: str,
        json: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Any:
        resp = self.session.post(
            url,
            json=json,
            timeout=timeout or self.cfg.timeout,
            headers=headers or {"Content-Type": "application/json"},
        )
        return self._check(resp, url, json)
