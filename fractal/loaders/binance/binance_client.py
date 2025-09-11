from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

FUTUTRE_SECTION = 'futures'
SPOT_SECTION = 'spot'

@dataclass
class HttpConfig:
    futures_base_url: str = "https://fapi.binance.com"
    timeout: int = 15
    max_retries: int = 5
    backoff_factor: float = 0.5


class BinanceHttp:

    def __init__(self, cfg: Optional[HttpConfig] = None) -> None:
        self.cfg = cfg or HttpConfig()
        self.session = requests.Session()

        retry = Retry(
            total=self.cfg.max_retries,
            read=self.cfg.max_retries,
            connect=self.cfg.max_retries,
            status=self.cfg.max_retries,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET",),
            backoff_factor=self.cfg.backoff_factor,
            raise_on_status=False,
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def get(self, section: str, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        if section.lower() == FUTUTRE_SECTION:
            base_url: str = self.cfg.futures_base_url
        else:
            raise ValueError(f"Unknown section {section} for Binance API")

        url = f"{base_url}{path}"
        resp = self.session.get(url, params=params or {}, timeout=self.cfg.timeout)
        try:
            resp.raise_for_status()
        except requests.HTTPError as e:
            # Attach response text for easier debugging while keeping the original error
            raise requests.HTTPError(
            f"HTTP {resp.status_code} for {url} with params={params}: {resp.text[:300]}"
            ) from e
        return resp.json()
