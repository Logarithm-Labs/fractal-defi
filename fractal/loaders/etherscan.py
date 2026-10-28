"""Etherscan V2 (multichain) API client for event logs and block-by-time.

One key, any supported chain (``chainid``), for the two reads a log
backfill needs when no archive node is at hand (or as a fallback for
one): the block at a timestamp and filtered event logs::

    es = EtherscanClient(chain_id=1)  # key from ETHERSCAN_API_KEY
    block = es.block_by_time(1_700_000_000, closest="before")
    logs = es.get_logs(CONTRACT, block - 7_200, block, topic0=EVENT_TOPIC)

The key comes from the ``api_key`` argument or the ``ETHERSCAN_API_KEY``
environment variable. It travels only as the ``apikey`` query parameter:
it is never part of a cache key or ``repr``, request failures are
re-raised as redacted copies detached from the original, and urllib3 log
records are scrubbed while a request runs
(:func:`fractal.loaders._http.call_redacted`).

Calls are spaced by ``min_interval`` seconds (the free tier allows a few
calls per second per key). Etherscan reports throttling inside an HTTP 200
body (``"Max calls per sec rate limit reached"``); such answers are
retried with a growing pause, while HTTP 429/5xx are retried by
:class:`~fractal.loaders._http.HttpClient`.

Log rows come back in the JSON-RPC shape (``address``, ``topics``,
``data``, ``blockNumber``, ``logIndex``, ``transactionHash`` as hex), plus
``timeStamp`` as an integer, so code written for
:class:`~fractal.loaders.rpc.JsonRpcClient` logs accepts them unchanged.
"""
import os
import threading
import time
from itertools import combinations
from typing import Any

from fractal.loaders._http import KEY_REDACTOR, HttpClient, call_redacted
from fractal.loaders.rpc import hex_to_int

__all__ = ["ETHERSCAN_V2_URL", "EtherscanClient", "EtherscanError"]

ETHERSCAN_V2_URL = "https://api.etherscan.io/v2/api"
_PAGE_SIZE = 1_000
_MAX_PAGES = 10  # Etherscan caps page × offset at 10 000 rows
_EMPTY_MESSAGES = ("no records found", "no transactions found")
_RATE_LIMIT_MARKERS = ("rate limit", "max calls per sec", "too many")


class EtherscanError(RuntimeError):
    """Etherscan answered ``status: 0`` with an error, or a malformed payload."""


class EtherscanClient:
    """Throttled Etherscan V2 client for one chain.

    Args:
        api_key: Etherscan key; falls back to the ``ETHERSCAN_API_KEY``
            environment variable.
        chain_id: EVM chain id (``1`` Ethereum, ``8453`` Base, ...).
        http: injectable transport (offline tests pass a fake).
        base_url: API endpoint.
        min_interval: minimum seconds between two requests of this client.
        rate_limit_retries: retries of a body-level rate-limit answer.
    """

    def __init__(
        self,
        api_key: str | None = None,
        chain_id: int = 1,
        http: HttpClient | None = None,
        base_url: str = ETHERSCAN_V2_URL,
        min_interval: float = 0.35,
        rate_limit_retries: int = 5,
    ) -> None:
        resolved = api_key or os.getenv("ETHERSCAN_API_KEY")
        if not resolved:
            raise ValueError("Etherscan requires an API key: pass api_key= or set ETHERSCAN_API_KEY.")
        self._api_key: str = resolved
        self.chain_id = int(chain_id)
        self.base_url = base_url
        self.min_interval = float(min_interval)
        self.rate_limit_retries = int(rate_limit_retries)
        self._http = http or HttpClient()
        self._lock = threading.Lock()
        self._last_call = 0.0

    def __repr__(self) -> str:
        return f"EtherscanClient(chain_id={self.chain_id}, base_url={self.base_url!r})"

    # ---------------------------------------------------------- transport
    def _throttle(self) -> None:
        with self._lock:
            wait = self._last_call + self.min_interval - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()

    def scrub(self, text: str) -> str:
        """``text`` without this client's key (an error body may echo the request)."""
        return KEY_REDACTOR.scrub(text, self._api_key)

    def request(self, module: str, action: str, **params: Any) -> Any:
        """One API call; returns ``result`` (``[]`` for "no records")."""
        query = {"chainid": self.chain_id, "module": module, "action": action, **params, "apikey": self._api_key}
        for attempt in range(self.rate_limit_retries + 1):
            self._throttle()
            payload = call_redacted((self._api_key,), self._http.get, self.base_url, params=query)
            if not isinstance(payload, dict):
                raise EtherscanError(f"Etherscan {module}.{action}: non-object payload")
            if str(payload.get("status")) == "1":
                return payload.get("result")
            message = str(payload.get("message", ""))
            result = payload.get("result")
            if message.lower().startswith(_EMPTY_MESSAGES) or result == []:
                return []
            text = f"{message}: {result}"
            if any(marker in text.lower() for marker in _RATE_LIMIT_MARKERS) and attempt < self.rate_limit_retries:
                time.sleep(1.0 + attempt)
                continue
            raise EtherscanError(f"Etherscan {module}.{action}: {self.scrub(text)[:200]}")
        raise EtherscanError(f"Etherscan {module}.{action}: still rate limited")

    def _paged_logs(self, lo: int, hi: int, params: dict[str, Any]) -> list[dict[str, Any]]:
        """All log rows over ``[lo, hi]``; a range past the 10k-row window is split in two."""
        rows: list[dict[str, Any]] = []
        for page in range(1, _MAX_PAGES + 1):
            batch = self.request("logs", "getLogs", fromBlock=lo, toBlock=hi, page=page, offset=_PAGE_SIZE,
                                 **params)
            if not isinstance(batch, list):
                raise EtherscanError(f"Etherscan logs.getLogs: expected a list, got {batch!r}"[:200])
            rows += batch
            if len(batch) < _PAGE_SIZE:
                return rows
        if hi <= lo:
            raise EtherscanError(f"Etherscan logs.getLogs: more than 10 000 rows in block {lo}")
        mid = (lo + hi) // 2
        return self._paged_logs(lo, mid, params) + self._paged_logs(mid + 1, hi, params)

    # -------------------------------------------------------------- reads
    def block_by_time(self, timestamp: int, closest: str = "before") -> int:
        """Block at a UNIX timestamp (``closest`` = ``"before"`` or ``"after"``)."""
        if closest not in ("before", "after"):
            raise ValueError(f"closest must be 'before' or 'after', got {closest!r}")
        result = self.request("block", "getblocknobytime", timestamp=int(timestamp), closest=closest)
        try:
            return int(result)
        except (TypeError, ValueError) as exc:
            raise EtherscanError(f"Etherscan getblocknobytime returned {result!r}") from exc

    def get_logs(
        self,
        address: str | None,
        from_block: int,
        to_block: int,
        topic0: str | None = None,
        topics: dict[int, str] | None = None,
    ) -> list[dict[str, Any]]:
        """Logs over ``[from_block, to_block]``, oldest first.

        ``address`` may be ``None`` (any emitter) when a topic is given.
        ``topics`` maps a position (1-3) to a required value; all given
        topics are AND-ed. Etherscan takes one value per position: expand
        an OR-list into one call per value (as
        :class:`~fractal.loaders.onchain.LogReader` does). Rows are
        normalised to the JSON-RPC log shape.
        """
        given = dict(topics or {})
        if any(pos not in (1, 2, 3) for pos in given):
            raise ValueError(f"topic positions must be 1, 2 or 3, got {sorted(given)}")
        if topic0 is not None:
            given[0] = topic0
        if address is None and not given:
            raise ValueError("Etherscan getLogs needs an address or at least one topic")
        params: dict[str, Any] = {} if address is None else {"address": address}
        for pos, value in given.items():
            params[f"topic{pos}"] = value
        for a, b in combinations(sorted(given), 2):
            params[f"topic{a}_{b}_opr"] = "and"
        raw = self._paged_logs(int(from_block), int(to_block), params)
        logs = [self._normalise_log(row) for row in raw]
        return sorted(logs, key=lambda log: (int(log["blockNumber"], 16), int(log["logIndex"], 16)))

    @staticmethod
    def _normalise_log(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "address": str(row["address"]).lower(),
            "topics": [t.lower() for t in row.get("topics", []) if t],
            "data": row.get("data") or "0x",
            "blockNumber": hex(hex_to_int(row["blockNumber"])),
            "logIndex": hex(hex_to_int(row.get("logIndex"))),
            "transactionHash": str(row["transactionHash"]).lower(),
            "timeStamp": hex_to_int(row.get("timeStamp")),
        }
