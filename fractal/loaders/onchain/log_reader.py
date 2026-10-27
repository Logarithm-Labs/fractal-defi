"""Event logs over a block range from a JSON-RPC node, with Etherscan as fallback.

:class:`LogReader` reads ``eth_getLogs`` in chunks of ``chunk_size``
blocks. A failed chunk is classified by its error text
(:func:`~fractal.loaders.rpc.is_splittable_error`,
:func:`~fractal.loaders.rpc.is_transient_error`):

* a range/limit rejection, 413 or timeout halves the range and reads both
  halves (down to ``min_chunk`` blocks);
* a transient failure (overload, HTTP 5xx/429, dropped connection) is
  retried as-is with exponential backoff, up to ``transient_retries`` times;
* anything else propagates.

A range that still fails after splitting or retrying is read from
Etherscan (:class:`~fractal.loaders.etherscan.EtherscanClient`) when one is
given. Etherscan filters one value per topic position, so an OR-list
filter becomes one Etherscan query per combination. With
``prefer="etherscan"`` the order is reversed: the whole range is read from
Etherscan and falls back to the node on an Etherscan failure.

Rows are JSON-RPC shaped logs, de-duplicated by ``(transactionHash,
logIndex)`` and sorted by ``(blockNumber, logIndex)``. Keys stay out of
errors and log records: the node client redacts a keyed URL and the
Etherscan client its ``apikey``; fallback warnings repeat only that
redacted text.
"""
import itertools
import logging
import time
from collections import Counter
from collections.abc import Callable, Sequence
from typing import Any

import requests

from fractal.loaders._http import KEY_REDACTOR, LoaderHttpError, url_secrets
from fractal.loaders.etherscan import EtherscanClient, EtherscanError
from fractal.loaders.rpc import JsonRpcClient, RpcCallError, hex_to_int, is_splittable_error, is_transient_error

__all__ = ["LogReader"]

_LOG = logging.getLogger(__name__)
_FAILURES = (RpcCallError, LoaderHttpError, EtherscanError, requests.RequestException)

Topics = Sequence[str | Sequence[str] | None]


class LogReader:
    """``eth_getLogs`` with retries, range splitting and an Etherscan fallback.

    Args:
        rpc: node URL or client; ``None`` reads from Etherscan only.
        etherscan: fallback (or, with ``prefer="etherscan"``, primary) source.
        prefer: ``"rpc"`` (default) or ``"etherscan"``.
        chunk_size: blocks per ``eth_getLogs`` request.
        min_chunk: smallest range still split on a range/limit error.
        transient_retries: attempts per range on transient errors.
        backoff: first retry pause in seconds, doubled per attempt
            (capped at ``max_backoff``).
        sleep: pause function (tests pass a no-op).
    """

    def __init__(
        self,
        rpc: JsonRpcClient | str | None = None,
        etherscan: EtherscanClient | None = None,
        *,
        prefer: str = "rpc",
        chunk_size: int = 10_000,
        min_chunk: int = 1,
        transient_retries: int = 6,
        backoff: float = 3.0,
        max_backoff: float = 60.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if rpc is None and etherscan is None:
            raise ValueError("LogReader needs an RPC endpoint, an Etherscan client or both")
        if prefer not in ("rpc", "etherscan"):
            raise ValueError(f"prefer must be 'rpc' or 'etherscan', got {prefer!r}")
        if chunk_size < 1 or min_chunk < 1 or transient_retries < 1:
            raise ValueError("chunk_size, min_chunk and transient_retries must be positive")
        self.rpc = JsonRpcClient(rpc) if isinstance(rpc, str) else rpc
        self.etherscan = etherscan
        self.prefer = prefer
        self.chunk_size, self.min_chunk = int(chunk_size), int(min_chunk)
        self.transient_retries = int(transient_retries)
        self.backoff, self.max_backoff = float(backoff), float(max_backoff)
        self._sleep = sleep
        self.calls: Counter = Counter()  # requests per source, for diagnostics

    def __repr__(self) -> str:
        return f"LogReader(rpc={'yes' if self.rpc else 'no'}, etherscan={self.etherscan!r}, prefer={self.prefer!r})"

    # ---------------------------------------------------------------- public
    def get_logs(
        self,
        address: str | Sequence[str],
        topics: Topics | None,
        from_block: int,
        to_block: int,
    ) -> list[dict[str, Any]]:
        """Logs of ``address`` matching ``topics`` over ``[from_block, to_block]``.

        ``topics`` follows ``eth_getLogs``: one entry per position, each a
        value, an OR-list or ``None`` (any).
        """
        lo, hi = int(from_block), int(to_block)
        if hi < lo:
            return []
        if self.prefer == "etherscan" and self.etherscan is not None:
            try:
                return self._finish(self._etherscan(address, topics, lo, hi))
            except _FAILURES as exc:
                if self.rpc is None:
                    raise
                self._warn("etherscan", "rpc", lo, hi, exc)
        if self.rpc is None:
            return self._finish(self._etherscan(address, topics, lo, hi))
        logs: list[dict[str, Any]] = []
        for start in range(lo, hi + 1, self.chunk_size):
            logs += self._rpc_range(address, topics, start, min(hi, start + self.chunk_size - 1))
        return self._finish(logs)

    # ------------------------------------------------------------------ RPC
    def _rpc_range(self, address: Any, topics: Topics | None, lo: int, hi: int) -> list[dict[str, Any]]:
        rpc = self.rpc
        if rpc is None:
            raise ValueError("no RPC endpoint configured")
        for attempt in range(self.transient_retries):
            try:
                self.calls["rpc"] += 1
                return rpc.get_logs(address, topics, lo, hi)
            except _FAILURES as exc:
                if is_splittable_error(exc) and hi - lo + 1 > self.min_chunk:
                    mid = (lo + hi) // 2
                    return self._rpc_range(address, topics, lo, mid) + self._rpc_range(address, topics, mid + 1, hi)
                retry = is_transient_error(exc) and attempt < self.transient_retries - 1
                if not retry:
                    if self.etherscan is None or not (is_splittable_error(exc) or is_transient_error(exc)):
                        raise
                    self._warn("rpc", "etherscan", lo, hi, exc)
                    return self._etherscan(address, topics, lo, hi)
                self._sleep(min(self.max_backoff, self.backoff * 2 ** attempt))
        raise RpcCallError(f"eth_getLogs [{lo}, {hi}]: no attempt made")  # transient_retries >= 1

    # ------------------------------------------------------------ Etherscan
    def _etherscan(self, address: Any, topics: Topics | None, lo: int, hi: int) -> list[dict[str, Any]]:
        if self.etherscan is None:
            raise ValueError("no Etherscan client configured")
        addresses = [address] if isinstance(address, str) else list(address)
        positions = [[None] if t is None else [t] if isinstance(t, str) else list(t) for t in (topics or [])]
        logs: list[dict[str, Any]] = []
        for addr in addresses:
            for combo in itertools.product(*positions):
                fixed = {pos: value for pos, value in enumerate(combo) if value is not None and pos > 0}
                self.calls["etherscan"] += 1
                logs += self.etherscan.get_logs(addr, lo, hi, topic0=combo[0] if combo else None, topics=fixed)
        return logs

    # -------------------------------------------------------------- helpers
    @staticmethod
    def _finish(logs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        unique = {(str(log.get("transactionHash", "")).lower(), hex_to_int(log["logIndex"])): log for log in logs}
        return sorted(unique.values(), key=lambda log: (hex_to_int(log["blockNumber"]), hex_to_int(log["logIndex"])))

    def _warn(self, source: str, fallback: str, lo: int, hi: int, exc: BaseException) -> None:
        secrets = url_secrets(self.rpc.rpc_url) if self.rpc is not None else ()
        text = KEY_REDACTOR.scrub(str(exc), *secrets)
        _LOG.warning("logs [%d, %d] via %s failed, falling back to %s: %s", lo, hi, source, fallback, text[:200])
