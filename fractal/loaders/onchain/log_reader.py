"""Event logs over a block or time range from a JSON-RPC node, with Etherscan as fallback.

:class:`LogReader` is protocol-agnostic: give it an emitter address (or a
list, or ``None`` for any) and an ``eth_getLogs`` topic filter, and it
returns the raw logs. Decoding is left to the caller (see the ABI helpers
in :mod:`fractal.loaders.rpc`)::

    reader = LogReader(os.environ["ETH_RPC_URL"], EtherscanClient(chain_id=1))
    topic = event_topic("Transfer(address,address,uint256)")
    logs = reader.get_logs(TOKEN, [topic, None, [ADDR_A, ADDR_B]], 21_000_000, 21_100_000)
    logs = reader.get_logs_between(TOKEN, [topic], "2024-01-01", "2024-01-02", with_timestamps=True)

The node is read with ``eth_getLogs`` in chunks of ``chunk_size``
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
filter becomes one Etherscan query per combination (and per address).
With ``prefer="etherscan"`` the order is reversed: the whole range is read
from Etherscan and falls back to the node on an Etherscan failure.

Rows are JSON-RPC shaped logs (``address``, ``topics``, ``data``,
``blockNumber``, ``logIndex``, ``transactionHash``, ...), de-duplicated by
``(transactionHash, logIndex)`` and sorted by ``(blockNumber, logIndex)``.
``with_timestamps=True`` adds an integer ``timeStamp`` (UNIX seconds) from
the log itself when the source provides it (Etherscan ``timeStamp``, a
node's ``blockTimestamp``), else from the block header (one cached
``eth_getBlockByNumber`` per distinct block).

Keys stay out of errors and log records: the node client redacts a keyed
URL and the Etherscan client its ``apikey``; fallback warnings repeat only
that redacted text.
"""
import itertools
import logging
import time
from collections import Counter
from collections.abc import Callable, Sequence
from datetime import datetime
from typing import Any

import pandas as pd
import requests

from fractal.loaders._http import KEY_REDACTOR, LoaderHttpError, url_secrets
from fractal.loaders.etherscan import EtherscanClient, EtherscanError
from fractal.loaders.rpc import JsonRpcClient, RpcCallError, hex_to_int, is_splittable_error, is_transient_error

__all__ = ["LogReader"]

_LOG = logging.getLogger(__name__)
_FAILURES = (RpcCallError, LoaderHttpError, EtherscanError, requests.RequestException)

Address = str | Sequence[str] | None
Topics = Sequence[str | Sequence[str] | None]
TimeLike = int | float | str | datetime | pd.Timestamp


def _to_unix(value: TimeLike) -> int:
    """UNIX seconds from an epoch number, an ISO string or a datetime (naive = UTC)."""
    if isinstance(value, (int, float)):
        return int(value)
    stamp = pd.Timestamp(value)
    stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
    return int(stamp.timestamp())


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
        max_backoff: longest retry pause in seconds.
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
        address: Address,
        topics: Topics | None,
        from_block: int,
        to_block: int,
        *,
        with_timestamps: bool = False,
    ) -> list[dict[str, Any]]:
        """Logs matching ``address`` and ``topics`` over ``[from_block, to_block]``.

        ``address`` is one emitter, a list (any of them) or ``None`` (any
        emitter; Etherscan then needs a topic). ``topics`` follows
        ``eth_getLogs``: one entry per position, each a value, an OR-list
        or ``None`` (any).
        """
        lo, hi = int(from_block), int(to_block)
        if hi < lo:
            return []
        logs = self._read(address, topics, lo, hi)
        return self._with_timestamps(logs) if with_timestamps else logs

    def get_logs_between(
        self,
        address: Address,
        topics: Topics | None,
        start: TimeLike,
        end: TimeLike,
        *,
        with_timestamps: bool = False,
    ) -> list[dict[str, Any]]:
        """Logs of blocks mined in ``[start, end]`` (see :meth:`block_range`)."""
        from_block, to_block = self.block_range(start, end)
        return self.get_logs(address, topics, from_block, to_block, with_timestamps=with_timestamps)

    def block_range(self, start: TimeLike, end: TimeLike) -> tuple[int, int]:
        """``(first block at or after start, last block at or before end)``.

        Times are UNIX seconds, ISO strings or datetimes (naive = UTC).
        Uses :meth:`JsonRpcClient.block_at_timestamp` when a node is
        configured, else Etherscan ``getblocknobytime``. The result is an
        empty range (``to < from``) when no block was mined in the window.
        """
        ts_start, ts_end = _to_unix(start), _to_unix(end)
        if ts_end < ts_start:
            raise ValueError(f"end {end!r} is before start {start!r}")
        if self.rpc is not None:
            last = self.rpc.block_at_timestamp(ts_end, closest="before")
            try:
                first = self.rpc.block_at_timestamp(ts_start, closest="after", high=last)
            except ValueError:  # nothing mined in [start, end]
                return last + 1, last
            return first, last
        etherscan = self.etherscan
        if etherscan is None:  # unreachable: the constructor requires one source
            raise ValueError("no block source configured")
        return etherscan.block_by_time(ts_start, closest="after"), etherscan.block_by_time(ts_end, closest="before")

    # ------------------------------------------------------------ sources
    def _read(self, address: Address, topics: Topics | None, lo: int, hi: int) -> list[dict[str, Any]]:
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

    def _rpc_range(self, address: Address, topics: Topics | None, lo: int, hi: int) -> list[dict[str, Any]]:
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

    def _etherscan(self, address: Address, topics: Topics | None, lo: int, hi: int) -> list[dict[str, Any]]:
        if self.etherscan is None:
            raise ValueError("no Etherscan client configured")
        addresses: list[str | None] = [address] if address is None or isinstance(address, str) else list(address)
        positions = [[None] if t is None else [t] if isinstance(t, str) else list(t) for t in (topics or [])]
        logs: list[dict[str, Any]] = []
        for addr in addresses:
            for combo in itertools.product(*positions):
                fixed = {pos: value for pos, value in enumerate(combo) if value is not None and pos > 0}
                self.calls["etherscan"] += 1
                logs += self.etherscan.get_logs(addr, lo, hi, topic0=combo[0] if combo else None, topics=fixed)
        return logs

    # -------------------------------------------------------------- helpers
    def _with_timestamps(self, logs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Add an integer ``timeStamp`` to every log, reading each missing block once."""
        times: dict[int, int] = {}
        for log in logs:
            given = log.get("timeStamp", log.get("blockTimestamp"))
            if given not in (None, "", "0x"):
                times.setdefault(hex_to_int(log["blockNumber"]), hex_to_int(given))
        out = []
        for log in logs:
            block = hex_to_int(log["blockNumber"])
            if block not in times:
                if self.rpc is None:
                    raise ValueError(f"log in block {block} has no timestamp and no RPC is configured")
                self.calls["rpc"] += 1
                times[block] = self.rpc.block_timestamp(block)
            out.append({**log, "timeStamp": times[block]})
        return out

    @staticmethod
    def _finish(logs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        unique = {(str(log.get("transactionHash", "")).lower(), hex_to_int(log["logIndex"])): log for log in logs}
        return sorted(unique.values(), key=lambda log: (hex_to_int(log["blockNumber"]), hex_to_int(log["logIndex"])))

    def _warn(self, source: str, fallback: str, lo: int, hi: int, exc: BaseException) -> None:
        secrets = url_secrets(self.rpc.rpc_url) if self.rpc is not None else ()
        text = KEY_REDACTOR.scrub(str(exc), *secrets)
        if self.etherscan is not None:
            text = self.etherscan.scrub(text)
        _LOG.warning("logs [%d, %d] via %s failed, falling back to %s: %s", lo, hi, source, fallback, text[:200])
