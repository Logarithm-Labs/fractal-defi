"""Network gas price history from a JSON-RPC node (``eth_feeHistory``).

:class:`GasPriceHistoryLoader` returns, per bar (``freq="1h"``, ``"1D"``,
...), the EIP-1559 base fee and the priority fee (tip) a transaction paid
on one network, in gwei, as a :class:`~fractal.loaders.structs.GasPriceHistory`::

    loader = GasPriceHistoryLoader(os.environ["ETH_RPC_URL"], "ethereum",
                                   "2024-01-01", "2024-02-01", freq="1h")
    gas = loader.read(with_run=True)          # base_fee_gwei, priority_fee_gwei, ...

**What is sampled.** For each bar time ``t`` the loader finds the last block mined at or before
``t`` (:meth:`~fractal.loaders.rpc.JsonRpcClient.block_at_timestamp`, so
no look-ahead) and reads ``eth_feeHistory`` over the ``blocks_per_sample``
blocks ending there. With ``samples_per_bar = k > 1`` it does the same at
``k`` evenly spaced times in ``(t - bar, t]`` and pools the blocks. Per bar:

* ``base_fee_gwei`` is the mean ``baseFeePerGas`` of the pooled blocks;
* ``priority_fee_p<q>_gwei`` is, for each requested percentile ``q``, the
  median over the pooled blocks of the node's ``reward`` at ``q`` (the
  effective tip at the ``q``-th percentile of gas used in that block).
  Empty blocks (``gasUsedRatio == 0``), for which nodes report a zero
  reward, are left out unless every pooled block is empty;
* ``priority_fee_gwei`` is the column at ``cost_percentile`` and
  ``gas_price_gwei = base_fee_gwei + priority_fee_gwei``;
* ``gas_used_ratio`` is the mean block fullness, ``block`` the newest
  sampled block.

**Limits.**

* The value at ``t`` describes ``blocks_per_sample × samples_per_bar``
  blocks before ``t``, not the whole bar: on Ethereum the default (20
  blocks, one sample) covers four minutes. Raise either knob for a fuller
  bar average; each sample costs one ``eth_feeHistory`` call plus a few
  cached header reads.
* ``eth_feeHistory`` needs the block bodies and receipts of the range: use
  a node that keeps history (an archive node, or a full node without
  history expiry). Providers cap ``blockCount`` (1024 in geth), and some
  serve fee history only for recent blocks.
* Blocks before EIP-1559 report a zero base fee; the reward is then the
  whole gas price, so ``gas_price_gwei`` stays meaningful.
* Bars after the chain tip and before the first block are dropped.

**L2 networks.**

* **OP Stack** (Optimism, Base, ...): base and priority fees price L2
  execution and are meaningful, but a transaction also pays an L1 data
  fee (the receipt's ``l1Fee``) that is not part of this series, so
  ``gas_units × gas_price`` understates the full cost.
* **Arbitrum**: the sequencer ignores tips (they are refunded), so the
  reward percentiles are not a tip market and ``base_fee_gwei`` is
  effectively the gas price (it sits at its floor most of the time). The
  L1 component is charged as extra gas units, not as a higher price.
* Other L2s and sidechains implement ``eth_feeHistory`` with their own
  semantics (fixed or zero base fees, enforced minimum tips); read their
  docs before comparing networks.

**USD.** Nothing fetches prices. With ``native_price(t) -> float`` (USD per native
token at bar ``t``), :meth:`GasPriceHistoryLoader.read` adds
``native_price_usd`` and ``tx_cost_usd``; ``tx_cost_native = gas_units ×
gas_price`` (native tokens) is always added. The cache holds only the
chain data, so the price function never enters a cache key.
"""
from collections.abc import Callable, Sequence
from datetime import datetime
from itertools import pairwise
from typing import Any

import numpy as np
import pandas as pd

from fractal.loaders._http import HttpClient
from fractal.loaders.base_loader import Loader, LoaderType
from fractal.loaders.rpc import JsonRpcClient, hex_to_int
from fractal.loaders.structs import GasPriceHistory

__all__ = ["GasPriceHistoryLoader"]

GWEI = 1e9
_MAX_BLOCK_COUNT = 1024  # geth's eth_feeHistory cap; most providers match or go lower
_COST_COLUMNS = ("native_price_usd", "tx_cost_usd")


def _utc(value: str | datetime | pd.Timestamp) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def percentile_column(q: float) -> str:
    """Column name of the priority fee at percentile ``q`` (``25`` → ``priority_fee_p25_gwei``)."""
    return f"priority_fee_p{float(q):g}_gwei"


class GasPriceHistoryLoader(Loader):
    """Base and priority fee per bar for one network (see the module docs).

    Args:
        rpc_url: JSON-RPC endpoint that serves ``eth_feeHistory`` over the
            range. A keyed URL is redacted from errors and logs and is
            never part of the cache key.
        chain: label for the cache key (``"ethereum"``, ``"base"``, ...).
        start_time: first bar (naive = UTC).
        end_time: last bar, inclusive.
        freq: bar length, a fixed ``pd.Timedelta`` string (``"1h"``, ``"1D"``).
        blocks_per_sample: blocks per ``eth_feeHistory`` call (1-1024).
        samples_per_bar: evenly spaced samples pooled per bar.
        percentiles: priority-fee percentiles to read, increasing, in [0, 100].
        cost_percentile: the percentile used for ``priority_fee_gwei``
            and the cost columns; must be one of ``percentiles``.
        gas_units: gas of the reference transaction for the cost columns
            (21 000 is a plain native transfer).
        native_price: optional ``f(bar time) -> USD per native token``.
        loader_type: cache format.
        http: injectable transport (offline tests pass a fake).
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        rpc_url: str,
        chain: str,
        start_time: str | datetime | pd.Timestamp,
        end_time: str | datetime | pd.Timestamp,
        freq: str = "1h",
        *,
        blocks_per_sample: int = 20,
        samples_per_bar: int = 1,
        percentiles: Sequence[float] = (25.0, 50.0, 75.0),
        cost_percentile: float = 50.0,
        gas_units: int = 21_000,
        native_price: Callable[[pd.Timestamp], float | None] | None = None,
        loader_type: LoaderType = LoaderType.CSV,
        http: HttpClient | None = None,
    ) -> None:
        super().__init__(loader_type=loader_type)
        if not chain or not str(chain).replace("-", "").replace("_", "").isalnum():
            raise ValueError(f"chain must be a plain label such as 'ethereum', got {chain!r}")
        self.chain = str(chain).lower()
        self.start_time, self.end_time = _utc(start_time), _utc(end_time)
        if self.end_time < self.start_time:
            raise ValueError("end_time is before start_time")
        self.freq = freq
        self.bar_length = pd.Timedelta(freq)
        if self.bar_length <= pd.Timedelta(0):
            raise ValueError(f"freq must be a positive fixed duration, got {freq!r}")
        if not 1 <= int(blocks_per_sample) <= _MAX_BLOCK_COUNT:
            raise ValueError(f"blocks_per_sample must be in [1, {_MAX_BLOCK_COUNT}], got {blocks_per_sample}")
        if int(samples_per_bar) < 1:
            raise ValueError(f"samples_per_bar must be >= 1, got {samples_per_bar}")
        qs = [float(q) for q in percentiles]
        if not qs or any(not 0 <= q <= 100 for q in qs) or any(a >= b for a, b in pairwise(qs)):
            raise ValueError(f"percentiles must be strictly increasing values in [0, 100], got {percentiles!r}")
        if float(cost_percentile) not in qs:
            raise ValueError(f"cost_percentile {cost_percentile} is not one of the percentiles {qs}")
        self.blocks_per_sample, self.samples_per_bar = int(blocks_per_sample), int(samples_per_bar)
        self.percentiles, self.cost_percentile = tuple(qs), float(cost_percentile)
        self.gas_units = int(gas_units)
        self.native_price = native_price
        self._rpc = JsonRpcClient(rpc_url, http=http)

    def __repr__(self) -> str:
        return (f"GasPriceHistoryLoader(chain={self.chain!r}, start={self.start_time}, end={self.end_time}, "
                f"freq={self.freq!r})")

    # ------------------------------------------------------------ pipeline
    def _cache_key(self) -> str:
        qs = "_".join(f"{q:g}" for q in self.percentiles)
        return (f"{self.chain}-gas-{int(self.start_time.timestamp())}-{int(self.end_time.timestamp())}"
                f"-{int(self.bar_length.total_seconds())}s-{self.blocks_per_sample}x{self.samples_per_bar}-p{qs}")

    def _sample_times(self, bar_end: int) -> list[int]:
        step = self.bar_length.total_seconds() / self.samples_per_bar
        return [bar_end - round(step * (self.samples_per_bar - 1 - j)) for j in range(self.samples_per_bar)]

    def _bar_row(self, bar_end: int, low: int, tip: int) -> tuple[dict[str, Any] | None, int]:
        """One bar from its samples; returns ``(row or None, newest block)``."""
        base: list[int] = []
        ratios: list[float] = []
        rewards: list[list[int]] = []
        seen: set[int] = set()
        newest = low
        for ts in self._sample_times(bar_end):
            try:
                block = self._rpc.block_at_timestamp(ts, closest="before", low=low, high=tip)
            except ValueError:  # before the first block
                continue
            low = newest = block
            if block in seen:
                continue
            seen.add(block)
            history = self._rpc.fee_history(min(self.blocks_per_sample, block + 1), block, self.percentiles)
            count = len(history.get("gasUsedRatio") or [])
            base += [hex_to_int(v) for v in history["baseFeePerGas"][:count]]
            ratios += [float(r) for r in history["gasUsedRatio"]]
            reward = history.get("reward") or [[0] * len(self.percentiles)] * count
            rewards += [[hex_to_int(v) for v in row] for row in reward[:count]]
        if not ratios:
            return None, newest
        full = [row for row, ratio in zip(rewards, ratios, strict=False) if ratio > 0] or rewards
        tips = np.median(np.asarray(full, dtype=float), axis=0) / GWEI
        row: dict[str, Any] = {
            "time": bar_end,
            "block": newest,
            "n_blocks": len(ratios),
            "base_fee_gwei": float(np.mean(base)) / GWEI,
            "gas_used_ratio": float(np.mean(ratios)),
        }
        row.update({percentile_column(q): float(tip) for q, tip in zip(self.percentiles, tips, strict=True)})
        return row, newest

    def extract(self) -> None:
        tip = self._rpc.block_number()
        tip_time = self._rpc.block_timestamp(tip)
        rows: list[dict[str, Any]] = []
        low = 0
        for bar_time in pd.date_range(self.start_time, self.end_time, freq=self.bar_length):
            bar_end = int(bar_time.timestamp())
            if bar_end > tip_time:  # the chain has not reached this bar yet
                break
            row, low = self._bar_row(bar_end, low, tip)
            if row is not None:
                rows.append(row)
        self._data = pd.DataFrame(rows)

    def transform(self) -> None:
        cols = ["time", "block", "n_blocks", "base_fee_gwei", "gas_used_ratio",
                *(percentile_column(q) for q in self.percentiles)]
        if self._data is None or self._data.empty:
            self._data = pd.DataFrame(columns=cols)
            return
        self._data = self._data[cols].sort_values("time").reset_index(drop=True)

    def read(self, with_run: bool = False) -> GasPriceHistory:
        if with_run:
            self.run()
        else:
            self._read(self._cache_key())
        data = self._data
        tip_column = percentile_column(self.cost_percentile)
        if data is None or data.empty:
            empty: list[float] = []
            return GasPriceHistory(time=pd.DatetimeIndex([], tz="UTC"), base_fee_gwei=empty, priority_fee_gwei=empty,
                                   gas_used_ratio=empty, block=empty,
                                   priority_fee_percentiles={percentile_column(q): empty for q in self.percentiles},
                                   tx_cost_native=empty,
                                   **({name: empty for name in _COST_COLUMNS} if self.native_price else {}))
        time = self._utc_index("time")
        gas_price = data["base_fee_gwei"].to_numpy(float) + data[tip_column].to_numpy(float)
        tx_cost_native = self.gas_units * gas_price / GWEI
        usd: dict[str, np.ndarray] = {}
        if self.native_price is not None:
            prices = np.array([np.nan if (p := self.native_price(t)) is None else float(p) for t in time])
            usd = {"native_price_usd": prices, "tx_cost_usd": tx_cost_native * prices}
        return GasPriceHistory(
            time=time,
            base_fee_gwei=data["base_fee_gwei"].to_numpy(float),
            priority_fee_gwei=data[tip_column].to_numpy(float),
            gas_used_ratio=data["gas_used_ratio"].to_numpy(float),
            block=data["block"].to_numpy("int64"),
            priority_fee_percentiles={percentile_column(q): data[percentile_column(q)].to_numpy(float)
                                      for q in self.percentiles},
            tx_cost_native=tx_cost_native,
            **usd,
        )
