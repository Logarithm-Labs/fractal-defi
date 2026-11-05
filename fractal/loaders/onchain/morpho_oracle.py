"""Oracle price history of a Morpho Blue market, read on-chain.

Morpho Blue checks health at the market oracle's ``price()``, not at the
collateral's market price, and the Morpho API keeps no history of it.
:class:`MorphoOraclePriceLoader` reads ``price()`` at the last block mined
at or before each bar (no look-ahead) and returns it in loan-token units
per collateral unit as a :class:`~fractal.loaders.structs.PriceHistory`,
ready for ``MorphoGlobalState.collateral_price``::

    loader = MorphoOraclePriceLoader(os.environ["ETH_ARCHIVE_RPC_URL"], market_id,
                                     "2025-06-01", "2025-12-01", freq="1D")
    oracle = loader.read(with_run=True)["price"]

The oracle address and the token decimals come from
:func:`~fractal.loaders.morpho.get_market_info` (a market's oracle is
immutable). ``eth_call`` on a past block needs an **archive node**; each
bar costs one call plus a few cached header reads, so daily bars keep a
year of history at about 365 calls.

**Hourly bars from daily reads.** A PT oracle is usually a closed-form
feed of the Pendle implied rate (:func:`~fractal.core.entities.models.pendle_math.linear_discount_oracle_price`,
:func:`~fractal.core.entities.models.pendle_math.twap_oracle_price`,
:func:`~fractal.core.entities.models.pendle_math.hybrid_oracle_price`, or the
minimum of two feeds), sometimes multiplied by a price feed of the
underlying asset. :func:`calibrate_oracle_price` scales such a model,
computed on the backtest's own grid, to the exact daily reads: the ratio
``κ = oracle / model`` at each read is interpolated in ``ln κ`` between
reads, so the underlying feed and any adapter the formula leaves out are
carried at daily resolution.
"""
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from fractal.loaders._http import HttpClient
from fractal.loaders.base_loader import Loader, LoaderType
from fractal.loaders.morpho import _validate_market_id, get_market_info
from fractal.loaders.rpc import JsonRpcClient, decode_words, function_selector
from fractal.loaders.structs import PriceHistory

__all__ = ["MorphoOraclePriceLoader", "calibrate_oracle_price"]

_PRICE_SELECTOR = function_selector("price()")


def _utc(value: str | datetime | pd.Timestamp) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


class MorphoOraclePriceLoader(Loader):
    """``price()`` of one Morpho Blue market's oracle per bar (see the module docs).

    Args:
        rpc_url: archive JSON-RPC endpoint of the market's chain. A keyed URL
            is redacted from errors and logs and is never part of the cache key.
        market_id: 32-byte market id (``0x…``).
        start_time: first bar (naive = UTC).
        end_time: last bar, inclusive. Bars after the chain tip are dropped.
        freq: bar length, a fixed ``pd.Timedelta`` string (``"1D"``, ``"1h"``).
        chain_id: EVM chain id of the market (Morpho API lookup).
        loader_type: cache format.
        http: injectable transport for the node and the Morpho API (offline tests).
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        rpc_url: str,
        market_id: str,
        start_time: str | datetime | pd.Timestamp,
        end_time: str | datetime | pd.Timestamp,
        freq: str = "1D",
        *,
        chain_id: int = 1,
        loader_type: LoaderType = LoaderType.CSV,
        http: HttpClient | None = None,
    ) -> None:
        super().__init__(loader_type=loader_type)
        self.market_id = _validate_market_id(market_id)
        self.chain_id = int(chain_id)
        self.start_time, self.end_time = _utc(start_time), _utc(end_time)
        if self.end_time < self.start_time:
            raise ValueError("end_time is before start_time")
        self.freq = freq
        self.bar_length = pd.Timedelta(freq)
        if self.bar_length <= pd.Timedelta(0):
            raise ValueError(f"freq must be a positive fixed duration, got {freq!r}")
        self._http = http
        self._rpc = JsonRpcClient(rpc_url, http=http)

    def __repr__(self) -> str:
        return (f"MorphoOraclePriceLoader(market_id={self.market_id!r}, chain_id={self.chain_id}, "
                f"start={self.start_time}, end={self.end_time}, freq={self.freq!r})")

    # ------------------------------------------------------------ pipeline
    def _cache_key(self) -> str:
        return (f"{self.chain_id}-{self.market_id}-oracle-{int(self.start_time.timestamp())}-"
                f"{int(self.end_time.timestamp())}-{int(self.bar_length.total_seconds())}s")

    def extract(self) -> None:
        info = get_market_info(self.market_id, self.chain_id, http=self._http)
        if not info.oracle_address:
            raise ValueError(f"market {self.market_id} on chain {self.chain_id} has no oracle address")
        tip = self._rpc.block_number()
        tip_time = self._rpc.block_timestamp(tip)
        rows: list[dict[str, Any]] = []
        low = 0
        for bar_time in pd.date_range(self.start_time, self.end_time, freq=self.bar_length):
            bar_end = int(bar_time.timestamp())
            if bar_end > tip_time:  # the chain has not reached this bar yet
                break
            try:
                block = self._rpc.block_at_timestamp(bar_end, closest="before", low=low, high=tip)
            except ValueError:  # before the first block
                continue
            low = block
            result = self._rpc.eth_call(info.oracle_address, _PRICE_SELECTOR, block)
            if result == "0x":
                raise ValueError(
                    f"oracle {info.oracle_address} returned no data at block {block}: the bar precedes "
                    "the oracle's deployment; start the window later"
                )
            rows.append({"time": bar_end, "block": block, "price": decode_words(result, 1)[0] / info.price_scale})
        self._data = pd.DataFrame(rows)

    def transform(self) -> None:
        cols = ["time", "block", "price"]
        if self._data is None or self._data.empty:
            self._data = pd.DataFrame(columns=cols)
            return
        self._data = self._data[cols].sort_values("time").reset_index(drop=True)

    def read(self, with_run: bool = False) -> PriceHistory:
        if with_run:
            self.run()
        else:
            self._read(self._cache_key())
        if self._data is None or self._data.empty:
            return PriceHistory(prices=[], time=[])
        return PriceHistory(prices=self._data["price"].to_numpy(float), time=self._utc_index("time"))


def calibrate_oracle_price(model: pd.Series, oracle: pd.Series) -> pd.Series:
    """Scale a modelled oracle price to exact oracle reads: ``model × κ``.

    Args:
        model: modelled oracle price on the backtest grid (UTC ``DatetimeIndex``),
            e.g. a Pendle feed formula evaluated per bar.
        oracle: exact ``price()`` reads (e.g. daily, from
            :class:`MorphoOraclePriceLoader`); every read time must be on the
            model grid.

    Returns:
        ``model × κ`` on the model grid, where ``κ = oracle / model`` at the
        reads, ``ln κ`` is linear in time between reads and held at the edge
        value before the first and after the last read.
    """
    oracle = oracle.sort_index()
    if oracle.empty:
        raise ValueError("calibration needs at least one oracle read")
    off_grid = oracle.index.difference(model.index)
    if len(off_grid):
        raise ValueError(f"{len(off_grid)} oracle read(s) are not on the model grid, first at {off_grid[0]}")
    ratio = oracle.to_numpy(float) / model.loc[oracle.index].to_numpy(float)
    if not np.all(np.isfinite(ratio) & (ratio > 0)):
        raise ValueError("oracle reads and model values must be positive and finite at every read")
    # same unit on both axes: the two indexes may carry different datetime resolutions
    ln_kappa = np.interp(model.index.as_unit("ns").asi8, oracle.index.as_unit("ns").asi8, np.log(ratio))
    return pd.Series(model.to_numpy(float) * np.exp(ln_kappa), index=model.index, name=model.name)
