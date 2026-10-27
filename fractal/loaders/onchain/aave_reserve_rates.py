"""Aave V3 reserve rates read on chain (``Pool.getReserveData``).

Works for Aave V3 and forks that keep its ``getReserveData(address)``
return layout (Spark, for example), on any chain, without a subgraph or
the Aave API: one ``eth_call`` per bar at the last block at or before the
bar time. Historical blocks need an archive RPC; pass its URL (for
example from an environment variable). A key embedded in the URL is kept
out of errors and logs by :class:`~fractal.loaders.rpc.JsonRpcClient`.

Layout read (``ReserveDataLegacy``, Aave V3.0-3.x, 15 words)::

    0 configuration        1 liquidityIndex (ray)    2 currentLiquidityRate (ray APR)
    3 variableBorrowIndex  4 currentVariableBorrowRate (ray APR)
    5 currentStableBorrowRate  6 lastUpdateTimestamp  7 id
    8 aTokenAddress        9 stableDebtTokenAddress  10 variableDebtTokenAddress
    11 interestRateStrategyAddress 12 accruedToTreasury 13 unbacked
    14 isolationModeTotalDebt

Aave V2 orders the first words differently and is not supported.
"""
import math
from dataclasses import dataclass
from datetime import datetime
from typing import ClassVar

import numpy as np
import pandas as pd

from fractal.loaders._dt import SECONDS_PER_YEAR, to_seconds, to_utc
from fractal.loaders._http import HttpClient
from fractal.loaders.base_loader import Loader, LoaderType
from fractal.loaders.rpc import JsonRpcClient, decode_address, decode_words, encode_address, function_selector
from fractal.loaders.structs import LendingHistory
from fractal.loaders.thegraph.base_graph_loader import validate_evm_address

__all__ = ["AaveV3ReserveRatesLoader", "ReserveData", "decode_reserve_data", "ray_apr_to_apy"]

RAY = 10 ** 27
GET_RESERVE_DATA = function_selector("getReserveData(address)")  # 0x35ea6a75
_RESERVE_WORDS = 15


def ray_apr_to_apy(rate_ray: int | float) -> float:
    """Annual rate in ray (the on-chain APR) → APY under continuous compounding, ``expm1(apr)``.

    Aave compounds borrow interest per second, which ``expm1`` matches to
    well under a basis point at any realistic rate.
    """
    return math.expm1(float(rate_ray) / RAY)


@dataclass(frozen=True)
class ReserveData:
    """Decoded ``getReserveData`` result; rates are ray APRs, indexes rays."""

    configuration: int
    liquidity_index: int
    liquidity_rate: int
    variable_borrow_index: int
    variable_borrow_rate: int
    last_update_timestamp: int
    a_token: str
    variable_debt_token: str
    interest_rate_strategy: str

    @property
    def supply_apy(self) -> float:
        return ray_apr_to_apy(self.liquidity_rate)

    @property
    def borrow_apy(self) -> float:
        return ray_apr_to_apy(self.variable_borrow_rate)


def decode_reserve_data(hex_result: str) -> ReserveData:
    """Decode the raw ``getReserveData`` return of an Aave V3-layout pool."""
    w = decode_words(hex_result, _RESERVE_WORDS)
    return ReserveData(
        configuration=w[0],
        liquidity_index=w[1],
        liquidity_rate=w[2],
        variable_borrow_index=w[3],
        variable_borrow_rate=w[4],
        last_update_timestamp=w[6],
        a_token=decode_address(w[8]),
        variable_debt_token=decode_address(w[10]),
        interest_rate_strategy=decode_address(w[11]),
    )


class AaveV3ReserveRatesLoader(Loader):
    """Supply and variable-borrow rates of one reserve on a regular time grid.

    Each bar ``t`` in ``[start_time, end_time]`` (step ``step_seconds``)
    reads the reserve at the last block with time ``<= t``; bars after the
    chain tip are dropped, and a bar before the pool existed fails to
    decode (start after deployment). ``read``
    returns a :class:`~fractal.loaders.structs.LendingHistory` whose
    ``lending_rate`` / ``borrowing_rate`` are per-bar continuous exponents
    ``apr · step / YEAR`` (the convention of the Morpho loaders), with the
    annual ``supply_apy`` / ``borrow_apy`` alongside.

    Args:
        rpc_url: archive JSON-RPC endpoint (never cached or logged).
        pool: the ``Pool`` proxy (Aave V3, Spark, ...).
        asset: underlying reserve asset.
        start_time / end_time: inclusive window (naive = UTC).
        step_seconds: bar spacing; defaults to one hour.
        loader_type: cache format.
        http: injectable transport (offline tests pass a fake).
    """

    _COLUMNS: ClassVar[list[str]] = ["time", "block", "supply_apr", "borrow_apr", "last_update"]

    def __init__(
        self,
        rpc_url: str,
        pool: str,
        asset: str,
        start_time: datetime,
        end_time: datetime,
        step_seconds: int = 3_600,
        loader_type: LoaderType = LoaderType.CSV,
        http: HttpClient | None = None,
    ) -> None:
        super().__init__(loader_type=loader_type)
        self.pool = validate_evm_address(pool, field="pool")
        self.asset = validate_evm_address(asset, field="asset")
        self.start_time = to_utc(start_time)
        self.end_time = to_utc(end_time)
        if self.end_time < self.start_time:
            raise ValueError("end_time must not precede start_time")
        if int(step_seconds) <= 0:
            raise ValueError("step_seconds must be positive")
        self.step_seconds = int(step_seconds)
        self._rpc = JsonRpcClient(rpc_url, http=http)

    def _cache_key(self) -> str:
        return (f"{self.pool}-{self.asset}-{to_seconds(self.start_time)}-{to_seconds(self.end_time)}"
                f"-{self.step_seconds}")

    def reserve_data(self, block: int | str = "latest") -> ReserveData:
        """Decoded reserve state at ``block``."""
        data = GET_RESERVE_DATA + encode_address(self.asset)
        return decode_reserve_data(self._rpc.eth_call(self.pool, data, block=block))

    def extract(self) -> None:
        start, end = to_seconds(self.start_time), to_seconds(self.end_time)
        high = self._rpc.block_number()
        end = min(end, self._rpc.block_timestamp(high))  # bars past the chain tip are not observable yet
        rows, low = [], 0
        for ts in range(start, end + 1, self.step_seconds):
            block = self._rpc.block_at_timestamp(ts, closest="before", low=low, high=high)
            low = block  # the grid increases, so later bars sit at or after this block
            reserve = self.reserve_data(block)
            rows.append({
                "time": ts,
                "block": block,
                # Ray integers exceed int64: divide as Python ints, store floats.
                "supply_apr": reserve.liquidity_rate / RAY,
                "borrow_apr": reserve.variable_borrow_rate / RAY,
                "last_update": reserve.last_update_timestamp,
            })
        self._data = pd.DataFrame(rows, columns=self._COLUMNS)

    def transform(self) -> None:
        if self._data is None or self._data.empty:
            self._data = pd.DataFrame(columns=self._COLUMNS)
            return
        self._data = self._data[self._COLUMNS].sort_values("time").reset_index(drop=True)

    def read(self, with_run: bool = False) -> LendingHistory:
        if with_run:
            self.run()
        else:
            self._read(self._cache_key())
        if self._data is None or self._data.empty:
            return LendingHistory(lending_rates=[], borrowing_rates=[], time=[])
        supply_apr = self._data["supply_apr"].astype(float).to_numpy()
        borrow_apr = self._data["borrow_apr"].astype(float).to_numpy()
        per_bar = self.step_seconds / SECONDS_PER_YEAR
        return LendingHistory(
            lending_rates=supply_apr * per_bar,
            borrowing_rates=borrow_apr * per_bar,
            time=self._utc_index("time"),
            supply_apy=np.expm1(supply_apr),
            borrow_apy=np.expm1(borrow_apr),
        )
