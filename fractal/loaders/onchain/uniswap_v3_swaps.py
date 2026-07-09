"""Uniswap V3 Swap-event loader (JSON-RPC ``eth_getLogs``).

Raw per-swap records for a V3-style pool: signed amounts, the pool's
active liquidity during the swap, post-swap tick. Works for any fork
emitting the canonical ``Swap`` event (Uniswap V3, Slipstream, ...).
Needs an RPC with ``eth_getLogs`` over historic ranges (no archive
state access).
"""
from typing import List, Optional

import pandas as pd

from fractal.loaders._http import HttpClient, LoaderHttpError
from fractal.loaders.base_loader import Loader, LoaderType
from fractal.loaders.structs import SwapsHistory
from fractal.loaders.thegraph.base_graph_loader import validate_evm_address


class RpcLoaderException(RuntimeError):
    pass


class UniswapV3SwapsLoader(Loader):
    """Per-swap events of one pool over a block range.

    Args:
        rpc_url (str): JSON-RPC endpoint (``eth_getLogs`` capable).
        pool (str): pool address.
        token0_decimals (int): token0 decimals (amounts scale to human).
        token1_decimals (int): token1 decimals.
        start_block (int): first block of the range (e.g. a mint block).
        end_block (Optional[int]): last block; latest when omitted.
        chunk_size (int): blocks per ``eth_getLogs`` call; halved
            automatically when the node rejects the range.
        loader_type (LoaderType): cache format.
    """

    SWAP_TOPIC = "0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67"

    def __init__(
        self,
        rpc_url: str,
        pool: str,
        token0_decimals: int,
        token1_decimals: int,
        start_block: int,
        end_block: Optional[int] = None,
        chunk_size: int = 10_000,
        loader_type: LoaderType = LoaderType.CSV,
        http: Optional[HttpClient] = None,
    ) -> None:
        super().__init__(loader_type=loader_type)
        self._rpc_url = rpc_url
        self.pool = validate_evm_address(pool, field="pool")
        self.token0_decimals = token0_decimals
        self.token1_decimals = token1_decimals
        self.start_block = start_block
        self.end_block = end_block
        self.chunk_size = chunk_size
        self._http = http or HttpClient()
        self._rpc_id = 0

    # ----------------------------------------------------------- transport
    def _rpc(self, method: str, params: list):
        self._rpc_id += 1
        payload = self._http.post(self._rpc_url, json={
            "jsonrpc": "2.0", "id": self._rpc_id, "method": method, "params": params,
        })
        if not isinstance(payload, dict) or "error" in payload:
            raise RpcLoaderException(f"RPC {method} failed: {payload!r}")
        return payload["result"]

    def _block_timestamp(self, number: int) -> int:
        block = self._rpc("eth_getBlockByNumber", [hex(number), False])
        return int(block["timestamp"], 16)

    # -------------------------------------------------------------- decode
    @staticmethod
    def _as_int(value: int, bits: int = 256) -> int:
        return value - (1 << bits) if value >= (1 << (bits - 1)) else value

    @classmethod
    def _decode_log(cls, log: dict) -> dict:
        """Decode one Swap log: (amount0 int256, amount1 int256,
        sqrtPriceX96 uint160, liquidity uint128, tick int24)."""
        data = log["data"]
        words = [int(data[2 + 64 * i: 2 + 64 * (i + 1)], 16) for i in range(5)]
        return {
            "amount0": cls._as_int(words[0]),
            "amount1": cls._as_int(words[1]),
            "sqrt_price_x96": words[2],
            "liquidity": words[3],
            "tick": cls._as_int(words[4]),
            "block": int(log["blockNumber"], 16),
            "log_index": int(log["logIndex"], 16),
        }

    # ------------------------------------------------------------ pipeline
    def _cache_key(self) -> str:
        end = self.end_block if self.end_block is not None else "latest"
        return f"{self.pool}-swaps-{self.start_block}-{end}"

    def extract(self) -> None:
        end_block = self.end_block
        if end_block is None:
            end_block = int(self._rpc("eth_blockNumber", []), 16)
        rows: List[dict] = []
        start, chunk = self.start_block, self.chunk_size
        while start <= end_block:
            end = min(start + chunk - 1, end_block)
            try:
                logs = self._rpc("eth_getLogs", [{
                    "address": self.pool,
                    "topics": [self.SWAP_TOPIC],
                    "fromBlock": hex(start),
                    "toBlock": hex(end),
                }])
            except (RpcLoaderException, LoaderHttpError):
                # Nodes reject oversized ranges either as a JSON-RPC
                # error or at the HTTP level — halve and retry the same
                # range in both cases.
                if chunk <= 500:
                    raise
                chunk //= 2
                continue
            rows.extend(self._decode_log(log) for log in logs)
            start = end + 1
            # Recover after transient rejections so one bad range does
            # not permanently multiply the number of calls.
            chunk = min(chunk * 2, self.chunk_size)
        self._extracted_range = (self.start_block, end_block)
        self._data = pd.DataFrame(rows)

    def transform(self) -> None:
        cols = ["time", "amount0", "amount1", "liquidity", "tick", "price", "block", "log_index"]
        if self._data is None or self._data.empty:
            self._data = pd.DataFrame(columns=cols)
            return
        df = self._data.sort_values(["block", "log_index"]).reset_index(drop=True)
        df["amount0"] = df["amount0"].astype(float) / 10 ** self.token0_decimals
        df["amount1"] = df["amount1"].astype(float) / 10 ** self.token1_decimals
        # token0-in-token1 price, human units, from the post-swap sqrt price
        sqrt_p = df["sqrt_price_x96"].astype(float) / 2 ** 96
        df["price"] = sqrt_p ** 2 * 10 ** (self.token0_decimals - self.token1_decimals)
        # Block -> timestamp: linear between range boundaries (exact on
        # fixed-block-time chains like Base).
        b0, b1 = self._extracted_range
        t0 = self._block_timestamp(b0)
        t1 = self._block_timestamp(b1) if b1 > b0 else t0
        rate = (t1 - t0) / (b1 - b0) if b1 > b0 else 0.0
        df["time"] = pd.to_datetime(
            (t0 + (df["block"] - b0) * rate).round().astype(int), unit="s", utc=True)
        self._data = df[cols]

    def read(self, with_run: bool = False) -> SwapsHistory:
        if with_run:
            self.run()
        else:
            self._read(self._cache_key())
        if self._data is None or self._data.empty:
            return SwapsHistory(amount0=[], amount1=[], liquidity=[], ticks=[],
                                prices=[], blocks=[], log_indexes=[], time=[])
        return SwapsHistory(
            amount0=self._data["amount0"].astype(float).values,
            amount1=self._data["amount1"].astype(float).values,
            liquidity=self._data["liquidity"].astype(float).values,
            ticks=self._data["tick"].astype(int).values,
            prices=self._data["price"].astype(float).values,
            blocks=self._data["block"].astype(int).values,
            log_indexes=self._data["log_index"].astype(int).values,
            time=pd.to_datetime(self._data["time"], utc=True).values,
        )
