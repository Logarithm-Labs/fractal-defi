"""Uniswap V3 Swap-event loader (JSON-RPC ``eth_getLogs``).

Raw per-swap records for a V3-style pool: signed amounts, the pool's
active liquidity during the swap, post-swap tick. Works for any fork
emitting the canonical ``Swap`` event (Uniswap V3, Slipstream, ...).
Needs an RPC with ``eth_getLogs`` over historic ranges (no archive
state access). Transport and decoding come from :mod:`fractal.loaders.rpc`.
"""
from typing import List, Optional

import pandas as pd

from fractal.loaders._http import HttpClient
from fractal.loaders.base_loader import Loader, LoaderType
from fractal.loaders.rpc import JsonRpcClient, RpcCallError, as_signed, decode_words, event_topic
from fractal.loaders.structs import SwapsHistory
from fractal.loaders.thegraph.base_graph_loader import validate_evm_address

RpcLoaderException = RpcCallError  # kept for callers that catch the old name


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

    SWAP_TOPIC = event_topic("Swap(address,address,int256,int256,uint160,uint128,int24)")

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
        self._rpc = JsonRpcClient(rpc_url, http=http)

    # -------------------------------------------------------------- decode
    @staticmethod
    def _decode_log(log: dict) -> dict:
        """Decode one Swap log: (amount0 int256, amount1 int256,
        sqrtPriceX96 uint160, liquidity uint128, tick int24)."""
        words = decode_words(log["data"], 5)
        return {
            "amount0": as_signed(words[0]),
            "amount1": as_signed(words[1]),
            "sqrt_price_x96": words[2],
            "liquidity": words[3],
            "tick": as_signed(words[4]),
            "block": int(log["blockNumber"], 16),
            "log_index": int(log["logIndex"], 16),
        }

    # ------------------------------------------------------------ pipeline
    def _cache_key(self) -> str:
        end = self.end_block if self.end_block is not None else "latest"
        return f"{self.pool}-swaps-{self.start_block}-{end}"

    def extract(self) -> None:
        end_block = self.end_block if self.end_block is not None else self._rpc.block_number()
        logs = self._rpc.iter_logs(self.pool, [self.SWAP_TOPIC], self.start_block, end_block,
                                   chunk_size=self.chunk_size)
        rows: List[dict] = [self._decode_log(log) for log in logs]
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
        t0 = self._rpc.block_timestamp(b0)
        t1 = self._rpc.block_timestamp(b1) if b1 > b0 else t0
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
