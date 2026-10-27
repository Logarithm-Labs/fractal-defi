"""Exact DEX fee tiers for a token pair, read on chain.

:class:`OnchainDexFeeTierReader` is the on-chain counterpart of the
DefiLlama ``poolMeta`` parser: instead of a free-text label it reads

* **Uniswap V3** (and forks with the same factory): ``getPool(a, b, fee)``
  on the factory for each fee tier, keeping the pools that exist, plus the
  pool's active ``liquidity()``;
* **Curve**: ``fee()`` of caller-listed pools (the base fee, ``1e10``
  denominator; stableswap-ng off-peg multipliers can raise the fee actually
  charged).

TVL is the pool's balance of the two pair tokens priced with the
caller's ``token_prices_usd`` (no price fetching here); without prices it
is ``NaN``. Like the DefiLlama reader, the result is the fee tier only, a
lower bound on swap cost (no price impact, hooks or gas).

``read`` returns the DefiLlama fee-tier reader's columns (``pool``,
``project``, ``symbol``, ``pool_meta``, ``fee_pct``, ``fee_resolution_pct``,
``fee_source``, ``tvl_usd``, ``eligible``) plus ``in_range_liquidity``
(Uniswap V3 only), eligible and cheapest first.
"""
from collections.abc import Iterable, Mapping
from typing import ClassVar

import numpy as np
import pandas as pd

from fractal.loaders._http import HttpClient
from fractal.loaders.rpc import BlockTag, JsonRpcClient, decode_address, decode_words, encode_address
from fractal.loaders.rpc import function_selector as _sel
from fractal.loaders.thegraph.base_graph_loader import validate_evm_address

__all__ = ["UNISWAP_V3_FEE_TIERS", "OnchainDexFeeTierReader"]

UNISWAP_V3_FEE_TIERS = (100, 500, 3_000, 10_000)  # hundredths of a basis point
_GET_POOL = _sel("getPool(address,address,uint24)")
_FEE = _sel("fee()")
_LIQUIDITY = _sel("liquidity()")
_BALANCE_OF = _sel("balanceOf(address)")
_DECIMALS = _sel("decimals()")


class OnchainDexFeeTierReader:
    """Fee tiers of a token pair's Uniswap V3 and Curve pools at one block.

    Args:
        rpc_url: JSON-RPC endpoint (archive when ``block`` is historical).
        token_a / token_b: the pair.
        uniswap_v3_factory: factory address on this chain; ``None`` skips V3.
        fee_tiers: V3 tiers to probe.
        curve_pools: Curve pool addresses that hold the pair.
        token_prices_usd: USD price per token address, for TVL.
        min_tvl_usd: TVL threshold for ``eligible`` (ignored when ``0``).
        block: block to read at.
        http: injectable transport (offline tests pass a fake).
    """

    COLUMNS: ClassVar[list[str]] = [
        "pool", "project", "symbol", "pool_meta", "fee_pct", "fee_resolution_pct", "fee_source", "tvl_usd",
        "eligible", "in_range_liquidity",
    ]

    def __init__(
        self,
        rpc_url: str,
        token_a: str,
        token_b: str,
        *,
        uniswap_v3_factory: str | None = None,
        fee_tiers: Iterable[int] = UNISWAP_V3_FEE_TIERS,
        curve_pools: Iterable[str] = (),
        token_prices_usd: Mapping[str, float] | None = None,
        min_tvl_usd: float = 0.0,
        block: BlockTag = "latest",
        http: HttpClient | None = None,
    ) -> None:
        self.tokens = (validate_evm_address(token_a, field="token_a"), validate_evm_address(token_b, field="token_b"))
        self.factory = validate_evm_address(uniswap_v3_factory, field="uniswap_v3_factory") \
            if uniswap_v3_factory else None
        self.fee_tiers = tuple(int(t) for t in fee_tiers)
        self.curve_pools = [validate_evm_address(p, field="curve_pools") for p in curve_pools]
        self.prices = {k.lower(): float(v) for k, v in (token_prices_usd or {}).items()}
        self.min_tvl_usd = float(min_tvl_usd)
        self.block = block
        self._rpc = JsonRpcClient(rpc_url, http=http)
        self._decimals: dict[str, int] = {}

    def _word(self, to: str, data: str) -> int:
        return decode_words(self._rpc.eth_call(to, data, block=self.block), 1)[0]

    def _tvl_usd(self, pool: str) -> float:
        total = 0.0
        for token in self.tokens:
            if token not in self.prices:
                return np.nan
            if token not in self._decimals:
                self._decimals[token] = self._word(token, _DECIMALS)
            balance = self._word(token, _BALANCE_OF + encode_address(pool))
            total += balance / 10 ** self._decimals[token] * self.prices[token]
        return total

    def _uniswap_v3(self) -> list[dict]:
        rows = []
        a, b = (encode_address(t) for t in self.tokens)
        for tier in self.fee_tiers:
            pool = decode_address(self._word(self.factory, _GET_POOL + a + b + format(tier, "064x")))
            if int(pool, 16) == 0:
                continue
            rows.append({"pool": pool, "project": "uniswap-v3", "pool_meta": f"fee tier {tier}",
                         "fee_pct": tier / 1e4, "tvl_usd": self._tvl_usd(pool),
                         "in_range_liquidity": float(self._word(pool, _LIQUIDITY))})
        return rows

    def _curve(self) -> list[dict]:
        return [{"pool": pool, "project": "curve-dex", "pool_meta": "fee()", "fee_pct": self._word(pool, _FEE) / 1e8,
                 "tvl_usd": self._tvl_usd(pool), "in_range_liquidity": np.nan} for pool in self.curve_pools]

    def read(self) -> pd.DataFrame:
        """Every pool found, eligible and cheapest first (ties to the larger TVL)."""
        rows = (self._uniswap_v3() if self.factory else []) + self._curve()
        frame = pd.DataFrame(rows, columns=["pool", "project", "pool_meta", "fee_pct", "tvl_usd",
                                            "in_range_liquidity"])
        frame["symbol"] = ""
        frame["fee_resolution_pct"] = 0.0
        frame["fee_source"] = "onchain"
        frame = frame.astype({"fee_pct": "float64", "tvl_usd": "float64", "in_range_liquidity": "float64"})
        tvl_ok = (frame["tvl_usd"] >= self.min_tvl_usd) if self.min_tvl_usd > 0 else True
        frame["eligible"] = frame["fee_pct"].notna() & tvl_ok
        frame = frame.sort_values(["eligible", "fee_pct", "tvl_usd"], ascending=[False, True, False], kind="stable")
        return frame[self.COLUMNS].reset_index(drop=True)

    def cheapest(self) -> pd.Series | None:
        """Cheapest eligible pool, or ``None``."""
        frame = self.read()
        if frame.empty or not bool(frame["eligible"].iloc[0]):
            return None
        return frame.iloc[0]
