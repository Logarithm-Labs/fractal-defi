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

:func:`cheapest_fee_tier` merges this frame with the DefiLlama reader's
(``DefiLlamaDexFeeTierLoader``, same columns): on-chain
rows replace DefiLlama's for the projects read on chain
(:attr:`OnchainDexFeeTierReader.covered_projects`), DefiLlama supplies the
other projects, and the cheapest eligible pool is kept per pair.
"""
from collections.abc import Collection, Iterable, Mapping
from typing import ClassVar

import numpy as np
import pandas as pd

from fractal.loaders._http import HttpClient
from fractal.loaders.rpc import BlockTag, JsonRpcClient, decode_address, decode_words, encode_address
from fractal.loaders.rpc import function_selector as _sel
from fractal.loaders.thegraph.base_graph_loader import validate_evm_address

__all__ = ["UNISWAP_V3_FEE_TIERS", "OnchainDexFeeTierReader", "cheapest_fee_tier", "merge_fee_tiers"]

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

    @property
    def covered_projects(self) -> frozenset[str]:
        """Projects this reader reads on chain (whether or not a pool exists)."""
        return frozenset({"uniswap-v3"} if self.factory else set()) | frozenset(
            {"curve-dex"} if self.curve_pools else set())

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


def merge_fee_tiers(
    defillama: pd.DataFrame | None,
    onchain: pd.DataFrame | None,
    *,
    covered_projects: Collection[str] | None = None,
    min_tvl_usd: float | None = None,
) -> pd.DataFrame:
    """Candidate pools of one or more pairs from both fee-tier sources.

    On-chain rows come first; DefiLlama rows are kept only for projects not
    in ``covered_projects`` (default: the projects present in ``onchain``;
    pass :attr:`OnchainDexFeeTierReader.covered_projects` so a project read
    on chain without a pool also hides its DefiLlama rows). With
    ``min_tvl_usd``, ``eligible`` is recomputed for both as a known fee and
    ``tvl_usd >= min_tvl_usd`` (an unknown TVL is not eligible); otherwise
    each source's ``eligible`` is kept. Extra columns (``chain``, ``pair``,
    ...) pass through.
    """
    frames = [f for f in (onchain, defillama) if f is not None]
    onchain = onchain if onchain is not None else pd.DataFrame(columns=OnchainDexFeeTierReader.COLUMNS)
    covered = set(onchain["project"]) if covered_projects is None else set(covered_projects)
    if defillama is not None:
        frames[-1] = defillama[~defillama["project"].isin(covered)]
    columns = list(dict.fromkeys(OnchainDexFeeTierReader.COLUMNS + [c for f in frames for c in f.columns]))
    frames = [f.reindex(columns=columns) for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(columns=columns)
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.astype({"fee_pct": "float64", "tvl_usd": "float64"})
    if min_tvl_usd is not None:
        merged["eligible"] = merged["fee_pct"].notna() & (merged["tvl_usd"] >= float(min_tvl_usd))
    merged["eligible"] = merged["eligible"].eq(True)  # a missing flag is not eligible
    return merged


def cheapest_fee_tier(
    defillama: pd.DataFrame | None,
    onchain: pd.DataFrame | None,
    *,
    covered_projects: Collection[str] | None = None,
    min_tvl_usd: float | None = None,
    by: str | list[str] | None = None,
) -> pd.DataFrame:
    """Cheapest eligible pool per pair from the DefiLlama and on-chain frames.

    Candidates are :func:`merge_fee_tiers`; among eligible ones the lowest
    ``fee_pct`` wins, ties to the larger ``tvl_usd`` (then on-chain first).
    ``by`` names the columns identifying a pair (``None``: the frames hold
    one pair). One row per pair that has an eligible pool; pairs without
    one are left out.
    """
    merged = merge_fee_tiers(defillama, onchain, covered_projects=covered_projects, min_tvl_usd=min_tvl_usd)
    live = merged[merged["eligible"]]
    if live.empty:
        return live.reset_index(drop=True)
    ranked = live.sort_values(["fee_pct", "tvl_usd"], ascending=[True, False], kind="stable", na_position="last")
    keys = [by] if isinstance(by, str) else list(by or [])
    best = ranked.groupby(keys, sort=True, dropna=False).head(1) if keys else ranked.head(1)
    if keys:
        best = best.sort_values(keys, kind="stable")
    return best.reset_index(drop=True)
