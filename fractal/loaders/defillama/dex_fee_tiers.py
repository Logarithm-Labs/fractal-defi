"""Cheapest published DEX fee tier for a token pair, from DefiLlama pools.

:class:`DefiLlamaDexFeeTierLoader` lists the DefiLlama yield pools on one
chain whose underlying tokens are exactly the requested pair (for example
two stablecoins) and parses each pool's fee from its free-text
``poolMeta`` label with :func:`parse_pool_fee_pct`. It reads the keyless
``yields.llama.fi/pools`` snapshot through the shared DefiLlama loader
plumbing.

``poolMeta`` is free text maintained per adapter, not a fee field, so the
parse is deliberately conservative. Only these shapes yield a fee:

* one percentage, e.g. ``"0.05%"``;
* a percentage labelled as the fee, e.g. ``"0.01% fee | CL range 0.09%"``
  or ``"fee 0.3%"``;
* one basis-point value, e.g. ``"5 bps"``.

Anything else (missing or non-string labels, ``"Dynamic fee (hook)"``,
several unlabelled percentages such as weighted-pool shares, values outside
``[0, 100)``) gives ``NaN``, never an exception. Projects with a protocol-wide
fixed fee and no label (``uniswap-v2``) can be filled from a constant map.

The result is the fee tier only: it excludes price impact, hook or dynamic
fees and gas, so it is a lower bound on the cost of a swap.
"""
import math
import re
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from fractal.loaders.base_loader import LoaderType
from fractal.loaders.defillama.defillama import DefiLlamaBaseLoader

YIELDS_BASE_URL = "https://yields.llama.fi"

# DefiLlama project slugs of spot DEXes whose pools carry a swap fee.
DEFAULT_DEX_PROJECTS = frozenset(
    {
        "aerodrome-slipstream",
        "aerodrome-v1",
        "balancer-v3",
        "camelot-v3",
        "curve-dex",
        "ekubo",
        "fluid-dex",
        "maverick-v2",
        "pancakeswap-amm-v3",
        "solidly-v3",
        "sushiswap-v3",
        "uniswap-v2",
        "uniswap-v3",
        "uniswap-v4",
    }
)
# Protocol-wide fixed fees (percent) for projects whose pools carry no fee label.
DEFAULT_FIXED_FEE_PCT: Mapping[str, float] = {"uniswap-v2": 0.30}

_NUMBER = r"(\d+(?:\.(\d+))?)"
_PERCENT = re.compile(_NUMBER + r"\s*%")
_LABELLED = (
    re.compile(_NUMBER + r"\s*%\s*fee\b", re.IGNORECASE),
    re.compile(r"\bfee\s*:?\s*" + _NUMBER + r"\s*%", re.IGNORECASE),
)
_BPS = re.compile(_NUMBER + r"\s*bps?\b", re.IGNORECASE)
_COLUMNS = ["pool", "project", "symbol", "pool_meta", "tvl_usd"]
_READ_COLUMNS = [
    "pool",
    "project",
    "symbol",
    "pool_meta",
    "fee_pct",
    "fee_resolution_pct",
    "fee_source",
    "tvl_usd",
    "eligible",
]


def _fee_from_match(match: re.Match, scale: float = 1.0) -> tuple[float, float]:
    decimals = len(match.group(2) or "")
    return float(match.group(1)) * scale, 0.5 * 10 ** (-decimals) * scale


def parse_pool_fee_pct(meta: Any) -> tuple[float, float]:
    """Fee percent and its display resolution from a DefiLlama ``poolMeta``.

    Returns ``(fee_pct, resolution_pct)``: ``"0.05%"`` gives ``(0.05, 0.005)``
    (half a unit of the last shown digit), ``"5 bps"`` gives ``(0.05, 0.005)``.
    Unknown, dynamic, ambiguous or out-of-range labels give ``(nan, nan)``;
    the function never raises.
    """
    nan = (math.nan, math.nan)
    if not isinstance(meta, str) or not meta.strip() or "dynamic" in meta.lower():
        return nan
    fee = nan
    for pattern in _LABELLED:
        labelled = pattern.findall(meta)
        if len(labelled) == 1:
            fee = _fee_from_match(pattern.search(meta))
            break
    else:
        percents = list(_PERCENT.finditer(meta))
        bps = list(_BPS.finditer(meta))
        if len(percents) == 1 and not bps:
            fee = _fee_from_match(percents[0])
        elif len(bps) == 1 and not percents:
            fee = _fee_from_match(bps[0], scale=0.01)
    if not (math.isfinite(fee[0]) and 0.0 <= fee[0] < 100.0):
        return nan
    return fee


class DefiLlamaDexFeeTierLoader(DefiLlamaBaseLoader):
    """DEX pools for one exact token set on one chain, cheapest fee first.

    Args:
        tokens: Underlying token addresses of the pool (at least two). Only
            pools whose ``underlyingTokens`` equal this set match, so a
            three-coin pool does not match a two-token request.
        chain: DefiLlama chain name, e.g. ``"Ethereum"`` or ``"Base"``
            (compared case-insensitively).
        projects: DefiLlama DEX project slugs to keep.
        min_tvl_usd: TVL a pool needs to be ``eligible``.
        fixed_fee_pct: Fee (percent) for projects whose pools carry no label.
        loader_type: Cache serialization format.
        base_url: DefiLlama yields host.

    ``read`` returns one row per matching pool with ``pool`` (DefiLlama pool
    id), ``project``, ``symbol``, ``pool_meta``, ``fee_pct`` and
    ``fee_resolution_pct`` (``NaN`` when not parseable), ``fee_source``
    (``"pool_meta"``, ``"protocol_constant"`` or ``"unknown"``), ``tvl_usd``
    and ``eligible`` (known fee and TVL at least ``min_tvl_usd``). Eligible
    rows come first, cheapest fee and then largest TVL first, so
    :meth:`cheapest` is the first row. The endpoint is a live snapshot: the
    cache holds the last fetched snapshot and is replaced by
    ``read(with_run=True)``.
    """

    def __init__(
        self,
        tokens: Iterable[str],
        chain: str,
        projects: Iterable[str] = DEFAULT_DEX_PROJECTS,
        min_tvl_usd: float = 100_000.0,
        fixed_fee_pct: Mapping[str, float] | None = None,
        loader_type: LoaderType = LoaderType.CSV,
        base_url: str = YIELDS_BASE_URL,
    ) -> None:
        super().__init__(loader_type=loader_type, base_url=base_url)
        self.tokens = frozenset(token.lower() for token in tokens)
        if len(self.tokens) < 2:
            raise ValueError("tokens must name at least two distinct addresses")
        if not chain:
            raise ValueError("chain is required")
        self.chain = chain
        self.projects = frozenset(projects)
        if not math.isfinite(min_tvl_usd) or min_tvl_usd < 0:
            raise ValueError(f"min_tvl_usd must be a finite non-negative number, got {min_tvl_usd!r}")
        self.min_tvl_usd = float(min_tvl_usd)
        self.fixed_fee_pct = dict(DEFAULT_FIXED_FEE_PCT if fixed_fee_pct is None else fixed_fee_pct)

    def _cache_subject(self) -> str:
        return f"dex-fee-tiers-v1-{self.chain.lower()}-{'-'.join(sorted(self.tokens))}"

    def extract(self) -> None:
        payload = self._get_json("/pools")
        pools = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(pools, list):
            raise ValueError("DefiLlama /pools: no data array")
        rows = []
        for pool in pools:
            if not isinstance(pool, dict) or str(pool.get("chain", "")).lower() != self.chain.lower():
                continue
            underlying = pool.get("underlyingTokens")
            if not isinstance(underlying, list):
                continue
            if {str(token).lower() for token in underlying} != self.tokens:
                continue
            try:
                tvl = float(pool.get("tvlUsd") or 0.0)
            except (TypeError, ValueError):
                tvl = math.nan
            rows.append(
                {
                    "pool": str(pool.get("pool", "")),
                    "project": str(pool.get("project", "")),
                    "symbol": str(pool.get("symbol", "")),
                    "pool_meta": pool.get("poolMeta") if isinstance(pool.get("poolMeta"), str) else None,
                    "tvl_usd": tvl,
                }
            )
        self._data = pd.DataFrame(rows, columns=_COLUMNS)

    def transform(self) -> None:
        if self._data is None or self._data.empty:
            self._data = pd.DataFrame(columns=_COLUMNS)

    def read(self, with_run: bool = False) -> pd.DataFrame:
        """Return matching pools, eligible and cheapest first."""
        if with_run:
            self.run()
        else:
            self._read(self._cache_key())
        data = self._data if self._data is not None else pd.DataFrame(columns=_COLUMNS)
        data = data[data["project"].fillna("").astype(str).isin(self.projects)]
        fees = [parse_pool_fee_pct(meta) for meta in data["pool_meta"]]
        frame = pd.DataFrame(
            {
                "pool": data["pool"].fillna("").astype(str).to_numpy(),
                "project": data["project"].fillna("").astype(str).to_numpy(),
                "symbol": data["symbol"].fillna("").astype(str).to_numpy(),
                "pool_meta": [meta if isinstance(meta, str) else None for meta in data["pool_meta"]],
                "fee_pct": pd.Series([fee for fee, _ in fees], dtype="float64").to_numpy(),
                "fee_resolution_pct": pd.Series([res for _, res in fees], dtype="float64").to_numpy(),
                "tvl_usd": pd.to_numeric(data["tvl_usd"], errors="coerce").astype("float64").to_numpy(),
            }
        )
        parsed = frame["fee_pct"].notna().to_numpy()
        constant = ~parsed & frame["project"].isin(self.fixed_fee_pct).to_numpy()
        fixed = frame["project"].map(self.fixed_fee_pct).astype("float64").to_numpy()
        frame["fee_pct"] = np.where(constant, fixed, frame["fee_pct"].to_numpy())
        frame["fee_resolution_pct"] = np.where(constant, 0.0, frame["fee_resolution_pct"].to_numpy())
        frame["fee_source"] = np.where(parsed, "pool_meta", np.where(constant, "protocol_constant", "unknown"))
        frame["eligible"] = frame["fee_pct"].notna() & (frame["tvl_usd"] >= self.min_tvl_usd)
        frame = frame.sort_values(
            ["eligible", "fee_pct", "tvl_usd"], ascending=[False, True, False], kind="stable"
        )
        return frame[_READ_COLUMNS].reset_index(drop=True)

    def cheapest(self, with_run: bool = False) -> pd.Series | None:
        """Cheapest eligible pool (ties to the larger TVL), or ``None``."""
        frame = self.read(with_run=with_run)
        if frame.empty or not bool(frame["eligible"].iloc[0]):
            return None
        return frame.iloc[0]


__all__ = [
    "DEFAULT_DEX_PROJECTS",
    "DEFAULT_FIXED_FEE_PCT",
    "DefiLlamaDexFeeTierLoader",
    "parse_pool_fee_pct",
]
