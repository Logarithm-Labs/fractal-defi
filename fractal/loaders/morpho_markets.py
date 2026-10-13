"""Morpho Blue market discovery across every chain the API serves.

The GraphQL ``markets`` list query is the single source used here, because the
REST ``/v1/blue/markets`` payload carries only addresses, ``lltv_wad`` and the
creation block — no loan/collateral symbols, no oracle type and no sizes — so a
symbol filter or an oracle-type allowlist would need one enrichment call per
market. The list query returns everything the universe decision needs in one
page, and the API's own ``listed`` flag is applied server-side.

Historical data is *not* reachable from this query (documented restriction:
``historicalState`` only exists on ``marketById``/``market``), which is why
:class:`fractal.loaders.morpho.MorphoMarketLoader` stays the history loader and
this one only enumerates the universe.

Discovery is a data step, not a risk decision: the filters here (exact loan
symbol, ``listed``, an optional LLTV ceiling used as a download size cap) are
deliberately loose. The example writes an address-pinned registry CSV from the
result and a human reviews it before a backtest reads it.
"""
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from fractal.core.entities.models.morpho_math import from_wad
from fractal.loaders._http import HttpClient
from fractal.loaders.base_loader import Loader, LoaderType
from fractal.loaders.morpho import MorphoLoaderException, graphql_query

__all__ = [
    "MorphoMarketListing",
    "MorphoMarketsLoader",
    "discover_chains",
]

_CHAINS_QUERY = """
query Chains { chains { id network } }
"""

_MARKETS_QUERY = """
query Markets($first: Int!, $skip: Int!, $where: MarketFilters) {
  markets(first: $first, skip: $skip, where: $where) {
    items {
      marketId lltv listed creationTimestamp
      loanAsset { address symbol decimals }
      collateralAsset { address symbol decimals }
      oracle { type }
      state { supplyAssetsUsd borrowAssetsUsd liquidityAssetsUsd }
    }
  }
}
"""


@dataclass(frozen=True)
class MorphoMarketListing:
    """One discoverable market: static config, oracle type and current USD sizes.

    Attributes:
        listed: the API's own listing flag (the field is named ``listed`` on
            the deployed schema; older SDK notes call it ``whitelisted``).
        lltv: liquidation LTV as a fraction (``lltv_wad / 1e18``).
        supply/borrow/liquidity_assets_usd: current market sizes in USD
            notional; ``None`` when the API omits the size series.
    """
    chain_id: int
    market_id: str
    loan_address: str
    loan_symbol: str
    loan_decimals: int
    collateral_address: str
    collateral_symbol: str
    collateral_decimals: int
    lltv: float
    oracle_type: str | None
    listed: bool
    creation_timestamp: int | None = None
    supply_assets_usd: float | None = None
    borrow_assets_usd: float | None = None
    liquidity_assets_usd: float | None = None


def discover_chains(http: HttpClient | None = None) -> list[tuple[int, str]]:
    """``[(chain_id, network_name), ...]`` for every chain the API serves."""
    data = graphql_query(http or HttpClient(), _CHAINS_QUERY, {})
    chains = data.get("chains")
    if not isinstance(chains, list):
        raise MorphoLoaderException("Morpho GraphQL: 'chains' missing from the response")
    return sorted((int(chain["id"]), str(chain["network"])) for chain in chains)


def _listing(chain_id: int, item: dict[str, Any]) -> MorphoMarketListing:
    loan = item.get("loanAsset") or {}
    collateral = item.get("collateralAsset") or {}
    state = item.get("state") or {}

    def size(key: str) -> float | None:
        return None if state.get(key) is None else float(state[key])

    return MorphoMarketListing(
        chain_id=int(chain_id),
        market_id=str(item["marketId"]).lower(),
        loan_address=str(loan.get("address", "")).lower(),
        loan_symbol=str(loan.get("symbol", "")),
        loan_decimals=int(loan.get("decimals", 0)),
        collateral_address=str(collateral.get("address", "")).lower(),
        collateral_symbol=str(collateral.get("symbol", "")),
        collateral_decimals=int(collateral.get("decimals", 0)),
        lltv=from_wad(item["lltv"]),
        oracle_type=(item.get("oracle") or {}).get("type"),
        listed=bool(item.get("listed")),
        creation_timestamp=item.get("creationTimestamp"),
        supply_assets_usd=size("supplyAssetsUsd"),
        borrow_assets_usd=size("borrowAssetsUsd"),
        liquidity_assets_usd=size("liquidityAssetsUsd"),
    )


class MorphoMarketsLoader(Loader):
    """Enumerate listed Morpho Blue markets for a set of loan assets.

    Args:
        chain_ids: chains to scan; ``None`` discovers every chain the API
            serves (``discover_chains``).
        loan_symbols: exact loan-asset symbols to keep (``{"USDC", "USDT"}``).
            Matching is exact on purpose — bridged variants (``USDC.e``) must be
            added by hand rather than swept in by a substring match.
        listed: request only the API's listed markets (server-side filter).
        lltv_max: optional download size cap on LLTV (inclusive, as a fraction).
            This is *not* the strategy's risk boundary — that is a strategy
            hyperparameter (``MAX_LLTV``) so one registry serves every run.
        page_size / max_markets: paging controls (``max_markets=None`` = all).
        loader_type: the cache format; ``PICKLE`` (default) because the payload
            is a list of dataclasses, not a frame.
        http: injectable client for offline tests.
    """

    def __init__(
        self,
        chain_ids: Sequence[int] | None = None,
        loan_symbols: Iterable[str] = ("USDC", "USDT"),
        listed: bool = True,
        lltv_max: float | None = None,
        *,
        page_size: int = 100,
        max_markets: int | None = None,
        loader_type: LoaderType = LoaderType.PICKLE,
        http: HttpClient | None = None,
    ) -> None:
        super().__init__(loader_type=loader_type)
        if page_size <= 0:
            raise ValueError(f"page_size must be > 0, got {page_size}")
        if max_markets is not None and max_markets <= 0:
            raise ValueError(f"max_markets must be > 0 or None, got {max_markets}")
        if lltv_max is not None and not 0 < lltv_max <= 1:
            raise ValueError(f"lltv_max must be in (0, 1] or None, got {lltv_max}")
        symbols = tuple(dict.fromkeys(str(symbol) for symbol in loan_symbols))
        if not symbols:
            raise ValueError("loan_symbols must not be empty")
        self.chain_ids = None if chain_ids is None else tuple(int(chain) for chain in chain_ids)
        self.loan_symbols = symbols
        self.listed = bool(listed)
        self.lltv_max = lltv_max
        self.page_size = int(page_size)
        self.max_markets = max_markets
        self._http = http or HttpClient()

    def _cache_key(self) -> str:
        chains = "all" if self.chain_ids is None else "_".join(str(chain) for chain in self.chain_ids)
        lltv = "all" if self.lltv_max is None else f"{self.lltv_max:g}"
        return f"chains-{chains}-listed-{self.listed}-loan-{'_'.join(self.loan_symbols)}-lltv-{lltv}"

    def extract(self) -> None:
        chain_ids = self.chain_ids if self.chain_ids is not None else [id_ for id_, _ in discover_chains(self._http)]
        items: list[dict[str, Any]] = []
        for chain_id in chain_ids:
            skip = 0
            while True:
                where = {"chainId_in": [chain_id], "listed": True} if self.listed else {"chainId_in": [chain_id]}
                payload = graphql_query(self._http, _MARKETS_QUERY, {
                    "first": self.page_size, "skip": skip, "where": where,
                })
                markets = payload.get("markets")
                if not isinstance(markets, dict):
                    raise MorphoLoaderException(f"Morpho GraphQL: 'markets' missing for chain {chain_id}")
                page = markets.get("items") or []
                items.extend({"chain_id": chain_id, **item} for item in page)
                skip += self.page_size
                if len(page) < self.page_size:
                    break
                if self.max_markets is not None and len(items) >= self.max_markets:
                    break
            if self.max_markets is not None and len(items) >= self.max_markets:
                break
        self._data = items

    def transform(self) -> None:
        listings = [
            _listing(item["chain_id"], item)
            for item in (self._data or [])
            if str((item.get("loanAsset") or {}).get("symbol", "")) in self.loan_symbols
            and (self.lltv_max is None or from_wad(item["lltv"]) <= self.lltv_max)
            and (not self.listed or bool(item.get("listed")))
        ]
        self._data = sorted(
            listings,
            key=lambda listing: (listing.chain_id, -(listing.liquidity_assets_usd or 0.0), listing.market_id),
        )

    def read(self, with_run: bool = False) -> list[MorphoMarketListing]:
        if with_run:
            self.run()
        else:
            self._read(self._cache_key())
        return list(self._data or [])
