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

The listing's sizes are *current*: ranking a past universe by them is
survivorship-biased. :func:`market_sizes_as_of` reads each market's USD sizes
at a past timestamp from ``marketById { historicalState }`` (batched with
GraphQL aliases) so a universe can be ranked point-in-time.
"""
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from fractal.core.entities.models.morpho_math import from_wad
from fractal.loaders._dt import to_seconds, to_utc
from fractal.loaders._http import HttpClient
from fractal.loaders.base_loader import Loader, LoaderType
from fractal.loaders.morpho import MorphoLoaderException, _validate_market_id, graphql_query

__all__ = [
    "AS_OF_RULES",
    "MorphoMarketListing",
    "MorphoMarketsLoader",
    "discover_chains",
    "market_sizes_as_of",
]

#: Snapshot rules of :func:`market_sizes_as_of`.
AS_OF_RULES = ("at_or_before", "end_of_day")
_DAY = timedelta(days=1)
_AS_OF_COLUMNS = ["chain_id", "market_id", "time", "supply_assets_usd", "borrow_assets_usd"]

_CHAINS_QUERY = """
query Chains { chains { id network } }
"""

_MARKETS_QUERY = """
query Markets($first: Int!, $skip: Int!, $where: MarketFilters) {
  markets(first: $first, skip: $skip, orderBy: UniqueKey, where: $where) {
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
        page_size / max_markets: paging controls. ``max_markets`` caps the
            markets that pass the filters above, counted across chains in
            order (``None`` = all).
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
        limit = "all" if self.max_markets is None else str(self.max_markets)
        return (
            f"schema-v3-chains-{chains}-listed-{self.listed}-loan-{'_'.join(self.loan_symbols)}"
            f"-lltv-{lltv}-max-{limit}"
        )

    def extract(self) -> None:
        chain_ids = self.chain_ids if self.chain_ids is not None else [id_ for id_, _ in discover_chains(self._http)]
        items: list[dict[str, Any]] = []
        seen: set[tuple[int, str]] = set()
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
                new_items = []
                for item in page:
                    market = (chain_id, str(item.get("marketId", "")).lower())
                    if market not in seen:
                        seen.add(market)
                        new_items.append({"chain_id": chain_id, **item})
                if not new_items:
                    break
                skip += self.page_size
                new_items = [item for item in new_items if self._keep(item)]
                if self.max_markets is not None:
                    new_items = new_items[:self.max_markets - len(items)]
                items.extend(new_items)
                if len(page) < self.page_size:
                    break
                if self.max_markets is not None and len(items) >= self.max_markets:
                    break
            if self.max_markets is not None and len(items) >= self.max_markets:
                break
        self._data = items

    def _keep(self, item: dict[str, Any]) -> bool:
        """Client-side filters: exact loan symbol, LLTV cap and the listed flag."""
        return (
            str((item.get("loanAsset") or {}).get("symbol", "")) in self.loan_symbols
            and (self.lltv_max is None or from_wad(item["lltv"]) <= self.lltv_max)
            and (not self.listed or bool(item.get("listed")))
        )

    def transform(self) -> None:
        listings: dict[tuple[int, str], MorphoMarketListing] = {}
        for item in self._data or []:
            if self._keep(item):
                listing = _listing(item["chain_id"], item)
                listings.setdefault((listing.chain_id, listing.market_id), listing)
        self._data = sorted(
            listings.values(),
            key=lambda listing: (listing.chain_id, -(listing.liquidity_assets_usd or 0.0), listing.market_id),
        )

    def read(self, with_run: bool = False) -> list[MorphoMarketListing]:
        if with_run:
            self.run()
        else:
            self._read(self._cache_key())
        return list(self._data or [])


def _as_of_window(timestamp: datetime, rule: str, max_staleness: timedelta) -> tuple[int, int]:
    if rule == "at_or_before":
        return to_seconds(timestamp - max_staleness), to_seconds(timestamp)
    day = timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
    return to_seconds(day), to_seconds(day + _DAY)


def _snapshot(history: dict[str, Any], lo: int, hi: int) -> tuple[int | None, float, float]:
    """Latest point in ``[lo, hi]`` with a supply value, and the borrow value at that same point."""
    def points(key: str) -> dict[int, float]:
        out = {}
        for point in history.get(key) or []:
            if point.get("x") is not None and point.get("y") is not None and lo <= int(point["x"]) <= hi:
                out[int(point["x"])] = float(point["y"])
        return out

    supply, borrow = points("supplyAssetsUsd"), points("borrowAssetsUsd")
    if not supply:
        return None, float("nan"), float("nan")
    x = max(supply)
    return x, supply[x], borrow.get(x, float("nan"))


def market_sizes_as_of(
    markets: Iterable[MorphoMarketListing | tuple[int, str]],
    timestamp: datetime,
    *,
    rule: str = "at_or_before",
    max_staleness: timedelta = _DAY,
    interval: str = "HOUR",
    batch: int = 25,
    http: HttpClient | None = None,
) -> pd.DataFrame:
    """Point-in-time USD supply/borrow sizes of many markets, for survivorship-free ranking.

    Each market's ``historicalState { supplyAssetsUsd, borrowAssetsUsd }`` is
    read over a short window (``batch`` markets per aliased GraphQL request)
    and one snapshot point is kept: the **latest** point inside the window
    that carries a supply value; the borrow size is taken at that same point.

    Rules (``rule``):
        ``"at_or_before"`` — window ``[timestamp − max_staleness, timestamp]``:
            the last observation at or before ``timestamp``, so no data after
            the as-of instant is used.
        ``"end_of_day"`` — window ``[D, D + 1 day]`` with ``D`` the UTC day of
            ``timestamp``: the day's closing point (the next midnight when the
            API has it). This uses up to one day of data *after* ``D`` and
            matches a "size on date D" convention; prefer ``"at_or_before"``
            when look-ahead matters.

    Args:
        markets: :class:`MorphoMarketListing` items or ``(chain_id, market_id)``
            pairs. The result is only as survivorship-free as this list: a list
            from ``MorphoMarketsLoader(listed=True)`` already drops markets that
            are no longer listed.
        timestamp: the as-of instant (naive datetimes are UTC).
        max_staleness: look-back of ``"at_or_before"`` (> 0).
        interval: API interval, ``"HOUR"`` or ``"DAY"``.
        batch: markets per request.
        http: injectable client for offline tests.

    Returns:
        One row per input market, in input order, with columns ``chain_id``,
        ``market_id``, ``time`` (UTC time of the snapshot point, ``NaT`` when
        none) and ``supply_assets_usd`` / ``borrow_assets_usd`` (NaN when the
        window has no point, e.g. before the market's creation). A market the
        API does not know raises :class:`MorphoLoaderException`.
    """
    if rule not in AS_OF_RULES:
        raise ValueError(f"rule must be one of {AS_OF_RULES}, got {rule!r}")
    if interval not in ("HOUR", "DAY"):
        raise ValueError(f"interval must be 'HOUR' or 'DAY', got {interval!r}")
    if batch <= 0:
        raise ValueError(f"batch must be > 0, got {batch}")
    if max_staleness <= timedelta(0):
        raise ValueError(f"max_staleness must be positive, got {max_staleness}")
    pairs = [
        (int(m.chain_id), _validate_market_id(m.market_id)) if isinstance(m, MorphoMarketListing)
        else (int(m[0]), _validate_market_id(m[1]))
        for m in markets
    ]
    lo, hi = _as_of_window(to_utc(timestamp), rule, max_staleness)
    client = http or HttpClient()
    rows = []
    for start in range(0, len(pairs), batch):
        chunk = pairs[start:start + batch]
        # Market ids are regex-validated hex and chain ids are ints, so inlining
        # them as alias arguments cannot inject GraphQL.
        aliases = " ".join(
            f'm{i}: marketById(marketId: "{market_id}", chainId: {chain_id}) {{ historicalState {{ '
            f"supplyAssetsUsd(options: $o) {{ x y }} borrowAssetsUsd(options: $o) {{ x y }} }} }}"
            for i, (chain_id, market_id) in enumerate(chunk)
        )
        data = graphql_query(client, f"query MarketSizesAsOf($o: TimeseriesOptions) {{ {aliases} }}", {
            "o": {"startTimestamp": lo, "endTimestamp": hi, "interval": interval},
        })
        for i, (chain_id, market_id) in enumerate(chunk):
            market = data.get(f"m{i}")
            if not market:
                raise MorphoLoaderException(f"market {market_id} not found on chain {chain_id}")
            x, supply, borrow = _snapshot(market.get("historicalState") or {}, lo, hi)
            rows.append({
                "chain_id": chain_id, "market_id": market_id,
                "time": pd.NaT if x is None else pd.Timestamp(x, unit="s", tz="UTC"),
                "supply_assets_usd": supply, "borrow_assets_usd": borrow,
            })
    frame = pd.DataFrame(rows, columns=_AS_OF_COLUMNS)
    frame["chain_id"] = frame["chain_id"].astype("int64")
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    frame[["supply_assets_usd", "borrow_assets_usd"]] = frame[["supply_assets_usd", "borrow_assets_usd"]].astype(float)
    return frame
