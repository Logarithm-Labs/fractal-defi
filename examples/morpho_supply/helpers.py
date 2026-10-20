"""Data plumbing for the Morpho supply example.

Registry CSVs (written by ``discover.py``, reviewed by a human) are read back
into listings, hourly history is loaded per market, and everything is stitched
into the :class:`Observation` stream the strategy consumes.

Two data facts drive the design, both verified live:

* the observed ``supply_apy`` is already net of the market fee, and the entity's
  own-impact rate is that observed APY scaled by a ratio of IRM curves in which
  ``(1 − fee)`` cancels, so the fee does not move any rate here. The API has no
  fee history anyway; ``fetch_fee=True`` records the current fee for inspection;
* markets are on different chains and can have different bar coverage, so a
  market missing a bar is simply left out of that observation
  (``STRICT_OBSERVATIONS = False`` on the strategy) rather than aborting the run.
"""
import dataclasses
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from fractal.core.base import Observation
from fractal.core.entities.protocols.morpho_supply import MorphoSupplyGlobalState
from fractal.core.entities.simple.spot import SimpleSpotExchangeGlobalState
from fractal.loaders.morpho import MorphoLoaderException, MorphoMarketLoader, get_market_info
from fractal.loaders.morpho_markets import MorphoMarketListing
from fractal.strategies.morpho_supply_rotation import market_entity_name

HERE = os.path.dirname(os.path.abspath(__file__))
REGISTRY_DIR = os.path.join(HERE, "registry")
CASH_NAME = "CASH"
# Illustrative USD gas per supply/withdraw leg by chain id; unlisted chains fall
# back to DEFAULT_GAS_USD_PER_LEG. Set your own figures for an experiment.
GAS_USD_BY_CHAIN = {1: 0.50, 8453: 0.03, 42161: 0.05, 999: 0.01, 143: 0.01, 5042: 0.01}
DEFAULT_GAS_USD_PER_LEG = 0.50
OPTIONAL_FLOAT_FIELDS = ("creation_timestamp", "supply_assets_usd", "borrow_assets_usd", "liquidity_assets_usd")
_HISTORY_COLUMNS = ("supply_apy", "borrow_apy", "supply_rate", "utilization", "rate_at_target",
                    "supply_assets_usd", "borrow_assets_usd", "liquidity_assets_usd", "net_supply_apy")
_REQUIRED_COLUMNS = tuple(c for c in _HISTORY_COLUMNS if c != "net_supply_apy")


@dataclass
class MarketPanel:
    """One market's history (``attrs["fee"]`` holds the current fee when it was fetched)."""
    listing: MorphoMarketListing
    history: pd.DataFrame

    @property
    def name(self) -> str:
        return market_entity_name(self.listing)


def load_registry(chain_ids: list[int] | None = None, registry_dir: str = REGISTRY_DIR) -> list[MorphoMarketListing]:
    """Read the reviewed registry CSVs back into listings (idle markets dropped).

    Fails loudly on a registry written by an older column set rather than
    filling the gap with a guess: the registry *is* the universe.
    """
    if not os.path.isdir(registry_dir):
        raise FileNotFoundError(f"no registry at {registry_dir}; run discover.py first")
    required = {field.name for field in dataclasses.fields(MorphoMarketListing)}
    listings: list[MorphoMarketListing] = []
    for file_name in sorted(os.listdir(registry_dir)):
        if not file_name.endswith(".csv"):
            continue
        frame = pd.read_csv(os.path.join(registry_dir, file_name))
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"registry {file_name} is missing {sorted(missing)}; re-run discover.py")
        for row in frame.to_dict("records"):
            for field in OPTIONAL_FLOAT_FIELDS:
                if field in row and pd.isna(row[field]):
                    row[field] = None
            listing = MorphoMarketListing(**row)
            if chain_ids and listing.chain_id not in chain_ids:
                continue
            if listing.lltv <= 0.0:  # idle market: no IRM, nothing borrowable, earns ~0
                continue
            listings.append(listing)
    if not listings:
        raise ValueError("registry holds no usable markets for the requested chains")
    return listings


def select(listings: list[MorphoMarketListing], top_n: int, min_liquidity_usd: float) -> list[MorphoMarketListing]:
    """Markets by current withdrawable liquidity, largest first.

    The APY band is enforced bar-by-bar in :func:`observations` instead: a
    market's *current* rate says nothing about whether its history is usable.
    """
    liquid = [listing for listing in listings
              if (listing.liquidity_assets_usd or 0.0) >= min_liquidity_usd]
    return sorted(liquid, key=lambda listing: -(listing.liquidity_assets_usd or 0.0))[:top_n]


def load_panels(listings: list[MorphoMarketListing], start, end, fetch_fee: bool = False) -> list[MarketPanel]:
    """Hourly history per market.

    ``fetch_fee`` records each market's current fee in ``attrs`` at the cost of
    one extra API call per market. It is off by default because the fee cancels
    out of the anchored rate model (see the module docstring).
    """
    panels, skipped = [], []
    for listing in listings:
        try:
            history = _load_history(listing, start, end)
        except (ValueError, MorphoLoaderException) as exc:  # grid gap or API error: report, do not fill
            skipped.append((listing, f"{type(exc).__name__}: {exc}"))
            continue
        if len(history) == 0:
            skipped.append((listing, "empty history"))
            continue
        frame = pd.DataFrame({column: history[column].astype(float) for column in _HISTORY_COLUMNS},
                             index=history.index)
        if fetch_fee:
            frame.attrs["fee"] = _current_fee(listing)
        panels.append(MarketPanel(listing=listing, history=frame))
    for listing, reason in skipped:
        print(f"  skipped {listing.chain_id}:{listing.market_id[:10]} — {reason}")
    if skipped:
        print(f"  {len(skipped)} of {len(listings)} markets skipped (listed above); the panel is narrower, not filled")
    if not panels:
        raise ValueError("no market returned a usable hourly panel")
    return panels


def _load_history(listing: MorphoMarketListing, start, end):
    """Cache first, API only on a miss.

    ``read(with_run=True)`` always re-runs ``extract`` (a live GraphQL call per
    market); ``read()`` reads the on-disk cache keyed by market, window and
    column schema and raises ``FileNotFoundError`` when it is absent. A pinned
    window therefore costs the API once and every later run is a CSV read.
    """
    loader = MorphoMarketLoader(listing.market_id, listing.chain_id, start, end)
    try:
        return loader.read()
    except FileNotFoundError:
        return loader.read(with_run=True)


def _current_fee(listing: MorphoMarketListing) -> float:
    """Current market fee, recorded as a constant (the API has no fee history).

    Raises on an API failure rather than assuming zero: a silently wrong fee
    is a wrong rate, not a missing decoration.
    """
    return float(get_market_info(listing.market_id, listing.chain_id).fee or 0.0)


def observations(panels: list[MarketPanel], cash_name: str = CASH_NAME,
                 max_supply_apy: float = 1.0) -> list[Observation]:
    """One observation per bar, carrying a state for every market with data then.

    Bars outside the configured rate range are omitted. Strategies retain the
    last state of an omitted market, so omission does not guarantee exclusion
    from subsequent decisions. Missing reward observations are treated as zero.
    """
    timeline = sorted({timestamp for panel in panels for timestamp in panel.history.index})
    cash_state = SimpleSpotExchangeGlobalState(close=1.0)
    stream = []
    for timestamp in timeline:
        states = {}
        for panel in panels:
            if timestamp not in panel.history.index:
                continue
            row = panel.history.loc[timestamp]
            # A bar with any NaN field (size series not backfilled, idle market
            # without an IRM) is left out: the entity rejects non-finite state,
            # and a NaN size would otherwise read as an open gate.
            if row[list(_REQUIRED_COLUMNS)].isna().any() or not 0.0 < float(row["supply_apy"]) <= max_supply_apy:
                continue
            # Reward campaigns: netSupplyApy − supplyApy. A missing netSupplyApy point means the API has no
            # reward figure for that bar, which is treated as "no rewards" (rewards are optional upside,
            # never a substitute for a missing base rate).
            net = row.get("net_supply_apy", float("nan"))
            reward = max(float(net) - float(row["supply_apy"]), 0.0) if pd.notna(net) else 0.0
            states[panel.name] = MorphoSupplyGlobalState(
                utilization=float(row["utilization"]),
                supply_rate=float(row["supply_rate"]),
                supply_apy=float(row["supply_apy"]),
                borrow_apy=float(row["borrow_apy"]),
                rate_at_target=float(row["rate_at_target"]),
                fee=float(panel.history.attrs.get("fee", 0.0)),
                supply_assets_usd=float(row["supply_assets_usd"]),
                borrow_assets_usd=float(row["borrow_assets_usd"]),
                liquidity_assets_usd=float(row["liquidity_assets_usd"]),
                reward_apy=reward,
            )
        if not states:
            continue
        states[cash_name] = cash_state
        stream.append(Observation(timestamp=timestamp, states=states))
    return stream


def static_reference_apy(panels: list[MarketPanel]) -> float:
    """Equal-weight mean realised supply APY across the same markets.

    A naive reference only: no costs, no exit gate, no own-position impact and
    no rotation. This is a descriptive rate average, not a compounded portfolio
    return or an implementable benchmark. Vault allocations are not modelled.
    """
    means = [float(panel.history["supply_apy"].dropna().mean()) for panel in panels]
    means = [mean for mean in means if not np.isnan(mean)]
    return float(np.mean(means)) if means else float("nan")
