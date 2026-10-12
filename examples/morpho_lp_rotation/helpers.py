"""Data plumbing for the Morpho LP rotation example.

Registry CSVs (written by ``discover.py``, reviewed by a human) are read back
into listings, hourly history is loaded per market, and everything is stitched
into the :class:`Observation` stream the strategy consumes.

Two data facts drive the design, both verified live:

* the API exposes **no** fee history (``fee`` is not a field on ``MarketHistory``),
  so the market's current fee is fetched once and applied as a constant;
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
from fractal.core.entities.protocols.morpho_lp import MorphoLpGlobalState
from fractal.core.entities.simple.spot import SimpleSpotExchangeGlobalState
from fractal.loaders.morpho import MorphoMarketLoader, get_market_info
from fractal.loaders.morpho_markets import MorphoMarketListing
from fractal.strategies.morpho_lp_rotation import market_entity_name

HERE = os.path.dirname(os.path.abspath(__file__))
REGISTRY_DIR = os.path.join(HERE, "registry")
CASH_NAME = "CASH"
OPTIONAL_FLOAT_FIELDS = ("creation_timestamp", "supply_assets_usd", "borrow_assets_usd", "liquidity_assets_usd")
_HISTORY_COLUMNS = ("supply_apy", "supply_rate", "utilization", "rate_at_target",
                    "supply_assets_usd", "borrow_assets_usd", "liquidity_assets_usd")


@dataclass
class MarketPanel:
    """One market's history plus the constant fee that history cannot carry."""
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


def load_panels(listings: list[MorphoMarketListing], start, end, fetch_fee: bool = True) -> list[MarketPanel]:
    """Hourly history per market; ``fetch_fee`` costs one extra API call per market."""
    panels = []
    for listing in listings:
        try:
            history = MorphoMarketLoader(listing.market_id, listing.chain_id, start, end).read(with_run=True)
        except Exception as exc:  # pylint: disable=broad-except - one bad market must not kill the panel
            print(f"  skipped {listing.chain_id}:{listing.market_id[:10]} ({type(exc).__name__})")
            continue
        if len(history) == 0:
            continue
        frame = pd.DataFrame({column: history[column].astype(float) for column in _HISTORY_COLUMNS},
                             index=history.index)
        if fetch_fee:
            frame.attrs["fee"] = _current_fee(listing)
        panels.append(MarketPanel(listing=listing, history=frame))
    if not panels:
        raise ValueError("no market returned a usable hourly panel")
    return panels


def _current_fee(listing: MorphoMarketListing) -> float:
    """Current market fee, applied as a constant (the API has no fee history)."""
    try:
        return float(get_market_info(listing.market_id, listing.chain_id).fee or 0.0)
    except Exception:  # pylint: disable=broad-except - fee is a second-order term here
        return 0.0


def observations(panels: list[MarketPanel], cash_name: str = CASH_NAME,
                 max_supply_apy: float = 1.0) -> list[Observation]:
    """One observation per bar, carrying a state for every market with data then.

    Markets whose rate is absurd (the zombie stratum RQ1 identified) are left
    out of that bar so they can never be selected.
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
            if pd.isna(row["supply_apy"]) or not 0.0 < float(row["supply_apy"]) <= max_supply_apy:
                continue
            states[panel.name] = MorphoLpGlobalState(
                utilization=float(row["utilization"]),
                supply_rate=float(row["supply_rate"]),
                supply_apy=float(row["supply_apy"]),
                borrow_apy=float(row["borrow_apy"]) if "borrow_apy" in row else float(row["supply_apy"]),
                rate_at_target=float(row["rate_at_target"]),
                fee=float(panel.history.attrs.get("fee", 0.0)),
                supply_assets_usd=float(row["supply_assets_usd"]),
                borrow_assets_usd=float(row["borrow_assets_usd"]),
                liquidity_assets_usd=float(row["liquidity_assets_usd"]),
            )
        if not states:
            continue
        states[cash_name] = cash_state
        stream.append(Observation(timestamp=timestamp, states=states))
    return stream


def static_reference_apy(panels: list[MarketPanel]) -> float:
    """Equal-weight mean realised supply APY across the same markets.

    A naive reference only: no costs, no exit gate, no own-position impact and
    no rotation. It is the honest order of magnitude to compare the strategy
    against, not an implementable benchmark — the implementable alternative is
    a MetaMorpho vault (dismissed in CONTEXT §7), which is not modelled here.
    """
    means = [float(panel.history["supply_apy"].dropna().mean()) for panel in panels]
    means = [mean for mean in means if not np.isnan(mean)]
    return float(np.mean(means)) if means else float("nan")
