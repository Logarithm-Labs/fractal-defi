"""Cross-venue route costs and transfer-friction arithmetic.

A *venue* is where capital can sit: one asset on one chain, ``(asset, chain_id)``.
Moving capital between venues crosses a chain (a bridge), an asset (a
conversion), or both, costs gas, and may leave the capital in transit for a few
bars, earning nothing. This module prices those moves; it does not execute them
(see :class:`fractal.core.entities.simple.transit.InTransitEntity` and
:class:`~fractal.core.entities.simple.transit.VenueCashEntity` for the
accounting side).

:class:`RouteLeg` is one directed leg. Its cost is affine in the amount moved:
``fixed + bps · amount / 1e4``, where the bridge part (``bridge_fixed_usd`` +
``bridge_bps``) applies only across chains, ``conversion_bps`` only across assets,
and ``gas_usd`` on every leg. ``latency_bars`` is how many whole bars the capital
spends in transit. Every component can be switched off by leaving it out of
``components`` (useful for cost-sensitivity runs).

:class:`RouteCosts` routes everything through a *home* venue: ``entry[v]``
prices home → ``v`` and ``exit[v]`` prices ``v`` → home. A move between two
non-home venues is priced as exit plus entry, which is an upper bound when a
direct route exists.

The functions at the bottom are plain friction arithmetic on annualised decimal
rates (``0.05`` = 5 %) over a 365-day year: :func:`amortise` (a round trip as an
annual rate), :func:`break_even` (days a spread must persist to repay a round
trip), :func:`amortised_cost_days` (how often a spread series fails to repay its
amortised cost), :func:`latency_cost` (annual in-transit drag of a turnover),
:func:`fast_break_even` (the spread above which a faster, dearer transfer pays)
and :func:`conversion_cost` (the cheaper of a DEX swap and an issuer
redemption). :func:`fit_fee_schedule` (a :class:`FeeSchedule`) and
:func:`friction_table` turn observed transfers into ``fixed + bps`` fees and
latency quantiles per route, the inputs of a :class:`RouteLeg`. This module is
the canonical home of the fee fit: ``fractal.core`` never imports from
``fractal.loaders``, so loaders that fit fees import it from here.
:func:`basis_pnl` marks exposure held in non-home assets to their price in the
holder's unit.

Costs are fee-only: price impact (slippage) on a conversion is not modelled.
No fitted costs, latencies or thresholds ship with this module.
"""
import math
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from itertools import pairwise
from typing import NamedTuple

import numpy as np
import pandas as pd

__all__ = [
    "CASH_COMPONENTS",
    "COST_COMPONENTS",
    "DAYS_PER_YEAR",
    "HOURS_PER_YEAR",
    "ZERO_LEG",
    "FeeSchedule",
    "RouteCosts",
    "RouteLeg",
    "Venue",
    "amortise",
    "amortised_cost_days",
    "basis_pnl",
    "break_even",
    "conversion_cost",
    "fast_break_even",
    "fit_fee_schedule",
    "friction_table",
    "latency_cost",
    "venue_crossings",
]

#: Every cost component a leg can charge; ``latency`` is an opportunity cost, the rest are cash.
COST_COMPONENTS: tuple[str, ...] = ("bridge", "conversion", "gas", "latency")
#: Components paid in cash out of the moved capital.
CASH_COMPONENTS: tuple[str, ...] = ("bridge", "conversion", "gas")
DAYS_PER_YEAR: float = 365.0
HOURS_PER_YEAR: float = 24.0 * DAYS_PER_YEAR

#: ``(asset, chain_id)`` — one asset on one chain.
Venue = tuple[str, int]

_DIRECTIONS = ("entry", "exit")


def _check_components(components: Collection[str]) -> frozenset[str]:
    chosen = frozenset(components)
    unknown = chosen - set(COST_COMPONENTS)
    if unknown:
        raise ValueError(f"unknown cost components {sorted(unknown)}; expected a subset of {COST_COMPONENTS}")
    return chosen


def venue_crossings(venue: Venue, home: Venue) -> tuple[bool, bool]:
    """``(cross_chain, cross_asset)`` of a move between ``home`` and ``venue``."""
    return int(venue[1]) != int(home[1]), str(venue[0]) != str(home[0])


@dataclass(frozen=True)
class RouteLeg:
    """Cost of one directed route leg (fee-only, affine in the amount).

    Attributes:
        bridge_fixed_usd: fixed bridge fee per transfer, charged across chains only.
        bridge_bps: proportional bridge fee in basis points, across chains only.
        conversion_bps: asset-conversion fee in basis points, across assets only.
        gas_usd: the leg's own transaction gas (bridge burn/mint, swap), charged on
            every non-empty leg.
        latency_bars: whole bars the capital is in transit (earning nothing).
    """
    bridge_fixed_usd: float = 0.0
    bridge_bps: float = 0.0
    conversion_bps: float = 0.0
    gas_usd: float = 0.0
    latency_bars: int = 0

    def __post_init__(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"RouteLeg.{item.name} must be a number, got {value!r}")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"RouteLeg.{item.name} must be finite and >= 0, got {value!r}")
        if not isinstance(self.latency_bars, int):
            raise ValueError(f"RouteLeg.latency_bars must be an integer, got {self.latency_bars!r}")
        if self.bridge_bps + self.conversion_bps >= 1e4:
            raise ValueError("RouteLeg bridge_bps + conversion_bps must stay below 10000 (100 %)")

    def fixed_and_bps(self, *, cross_chain: bool = True, cross_asset: bool = True,
                      components: Collection[str] = COST_COMPONENTS) -> tuple[float, float]:
        """``(fixed USD, bps)`` of the cash cost: :meth:`costs` sums to ``fixed + bps · amount / 1e4``."""
        chosen = _check_components(components)
        bridge = "bridge" in chosen and cross_chain
        fixed = (self.bridge_fixed_usd if bridge else 0.0) + (self.gas_usd if "gas" in chosen else 0.0)
        bps = (self.bridge_bps if bridge else 0.0) \
            + (self.conversion_bps if "conversion" in chosen and cross_asset else 0.0)
        return fixed, bps

    def costs(self, amount: float, *, cross_chain: bool = True, cross_asset: bool = True,
              components: Collection[str] = COST_COMPONENTS) -> dict[str, float]:
        """Cash cost of moving ``amount`` (gross, USD) on this leg, by component.

        Keys are :data:`CASH_COMPONENTS`; an empty or negative amount costs nothing.
        """
        chosen = _check_components(components)
        out = dict.fromkeys(CASH_COMPONENTS, 0.0)
        if not amount > 0.0:
            return out
        if "bridge" in chosen and cross_chain:
            out["bridge"] = self.bridge_fixed_usd + self.bridge_bps * amount / 1e4
        if "conversion" in chosen and cross_asset:
            out["conversion"] = self.conversion_bps * amount / 1e4
        if "gas" in chosen:
            out["gas"] = self.gas_usd
        return out

    def gross_for_net(self, net: float, *, cross_chain: bool = True, cross_asset: bool = True,
                      components: Collection[str] = COST_COMPONENTS) -> float:
        """Gross amount to send so that ``net`` USD arrives (``0`` for ``net <= 0``)."""
        if not net > 0.0:
            return 0.0
        fixed, bps = self.fixed_and_bps(cross_chain=cross_chain, cross_asset=cross_asset, components=components)
        return (net + fixed) / (1.0 - bps / 1e4)

    def latency(self, components: Collection[str] = COST_COMPONENTS) -> int:
        """Bars in transit, or ``0`` when ``latency`` is switched off."""
        return int(self.latency_bars) if "latency" in _check_components(components) else 0


#: A free, instant leg (any move inside the home venue).
ZERO_LEG = RouteLeg()


@dataclass(frozen=True)
class RouteCosts:
    """Home-routed costs: ``entry[v]`` prices home → ``v``, ``exit[v]`` prices ``v`` → home.

    The home venue itself always gets :data:`ZERO_LEG`; any other venue missing
    from the table raises ``KeyError`` (a silent zero would make an unpriced
    venue look free).
    """
    entry: Mapping[Venue, RouteLeg] = field(default_factory=dict)
    exit: Mapping[Venue, RouteLeg] = field(default_factory=dict)

    def leg(self, venue: Venue, direction: str, home: Venue) -> RouteLeg:
        """The ``"entry"`` or ``"exit"`` leg of ``venue`` relative to ``home``."""
        if direction not in _DIRECTIONS:
            raise ValueError(f"direction must be 'entry' or 'exit', got {direction!r}")
        if tuple(venue) == tuple(home):
            return ZERO_LEG
        table = self.entry if direction == "entry" else self.exit
        if venue not in table:
            raise KeyError(f"no {direction} route cost for venue {venue!r}")
        return table[venue]

    def decision_cost(self, venue: Venue, home: Venue, delta, *, opportunity_apy: float = 0.0,
                      bars_per_year: float = HOURS_PER_YEAR,
                      components: Collection[str] = COST_COMPONENTS) -> np.ndarray:
        """Cost of changing a position at ``venue`` by ``delta`` USD (vectorised), home-routed.

        A positive delta pays the entry leg, a negative one the exit leg, each as
        ``fixed + bps · |delta| / 1e4`` plus, with ``latency`` on, the transit time
        valued at ``opportunity_apy``: ``opportunity_apy · latency_bars / bars_per_year
        · |delta|``. The home venue and a zero delta cost nothing.
        """
        delta = np.asarray(delta, dtype=float)
        out = np.zeros(delta.shape)
        if tuple(venue) == tuple(home):
            return out
        if not math.isfinite(bars_per_year) or bars_per_year <= 0:
            raise ValueError(f"bars_per_year must be finite and > 0, got {bars_per_year!r}")
        cross_chain, cross_asset = venue_crossings(venue, home)
        size = np.abs(delta)
        for direction, mask in (("entry", delta > 1e-9), ("exit", delta < -1e-9)):
            if not mask.any():
                continue
            leg = self.leg(venue, direction, home)
            fixed, bps = leg.fixed_and_bps(cross_chain=cross_chain, cross_asset=cross_asset, components=components)
            years = leg.latency(components) / bars_per_year
            out = np.where(mask, fixed + bps * size / 1e4 + opportunity_apy * years * size, out)
        return out

    def scaled(self, component: str, multiple: float) -> "RouteCosts":
        """Every leg with one component multiplied by ``multiple`` (cost sensitivity).

        Latency is rounded up to whole bars.
        """
        attrs = {"bridge": ("bridge_fixed_usd", "bridge_bps"), "conversion": ("conversion_bps",),
                 "gas": ("gas_usd",), "latency": ("latency_bars",)}
        if component not in attrs:
            raise ValueError(f"unknown cost component {component!r}; expected one of {COST_COMPONENTS}")
        if not math.isfinite(multiple) or multiple < 0:
            raise ValueError(f"multiple must be finite and >= 0, got {multiple!r}")

        def scale(leg: RouteLeg) -> RouteLeg:
            changes = {name: getattr(leg, name) * multiple for name in attrs[component]}
            if "latency_bars" in changes:
                changes["latency_bars"] = math.ceil(changes["latency_bars"] - 1e-9)
            return replace(leg, **changes)

        return RouteCosts(entry={venue: scale(leg) for venue, leg in self.entry.items()},
                          exit={venue: scale(leg) for venue, leg in self.exit.items()})


# ----------------------------------------------------------------- frictions
def _as_output(value: np.ndarray):
    return float(value) if value.ndim == 0 else value


def _positive(name: str, value) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if np.any(~np.isfinite(array)) or np.any(array <= 0.0):
        raise ValueError(f"{name} must be finite and > 0")
    return array


def amortise(round_trip_cost_usd, size_usd, horizon_days: float):
    """A round-trip cost spread over a holding horizon, as an annualised rate.

    ``cost / size · 365 / horizon_days``; broadcasts over arrays.
    """
    if not math.isfinite(horizon_days) or horizon_days <= 0:
        raise ValueError(f"horizon_days must be finite and > 0, got {horizon_days!r}")
    out = np.asarray(round_trip_cost_usd, dtype=float) / _positive("size_usd", size_usd) \
        * DAYS_PER_YEAR / horizon_days
    return _as_output(out)


def break_even(round_trip_cost_usd, size_usd, spread_net):
    """Days a net annual spread must persist to repay a round trip: ``cost / size / spread · 365``.

    ``inf`` where the spread is not positive; broadcasts over arrays.
    """
    cost_fraction = np.asarray(round_trip_cost_usd, dtype=float) / _positive("size_usd", size_usd)
    spread = np.asarray(spread_net, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(spread > 0.0, cost_fraction / spread * DAYS_PER_YEAR, np.inf)
    return _as_output(np.asarray(out))


def latency_cost(latency_hours, rate, turnover_per_year):
    """Annual in-transit drag as a fraction of NAV: ``turnover · rate · latency_hours / 8760``.

    ``turnover_per_year`` is the annualised one-way turnover (capital moved per
    year over NAV); ``rate`` is what the capital would earn if not in transit.
    """
    out = np.asarray(turnover_per_year, dtype=float) * np.asarray(rate, dtype=float) \
        * np.asarray(latency_hours, dtype=float) / HOURS_PER_YEAR
    return _as_output(out)


def fast_break_even(size_usd: float, fast_fee_usd: float, standard_fee_usd: float,
                    fast_latency_hours: float, standard_latency_hours: float) -> float:
    """Annual spread above which a fast transfer beats a standard one for ``size_usd``.

    Fast pays when the spread earned over the hours it saves exceeds its extra
    fee: ``(fast_fee − standard_fee) / size · 8760 / (standard_latency − fast_latency)``.
    ``0`` when fast is not dearer; ``inf`` when it is dearer but saves no time.
    """
    if not math.isfinite(size_usd) or size_usd <= 0:
        raise ValueError(f"size_usd must be finite and > 0, got {size_usd!r}")
    extra = (fast_fee_usd - standard_fee_usd) / size_usd
    if extra <= 0.0:
        return 0.0
    saved_hours = standard_latency_hours - fast_latency_hours
    if saved_hours <= 0.0:
        return float("inf")
    return extra * HOURS_PER_YEAR / saved_hours


def amortised_cost_days(cost_series: pd.Series, spread_series: pd.Series, horizon_days: float) -> float:
    """Share of days on which a spread is below its amortised round-trip cost.

    ``cost_series`` is the daily round-trip cost as a fraction of size; it is
    annualised over ``horizon_days`` (as :func:`amortise`) and compared with the
    annualised ``spread_series``. Only days present and finite in both enter;
    ``nan`` when none do.
    """
    if not math.isfinite(horizon_days) or horizon_days <= 0:
        raise ValueError(f"horizon_days must be finite and > 0, got {horizon_days!r}")
    both = pd.concat({"cost": cost_series, "spread": spread_series}, axis=1, join="inner")
    both = both.replace([np.inf, -np.inf], np.nan).dropna()
    if both.empty:
        return float("nan")
    amortised = both["cost"] * DAYS_PER_YEAR / horizon_days
    return float((both["spread"] < amortised).mean())


def conversion_cost(size_usd, dex_fee_bps, basis_bps, issuer_fee_bps, issuer_latency_hours, rate) -> dict:
    """Fee-only cost of converting ``size_usd`` between two assets: the cheaper of two routes.

    * DEX: ``size · (dex_fee_bps + basis_bps) / 1e4``; ``basis_bps`` is the signed
      deviation of the execution price from par, positive when adverse.
    * Issuer: ``size · issuer_fee_bps / 1e4 + size · rate · issuer_latency_hours / 8760``,
      the settlement time valued as in-transit opportunity cost at ``rate``.

    A route given as ``None`` / ``nan`` is unavailable. Linear in size (no slippage).
    Inputs broadcast. Returns ``cost_usd``, ``route`` (``"dex"``, ``"issuer"`` or
    ``"none"``), ``dex_cost_usd`` and ``issuer_cost_usd``; scalars for scalar inputs.
    """
    def as_array(value):
        return np.asarray(np.nan if value is None else value, dtype=float)

    size = as_array(size_usd)
    dex = size * (as_array(dex_fee_bps) + as_array(basis_bps)) / 1e4
    issuer = size * as_array(issuer_fee_bps) / 1e4 + size * as_array(rate) * as_array(issuer_latency_hours) \
        / HOURS_PER_YEAR
    dex, issuer = np.broadcast_arrays(dex, issuer)
    cost = np.fmin(dex, issuer)
    route = np.where(np.isnan(cost), "none", np.where(np.isnan(issuer) | (dex <= issuer), "dex", "issuer"))
    if cost.ndim == 0:
        return {"cost_usd": float(cost), "route": str(route), "dex_cost_usd": float(dex),
                "issuer_cost_usd": float(issuer)}
    return {"cost_usd": cost, "route": route, "dex_cost_usd": dex, "issuer_cost_usd": issuer}


class FeeSchedule(NamedTuple):
    """Affine fee schedule: ``fee = fixed + bps * size / 1e4`` (same units as ``size``).

    The canonical home of the type: loaders that fit observed fees (e.g. bridge
    transfers) import it from here, since ``fractal.core`` never imports loaders.
    """

    fixed: float
    bps: float

    def fee(self, size: float) -> float:
        """Fee for a transfer of ``size``."""
        return self.fixed + self.bps * size / 1e4

    def fee_bps_at(self, size: float) -> float:
        """All-in fee for a transfer of ``size``, in basis points of ``size``."""
        return self.fee(size) / size * 1e4


def fit_fee_schedule(amounts: Iterable[float], fees: Iterable[float]) -> FeeSchedule:
    """Least-squares ``fee = fixed + bps * amount / 1e4`` with both parts ``>= 0``.

    Pairs with a non-finite value or a non-positive amount are dropped.
    With no pairs the result is NaN. With a single distinct amount the
    proportional part is not identified, so the median fee is the fixed part.
    If the unconstrained fit gives a negative rate, the fit is the median fee
    as a fixed charge; if it gives a negative fixed part, it is the median
    proportional rate with no fixed charge.
    """
    amount = np.asarray(list(amounts), dtype=float)
    fee = np.asarray(list(fees), dtype=float)
    if amount.shape != fee.shape:
        raise ValueError(f"amounts and fees must have the same length, got {amount.size} and {fee.size}")
    keep = np.isfinite(amount) & np.isfinite(fee) & (amount > 0)
    amount, fee = amount[keep], fee[keep]
    if amount.size == 0:
        return FeeSchedule(math.nan, math.nan)
    if np.unique(amount).size < 2:
        return FeeSchedule(max(float(np.median(fee)), 0.0), 0.0)
    design = np.column_stack([np.ones_like(amount), amount / 1e4])
    fixed, bps = np.linalg.lstsq(design, fee, rcond=None)[0]
    if bps < 0:
        return FeeSchedule(max(float(np.median(fee)), 0.0), 0.0)
    if fixed < 0:
        return FeeSchedule(0.0, max(float(np.median(fee / amount)) * 1e4, 0.0))
    return FeeSchedule(float(fixed), float(bps))


def _size_label(size: float) -> str:
    """Compact USD size label for column names: ``1000000 -> "1m"``, ``50000 -> "50k"``."""
    if not math.isfinite(size):
        return f"{size:g}"
    for scale, suffix in ((1e9, "b"), (1e6, "m"), (1e3, "k")):
        if abs(size) >= scale:
            return f"{size / scale:g}{suffix}"
    return f"{size:g}"


_TRANSFER_KEYS = ("protocol", "asset", "src_chain", "dst_chain", "mode")


def friction_table(transfers: pd.DataFrame, sizes_usd: Sequence[float], *,
                   group_by: Sequence[str] = _TRANSFER_KEYS, quantiles: Sequence[float] = (0.1, 0.5, 0.9),
                   thin_below: int | None = None,
                   gas_buckets_usd: Sequence[float] | None = None) -> pd.DataFrame:
    """Per-route friction summary of observed transfers, one row per ``group_by`` key.

    ``transfers`` has the ``group_by`` columns plus ``amount_usd``, ``fee_usd`` and
    ``latency_s``; optional ``matched`` (bool, default all ``True``: a transfer whose
    arrival was observed) and ``gas_usd``. Statistics use matched transfers only;
    unmatched ones are counted.

    Columns: the keys, ``n``, ``n_unmatched``, ``thin`` (``n < thin_below``, only
    when ``thin_below`` is given), latency quantiles in hours (``latency_p50_h``
    ...), the :func:`fit_fee_schedule` fit ``fee_fixed_usd`` + ``fee_bps``, the
    fitted total fee in bps of size at each of ``sizes_usd`` (``fee_bps_at_1m``
    ...) and, with ``gas_buckets_usd`` (ascending bucket edges in USD of amount),
    the median gas in bps of amount per bucket (``gas_bps_<lo>_<hi>``).
    """
    keys = list(group_by)
    required = {*keys, "amount_usd", "fee_usd", "latency_s"}
    missing = required - set(transfers.columns)
    if missing:
        raise ValueError(f"transfers is missing columns {sorted(missing)}")
    sizes = [float(size) for size in sizes_usd]
    if any(not math.isfinite(size) or size <= 0.0 for size in sizes):
        raise ValueError("sizes_usd must be finite and > 0")
    if any(not 0.0 <= q <= 1.0 for q in quantiles):
        raise ValueError("quantiles must be in [0, 1]")
    edges = None if gas_buckets_usd is None else [float(edge) for edge in gas_buckets_usd]
    if edges is not None:
        if len(edges) < 2 or any(lo >= hi for lo, hi in pairwise(edges)):
            raise ValueError("gas_buckets_usd must hold at least two strictly increasing edges")
        if "gas_usd" not in transfers.columns:
            raise ValueError("gas_buckets_usd needs a gas_usd column")
        bucket_names = [f"gas_bps_{_size_label(lo)}_{_size_label(hi)}" for lo, hi in pairwise(edges)]
    frame = transfers.copy()
    frame["matched"] = frame["matched"].astype(bool) if "matched" in frame.columns else True
    rows = []
    for key, group in frame.groupby(keys, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        matched = group[group["matched"]]
        row = dict(zip(keys, key, strict=True))
        row.update(n=len(matched), n_unmatched=int((~group["matched"]).sum()))
        if thin_below is not None:
            row["thin"] = len(matched) < thin_below
        latency_h = matched["latency_s"].dropna().to_numpy(dtype=float) / 3600.0
        for q in quantiles:
            row[f"latency_p{round(q * 100)}_h"] = float(np.quantile(latency_h, q)) if latency_h.size else np.nan
        schedule = fit_fee_schedule(matched["amount_usd"].to_numpy(dtype=float),
                                    matched["fee_usd"].to_numpy(dtype=float))
        row.update(fee_fixed_usd=schedule.fixed, fee_bps=schedule.bps)
        for size in sizes:
            row[f"fee_bps_at_{_size_label(size)}"] = schedule.fee_bps_at(size)
        if edges is not None:
            gas = matched.dropna(subset=["gas_usd", "amount_usd"])
            gas = gas[gas["amount_usd"] > 0.0]
            bucket = pd.cut(gas["amount_usd"], edges, right=False, labels=bucket_names)
            medians = (gas["gas_usd"] / gas["amount_usd"] * 1e4).groupby(bucket, observed=False).median()
            for name in bucket_names:
                row[name] = float(medians.get(name, np.nan))
        rows.append(row)
    return pd.DataFrame(rows)


def basis_pnl(exposure: pd.DataFrame, prices: pd.DataFrame) -> pd.Series:
    """Per-bar mark-to-market P&L of exposure held in non-home assets.

    ``exposure`` is indexed by bar time, one column per asset, holding the USD
    notional (in the holder's unit, at par) held in that asset at the **end** of
    each bar. ``prices`` holds each asset's price in the holder's unit (for
    stablecoins, ``1 + basis``); it is sorted and forward-filled onto the
    exposure index, so a sparser price feed is fine. Bar ``t`` earns
    ``Σ_asset exposure[t−1] · (price[t] / price[t−1] − 1)``; the first bar earns 0.

    Assets without a price column, missing exposures and non-finite or
    non-positive prices contribute 0. Add the cumulative sum to an at-par NAV
    for a marked NAV.
    """
    out = pd.Series(0.0, index=exposure.index, name="basis_pnl")
    assets = [asset for asset in exposure.columns if asset in prices.columns]
    if not assets or len(exposure) < 2:
        return out
    marked = prices[assets].sort_index().reindex(exposure.index, method="ffill").astype(float)
    current = marked.to_numpy()
    previous = marked.shift(1).to_numpy()
    held = exposure[assets].shift(1).to_numpy(dtype=float)
    valid = np.isfinite(previous) & np.isfinite(current) & (previous > 0.0) & np.isfinite(held)
    with np.errstate(divide="ignore", invalid="ignore"):
        contribution = np.where(valid, held * (current / previous - 1.0), 0.0)
    out[:] = contribution.sum(axis=1)
    return out
