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
trip), :func:`latency_cost` (annual in-transit drag of a turnover) and
:func:`fast_break_even` (the spread above which a faster, dearer transfer pays).

Costs are fee-only: price impact (slippage) on a conversion is not modelled.
"""
import math
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field, fields, replace

import numpy as np

__all__ = [
    "CASH_COMPONENTS",
    "COST_COMPONENTS",
    "DAYS_PER_YEAR",
    "HOURS_PER_YEAR",
    "ZERO_LEG",
    "RouteCosts",
    "RouteLeg",
    "Venue",
    "amortise",
    "break_even",
    "fast_break_even",
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
