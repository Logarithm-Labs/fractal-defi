"""Income-optimal spread of a deposit across markets and the capacities it implies.

Each market ``m`` has a yield curve ``Y_m(P)`` on a grid of position sizes:
the yield the whole position ``P`` earns there (for example a horizon-average
supply APY from :mod:`fractal.core.entities.models.morpho_market_sim`). Its
income is ``I_m(P) = P · Y_m(P)``. :func:`concave_income_frontier` takes the
upper concave hull of each ``I_m`` and fills hull segments across markets in
order of decreasing marginal yield ``dI/dP``. The result is the income-optimal
allocation of the concave relaxation: exact at the hull vertices, where the
hull meets the grid income, and an upper bound on achievable income between
them. For non-concave income the grid search of
:class:`~fractal.strategies.morpho_supply_allocation.MorphoSupplyAllocationStrategy`
remains the exact solver.
"""
import itertools
from dataclasses import dataclass

import numpy as np

__all__ = ["IncomeFrontier", "capacity_at_rate", "concave_income_frontier"]


@dataclass(frozen=True)
class IncomeFrontier:
    """Frontier vertices after each filled segment, in fill order.

    ``total`` is the cumulative deposit, ``average_yield`` the frontier
    income over ``total`` and ``marginal_yield`` the slope of the segment that
    ends at the vertex. ``segment_market`` and ``segment_size`` give the
    market and size of each segment.
    """

    total: np.ndarray
    average_yield: np.ndarray
    marginal_yield: np.ndarray
    segment_market: np.ndarray
    segment_size: np.ndarray
    n_markets: int

    def allocation(self, totals) -> np.ndarray:
        """Per-market positions, shape ``(len(totals), n_markets)``, of the frontier fill.

        The last segment is filled pro rata. Rows for totals beyond the frontier
        (``total[-1]``) are ``NaN``.
        """
        totals = np.atleast_1d(np.asarray(totals, dtype=float))
        if np.any(~np.isfinite(totals)) or np.any(totals < 0.0):
            raise ValueError("totals must be finite and >= 0")
        out = np.zeros((totals.size, self.n_markets))
        starts = self.total - self.segment_size
        top = float(self.total[-1]) if self.total.size else 0.0
        for row, total in enumerate(totals):
            if total > top * (1.0 + 1e-12):
                out[row] = np.nan
                continue
            taken = np.clip(total - starts, 0.0, self.segment_size)
            np.add.at(out[row], self.segment_market, taken)
        return out


def _upper_hull(positions: np.ndarray, income: np.ndarray) -> list[int]:
    hull: list[int] = []
    for index in range(positions.size):
        hull.append(index)
        while len(hull) >= 3:
            a, b, c = hull[-3], hull[-2], hull[-1]
            if (income[b] - income[a]) * (positions[c] - positions[a]) <= (
                (income[c] - income[a]) * (positions[b] - positions[a])
            ):
                hull.pop(-2)
            else:
                break
    return hull


def concave_income_frontier(positions, yields) -> IncomeFrontier:
    """Income-optimal frontier of a deposit spread over markets (concave relaxation).

    Args:
        positions: grid of position sizes, strictly increasing and ``>= 0``;
            shared ``(grid,)`` or per market ``(markets, grid)``. The origin
            ``(0, 0)`` is added when the grid starts above zero.
        yields: ``Y_m(P)`` on the grid, shape ``(markets, grid)`` or ``(grid,)``
            for one market.

    Only segments with positive marginal income are filled. Ties in marginal
    yield are filled in market order.
    """
    yields = np.asarray(yields, dtype=float)
    if yields.ndim == 1:
        yields = yields[None, :]
    if yields.ndim != 2 or yields.shape[1] == 0:
        raise ValueError("yields must have shape (markets, grid) or (grid,)")
    try:
        grid = np.broadcast_to(np.asarray(positions, dtype=float), yields.shape)
    except ValueError as exc:
        raise ValueError("positions must have shape (grid,) or (markets, grid) matching yields") from exc
    if not (np.all(np.isfinite(grid)) and np.all(np.isfinite(yields))):
        raise ValueError("positions and yields must be finite")
    if np.any(grid < 0.0) or np.any(np.diff(grid, axis=1) <= 0.0):
        raise ValueError("positions must be >= 0 and strictly increasing")
    slopes, sizes, markets = [], [], []
    for market in range(yields.shape[0]):
        x = grid[market]
        income = x * yields[market]
        if x[0] > 0.0:
            x, income = np.r_[0.0, x], np.r_[0.0, income]
        hull = _upper_hull(x, income)
        for a, b in itertools.pairwise(hull):
            gain = income[b] - income[a]
            if gain > 0.0:
                slopes.append(gain / (x[b] - x[a]))
                sizes.append(x[b] - x[a])
                markets.append(market)
    order = np.argsort(-np.asarray(slopes, dtype=float), kind="stable")
    slope = np.asarray(slopes, dtype=float)[order]
    size = np.asarray(sizes, dtype=float)[order]
    total = np.cumsum(size)
    income_total = np.cumsum(slope * size)
    average = np.divide(income_total, total, out=np.zeros_like(total), where=total > 0.0)
    return IncomeFrontier(total, average, slope, np.asarray(markets, dtype=int)[order], size, yields.shape[0])


def capacity_at_rate(frontier: IncomeFrontier, rate: float, marginal: bool = False) -> float:
    """Largest frontier deposit whose average (or marginal) yield is still ``>= rate``.

    ``marginal=False``: the largest vertex with ``I(P) / P >= rate``.
    ``marginal=True``: the largest vertex with ``dI/dP >= rate`` — the deposit
    at which the next dollar stops beating an outside option ``rate``.
    Capacities are frontier vertices (refine the grid for resolution) and are
    censored at ``frontier.total[-1]``; ``0`` when no vertex qualifies.
    """
    if np.isnan(rate):
        raise ValueError("rate must not be NaN")
    values = frontier.marginal_yield if marginal else frontier.average_yield
    qualifying = np.nonzero(values >= rate)[0]
    return float(frontier.total[qualifying].max()) if qualifying.size else 0.0
