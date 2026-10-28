"""Synthetic unit tests for the concave income frontier and frontier capacities."""
import itertools

import numpy as np
import pytest

from fractal.core.entities.models.allocation_math import capacity_at_rate, concave_income_frontier

pytestmark = pytest.mark.core

GRID = np.array([0.0, 1.0, 2.0, 3.0, 4.0])


def _concave_yields():
    # incomes P*Y: market 0 -> 0, 5, 9, 12, 14 ; market 1 -> 0, 4, 7.5, 10.5, 13
    income = np.array([[0.0, 5.0, 9.0, 12.0, 14.0], [0.0, 4.0, 7.5, 10.5, 13.0]])
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(GRID > 0, income / np.where(GRID > 0, GRID, 1.0), 0.0), income


def test_frontier_fills_segments_by_marginal_yield():
    yields, _ = _concave_yields()
    frontier = concave_income_frontier(GRID, yields)
    assert frontier.marginal_yield.tolist() == [5.0, 4.0, 4.0, 3.5, 3.0, 3.0, 2.5, 2.0]
    assert frontier.segment_market.tolist() == [0, 0, 1, 1, 0, 1, 1, 0]
    assert frontier.total.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    assert frontier.average_yield[2] == pytest.approx((5 + 4 + 4) / 3)


def test_frontier_matches_brute_force_optimum_at_vertices_for_concave_income():
    yields, income = _concave_yields()
    frontier = concave_income_frontier(GRID, yields)
    for total, average in zip(frontier.total, frontier.average_yield, strict=True):
        best = max(income[0, i] + income[1, j] for i, j in itertools.product(range(5), repeat=2)
                   if GRID[i] + GRID[j] == total)
        assert total * average == pytest.approx(best)


def test_hull_relaxation_is_an_upper_bound_for_non_concave_income():
    income = np.array([0.0, 1.0, 6.0, 7.0])           # convex start: the hull skips P = 1
    grid = np.array([0.0, 1.0, 2.0, 3.0])
    yields = np.r_[0.0, income[1:] / grid[1:]]
    frontier = concave_income_frontier(grid, yields)
    assert frontier.total.tolist() == [2.0, 3.0]
    assert frontier.marginal_yield.tolist() == [3.0, 1.0]
    halfway = frontier.allocation([1.0])[0, 0]
    assert halfway == 1.0 and 3.0 * halfway > income[1]  # relaxation income 3 > grid income 1 between vertices


def test_allocation_sums_to_total_and_is_nan_beyond_the_frontier():
    yields, _ = _concave_yields()
    frontier = concave_income_frontier(GRID, yields)
    alloc = frontier.allocation([0.0, 2.5, 8.0, 9.0])
    assert alloc[0].tolist() == [0.0, 0.0]
    assert alloc[1].tolist() == [2.0, 0.5]
    assert alloc[2].tolist() == [4.0, 4.0]
    assert np.isnan(alloc[3]).all()


def test_capacity_average_and_marginal():
    yields, _ = _concave_yields()
    frontier = concave_income_frontier(GRID, yields)
    assert capacity_at_rate(frontier, 4.0, marginal=True) == 3.0
    assert capacity_at_rate(frontier, 3.0, marginal=True) == 6.0
    assert capacity_at_rate(frontier, 4.0) == 4.0       # averages 4.125 at P = 4, 3.9 at P = 5
    assert capacity_at_rate(frontier, 0.0, marginal=True) == 8.0   # censored at the grid top
    assert capacity_at_rate(frontier, 9.0) == 0.0
    assert capacity_at_rate(frontier, 4.0) >= capacity_at_rate(frontier, 4.0, marginal=True)


def test_grid_without_origin_and_per_market_grids():
    frontier = concave_income_frontier([[1.0, 2.0], [2.0, 4.0]], [[0.05, 0.04], [0.03, 0.02]])
    assert frontier.total[-1] == pytest.approx(6.0)
    assert frontier.marginal_yield[0] == pytest.approx(0.05)
    zero_yield = concave_income_frontier(GRID, np.zeros(5))
    assert zero_yield.total.size == 0
    assert capacity_at_rate(zero_yield, 0.0) == 0.0
    assert np.isnan(zero_yield.allocation([1.0])).all()


@pytest.mark.parametrize(
    "positions,yields",
    [
        ([0.0, 2.0, 1.0], [0.1, 0.1, 0.1]),
        ([-1.0, 1.0], [0.1, 0.1]),
        ([0.0, 1.0], [0.1, np.nan]),
        ([0.0, 1.0, 2.0], [[0.1, 0.1]]),
    ],
)
def test_frontier_rejects_invalid_grids(positions, yields):
    with pytest.raises(ValueError):
        concave_income_frontier(positions, yields)
