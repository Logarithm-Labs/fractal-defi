"""Morpho Blue supply allocation across several markets.

The single-position rotation (:mod:`fractal.strategies.morpho_supply_rotation`)
is bounded by the depth of one market: at size it degenerates into holding the
deepest pool. This strategy instead holds several positions. ``"average"`` mode
water-fills them to one common whole-position APY; ``"marginal"`` mode maximises
expected income net of switching costs on a discrete capital grid.

Water-filling. Each market's supply APY falls as we add capital (self-dilution,
:meth:`MorphoSupplyEntity.projected_supply_apy`). Pour capital into the
best-paying market until its rate falls to the second-best, then into both until
they meet the third, and so on, so that all funded markets pay one common *average*
rate ``λ`` on our position. (This is not the income optimum when our share of a pool
is large: that equalises marginal revenue, see ``"marginal"`` below.) For a given
``λ`` the amount market ``i`` takes is :meth:`MorphoSupplyEntity.capacity_at`
(the curve inverted), capped by its entry-gate liquidity; ``λ`` is found on a
shared grid of required position APYs, refined once inside the crossing bracket, so
the amounts sum to the capital.

Threshold as the stop. Pure water-filling would place any amount of capital by
driving ``λ`` toward zero. Below ``TARGET_APY`` the capital is better left idle
(a passive alternative, gas, risk), so ``λ`` never goes below it: if the
markets cannot absorb the whole book at ``TARGET_APY``, the rest stays in cash.
With little capital the threshold is never reached and the rule is pure
equalisation.

What this changes for capacity: the book's capacity is the **sum** of what each
market takes at the target rate, not the depth of the single best one.

Signal discipline is the rotation strategy's: the market snapshot used for
sizing is the mean of the ``SIGNAL_WINDOW_BARS`` bars ending ``SIGNAL_LAG_BARS``
observed bars ago, while the liquidity
gates read the current bar — a physical constraint at execution time, not a
forecast. Rebalancing happens every ``REBALANCE_BARS`` bars and only moves
amounts above ``MIN_REBALANCE_FRACTION`` of NAV, so costs are paid for
reallocations that matter. Costs are charged by the entities.

This is a discrete research model, not a live execution system. Sparse markets
retain their last state; cross-chain transfers omit bridge costs and latency.
Rewards assume a fixed campaign budget diluted by total supplied capital.
Defaults illustrate configuration and are not recommendations.

Two allocation modes (``ALLOCATION_MODE``):

``"average"`` (default)
    The water-filling above: equalise the *average* rate every funded pool pays
    on our whole position there. Fast, but not income-maximising: a dollar added
    to a pool lowers the rate on every dollar already in it, so the last dollar's
    contribution (the marginal revenue) is below the pool's average rate, and the
    rule over-allocates to small pools with steep dilution — exactly the hot,
    high-utilization pools a spike makes attractive. Churn is controlled only by
    ``MIN_REBALANCE_FRACTION``.

``"marginal"``
    Approximate expected income net of switching costs by dynamic programming
    over pools on a capital grid of ``ALLOC_CHUNKS`` chunks::

        max  h · Σ_i R_i(P_i) − Σ_i gas_i(P_i) + h · TARGET_APY · idle
        R_i(P)   = P · y_i(P)                    income of pool i at position P (dilution included)
        gas_i(P) = 0 if P is the current position, else GAS_USD_PER_LEG + bps · |ΔP|
        P_i ≥ P_i_now − withdrawable_i           (the exit gate)
        P_i ≤ liquidity cap, Σ P_i + idle = NAV,  h = HOLD_HORIZON_BARS / BARS_PER_YEAR

    No concavity is assumed. That matters: above the IRM target the curve is four
    times steeper, so a hot pool's income can *fall* as we add capital and rise
    again once utilization drops below target — the best position is then either
    small (harvest the spike) or large (push it below target), never in between,
    and equalising rates cannot find that. Idle "earns" ``TARGET_APY`` in the
    objective only, as the opportunity cost: a dollar is placed only if its
    marginal contribution beats the target, so the floor applies to the marginal
    dollar. The gas term makes holding the default: a reallocation happens only
    when its expected gain over ``HOLD_HORIZON_BARS`` pays for its legs.
    ``MIN_REBALANCE_FRACTION`` and ``MAX_POSITIONS`` are not used in this mode.
    Off-grid positions consume rounded-up budget units; the unchanged portfolio
    is also considered so rounding alone cannot force a trade. Targets exclude
    execution fees, which reduce the amounts actually supplied.

    Two opt-in options change the grid and the valuation; their defaults
    reproduce the description above exactly:

    * ``CHUNK_USD`` replaces the ``NAV / ALLOC_CHUNKS`` chunk with a fixed USD
      chunk, so the grid resolution no longer scales with the book. The grid
      holds ``floor(NAV / CHUNK_USD)`` chunks, capped at ``MAX_CHUNKS`` (a capped
      re-solve is flagged in :attr:`MorphoSupplyAllocationStrategy.resolve_log`);
      the budget left off the grid stays idle.
    * ``VALUATION = "horizon"`` values a position at the APY it would realise over
      ``HOLD_HORIZON_BARS`` on the frozen snapshot
      (:meth:`MorphoSupplyEntity.horizon_supply_apy`: own growth and, with
      ``IRM_FEEDBACK``, the ``rateAtTarget`` drift it causes) instead of the
      instant curve. It is a zero-market-response forecast; the dominance filter
      uses the same valuation at ``P = 0``. ``lambda_apy`` stays on the instant curve.
"""
import math
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, fields

import numpy as np

from fractal.core.base.entity import BaseEntity
from fractal.core.base.strategy import ActionToTake, BaseStrategy, NamedEntity
from fractal.core.entities.protocols.morpho_supply import MorphoSupplyGlobalState
from fractal.loaders.morpho_markets import MorphoMarketListing
from fractal.strategies.morpho_supply_common import (
    MorphoSupplyBaseParams,
    build_supply_entity,
    is_selectable,
    market_entity_name,
    validate_common_params,
)

__all__ = [
    "MorphoSupplyAllocationParams",
    "MorphoSupplyAllocationStrategy",
    "water_fill",
]

_BISECTION_STEPS = 60
_LAMBDA_POINTS = 4001          # required-rate grid of the vectorised average-mode solver
_REFINE_POINTS = 257           # second grid inside the crossing bracket (plus linear interpolation)


def water_fill(
    capital: float,
    take: dict[str, Callable[[float], float]],
    lambda_floor: float,
    lambda_ceiling: float,
) -> tuple[float, dict[str, float]]:
    """Common required position APY ``λ`` and per-market amounts that place ``capital``.

    Args:
        capital: USD to place.
        take: ``name -> f(λ)``, the amount market ``name`` absorbs while still
            paying ``λ`` (non-increasing in ``λ``, may return ``inf``).
        lambda_floor: the target rate; ``λ`` never goes below it. Capital the
            markets cannot take at this rate is returned as unplaced.
        lambda_ceiling: a rate above which nothing is taken (the best market's
            undiluted rate); the bisection's upper bound.

    Returns:
        ``(λ, amounts)``; ``sum(amounts) <= capital`` with equality unless the
        floor binds.
    """
    if capital <= 0.0 or not take:
        return lambda_floor, dict.fromkeys(take, 0.0)

    def total(rate: float) -> float:
        return sum(take[name](rate) for name in take)

    if total(lambda_floor) <= capital:
        return lambda_floor, {name: take[name](lambda_floor) for name in take}
    low, high = lambda_floor, max(lambda_ceiling, lambda_floor)
    for _ in range(_BISECTION_STEPS):
        mid = 0.5 * (low + high)
        if total(mid) > capital:
            low = mid
        else:
            high = mid
    # ``high`` is the smallest rate at which the markets take no more than the
    # capital. If a take curve jumps at that rate (a step: exogenous mode, or
    # an unbounded ``inf`` take), the capital left over is split across the
    # jumping markets in proportion to their jump, capped by it.
    amounts = {name: take[name](high) for name in take}
    remaining = capital - sum(amounts.values())
    if remaining > 1e-9:
        jumps = {name: take[name](low) - amounts[name] for name in take}
        jumps = {name: jump for name, jump in jumps.items() if jump > 0.0}
        if jumps:
            unbounded = [name for name, jump in jumps.items() if math.isinf(jump)]
            if unbounded:
                for name in unbounded:
                    amounts[name] += remaining / len(unbounded)
            else:
                total_jump = sum(jumps.values())
                for name, jump in jumps.items():
                    amounts[name] += min(jump, remaining * jump / total_jump)
    placed = sum(amounts.values())
    if placed > capital:
        amounts = {name: amount * capital / placed for name, amount in amounts.items()}
    return high, amounts


@dataclass
class MorphoSupplyAllocationParams(MorphoSupplyBaseParams):
    """Hyperparameters of the water-filling allocator.

    The shared eligibility, signal and entity fields live on
    :class:`MorphoSupplyBaseParams`; this class adds the allocator's own
    solving and mode parameters.

    Attributes:
        TARGET_APY: in ``"average"`` mode, the floor on the common whole-position
            APY ``λ``; in ``"marginal"`` mode, the idle-capital opportunity-cost
            APY. Capital whose mode-specific objective does not clear it stays in cash.
        MAX_POSITIONS: markets funded at once (the top ones by projected APY at
            ``NAV / MAX_POSITIONS``). Each funded market costs two legs per
            rebalance that touches it. Default 100 = no cap in practice.
        REBALANCE_BARS: bars between re-solving the allocation (default every bar).
        MIN_REBALANCE_FRACTION: resize deltas below this fraction of NAV are
            not traded (churn guard; pays for itself against ``COST_BPS_PER_LEG``).
            First entries and full exits are exempt, so the guard can never strand
            capital in cash or in a market the allocator no longer targets.
        SIGNAL_WINDOW_BARS: bars averaged into the sizing snapshot; ``1`` by
            default here (react bar-by-bar), while the rotation keeps the base
            default ``24`` for a steadier signal.
        ALLOCATION_MODE: ``"average"`` (equal average rates, churn guard
            ``MIN_REBALANCE_FRACTION``) or ``"marginal"`` (income-maximising with
            an expected-gas switching cost); see the module docstring.
        HOLD_HORIZON_BARS: ``"marginal"`` only — expected holding period, in
            bars, over which a rate gain must repay the gas of the legs it needs.
            Illustrative default: six bars. This is a configurable modeling
            horizon, not an estimate of future rate persistence.
        ALLOC_CHUNKS: ``"marginal"`` only — capital grid of the optimiser
            (100 = 1 % of NAV per chunk). Ignored when ``CHUNK_USD`` is set.
        CHUNK_USD: ``"marginal"`` only — fixed USD chunk of the grid; ``None``
            (default) keeps the ``NAV / ALLOC_CHUNKS`` chunk.
        MAX_CHUNKS: ``"marginal"`` with ``CHUNK_USD`` only — cap on the grid's
            chunk count (solver cost grows with it); budget above the cap stays idle.
        VALUATION: ``"marginal"`` only — ``"instant"`` (default: the whole-position
            APY at the snapshot) or ``"horizon"`` (the APY realised over
            ``HOLD_HORIZON_BARS`` on the frozen snapshot).
    """
    TARGET_APY: float = 0.06
    MAX_POSITIONS: int = 100
    REBALANCE_BARS: int = 1
    MIN_REBALANCE_FRACTION: float = 0.05
    SIGNAL_WINDOW_BARS: int = 1
    ALLOCATION_MODE: str = "average"
    HOLD_HORIZON_BARS: int = 6
    ALLOC_CHUNKS: int = 100
    CHUNK_USD: float | None = None
    MAX_CHUNKS: int = 1000
    VALUATION: str = "instant"


class MorphoSupplyAllocationStrategy(BaseStrategy[MorphoSupplyAllocationParams]):
    """Allocate across Morpho supply markets by average-rate water fill or marginal income."""

    STRICT_OBSERVATIONS: bool = False

    def __init__(
        self,
        markets: Sequence[MorphoMarketListing],
        cash_entity: BaseEntity,
        **kwargs,
    ) -> None:
        if not markets:
            raise ValueError("markets must not be empty")
        self._markets = tuple(markets)
        self._cash_entity = cash_entity
        self._history: dict[str, deque[MorphoSupplyGlobalState]] = {
            market_entity_name(listing): deque() for listing in self._markets
        }
        self._metadata: dict[str, MorphoMarketListing] = {}
        self._bar_index: int = 0
        self._fresh: set[str] = set()
        self._last_lambda: float | None = None
        self._last_targets: dict[str, float] = {}
        self._deferrals: list[dict] = []
        self._resolve_log: list[dict] = []
        super().__init__(**kwargs)

    # ------------------------------------------------------------- registry
    def set_up(self) -> None:
        """Register one entity per market plus the cash entity."""
        params = self._params
        validate_common_params(params)
        if params.ALLOCATION_MODE not in ("average", "marginal"):
            raise ValueError(f"ALLOCATION_MODE must be 'average' or 'marginal', got {params.ALLOCATION_MODE!r}")
        if params.VALUATION not in ("instant", "horizon"):
            raise ValueError(f"VALUATION must be 'instant' or 'horizon', got {params.VALUATION!r}")
        chunk_usd = params.CHUNK_USD
        if chunk_usd is not None and (isinstance(chunk_usd, bool) or not isinstance(chunk_usd, (int, float))
                                      or not math.isfinite(chunk_usd) or chunk_usd <= 0.0):
            raise ValueError(f"CHUNK_USD must be None or a positive USD amount, got {chunk_usd!r}")
        for field_name in ("ALLOC_CHUNKS", "HOLD_HORIZON_BARS", "REBALANCE_BARS", "MAX_POSITIONS", "MAX_CHUNKS"):
            value = getattr(params, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field_name} must be an integer >= 1, got {value!r}")
        if not math.isfinite(params.TARGET_APY) or params.TARGET_APY < 0.0:
            raise ValueError(f"TARGET_APY must be a finite non-negative rate, got {params.TARGET_APY!r}")
        if not 0.0 <= params.MIN_REBALANCE_FRACTION < 1.0:
            raise ValueError(
                f"MIN_REBALANCE_FRACTION must be in [0, 1), got {params.MIN_REBALANCE_FRACTION!r}")
        for listing in self._markets:
            self.register_entity(NamedEntity(
                entity_name=market_entity_name(listing),
                entity=build_supply_entity(params, listing),
            ))
        self.register_entity(NamedEntity(entity_name=params.CASH_ENTITY_NAME, entity=self._cash_entity))
        self._metadata = {market_entity_name(listing): listing for listing in self._markets}

    # ------------------------------------------------------------ telemetry
    @property
    def deferrals(self) -> list[dict]:
        """Rebalance legs the exit gate cut short: ``{market, bar, wanted, allowed}``."""
        return [dict(event) for event in self._deferrals]

    @property
    def resolve_log(self) -> list[dict]:
        """One row per ``"marginal"`` re-solve.

        Keys: ``bar``, ``budget`` (USD), ``chunk`` (USD), ``capacity_units`` (grid
        chunks), ``capped`` (``MAX_CHUNKS`` bound the grid), ``idle_usd`` (budget
        the solution leaves in cash) and ``idle_chunks`` (the same in chunks).
        """
        return [dict(row) for row in self._resolve_log]

    def allocation_report(self) -> dict:
        """Last solution, including its mode-specific ``lambda_apy``.

        ``lambda_apy`` is the common whole-position APY in ``"average"`` mode
        and the lowest last-chunk marginal APY among funded pools in
        ``"marginal"`` mode. It is ``None`` before a solution exists.
        """
        nav = self.total_balance
        cash = self.get_entity(self._params.CASH_ENTITY_NAME).balance
        return {
            "lambda_apy": self._last_lambda,
            "targets": dict(self._last_targets),
            "positions": sum(1 for name in self._history if self.get_entity(name).balance > 1.0),
            "cash_fraction": cash / nav if nav > 0 else 0.0,
            "gated_legs": len(self._deferrals),
        }

    # ---------------------------------------------------------------- step
    def _validate_observation(self, observation) -> None:
        super()._validate_observation(observation)
        self._fresh = set(observation.states)

    def predict(self) -> list[ActionToTake]:
        """Every ``REBALANCE_BARS`` bars: re-solve the water-filling and trade the deltas."""
        params = self._params
        self._bar_index += 1
        self._record_history()
        snapshots = self._snapshots()
        nav = self.total_balance
        if nav <= 0.0:
            return []
        if not snapshots:
            return []
        due = (self._bar_index - 1) % params.REBALANCE_BARS == 0
        if not due and self._last_targets:
            return []
        if params.ALLOCATION_MODE == "marginal":
            return self._rebalance(self._solve_marginal(snapshots, nav), threshold=0.0)
        targets = self._solve(snapshots, nav)
        return self._rebalance(targets, threshold=params.MIN_REBALANCE_FRACTION * nav)

    # ------------------------------------------------------------- solving
    def _solve(self, snapshots: dict[str, MorphoSupplyGlobalState], nav: float) -> dict[str, float]:
        """Water-fill the unplaced capital across the fresh eligible markets on a shared ``λ`` grid.

        Only markets with a fresh state on this bar may take new capital; a
        holding whose feed stopped is pinned at its current balance so
        :meth:`_rebalance` never trades it on stale liquidity.

        Vectorised equivalent of :func:`water_fill` over :meth:`MorphoSupplyEntity.capacity_at`:
        every market's take ``P_i(λ)`` is tabulated on one grid of required position APYs
        from ``TARGET_APY`` up to the best undiluted rate (the IRM curve inverted
        by interpolation on a fine utilization grid), the totals are summed as a
        vector and ``λ*`` is read off where the total crosses the unplaced
        capital — coarse grid, a second grid inside the crossing bracket, then
        linear interpolation. Runs in milliseconds for ~100 markets,
        which is what re-solving every bar needs.
        """
        params = self._params
        eligible = {name: snap for name, snap in snapshots.items()
                    if name in self._fresh and self._selectable(name)}
        pinned = {name: self.get_entity(name).balance for name in self._history
                  if name not in self._fresh and self.get_entity(name).balance > 1e-9}
        free = nav - sum(pinned.values())
        if not eligible or free <= 0.0:
            self._last_lambda, self._last_targets = None, pinned
            return pinned
        nominal = nav / max(params.MAX_POSITIONS, 1)
        ranked = sorted(
            eligible,
            key=lambda name: self.get_entity(name).projected_supply_apy(nominal, eligible[name]),
            reverse=True,
        )[:params.MAX_POSITIONS]
        ceiling = max(self.get_entity(name).projected_supply_apy(0.0, eligible[name]) for name in ranked)
        floor = params.TARGET_APY
        if ceiling <= floor:
            self._last_lambda, self._last_targets = floor, dict.fromkeys(ranked, 0.0) | pinned
            return self._last_targets
        lambdas = np.linspace(floor, ceiling, _LAMBDA_POINTS)
        takes = np.vstack([self._take_curve(name, eligible[name], lambdas, free) for name in ranked])
        total = takes.sum(axis=0)
        if total[0] <= free:                                  # the floor binds: place what the markets take at TARGET
            lam, amounts = floor, takes[:, 0]
        else:
            # cross ``free`` on the coarse grid, then refine on a second grid inside the
            # bracket: a spike pool's take moves by whole percents per coarse step
            hi = int(np.argmax(total <= free)) if (total <= free).any() else _LAMBDA_POINTS - 1
            lo = max(hi - 1, 0)
            fine = np.linspace(lambdas[lo], lambdas[hi], _REFINE_POINTS)
            fine_takes = np.vstack([self._take_curve(name, eligible[name], fine, free) for name in ranked])
            fine_total = fine_takes.sum(axis=0)
            j = int(np.argmax(fine_total <= free)) if (fine_total <= free).any() else len(fine) - 1
            k = max(j - 1, 0)
            if j == k or fine_total[k] <= fine_total[j]:
                lam, amounts = float(fine[j]), fine_takes[:, j]
            else:
                frac = float(np.clip((fine_total[k] - free) / (fine_total[k] - fine_total[j]), 0.0, 1.0))
                lam = float(fine[k] + frac * (fine[j] - fine[k]))
                amounts = fine_takes[:, k] + frac * (fine_takes[:, j] - fine_takes[:, k])
        amounts = np.maximum(np.nan_to_num(np.asarray(amounts, dtype=float), nan=0.0, posinf=0.0), 0.0)
        placed = float(amounts.sum())
        if placed > free > 0.0:
            amounts = amounts * (free / placed)
        self._last_lambda = lam
        self._last_targets = {name: float(amount) for name, amount in zip(ranked, amounts, strict=True)} | pinned
        return self._last_targets

    def _take_curve(self, name: str, snap: MorphoSupplyGlobalState, lambdas: np.ndarray, nav: float) -> np.ndarray:
        """``P(λ)`` for one market on the ``λ`` grid: the entity's take, capped by the entry gate and ``nav``."""
        cap = min(self._liquidity_cap(name), nav)
        return self.get_entity(name).take_curve(lambdas, snap, cap)

    # ---------------------------------------------------- marginal optimiser
    def _solve_marginal(self, snapshots: dict[str, MorphoSupplyGlobalState], nav: float) -> dict[str, float]:
        """Budget-feasible grid allocation net of expected costs, with an exact hold alternative."""
        params = self._params
        chunk, capacity, capped = self._grid(nav)
        fixed_chunk = params.CHUNK_USD is not None
        # a fixed chunk can leave budget off the grid: no pool option may exceed the grid itself
        grid_budget = min(nav, capacity * chunk) if fixed_chunk else nav
        horizon = params.HOLD_HORIZON_BARS / params.BARS_PER_YEAR
        held = {name for name in self._history if self.get_entity(name).balance > 1e-9}
        options: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []   # name, units, amounts, values
        for name in sorted(set(snapshots) | held):
            snap = snapshots.get(name) if name in self._fresh else None    # stale: never re-solved on frozen data
            option = self._pool_options(name, snap, grid_budget, chunk, horizon, name in held)
            if option is not None:
                options.append(option)
        best = np.full(capacity + 1, -np.inf)
        best[0] = 0.0
        choices = []
        ks = np.arange(capacity + 1)
        for _, units, _, values in options:
            idx = ks[None, :] - units[:, None]
            table = np.where(idx >= 0, best[np.clip(idx, 0, None)], -np.inf) + values[:, None]
            choices.append(table.argmax(axis=0))
            best = table.max(axis=0)
        if fixed_chunk:
            off_grid = max(nav - capacity * chunk, 0.0)
            idle_value = horizon * params.TARGET_APY * (chunk * np.clip(capacity - ks, 0, None) + off_grid)
        else:
            idle_value = horizon * params.TARGET_APY * chunk * np.clip(capacity - ks, 0, None)
        objective = best + idle_value
        k = int(np.argmax(objective))
        # Rounding several locked holdings up can exceed the grid budget even
        # though keeping them is feasible in USD. Always retain that alternative:
        # each pool at its option nearest to its balance, which is the balance
        # itself unless the pool only offers a forced exit.
        hold: dict[str, float] = {}
        hold_value = 0.0
        for name, _, amounts, values in options:
            nearest = int(np.argmin(np.abs(amounts - self.get_entity(name).balance)))
            hold[name] = float(amounts[nearest])
            hold_value += float(values[nearest])
        hold_value += horizon * params.TARGET_APY * max(nav - sum(hold.values()), 0.0)
        targets: dict[str, float] = {}
        if hold_value >= objective[k]:
            targets = hold
        else:
            for (name, units, amounts, _), choice in zip(reversed(options), reversed(choices), strict=True):
                c = int(choice[k])
                targets[name] = float(amounts[c])
                k -= int(units[c])
        self._last_targets = targets
        self._last_lambda = self._marginal_rate(targets, snapshots, chunk)
        idle = max(nav - sum(targets.values()), 0.0)
        self._resolve_log.append({
            "bar": self._bar_index, "budget": nav, "chunk": chunk, "capacity_units": capacity, "capped": capped,
            "idle_usd": idle, "idle_chunks": idle / chunk,
        })
        return targets

    def _grid(self, budget: float) -> tuple[float, int, bool]:
        """``(chunk USD, grid chunks, capped)`` of the marginal grid for ``budget``."""
        params = self._params
        if params.CHUNK_USD is None:
            return budget / params.ALLOC_CHUNKS, params.ALLOC_CHUNKS, False
        chunk = float(params.CHUNK_USD)
        needed = max(math.floor(budget / chunk + 1e-9), 0)
        return chunk, min(needed, params.MAX_CHUNKS), needed > params.MAX_CHUNKS

    def _valued_apy(self, name: str, amounts: np.ndarray, snap: MorphoSupplyGlobalState) -> np.ndarray:
        """Whole-position APY of each candidate amount under ``VALUATION`` (``0`` for an empty position)."""
        entity = self.get_entity(name)
        if self._params.VALUATION == "horizon":
            rates = entity.horizon_supply_apy(amounts, self._params.HOLD_HORIZON_BARS, snap)
            return np.where(amounts > 0, rates, 0.0)
        return np.array([entity.projected_supply_apy(float(a), snap) if a > 0 else 0.0 for a in amounts])

    def _pool_options(self, name: str, snap: MorphoSupplyGlobalState | None, nav: float, chunk: float,
                      horizon: float, holding: bool):
        """Candidate positions of one pool: ``(name, units, amounts, values)`` or ``None`` if irrelevant."""
        entity = self.get_entity(name)
        now = entity.balance
        gas, bps = entity.leg_cost_model
        stuck = max(now - min(now, entity.withdrawable), 0.0)          # what the exit gate will not release
        eligible = snap is not None and self._selectable(name)
        if not eligible:
            if not holding:
                return None
            if snap is None:
                amounts = np.array([now])       # stale: held, never traded on frozen liquidity
            elif stuck < now:
                amounts = np.array([stuck])     # fresh but ineligible: forced out as far as the gate allows
            else:
                amounts = np.array([now])       # fully gated: keep what cannot leave
        else:
            if not holding and self._undiluted_apy(name, snap) <= self._params.TARGET_APY:
                return None                                            # dominated by idle at any size
            cap = max(min(self._liquidity_cap(name), nav), now)
            grid = np.arange(math.ceil(stuck / chunk - 1e-9), math.floor(cap / chunk + 1e-9) + 1) * chunk
            amounts = np.unique(np.concatenate([grid[(grid >= stuck - 1e-9) & (grid <= cap + 1e-9)], [now]]))
        snap = snap if snap is not None else entity.global_state
        income = amounts * self._valued_apy(name, amounts, snap)
        moved = np.abs(amounts - now)
        cost = np.where(moved > 1e-9, gas + bps * moved / 1e4, 0.0)
        values = horizon * income - cost + np.where(moved <= 1e-9, 1e-9, 0.0)   # tie-break toward holding
        units = np.ceil(amounts / chunk - 1e-12).astype(int)
        return name, units, amounts, values

    def _undiluted_apy(self, name: str, snap: MorphoSupplyGlobalState) -> float:
        """APY of the first dollar under ``VALUATION`` (the dominance filter's test)."""
        entity = self.get_entity(name)
        if self._params.VALUATION == "horizon":
            return float(entity.horizon_supply_apy(0.0, self._params.HOLD_HORIZON_BARS, snap))
        return entity.projected_supply_apy(0.0, snap)

    def _marginal_rate(self, targets: dict[str, float], snapshots: dict[str, MorphoSupplyGlobalState],
                       chunk: float) -> float | None:
        """Lowest last-chunk marginal APY among funded pools (the solution's ``λ``), for reporting."""
        rates = []
        for name, amount in targets.items():
            snap = snapshots.get(name)
            if snap is None or amount < chunk:
                continue
            entity = self.get_entity(name)
            hi = amount * entity.projected_supply_apy(amount, snap)
            lo = (amount - chunk) * entity.projected_supply_apy(amount - chunk, snap)
            rates.append((hi - lo) / chunk)
        return min(rates) if rates else None

    def _liquidity_cap(self, name: str) -> float:
        """Largest position the entry gate allows in ``name`` on the current bar."""
        multiple = self._params.MIN_LIQUIDITY_MULTIPLE
        if multiple <= 0.0:
            return float("inf")
        return self.get_entity(name).liquidity_capacity(multiple)

    # ---------------------------------------------------------- rebalancing
    def _rebalance(self, targets: dict[str, float], threshold: float) -> list[ActionToTake]:
        params = self._params
        cash = params.CASH_ENTITY_NAME
        actions: list[ActionToTake] = []
        # 1. withdraw from over-allocated markets (and from markets no longer targeted);
        #    the churn guard applies to resizes only, never to full exits
        for name in self._history:
            entity = self.get_entity(name)
            target = targets.get(name, 0.0)
            excess = entity.balance - target
            if excess <= 0.0 or (target > 0.0 and excess <= threshold):
                continue
            gas, bps = entity.leg_cost_model
            wanted = max((excess - gas) / (1.0 + bps / 1e4), 0.0)
            allowed = entity.max_withdrawable_notional
            if allowed < wanted * (1 - 1e-9):
                self._deferrals.append({"market": name, "bar": self._bar_index, "wanted": wanted,
                                        "allowed": allowed})
            if allowed <= 0.0 or wanted <= 0.0:
                continue
            actions += self.transfer(
                from_entity=name, to_entity=cash,
                amount_in_notional=lambda strategy, source=name, desired=wanted: min(
                    desired, strategy.get_entity(source).max_withdrawable_notional),
            )
        # 2. deposit into under-allocated markets, best first, out of the cash available at
        #    execute time; the churn guard applies to resizes only, never to first entries
        for name, target in sorted(targets.items(), key=lambda item: -item[1]):
            balance = self.get_entity(name).balance
            shortfall = target - balance
            if shortfall <= 0.0 or (balance > 0.0 and shortfall <= threshold):
                continue
            actions += self.transfer(
                from_entity=cash, to_entity=name,
                amount_in_notional=lambda strategy, dest=name, wanted=shortfall: self._deposit_amount(dest, wanted),
            )
        return actions

    def _deposit_amount(self, dest: str, wanted: float) -> float:
        """What a cash → ``dest`` leg moves at execute time: capped by cash, ``0`` if it cannot pay its own gas."""
        amount = min(wanted, self.get_entity(self._params.CASH_ENTITY_NAME).balance)
        gas, _ = self.get_entity(dest).leg_cost_model
        return amount if amount > 2.0 * gas else 0.0

    # -------------------------------------------------------------- signals
    def _record_history(self) -> None:
        cap = self._params.SIGNAL_LAG_BARS + self._params.SIGNAL_WINDOW_BARS
        for name, series in self._history.items():
            if name not in self._fresh:
                continue
            series.append(self.get_entity(name).global_state)
            while len(series) > cap:
                series.popleft()

    def _snapshots(self) -> dict[str, MorphoSupplyGlobalState]:
        """Window-mean market snapshot per market, ending ``SIGNAL_LAG_BARS`` bars ago."""
        lag, window = self._params.SIGNAL_LAG_BARS, self._params.SIGNAL_WINDOW_BARS
        out = {}
        for name, series in self._history.items():
            values = list(series)
            if len(values) < lag + window:
                continue
            chunk = values[len(values) - lag - window:len(values) - lag]
            mean = {
                field.name: sum(getattr(state, field.name) for state in chunk) / len(chunk)
                for field in fields(MorphoSupplyGlobalState)
            }
            if mean["supply_assets_usd"] > 0.0:  # keep u consistent with the averaged sizes
                mean["utilization"] = min(max(mean["borrow_assets_usd"] / mean["supply_assets_usd"], 0.0), 1.0)
            out[name] = MorphoSupplyGlobalState(**mean)
        return out

    def _selectable(self, name: str) -> bool:
        """Static (LLTV, oracle, loan asset) and current-bar (pool size) eligibility of a market."""
        return is_selectable(self._params, self._metadata[name], self.get_entity(name).global_state)
