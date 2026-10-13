"""Morpho Blue supply allocation: water-fill capital across markets to a target rate.

The single-position rotation (:mod:`fractal.strategies.morpho_supply_rotation`)
is bounded by the depth of one market: at size it degenerates into holding the
deepest pool. This strategy holds **several** positions and sizes each one from
the IRM curve so that every market it is in pays the same rate on our position there
(mode ``"average"``) or so that total income net of gas is maximised (mode ``"marginal"``).

Water-filling. Each market's supply APY falls as we add capital (self-dilution,
:meth:`MorphoSupplyEntity.projected_supply_apy`). Pour capital into the
best-paying market until its rate falls to the second-best, then into both until
they meet the third, and so on, so that all funded markets pay one common *average*
rate ``λ`` on our position. (This is not the income optimum when our share of a pool
is large: that equalises marginal revenue, see ``"marginal"`` below.) For a given
``λ`` the amount market ``i`` takes is :meth:`MorphoSupplyEntity.capacity_at`
(the curve inverted), capped by its entry-gate liquidity; ``λ`` is found by
bisection so the amounts sum to the capital.

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
"""
import math
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, fields

import numpy as np

from fractal.core.base.entity import BaseEntity
from fractal.core.base.strategy import ActionToTake, BaseStrategy, BaseStrategyParams, NamedEntity
from fractal.core.entities.models.morpho_math import CURVE_STEEPNESS, TARGET_UTILIZATION
from fractal.core.entities.protocols.morpho_supply import MorphoSupplyEntity, MorphoSupplyGlobalState
from fractal.loaders.morpho_markets import MorphoMarketListing
from fractal.strategies.morpho_supply_rotation import market_entity_name

__all__ = [
    "MorphoSupplyAllocationParams",
    "MorphoSupplyAllocationStrategy",
    "water_fill",
]

_BISECTION_STEPS = 60
_LAMBDA_POINTS = 4001          # marginal-rate grid of the vectorised solver
_U_GRID = np.linspace(0.0, 1.0, 5001)   # utilization grid the IRM curve is inverted on (2e-4 in u)


def water_fill(
    capital: float,
    take: dict[str, Callable[[float], float]],
    lambda_floor: float,
    lambda_ceiling: float,
) -> tuple[float, dict[str, float]]:
    """Common marginal rate ``λ`` and per-market amounts that place ``capital``.

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
class MorphoSupplyAllocationParams(BaseStrategyParams):
    """Hyperparameters of the water-filling allocator.

    Attributes:
        TARGET_APY: the floor on the common marginal rate ``λ``. Capital that
            cannot be placed at this rate stays in cash.
        MAX_POSITIONS: markets funded at once (the top ones by projected APY at
            ``NAV / MAX_POSITIONS``). Each funded market costs two legs per
            rebalance that touches it. Default 100 = no cap in practice.
        MAX_LLTV / ORACLE_TYPES / LOAN_SYMBOLS / MIN_POOL_SIZE_USD: eligibility
            filters, as in the rotation; pool size is read on the current bar.
        MIN_LIQUIDITY_MULTIPLE: a market may hold at most
            ``withdrawable / MIN_LIQUIDITY_MULTIPLE`` of our capital, read on
            the current bar; ``0`` disables the cap.
        REBALANCE_BARS: bars between re-solving the allocation (default every bar).
        MIN_REBALANCE_FRACTION: deltas below this fraction of NAV are not
            traded (churn guard; pays for itself against ``COST_BPS_PER_LEG``).
        SIGNAL_LAG_BARS / SIGNAL_WINDOW_BARS: snapshot discipline, as in the
            rotation strategy.
        ALLOCATION_MODE: ``"average"`` (equal average rates, churn guard
            ``MIN_REBALANCE_FRACTION``) or ``"marginal"`` (income-maximising with
            an expected-gas switching cost); see the module docstring.
        HOLD_HORIZON_BARS: ``"marginal"`` only — expected holding period, in
            bars, over which a rate gain must repay the gas of the legs it needs.
            Illustrative default: six bars. This is a configurable modeling
            horizon, not an estimate of future rate persistence.
        ALLOC_CHUNKS: ``"marginal"`` only — capital grid of the optimiser
            (100 = 1 % of NAV per chunk).
        GAS_USD_PER_LEG: fixed USD gas of every non-empty deposit / withdraw,
            charged by the entities and priced by the ``"marginal"`` optimiser.
            The default for any chain not listed in ``GAS_USD_BY_CHAIN``.
        GAS_USD_BY_CHAIN: ``{chain_id: USD gas per leg}`` — per-network gas, so a
            Base market and an Ethereum market price their legs differently.
            Each market's entity gets the value for its listing's ``chain_id``.
        USE_REWARDS / REWARD_WEIGHT: credit market reward campaigns (diluted by
            our own supply) in both the accrual and the allocation decision;
            ``REWARD_WEIGHT`` is the share of the reward APY counted.
        CASH_ENTITY_NAME, OWN_IMPACT, ABSORPTION, IRM_FEEDBACK,
        COST_BPS_PER_LEG, BARS_PER_YEAR: passed to each
        :class:`MorphoSupplyEntity`.
    """
    TARGET_APY: float = 0.06
    MAX_POSITIONS: int = 100
    MAX_LLTV: float = 0.915
    ORACLE_TYPES: tuple[str, ...] = ("ChainlinkOracle", "ChainlinkOracleV2")
    LOAN_SYMBOLS: tuple[str, ...] = ("USDC", "USDT")
    MIN_POOL_SIZE_USD: float = 1_000_000.0
    MIN_LIQUIDITY_MULTIPLE: float = 1.0
    REBALANCE_BARS: int = 1
    MIN_REBALANCE_FRACTION: float = 0.05
    SIGNAL_LAG_BARS: int = 1
    SIGNAL_WINDOW_BARS: int = 1
    CASH_ENTITY_NAME: str = "CASH"
    OWN_IMPACT: bool = True
    ABSORPTION: float = 0.0
    IRM_FEEDBACK: bool = False
    USE_REWARDS: bool = False
    REWARD_WEIGHT: float = 1.0
    COST_BPS_PER_LEG: float = 0.0
    GAS_USD_PER_LEG: float = 0.0
    GAS_USD_BY_CHAIN: dict[int, float] = field(default_factory=dict)
    ALLOCATION_MODE: str = "average"
    HOLD_HORIZON_BARS: int = 6
    ALLOC_CHUNKS: int = 100
    BARS_PER_YEAR: float = 8760.0


class MorphoSupplyAllocationStrategy(BaseStrategy[MorphoSupplyAllocationParams]):
    """Hold several Morpho supply positions sized to one marginal rate."""

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
        super().__init__(**kwargs)

    # ------------------------------------------------------------- registry
    def set_up(self) -> None:
        """Register one entity per market plus the cash entity."""
        params = self._params
        if params.ALLOCATION_MODE not in ("average", "marginal"):
            raise ValueError(f"ALLOCATION_MODE must be 'average' or 'marginal', got {params.ALLOCATION_MODE!r}")
        if params.ALLOC_CHUNKS < 1 or params.HOLD_HORIZON_BARS < 1:
            raise ValueError("ALLOC_CHUNKS and HOLD_HORIZON_BARS must be >= 1")
        for listing in self._markets:
            self.register_entity(NamedEntity(
                entity_name=market_entity_name(listing),
                entity=MorphoSupplyEntity(
                    own_impact=params.OWN_IMPACT, absorption=params.ABSORPTION,
                    irm_feedback=params.IRM_FEEDBACK, use_rewards=params.USE_REWARDS,
                    reward_weight=params.REWARD_WEIGHT, cost_bps_per_leg=params.COST_BPS_PER_LEG,
                    gas_usd_per_leg=self._gas_for(listing), bars_per_year=params.BARS_PER_YEAR,
                ),
            ))
        self.register_entity(NamedEntity(entity_name=params.CASH_ENTITY_NAME, entity=self._cash_entity))
        self._metadata = {market_entity_name(listing): listing for listing in self._markets}

    def _gas_for(self, listing: MorphoMarketListing) -> float:
        """USD gas per leg on ``listing``'s chain (``GAS_USD_BY_CHAIN``, else ``GAS_USD_PER_LEG``)."""
        gas = float(self._params.GAS_USD_BY_CHAIN.get(listing.chain_id, self._params.GAS_USD_PER_LEG))
        if gas < 0 or not math.isfinite(gas):
            raise ValueError(
                f"gas per leg must be a finite non-negative USD amount, got {gas} for chain {listing.chain_id}")
        return gas

    # ------------------------------------------------------------ telemetry
    @property
    def deferrals(self) -> list[dict]:
        """Rebalance legs the exit gate cut short: ``{market, bar, wanted, allowed}``."""
        return [dict(event) for event in self._deferrals]

    def allocation_report(self) -> dict:
        """Last solved allocation: marginal rate, targets, idle cash share."""
        nav = self.total_balance
        cash = self.get_entity(self._params.CASH_ENTITY_NAME).balance
        return {
            "marginal_apy": self._last_lambda,
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
            return self._rebalance(self._solve_marginal(snapshots, nav), nav, threshold=0.0)
        targets = self._solve(snapshots, nav)
        return self._rebalance(targets, nav, threshold=params.MIN_REBALANCE_FRACTION * nav)

    # ------------------------------------------------------------- solving
    def _solve(self, snapshots: dict[str, MorphoSupplyGlobalState], nav: float) -> dict[str, float]:
        """Water-fill ``nav`` across the eligible markets on a shared ``λ`` grid.

        Vectorised equivalent of :func:`water_fill` over :meth:`MorphoSupplyEntity.capacity_at`:
        every market's take ``P_i(λ)`` is tabulated on one grid of marginal rates
        from ``TARGET_APY`` up to the best undiluted rate (the IRM curve inverted
        by interpolation on a fine utilization grid), the totals are summed as a
        vector and ``λ*`` is read off where the total crosses ``nav``, with linear
        interpolation between grid points. Runs in milliseconds for ~100 markets,
        which is what re-solving every bar needs.
        """
        params = self._params
        eligible = {name: snap for name, snap in snapshots.items() if self._selectable(name)}
        if not eligible:
            self._last_lambda, self._last_targets = None, {}
            return {}
        nominal = nav / max(params.MAX_POSITIONS, 1)
        ranked = sorted(
            eligible,
            key=lambda name: self.get_entity(name).projected_supply_apy(nominal, eligible[name]),
            reverse=True,
        )[:params.MAX_POSITIONS]
        ceiling = max(self.get_entity(name).projected_supply_apy(0.0, eligible[name]) for name in ranked)
        floor = params.TARGET_APY
        if ceiling <= floor:
            self._last_lambda, self._last_targets = floor, dict.fromkeys(ranked, 0.0)
            return self._last_targets
        lambdas = np.linspace(floor, ceiling, _LAMBDA_POINTS)
        takes = np.vstack([self._take_curve(name, eligible[name], lambdas, nav) for name in ranked])
        total = takes.sum(axis=0)
        if total[0] <= nav:                                  # the floor binds: place what the markets take at TARGET
            lam, amounts = floor, takes[:, 0]
        else:
            hi = int(np.argmax(total <= nav)) if (total <= nav).any() else _LAMBDA_POINTS - 1
            lo = max(hi - 1, 0)
            if hi == lo or total[lo] <= total[hi]:
                lam, amounts = float(lambdas[hi]), takes[:, hi]
            else:
                frac = (total[lo] - nav) / (total[lo] - total[hi])
                lam = float(lambdas[lo] + frac * (lambdas[hi] - lambdas[lo]))
                amounts = takes[:, lo] + frac * (takes[:, hi] - takes[:, lo])
        placed = float(amounts.sum())
        if placed > nav > 0.0:
            amounts = amounts * (nav / placed)
        self._last_lambda = lam
        self._last_targets = {name: float(amount) for name, amount in zip(ranked, amounts, strict=True)}
        return self._last_targets

    def _take_curve(self, name: str, snap: MorphoSupplyGlobalState, lambdas: np.ndarray, nav: float) -> np.ndarray:
        """``P(λ)`` for one market on the ``λ`` grid: capacity at each rate, capped by liquidity and ``nav``."""
        entity = self.get_entity(name)
        cap = min(self._liquidity_cap(name), nav)
        if not entity.own_impact:                            # exogenous: all-or-nothing on the headline rate
            return np.where(entity.projected_supply_apy(0.0, snap) >= lambdas, cap, 0.0)
        if entity.use_rewards and snap.reward_apy > 0.0:
            # The APY depends on size through the diluted rewards as well: invert it on a size grid.
            top = max(cap, 1.0) if math.isfinite(cap) else max(nav, 1.0)
            sizes = np.concatenate([[0.0], np.geomspace(1.0, top, 1200)])
            apy = np.array([entity.projected_supply_apy(float(x), snap) for x in sizes])
            apy = np.minimum.accumulate(apy)                 # enforce monotone decreasing against rounding noise
            take = np.interp(lambdas, apy[::-1], sizes[::-1], left=sizes[-1], right=0.0)
            return np.clip(take, 0.0, cap)
        rate = entity.curve_rate_at_target(snap)
        err = np.where(_U_GRID > TARGET_UTILIZATION, (_U_GRID - TARGET_UTILIZATION) / (1.0 - TARGET_UTILIZATION),
                       (_U_GRID - TARGET_UTILIZATION) / TARGET_UTILIZATION)
        coeff = np.where(_U_GRID > TARGET_UTILIZATION, CURVE_STEEPNESS - 1.0, 1.0 - 1.0 / CURVE_STEEPNESS)
        apy_grid = np.expm1(rate * (1.0 + coeff * err)) * _U_GRID * (1.0 - snap.fee)
        u_star = np.interp(lambdas, apy_grid, _U_GRID, left=0.0, right=1.0)
        alpha = min(entity.absorption, snap.utilization)
        with np.errstate(divide="ignore", invalid="ignore"):
            position = (snap.borrow_assets_usd - u_star * snap.supply_assets_usd) / (u_star - alpha)
        position = np.where(u_star <= alpha, cap, position)   # absorption keeps u above u*: unbounded, take the cap
        position = np.where(lambdas > apy_grid[-1], 0.0, position)  # unreachable even at u = 1
        return np.clip(np.nan_to_num(position, nan=0.0, posinf=cap), 0.0, cap)

    # ---------------------------------------------------- marginal optimiser
    def _solve_marginal(self, snapshots: dict[str, MorphoSupplyGlobalState], nav: float) -> dict[str, float]:
        """Budget-feasible grid allocation net of expected costs, with an exact hold alternative."""
        params = self._params
        chunks = params.ALLOC_CHUNKS
        chunk = nav / chunks
        horizon = params.HOLD_HORIZON_BARS / params.BARS_PER_YEAR
        held = {name for name in self._history if self.get_entity(name).balance > 1e-9}
        options: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []   # name, units, amounts, values
        for name in sorted(set(snapshots) | held):
            option = self._pool_options(name, snapshots.get(name), nav, chunk, horizon, name in held)
            if option is not None:
                options.append(option)
        capacity = chunks
        best = np.full(capacity + 1, -np.inf)
        best[0] = 0.0
        choices = []
        ks = np.arange(capacity + 1)
        for _, units, _, values in options:
            idx = ks[None, :] - units[:, None]
            table = np.where(idx >= 0, best[np.clip(idx, 0, None)], -np.inf) + values[:, None]
            choices.append(table.argmax(axis=0))
            best = table.max(axis=0)
        idle_value = horizon * params.TARGET_APY * chunk * np.clip(chunks - ks, 0, None)
        objective = best + idle_value
        k = int(np.argmax(objective))
        # Rounding several locked holdings up can exceed the grid budget even
        # though keeping them is feasible in USD. Always retain that alternative.
        current = {name: self.get_entity(name).balance for name, _, _, _ in options}
        hold_value = horizon * params.TARGET_APY * max(nav - sum(current.values()), 0.0)
        for name, _, amounts, values in options:
            hold_value += float(values[np.argmin(np.abs(amounts - current[name]))])
        targets: dict[str, float] = {}
        if hold_value >= objective[k]:
            targets = current
        else:
            for (name, units, amounts, _), choice in zip(reversed(options), reversed(choices), strict=True):
                c = int(choice[k])
                targets[name] = float(amounts[c])
                k -= int(units[c])
        self._last_targets = targets
        self._last_lambda = self._marginal_rate(targets, snapshots, chunk)
        return targets

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
            # no longer selectable (or no fresh snapshot): get out as far as the gate allows, else hold
            amounts = np.array([now, stuck]) if snap is not None and stuck < now else np.array([now])
        else:
            if not holding and entity.projected_supply_apy(0.0, snap) <= self._params.TARGET_APY:
                return None                                            # dominated by idle at any size
            cap = max(min(self._liquidity_cap(name), nav), now)
            grid = np.arange(math.ceil(stuck / chunk - 1e-9), math.floor(cap / chunk + 1e-9) + 1) * chunk
            amounts = np.unique(np.concatenate([grid[(grid >= stuck - 1e-9) & (grid <= cap + 1e-9)], [now]]))
        snap = snap if snap is not None else entity.global_state
        income = amounts * np.array([entity.projected_supply_apy(float(a), snap) if a > 0 else 0.0 for a in amounts])
        moved = np.abs(amounts - now)
        cost = np.where(moved > 1e-9, gas + bps * moved / 1e4, 0.0)
        values = horizon * income - cost + np.where(moved <= 1e-9, 1e-9, 0.0)   # tie-break toward holding
        units = np.ceil(amounts / chunk - 1e-12).astype(int)
        return name, units, amounts, values

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
        entity = self.get_entity(name)
        liquidity = entity.global_state.liquidity_assets_usd
        if not entity.own_impact:
            return liquidity / multiple
        # with our position in, withdrawable = liquidity + (1 − α)·P ≥ multiple·P
        unabsorbed = 1.0 - min(entity.absorption, entity.global_state.utilization)
        if multiple <= unabsorbed:
            return float("inf")
        return liquidity / (multiple - unabsorbed)

    # ---------------------------------------------------------- rebalancing
    def _rebalance(self, targets: dict[str, float], nav: float, threshold: float) -> list[ActionToTake]:  # pylint: disable=unused-argument
        params = self._params
        cash = params.CASH_ENTITY_NAME
        actions: list[ActionToTake] = []
        # 1. withdraw from over-allocated markets (and from markets no longer targeted)
        for name in self._history:
            entity = self.get_entity(name)
            excess = entity.balance - targets.get(name, 0.0)
            if excess <= threshold:
                continue
            allowed = entity.max_withdrawable_notional
            if allowed < excess * (1 - 1e-9):
                self._deferrals.append({"market": name, "bar": self._bar_index, "wanted": excess, "allowed": allowed})
            if allowed <= 0.0:
                continue
            actions += self.transfer(
                from_entity=name, to_entity=cash,
                amount_in_notional=lambda strategy, source=name, wanted=excess: min(
                    wanted, strategy.get_entity(source).max_withdrawable_notional),
            )
        # 2. deposit into under-allocated markets, best first, out of the cash available at execute time
        for name, target in sorted(targets.items(), key=lambda item: -item[1]):
            shortfall = target - self.get_entity(name).balance
            if shortfall <= threshold:
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
        params, listing = self._params, self._metadata[name]
        return (
            listing.lltv <= params.MAX_LLTV
            and listing.oracle_type in params.ORACLE_TYPES
            and listing.loan_symbol in params.LOAN_SYMBOLS
            and self.get_entity(name).global_state.supply_assets_usd >= params.MIN_POOL_SIZE_USD
        )
