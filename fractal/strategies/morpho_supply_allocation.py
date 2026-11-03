"""Morpho Blue supply allocation across several markets.

The single-position rotation (:mod:`fractal.strategies.morpho_supply_rotation`)
is bounded by the depth of one market: at size it degenerates into holding the
deepest pool. This strategy instead holds several positions, sized to maximise
expected income net of switching costs on a discrete capital grid.

Each market's supply APY falls as we add capital (self-dilution,
:meth:`MorphoSupplyEntity.projected_supply_apy`). A dollar added to a pool
lowers the rate on every dollar already in it, so the last dollar's
contribution (the marginal revenue) is below the pool's average rate. The
income optimum equalises marginal revenue across funded pools, not their
average rates: equalising average rates over-allocates to small pools with
steep dilution, exactly the hot, high-utilization pools a spike makes
attractive.

The allocator approximates expected income net of switching costs by dynamic
programming over pools on a capital grid of ``ALLOC_CHUNKS`` chunks::

    max  h · Σ_i R_i(P_i) − Σ_i gas_i(P_i) + h · TARGET_APY · idle
    R_i(P)   = P · y_i(P)                    income of pool i at position P (dilution included)
    gas_i(P) = 0 if P is the current position, else GAS_USD_PER_LEG + bps · |ΔP|
    P_i ≥ P_i_now − withdrawable_i           (the exit gate)
    P_i ≤ liquidity cap, Σ P_i + idle = NAV,  h = HOLD_HORIZON_BARS / BARS_PER_YEAR

No concavity is assumed. That matters: above the IRM target the curve is four
times steeper, so a hot pool's income can *fall* as we add capital and rise
again once utilization drops below target — the best position is then either
small (harvest the spike) or large (push it below target), never in between,
and equalising rates cannot find that.

Idle capital as the stop. Idle "earns" ``TARGET_APY`` in the objective only, as
the opportunity cost of capital (a passive alternative, gas, risk): a dollar is
placed only if its marginal contribution beats the target, so the hurdle
applies to the marginal dollar, and capital the markets cannot pay for at that
margin stays in cash. The gas term makes holding the default: a reallocation
happens only when its expected gain over ``HOLD_HORIZON_BARS`` pays for its
legs. Off-grid positions consume rounded-up budget units; the unchanged
portfolio is also considered so rounding alone cannot force a trade. Targets
exclude execution fees, which reduce the amounts actually supplied.

Signal discipline is the rotation strategy's: the market snapshot used for
sizing is the mean of the ``SIGNAL_WINDOW_BARS`` bars ending ``SIGNAL_LAG_BARS``
observed bars ago, while the liquidity
gates read the current bar — a physical constraint at execution time, not a
forecast. The allocation is re-solved every ``REBALANCE_BARS`` bars. Costs are
charged by the entities.

This is a discrete research model, not a live execution system. Sparse markets
retain their last state; cross-chain transfers omit bridge costs and latency.
Rewards assume a fixed campaign budget diluted by total supplied capital.
Defaults illustrate configuration and are not recommendations.
"""
import math
from collections import deque
from collections.abc import Sequence
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
]


@dataclass
class MorphoSupplyAllocationParams(MorphoSupplyBaseParams):
    """Hyperparameters of the marginal-income allocator.

    The shared eligibility, signal and entity fields live on
    :class:`MorphoSupplyBaseParams`; this class adds the allocator's own
    solving parameters.

    Attributes:
        TARGET_APY: the idle-capital opportunity-cost APY, the hurdle on
            marginal revenue: a chunk of capital is placed only if its
            marginal contribution beats it; the rest stays in cash.
        REBALANCE_BARS: bars between re-solving the allocation (default every bar).
        SIGNAL_WINDOW_BARS: bars averaged into the sizing snapshot; ``1`` by
            default here (react bar-by-bar), while the rotation keeps the base
            default ``24`` for a steadier signal.
        HOLD_HORIZON_BARS: expected holding period, in bars, over which a rate
            gain must repay the gas of the legs it needs. Illustrative
            default: six bars. This is a configurable modeling horizon, not an
            estimate of future rate persistence.
        ALLOC_CHUNKS: capital grid of the optimiser (100 = 1 % of NAV per chunk).
    """
    TARGET_APY: float = 0.06
    REBALANCE_BARS: int = 1
    SIGNAL_WINDOW_BARS: int = 1
    HOLD_HORIZON_BARS: int = 6
    ALLOC_CHUNKS: int = 100


class MorphoSupplyAllocationStrategy(BaseStrategy[MorphoSupplyAllocationParams]):
    """Allocate across Morpho supply markets by marginal income net of switching costs."""

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
        validate_common_params(params)
        for field_name in ("ALLOC_CHUNKS", "HOLD_HORIZON_BARS", "REBALANCE_BARS"):
            value = getattr(params, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field_name} must be an integer >= 1, got {value!r}")
        if not math.isfinite(params.TARGET_APY) or params.TARGET_APY < 0.0:
            raise ValueError(f"TARGET_APY must be a finite non-negative rate, got {params.TARGET_APY!r}")
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

    def allocation_report(self) -> dict:
        """Last solution, including its ``lambda_apy``.

        ``lambda_apy`` is the lowest last-chunk marginal APY among funded
        pools. It is ``None`` before a solution exists or when no pool holds
        a full chunk.
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
        """Every ``REBALANCE_BARS`` bars: re-solve the marginal allocation and trade the deltas."""
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
        return self._rebalance(self._solve_marginal(snapshots, nav))

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
            snap = snapshots.get(name) if name in self._fresh else None    # stale: never re-solved on frozen data
            option = self._pool_options(name, snap, nav, chunk, horizon, name in held)
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
            if snap is None:
                amounts = np.array([now])       # stale: held, never traded on frozen liquidity
            elif stuck < now:
                amounts = np.array([stuck])     # fresh but ineligible: forced out as far as the gate allows
            else:
                amounts = np.array([now])       # fully gated: keep what cannot leave
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
        return self.get_entity(name).liquidity_capacity(multiple)

    # ---------------------------------------------------------- rebalancing
    def _rebalance(self, targets: dict[str, float]) -> list[ActionToTake]:
        params = self._params
        cash = params.CASH_ENTITY_NAME
        actions: list[ActionToTake] = []
        # 1. withdraw from over-allocated markets (and from markets no longer targeted)
        for name in self._history:
            entity = self.get_entity(name)
            target = targets.get(name, 0.0)
            excess = entity.balance - target
            if excess <= 0.0:
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
        #    execute time
        for name, target in sorted(targets.items(), key=lambda item: -item[1]):
            balance = self.get_entity(name).balance
            shortfall = target - balance
            if shortfall <= 0.0:
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
