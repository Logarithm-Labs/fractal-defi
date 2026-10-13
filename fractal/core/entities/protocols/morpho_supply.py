"""Morpho Blue supply-side entity: loan-token supply with a liquidity gate.

The supplier-side mirror of
:class:`fractal.core.entities.protocols.morpho.MorphoEntity`: it holds loan-token
supply (USD notional), earns the market's supply rate, and — unlike every other
lending entity in this repo — **cannot always get its money back**. Morpho's
withdrawal cannot exceed available loan-token liquidity, so the
withdrawable amount is ``liquidityAssets = supplyAssets − borrowAssets``, which
is exactly 0 at full utilization. Entering is unconditional; exiting is not.

Two replay modes:

``own_impact=False``
    Exogenous replay: the historical ``supply_rate`` and
    ``liquidity_assets_usd`` are taken as given — the "our size changes nothing"
    baseline. Correct only for a position small relative to the market.

``own_impact=True`` (default)
    State-carrying replay: the position is added to the historical totals and
    the rate and the gate are recomputed with it included::

        S' = S + P
        B' = B + α·P            α = min(absorption, u)
        u' = B' / S'
        liquidity' = S' − B'
        borrow_apy' = AdaptiveCurveIRM(u', R*)
        supply_rate' = per-bar(borrow_apy' · u' · (1 − fee))

    ``absorption = 0`` models pure self-dilution. Effective absorption is
    capped at the historical utilization (``min(absorption, u)``), so
    ``absorption = 1`` preserves utilization. This cap is a modeling assumption,
    not a constraint on actual borrower demand.

    **The gate is vacuous at** ``absorption = 0``: with nobody borrowing our
    supply, ``liquidity' = liquidity + P`` and our own position is always
    withdrawable. Exit deferral is therefore only measurable with
    ``absorption > 0`` or ``own_impact=False``; a run at ``absorption = 0``
    reports zero exit deferrals by construction, not by evidence.

``irm_feedback=True`` (optional, with ``own_impact``)
    The AdaptiveCurve controller moves ``rateAtTarget`` by
    ``exp(50/yr · err(u) · Δt)`` every bar. The historical ``R*`` path already
    contains the market's own history; what it lacks is the *extra* drift our
    position causes by holding ``u'`` away from the historical ``u``. This mode
    accumulates exactly that differential, ``Σ 50/yr · (err(u') − err(u)) · Δt``,
    and applies it multiplicatively to the historical ``R*`` (clamped to the
    protocol bounds). A position that keeps a market 10 points below target for
    a month cuts its own ``R*`` by about a third on top of the curve dilution.
    The accumulated drift persists after the position leaves (the controller
    only closes the gap if utilization differs again), which is conservative
    for re-entering the same market. Borrower demand responding to the lower
    rate is still not modelled — that is what ``absorption`` brackets.

Costs: each non-empty leg charges a fixed ``gas_usd_per_leg`` (gas does not scale with
size) plus ``cost_bps_per_leg`` on the moved notional, and reports both through
``record_execution``, so
:class:`~fractal.core.base.strategy.result.StrategyResult`'s
``fees_paid``/``turnover``/``fee_drag`` pick it up unchanged. On a deposit the
fee is taken out of the amount supplied; on a withdraw ``amount_in_notional``
is what the *destination receives* and ``amount + fee`` leaves the position,
so the NAV drop of a transfer equals exactly the fees recorded. Size a withdraw
leg with :attr:`MorphoSupplyEntity.max_withdrawable_notional`.

Sizing helpers: :meth:`MorphoSupplyEntity.projected_supply_apy` is the APY a
whole position of a given size would earn here, and
:meth:`MorphoSupplyEntity.capacity_at` inverts it — the largest position that
still earns a target APY. Both accept an explicit market state so a strategy
can size on a lagged or smoothed snapshot instead of the current bar.
"""
import math
from dataclasses import dataclass, fields

from fractal.core.base.entity import BaseEntity, EntityException, GlobalState, InternalState
from fractal.core.entities.models.morpho_math import (
    ADJUSTMENT_SPEED,
    MAX_RATE_AT_TARGET,
    MIN_RATE_AT_TARGET,
    adaptive_curve_error,
    supply_apy_from_curve,
    utilization_for_supply_apy,
)

__all__ = [
    "MAX_MARKET_FEE",
    "MorphoSupplyEntity",
    "MorphoSupplyException",
    "MorphoSupplyGlobalState",
    "MorphoSupplyInternalState",
]

MAX_MARKET_FEE = 0.25  # ConstantsLib.MAX_FEE
BARS_PER_YEAR_HOURLY = 8760.0


class MorphoSupplyException(EntityException):
    """Errors raised by :class:`MorphoSupplyEntity`."""


@dataclass
class MorphoSupplyGlobalState(GlobalState):
    """Market context for one Morpho Blue market, all USD notional.

    Attributes:
        utilization: ``borrowAssets / supplyAssets``, in ``[0, 1]``.
        supply_rate: **per-bar** rate the market pays its suppliers — the
            supplier-side sibling of ``LendingHistory.borrowing_rate``. It is
            already net of the market fee (the API's ``supplyApy`` is), and is
            the rate applied in ``own_impact=False`` mode.
        supply_apy / borrow_apy: the market's annual effective rates, for
            reporting and formula consistency checks.
        rate_at_target: AdaptiveCurveIRM ``rateAtTarget`` as an **annual** APY
            (the loader converts the API's per-second WAD value).
        fee: market fee as a fraction (``[0, 0.25]``).
        supply_assets_usd / borrow_assets_usd / liquidity_assets_usd: market
            sizes; ``liquidity_assets_usd`` is the withdrawable amount and the
            gate input in exogenous mode; ``(1−u)·S`` is only a consistency check.
        reward_apy: annual reward rate of the market's incentive campaigns at the
            market's current supply (``netSupplyApy − supplyApy``), undiluted.
            Used only by entities with ``use_rewards=True``.

    Every field must be finite: a NaN size would silently open the gate (any
    comparison with NaN is False), so ``update_state`` rejects it loudly.
    """
    utilization: float = 0.0
    supply_rate: float = 0.0
    supply_apy: float = 0.0
    borrow_apy: float = 0.0
    rate_at_target: float = 0.0
    fee: float = 0.0
    supply_assets_usd: float = 0.0
    borrow_assets_usd: float = 0.0
    liquidity_assets_usd: float = 0.0
    reward_apy: float = 0.0


@dataclass
class MorphoSupplyInternalState(InternalState):
    """Position state.

    Attributes:
        supplied: loan-token supply in USD notional.
        rstar_log_drift: accumulated ``ln`` drift of ``rateAtTarget`` caused by
            this position (``irm_feedback`` mode); ``0`` otherwise.
    """
    supplied: float = 0.0
    rstar_log_drift: float = 0.0


class MorphoSupplyEntity(BaseEntity):
    """Supplier position in one isolated Morpho Blue market.

    Args:
        own_impact: recompute the rate and the exit gate with this position
            included (``True``) or replay the historical values (``False``).
        absorption: share of the position borrowers take up, in ``[0, 1]``;
            effective value is ``min(absorption, u)``. Only meaningful with
            ``own_impact=True``.
        use_rewards / reward_weight: credit the market's reward campaigns
            (``reward_apy``) on top of the base supply rate, counting
            ``reward_weight`` of them (a haircut for the reward token's price
            and liquidity risk). A campaign pays a fixed amount to the whole
            market, so our share of it dilutes as ``S / (S + P)`` — the same
            treatment Kamino's kvault-bot applies to farm rewards. Off by
            default: only base supply yield is credited.
        irm_feedback: also let the position's utilization offset drive the
            AdaptiveCurve controller's ``rateAtTarget`` (see module docstring).
            Requires ``own_impact=True``.
        cost_bps_per_leg: fee charged on the moved notional of every deposit
            and withdraw, in basis points. Costs live here rather than in the
            strategy so a fully deployed portfolio never needs idle cash to pay
            them, and so they surface in ``StrategyResult.fees_paid``.
        gas_usd_per_leg: fixed USD cost of every non-empty deposit or withdraw
            (the gas of one ``supply``/``withdraw`` call). Unlike the bps fee it
            does not shrink with the leg, so many small rebalances pay in full.
            A zero-amount leg costs nothing; a deposit smaller than its own
            fee is refused.
        bars_per_year: bars per year, used to turn an annual APY into the
            per-bar rate in ``own_impact`` mode. The default matches the
            hourly loader resolution; it must match the data's bar length.
    """

    def __init__(
        self,
        *,
        own_impact: bool = True,
        absorption: float = 0.0,
        irm_feedback: bool = False,
        use_rewards: bool = False,
        reward_weight: float = 1.0,
        cost_bps_per_leg: float = 0.0,
        gas_usd_per_leg: float = 0.0,
        bars_per_year: float = BARS_PER_YEAR_HOURLY,
    ) -> None:
        if not 0.0 <= absorption <= 1.0:
            raise MorphoSupplyException(f"absorption must be in [0, 1], got {absorption}")
        if cost_bps_per_leg < 0:
            raise MorphoSupplyException(f"cost_bps_per_leg must be >= 0, got {cost_bps_per_leg}")
        if gas_usd_per_leg < 0:
            raise MorphoSupplyException(f"gas_usd_per_leg must be >= 0, got {gas_usd_per_leg}")
        if bars_per_year <= 0:
            raise MorphoSupplyException(f"bars_per_year must be > 0, got {bars_per_year}")
        if not 0.0 <= reward_weight <= 1.0:
            raise MorphoSupplyException(f"reward_weight must be in [0, 1], got {reward_weight}")
        if irm_feedback and not own_impact:
            raise MorphoSupplyException("irm_feedback requires own_impact=True")
        self.own_impact: bool = bool(own_impact)
        self.absorption: float = float(absorption)
        self.irm_feedback: bool = bool(irm_feedback)
        self.use_rewards: bool = bool(use_rewards)
        self.reward_weight: float = float(reward_weight)
        self.cost_bps_per_leg: float = float(cost_bps_per_leg)
        self.gas_usd_per_leg: float = float(gas_usd_per_leg)
        self.bars_per_year: float = float(bars_per_year)
        super().__init__()

    _internal_state: MorphoSupplyInternalState
    _global_state: MorphoSupplyGlobalState

    def _initialize_states(self) -> None:
        self._internal_state = MorphoSupplyInternalState()
        self._global_state = MorphoSupplyGlobalState()

    @property
    def internal_state(self) -> MorphoSupplyInternalState:  # type: ignore[override]
        return self._internal_state

    @property
    def global_state(self) -> MorphoSupplyGlobalState:  # type: ignore[override]
        return self._global_state

    # ------------------------------------------------------------ readouts
    @property
    def balance(self) -> float:
        """Supplied notional in USD (the position's only asset)."""
        return self._internal_state.supplied

    @property
    def effective_utilization(self) -> float:
        """``u`` with this position included (the historical ``u`` when exogenous)."""
        return self._utilization_at(self._global_state, self._internal_state.supplied)

    @property
    def effective_rate_at_target(self) -> float:
        """Annual APY form of the ``rateAtTarget`` this position actually faces (drifted in ``irm_feedback``)."""
        return math.expm1(self._rate_at_target(self._global_state))

    @property
    def withdrawable(self) -> float:
        """USD this position can actually withdraw now — the exit gate.

        ``liquidity_assets_usd`` in exogenous mode; ``S' − B'`` with the
        position included otherwise (so our own supply is withdrawable as long
        as nobody borrows it against us).
        """
        state = self._global_state
        if not self.own_impact:
            return max(state.liquidity_assets_usd, 0.0)
        supplied = self._internal_state.supplied
        total_supply = state.supply_assets_usd + supplied
        total_borrow = state.borrow_assets_usd + self._absorbed(state) * supplied
        return max(total_supply - total_borrow, 0.0)

    @property
    def max_withdrawable_notional(self) -> float:
        """Largest ``amount_in_notional`` :meth:`action_withdraw` accepts right now.

        The destination receives this amount and ``amount + fee`` leaves the
        position, so it is ``(min(supplied, withdrawable) − gas) / (1 + bps)``;
        ``0`` when what could leave does not even cover the gas. This is what a
        strategy should size a withdraw leg to.
        """
        gross = min(self._internal_state.supplied, self.withdrawable)
        if gross <= self.gas_usd_per_leg:
            return 0.0
        return max((gross - self.gas_usd_per_leg) / (1.0 + self.cost_bps_per_leg / 1e4), 0.0)

    @property
    def effective_supply_rate(self) -> float:
        """The per-bar rate this position actually earns (self-diluted in ``own_impact`` mode)."""
        return self._effective_supply_rate(self._global_state, self._internal_state.supplied)

    @property
    def effective_supply_apy(self) -> float:
        """Annual effective form of :attr:`effective_supply_rate`."""
        return self.projected_supply_apy(self._internal_state.supplied)

    def projected_supply_apy(self, position_notional: float, state: MorphoSupplyGlobalState | None = None) -> float:
        """Annual supply APY a position of ``position_notional`` USD would earn here.

        The whole-position figure (not incremental): what the rate would be if
        our supply in this market *were* ``position_notional``. Under
        ``own_impact`` it carries the self-dilution of that size through the
        IRM curve; otherwise it is the market's ``supply_apy``. Rank markets on
        this with the capital you intend to deploy, not on the headline rate —
        the headline systematically favours the small, high-utilization markets
        that a position of your size would dilute the most.

        Args:
            position_notional: whole position size in USD.
            state: market snapshot to evaluate on; the current one by default.
                Pass a lagged or window-averaged snapshot to size without
                looking at the bar being traded.
        """
        if position_notional < 0:
            raise MorphoSupplyException(f"position_notional must be >= 0, got {position_notional}")
        state = self._global_state if state is None else state
        if not self.own_impact:
            return state.supply_apy + self._reward_apy(state, 0.0)
        # Continuous annualisation, the loader's own convention between
        # ``supply_rate`` and ``supply_apy`` — so this inverts exactly.
        return math.expm1(self._effective_supply_rate(state, position_notional) * self.bars_per_year)

    def capacity_at(self, target_apy: float, state: MorphoSupplyGlobalState | None = None) -> float:
        """Largest whole position (USD) that still earns at least ``target_apy`` here.

        Inverts the IRM curve: find ``u*`` with ``supply_apy(u*) = target``,
        then solve ``(B + α·P) / (S + P) = u*`` for ``P``. Returns ``0`` when
        the market already pays less than the target (or cannot reach it even
        at full utilization), ``inf`` when absorption keeps utilization above
        ``u*`` for any size (``α ≥ u*``: adding capital never dilutes below the
        target). In exogenous mode the market's headline rate decides:
        ``inf`` if it clears the target, ``0`` otherwise.
        """
        if target_apy < 0:
            raise MorphoSupplyException(f"target_apy must be >= 0, got {target_apy}")
        state = self._global_state if state is None else state
        if not self.own_impact:
            return float("inf") if self.projected_supply_apy(0.0, state) >= target_apy else 0.0
        if self.use_rewards and state.reward_apy > 0.0:
            return self._capacity_by_bisection(target_apy, state)
        u_star = utilization_for_supply_apy(target_apy, self._rate_at_target(state), state.fee)
        if u_star is None:
            return 0.0
        alpha = self._absorbed(state)
        if u_star <= alpha:
            return float("inf")
        position = (state.borrow_assets_usd - u_star * state.supply_assets_usd) / (u_star - alpha)
        return max(position, 0.0)

    # --------------------------------------------------------- actions
    def action_deposit(self, amount_in_notional: float) -> None:
        """Supply ``amount_in_notional`` USD, net of this leg's cost (gas + bps)."""
        if amount_in_notional < 0:
            raise MorphoSupplyException(f"deposit amount must be >= 0, got {amount_in_notional}")
        if amount_in_notional == 0.0:
            return  # an empty leg is not a transaction: no gas, nothing recorded
        fee = self._leg_fee(amount_in_notional)
        if fee > amount_in_notional:
            raise MorphoSupplyException(
                f"deposit of {amount_in_notional} does not cover its own cost {fee} (gas {self.gas_usd_per_leg})"
            )
        self._internal_state.supplied += amount_in_notional - fee
        self.record_execution("deposit", amount_in_notional, fee)

    def action_withdraw(self, amount_in_notional: float) -> None:
        """Withdraw so that ``amount_in_notional`` USD reaches the destination.

        ``amount + fee`` leaves the position; both must fit inside the position
        and inside the market's withdrawable liquidity.

        Raises:
            MorphoSupplyException: if ``amount + fee`` exceeds the position or
                the withdrawable amount. The gate is structural: a caller cannot
                withdraw liquidity the market does not have.
        """
        if amount_in_notional < 0:
            raise MorphoSupplyException(f"withdraw amount must be >= 0, got {amount_in_notional}")
        if amount_in_notional == 0.0:
            return  # an empty leg is not a transaction: no gas, nothing recorded
        fee = self._leg_fee(amount_in_notional)
        gross = amount_in_notional + fee
        if gross > self._internal_state.supplied * (1.0 + 1e-12):
            raise MorphoSupplyException(
                f"withdraw exceeds position: {amount_in_notional} + fee {fee} > {self._internal_state.supplied}"
            )
        withdrawable = self.withdrawable
        if gross > withdrawable * (1.0 + 1e-12):
            raise MorphoSupplyException(
                f"withdraw of {amount_in_notional} + fee {fee} exceeds withdrawable {withdrawable}: "
                "the market's utilization gates the exit"
            )
        self._internal_state.supplied = max(self._internal_state.supplied - gross, 0.0)
        self.record_execution("withdraw", amount_in_notional, fee)

    # --------------------------------------------------------- lifecycle
    def update_state(self, state: MorphoSupplyGlobalState) -> None:
        """Validate the snapshot, apply it, accrue one bar.

        The engine calls this before ``predict``, so the accrual uses the
        position as it stood at the end of the previous bar. In
        ``irm_feedback`` mode the controller drift for this bar is accumulated
        first, from the utilization offset the standing position causes.
        """
        for field in fields(MorphoSupplyGlobalState):
            value = getattr(state, field.name)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise MorphoSupplyException(f"{field.name} must be finite, got {value!r}")
        if not 0.0 <= state.utilization <= 1.0:
            raise MorphoSupplyException(f"utilization must be in [0, 1], got {state.utilization}")
        if not 0.0 <= state.fee <= MAX_MARKET_FEE:
            raise MorphoSupplyException(f"fee must be in [0, {MAX_MARKET_FEE}], got {state.fee}")
        if state.rate_at_target < 0.0:
            raise MorphoSupplyException(f"rate_at_target must be >= 0, got {state.rate_at_target}")
        if state.reward_apy < 0.0:
            raise MorphoSupplyException(f"reward_apy must be >= 0, got {state.reward_apy}")
        if state.supply_rate < -1.0:
            raise MorphoSupplyException(f"supply_rate must be >= -1, got {state.supply_rate}")
        if min(state.supply_assets_usd, state.borrow_assets_usd, state.liquidity_assets_usd) < 0.0:
            raise MorphoSupplyException("market sizes must be >= 0")
        if self.irm_feedback and self._internal_state.supplied > 0.0:
            own = self._utilization_at(state, self._internal_state.supplied)
            self._internal_state.rstar_log_drift += (
                ADJUSTMENT_SPEED * (adaptive_curve_error(own) - adaptive_curve_error(state.utilization))
                / self.bars_per_year
            )
        self._global_state = state
        rate = self._effective_supply_rate(state, self._internal_state.supplied)
        if rate < -1.0:
            raise MorphoSupplyException(
                f"effective supply rate would flip the balance negative: {rate} (check absorption)"
            )
        self._internal_state.supplied *= 1.0 + rate

    # ----------------------------------------------------------- helpers
    def _leg_fee(self, amount_in_notional: float) -> float:
        """Fixed gas plus the proportional fee; an empty leg is free (no transaction is sent)."""
        if amount_in_notional <= 0.0:
            return 0.0
        return self.gas_usd_per_leg + amount_in_notional * self.cost_bps_per_leg / 1e4

    @property
    def leg_cost_model(self) -> tuple[float, float]:
        """``(gas_usd_per_leg, cost_bps_per_leg)`` — what one leg costs, for strategies that price trades."""
        return self.gas_usd_per_leg, self.cost_bps_per_leg

    def _absorbed(self, state: MorphoSupplyGlobalState) -> float:
        """Effective absorption: ``min(absorption, u)``.

        Borrowers taking up more than ``u`` of our supply would push utilization
        *above* its historical level — new supply creating its own borrow demand
        and paying us a higher rate the bigger we are. That is not physical:
        adding supply can at most leave utilization where it was (the
        unmet-demand case at ``u ≈ 1``), never raise it. So ``absorption = 1``
        means "fully absorbed up to the prior utilization", i.e. no dilution.
        """
        return min(self.absorption, state.utilization)

    def _utilization_at(self, state: MorphoSupplyGlobalState, position_notional: float) -> float:
        if not self.own_impact:
            return state.utilization
        total_supply = state.supply_assets_usd + position_notional
        if total_supply <= 0.0:
            return 0.0
        total_borrow = state.borrow_assets_usd + self._absorbed(state) * position_notional
        return min(max(total_borrow / total_supply, 0.0), 1.0)

    def curve_rate_at_target(self, state: MorphoSupplyGlobalState | None = None) -> float:
        """Continuous annual ``rateAtTarget`` the IRM curve is evaluated with for ``state`` (drift applied).

        Public so a strategy can run the curve itself (e.g. vectorised over many
        markets) and stay consistent with what this entity accrues.
        """
        return self._rate_at_target(self._global_state if state is None else state)

    def _rate_at_target(self, state: MorphoSupplyGlobalState) -> float:
        """Continuous annual ``rateAtTarget`` the curve is evaluated with (drifted in ``irm_feedback``)."""
        rate = math.log1p(state.rate_at_target)
        if self.irm_feedback and self._internal_state.rstar_log_drift != 0.0:
            rate = min(max(rate * math.exp(self._internal_state.rstar_log_drift), MIN_RATE_AT_TARGET),
                       MAX_RATE_AT_TARGET)
        return rate

    def _effective_supply_rate(self, state: MorphoSupplyGlobalState, position_notional: float) -> float:
        """Per-bar rate a position of ``position_notional`` earns under ``state``.

        ``rate_at_target`` arrives as an annual APY (the loader's column), while
        the IRM curve is unit-preserving — it takes and returns a *rate*, not an
        APY. Convert APY → rate → curve → APY before deriving a per-bar rate.
        """
        if not self.own_impact:
            reward = self._reward_apy(state, 0.0)
            return state.supply_rate + (math.log1p(reward) / self.bars_per_year if reward > 0.0 else 0.0)
        utilization = self._utilization_at(state, position_notional)
        supply_apy = supply_apy_from_curve(utilization, self._rate_at_target(state), state.fee)
        return math.log1p(supply_apy + self._reward_apy(state, position_notional)) / self.bars_per_year

    def _reward_apy(self, state: MorphoSupplyGlobalState, position_notional: float) -> float:
        """Reward APY our position earns: ``weight · reward_apy · S / (S + P)`` (0 unless ``use_rewards``)."""
        if not self.use_rewards or state.reward_apy <= 0.0:
            return 0.0
        total = state.supply_assets_usd + (position_notional if self.own_impact else 0.0)
        share = state.supply_assets_usd / total if total > 0.0 else 1.0
        return self.reward_weight * state.reward_apy * share

    def _capacity_by_bisection(self, target_apy: float, state: MorphoSupplyGlobalState) -> float:
        """``capacity_at`` when rewards make the APY a non-closed-form function of size (still decreasing)."""
        if self.projected_supply_apy(0.0, state) < target_apy:
            return 0.0
        high = max(state.supply_assets_usd, 1.0)
        while self.projected_supply_apy(high, state) >= target_apy:
            high *= 4.0
            if high > 1e15:
                return float("inf")
        low = 0.0
        for _ in range(80):
            mid = 0.5 * (low + high)
            if self.projected_supply_apy(mid, state) >= target_apy:
                low = mid
            else:
                high = mid
        return low
