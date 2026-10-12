"""Morpho Blue supply-side ("LP") entity: loan-token supply with a liquidity gate.

The supplier-side mirror of
:class:`fractal.core.entities.protocols.morpho.MorphoEntity`: it holds loan-token
supply (USD notional), earns the market's supply rate, and — unlike every other
lending entity in this repo — **cannot always get its money back**. Morpho's
``withdraw`` reverts when ``totalBorrowAssets > totalSupplyAssets``, so the
withdrawable amount is ``liquidityAssets = supplyAssets − borrowAssets``, which
is exactly 0 at full utilization. Entering is unconditional; exiting is not.

Two replay modes (RQ4):

``own_impact=False``
    Exogenous replay: the historical ``supply_rate`` and
    ``liquidity_assets_usd`` are taken as given — the "our size changes nothing"
    baseline. Correct only for a position small relative to the market.

``own_impact=True`` (default)
    State-carrying replay: the position is added to the historical totals and
    the rate and the gate are recomputed with it included::

        S' = S + P
        B' = B + absorption·P
        u' = B' / S'
        liquidity' = S' − B'
        borrow_apy' = AdaptiveCurveIRM(u', rate_at_target)
        supply_rate' = per-bar(borrow_apy' · u' · (1 − fee))

    ``absorption = 0`` is pure self-dilution — the arithmetic of CONTEXT §2.3,
    where $1.0M into a $15.83M/`u`=92.7% market moves ``u`` to 87.2% and the
    position's own realised rate from 8.21% to ≈4.2%. ``absorption = u`` means
    borrowers take up the new supply and utilization is preserved. This is the
    mode that makes RQ4/RQ10 answerable instead of assumed.

Known limitation (RQ4 boundary): ``rate_at_target`` is replayed exogenously.
The controller's *faster* ``R*`` decay under our own lower utilization is not
modelled, so ``own_impact=True`` still overstates the rate we keep. Costs are
charged on the moved notional per leg and reported through ``record_execution``,
so :class:`~fractal.core.base.strategy.result.StrategyResult`'s
``fees_paid``/``turnover``/``fee_drag`` pick them up unchanged.
"""
import math
from dataclasses import dataclass

from fractal.core.base.entity import BaseEntity, EntityException, GlobalState, InternalState
from fractal.core.entities.models.morpho_math import adaptive_curve_borrow_apr

__all__ = [
    "MAX_MARKET_FEE",
    "MorphoLpEntity",
    "MorphoLpException",
    "MorphoLpGlobalState",
    "MorphoLpInternalState",
]

MAX_MARKET_FEE = 0.25  # ConstantsLib.MAX_FEE
BARS_PER_YEAR_HOURLY = 8760.0


class MorphoLpException(EntityException):
    """Errors raised by :class:`MorphoLpEntity`."""


@dataclass
class MorphoLpGlobalState(GlobalState):
    """Market context for one Morpho Blue market, all USD notional.

    Attributes:
        utilization: ``borrowAssets / supplyAssets``, in ``[0, 1]``.
        supply_rate: **per-bar** rate the market pays its suppliers — the
            supplier-side sibling of ``LendingHistory.borrowing_rate``. It is
            already net of the market fee (the API's ``supplyApy`` is), and is
            the rate applied in ``own_impact=False`` mode.
        supply_apy / borrow_apy: the market's annual effective rates, for
            reporting and for RQ1-style checks.
        rate_at_target: AdaptiveCurveIRM ``rateAtTarget`` as an **annual** APY
            (the loader converts the API's per-second WAD value).
        fee: market fee as a fraction (``[0, 0.25]``).
        supply_assets_usd / borrow_assets_usd / liquidity_assets_usd: market
            sizes; ``liquidity_assets_usd`` is the withdrawable amount and the
            only sound gate input (CONTEXT A7 — ``(1−u)·S`` is merely a sanity
            check, since it ignores accrued interest).
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


@dataclass
class MorphoLpInternalState(InternalState):
    """Position state.

    Attributes:
        supplied: loan-token supply in USD notional.
    """
    supplied: float = 0.0


class MorphoLpEntity(BaseEntity):
    """Supplier position in one isolated Morpho Blue market.

    Args:
        own_impact: recompute the rate and the exit gate with this position
            included (``True``) or replay the historical values (``False``).
        absorption: share of the position borrowers take up, in ``[0, 1]``;
            only meaningful with ``own_impact=True``.
        cost_bps_per_leg: fee charged on the moved notional of every deposit
            and withdraw, in basis points. Costs live here rather than in the
            strategy so a fully deployed portfolio never needs idle cash to pay
            them, and so they surface in ``StrategyResult.fees_paid``.
        bars_per_year: bars per year, used to turn an annual APY into the
            per-bar rate in ``own_impact`` mode. The default matches the
            hourly loader resolution; it must match the data's bar length.
    """

    def __init__(
        self,
        *,
        own_impact: bool = True,
        absorption: float = 0.0,
        cost_bps_per_leg: float = 0.0,
        bars_per_year: float = BARS_PER_YEAR_HOURLY,
    ) -> None:
        if not 0.0 <= absorption <= 1.0:
            raise MorphoLpException(f"absorption must be in [0, 1], got {absorption}")
        if cost_bps_per_leg < 0:
            raise MorphoLpException(f"cost_bps_per_leg must be >= 0, got {cost_bps_per_leg}")
        if bars_per_year <= 0:
            raise MorphoLpException(f"bars_per_year must be > 0, got {bars_per_year}")
        self.own_impact: bool = bool(own_impact)
        self.absorption: float = float(absorption)
        self.cost_bps_per_leg: float = float(cost_bps_per_leg)
        self.bars_per_year: float = float(bars_per_year)
        super().__init__()

    _internal_state: MorphoLpInternalState
    _global_state: MorphoLpGlobalState

    def _initialize_states(self) -> None:
        self._internal_state = MorphoLpInternalState()
        self._global_state = MorphoLpGlobalState()

    # ------------------------------------------------------------ readouts
    @property
    def balance(self) -> float:
        """Supplied notional in USD (the position's only asset)."""
        return self._internal_state.supplied

    @property
    def effective_utilization(self) -> float:
        """``u`` with this position included (the historical ``u`` when exogenous)."""
        state = self._global_state
        if not self.own_impact:
            return state.utilization
        total_supply = state.supply_assets_usd + self._internal_state.supplied
        if total_supply <= 0.0:
            return 0.0
        total_borrow = state.borrow_assets_usd + self.absorption * self._internal_state.supplied
        return min(max(total_borrow / total_supply, 0.0), 1.0)

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
        total_supply = state.supply_assets_usd + self._internal_state.supplied
        total_borrow = state.borrow_assets_usd + self.absorption * self._internal_state.supplied
        return max(total_supply - total_borrow, 0.0)

    @property
    def effective_supply_rate(self) -> float:
        """The per-bar rate this position actually earns (self-diluted in ``own_impact`` mode)."""
        return self._effective_supply_rate(self._global_state)

    @property
    def effective_supply_apy(self) -> float:
        """Annual effective form of :attr:`effective_supply_rate`."""
        return (1.0 + self._effective_supply_rate(self._global_state)) ** self.bars_per_year - 1.0

    # --------------------------------------------------------- actions
    def action_deposit(self, amount_in_notional: float) -> None:
        """Supply ``amount_in_notional`` USD, net of this leg's cost."""
        if amount_in_notional < 0:
            raise MorphoLpException(f"deposit amount must be >= 0, got {amount_in_notional}")
        fee = self._leg_fee(amount_in_notional)
        self._internal_state.supplied += amount_in_notional - fee
        self.record_execution("deposit", amount_in_notional, fee)

    def action_withdraw(self, amount_in_notional: float) -> None:
        """Withdraw ``amount_in_notional`` USD, gated by the market's liquidity.

        Raises:
            MorphoLpException: if the amount exceeds the position or the
                withdrawable amount. The gate is structural: a caller cannot
                withdraw liquidity the market does not have.
        """
        if amount_in_notional < 0:
            raise MorphoLpException(f"withdraw amount must be >= 0, got {amount_in_notional}")
        if amount_in_notional > self._internal_state.supplied:
            raise MorphoLpException(
                f"withdraw exceeds position: {amount_in_notional} > {self._internal_state.supplied}"
            )
        withdrawable = self.withdrawable
        if amount_in_notional > withdrawable:
            raise MorphoLpException(
                f"withdraw of {amount_in_notional} exceeds withdrawable {withdrawable}: "
                "the market's utilization gates the exit"
            )
        fee = self._leg_fee(amount_in_notional)
        self._internal_state.supplied -= amount_in_notional
        self.record_execution("withdraw", amount_in_notional, fee)

    # --------------------------------------------------------- lifecycle
    def update_state(self, state: MorphoLpGlobalState) -> None:
        """Apply the market state, accrue one bar, then re-evaluate the position.

        The engine calls this before ``predict``, so the accrual uses the
        position as it stood at the end of the previous bar.
        """
        if not 0.0 <= state.fee <= MAX_MARKET_FEE:
            raise MorphoLpException(f"fee must be in [0, {MAX_MARKET_FEE}], got {state.fee}")
        if state.rate_at_target < 0.0:
            raise MorphoLpException(f"rate_at_target must be >= 0, got {state.rate_at_target}")
        if state.supply_rate < -1.0:
            raise MorphoLpException(f"supply_rate must be >= -1, got {state.supply_rate}")
        self._global_state = state
        rate = self._effective_supply_rate(state)
        if rate < -1.0:
            raise MorphoLpException(
                f"effective supply rate would flip the balance negative: {rate} (check absorption)"
            )
        self._internal_state.supplied *= 1.0 + rate

    # ----------------------------------------------------------- helpers
    def _leg_fee(self, amount_in_notional: float) -> float:
        return amount_in_notional * self.cost_bps_per_leg / 1e4

    def _effective_supply_rate(self, state: MorphoLpGlobalState) -> float:
        """Per-bar rate this position earns, with the position's own impact applied.

        ``rate_at_target`` arrives as an annual APY (the loader's column), while
        :func:`adaptive_curve_borrow_apr` is unit-preserving — it takes and
        returns a *rate*, not an APY. So: APY → rate → curve → APY, exactly as
        the RQ1 validator does against live data (0.002 bp median residual).
        """
        if not self.own_impact:
            return state.supply_rate
        utilization = self.effective_utilization
        borrow_rate = adaptive_curve_borrow_apr(utilization, math.log1p(state.rate_at_target))
        supply_apy = math.expm1(borrow_rate) * utilization * (1.0 - state.fee)
        return math.log1p(supply_apy) / self.bars_per_year
