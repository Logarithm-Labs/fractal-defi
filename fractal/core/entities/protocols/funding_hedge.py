from dataclasses import dataclass
from typing import Literal, Optional

from fractal.core.base.entity import (
    BaseEntity,
    EntityException,
    GlobalState,
    InternalState,
)


# Same convention as MorphoEntity / PendlePTEntity: Julian year so the
# three entities share an annualisation base.
SECONDS_PER_YEAR: float = 365.25 * 24 * 3600


@dataclass
class FundingHedgeGlobalState(GlobalState):
    """Market context for a perpetual-funding carry leg.

    Attributes:
        funding_rate: Annualised funding rate in decimal (``0.10`` =
            10% APR). Hyperliquid quotes a per-hour funding rate; the
            loader multiplies by ``24 * 365.25`` before populating this
            field, so the entity sees a smooth annualised rate.
        timestamp_seconds: Unix epoch seconds of the observation. The
            strategy populates this from ``Observation.timestamp`` so
            :meth:`FundingHedgeEntity.update_state` can compute
            ``dt`` between successive updates.
    """

    funding_rate: float = 0.0
    timestamp_seconds: float = 0.0


@dataclass
class FundingHedgeInternalState(InternalState):
    """Position state inside the funding-rate hedge leg.

    Attributes:
        notional: Current long-funding-rate exposure size in USDC.
            Set by :meth:`FundingHedgeEntity.action_deposit` /
            :meth:`FundingHedgeEntity.action_withdraw`.
        accrued_pnl: Running PnL accrued from funding payments, in
            USDC. Signed: positive under favourable funding for the
            configured side, negative otherwise. Persists through
            ``action_withdraw`` (realised PnL is *not* unwound by
            closing notional).
        last_timestamp: Unix epoch seconds of the last observation.
            ``None`` until the first :meth:`FundingHedgeEntity.update_state`
            call; thereafter the entity accrues PnL over
            ``timestamp_seconds - last_timestamp``.
    """

    notional: float = 0.0
    accrued_pnl: float = 0.0
    last_timestamp: Optional[float] = None


_ALLOWED_DIRECTIONS = ("long", "short")


@dataclass
class FundingHedgeConfig:
    """Configuration for the funding-rate hedge entity.

    Attributes:
        direction: ``"long"`` (default) means the position *receives*
            funding when ``funding_rate > 0`` — the side a Boros
            funding-token holder takes. ``"short"`` flips the sign:
            the position *pays* funding when the rate is positive.
            Any other value is rejected by :class:`FundingHedgeEntity`
            at construction.
    """

    direction: Literal["long", "short"] = "long"


class FundingHedgeEntity(
    BaseEntity[FundingHedgeGlobalState, FundingHedgeInternalState]
):
    """Perpetual-funding-rate carry leg.

    Notional unit: USDC. The entity tracks a notional exposure size
    and an accrued PnL stream; its equity contribution to the loop's
    NAV is exactly :attr:`accrued_pnl` (no cash leg held).
    """

    _internal_state: FundingHedgeInternalState
    _global_state: FundingHedgeGlobalState

    def __init__(self, config: Optional[FundingHedgeConfig] = None) -> None:
        cfg = config or FundingHedgeConfig()
        if cfg.direction not in _ALLOWED_DIRECTIONS:
            raise EntityException(
                f"FundingHedgeConfig.direction must be one of "
                f"{_ALLOWED_DIRECTIONS}, got {cfg.direction!r}"
            )
        self._config = cfg
        super().__init__()

    def _initialize_states(self) -> None:
        self._global_state = FundingHedgeGlobalState()
        self._internal_state = FundingHedgeInternalState()

    # ------------------------------------------------------------------
    # Time evolution
    # ------------------------------------------------------------------

    def update_state(self, state: FundingHedgeGlobalState) -> None:
        """Apply new market context and accrue funding PnL.

        Flow:
            1. On the first call (``last_timestamp is None``) we cannot
               compute ``dt``, so we simply store the new state and the
               timestamp — no accrual happens yet. This matches the
               first-observation convention used by :class:`MorphoEntity`.
            2. On subsequent calls we accrue
               ``s * notional * funding_rate * dt_years`` into
               ``accrued_pnl``, where ``s = +1`` for ``direction="long"``
               and ``s = -1`` for ``direction="short"``.

        The accrual uses *current* ``notional`` — deposits / withdraws
        applied between observations affect only PnL from the next step
        onward, which mirrors how a real perp position accrues on its
        live notional at each funding interval.
        """
        if self._internal_state.last_timestamp is not None:
            dt_years = (
                state.timestamp_seconds - self._internal_state.last_timestamp
            ) / SECONDS_PER_YEAR
            sign = 1.0 if self._config.direction == "long" else -1.0
            self._internal_state.accrued_pnl += (
                sign
                * self._internal_state.notional
                * state.funding_rate
                * dt_years
            )
        self._internal_state.last_timestamp = state.timestamp_seconds
        self._global_state = state

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------

    @property
    def balance(self) -> float:
        """Equity contribution of the hedge leg in USDC.

        The entity holds no cash leg of its own (the delta-hedge offset
        is assumed perfect and lives outside this entity, by design —
        see module docstring). Its equity contribution to the loop's
        NAV is therefore exactly the realised + unrealised funding PnL.
        """
        return self._internal_state.accrued_pnl

    # ------------------------------------------------------------------
    # Action methods
    # ------------------------------------------------------------------

    def action_deposit(self, amount_in_notional: float) -> None:
        """Open or extend the funding-rate position by ``amount_in_notional`` USDC.

        Note we do not track a cash balance — ``amount_in_notional`` is
        the size of the funding-leg notional being added, not a USDC
        deposit. The implicit assumption is that the delta-neutral
        offset (spot / futures) is funded outside this entity.
        """
        if amount_in_notional < 0:
            raise EntityException(
                "action_deposit: amount must be non-negative, "
                f"got {amount_in_notional}"
            )
        self._internal_state.notional += amount_in_notional

    def action_withdraw(self, amount_in_notional: float) -> None:
        """Close or reduce the notional by ``amount_in_notional`` USDC.

        Realised PnL stays in :attr:`accrued_pnl` — closing the
        position does *not* unwind past funding payments, just as a
        perp close does not claw back collected funding.
        """
        if amount_in_notional < 0:
            raise EntityException(
                "action_withdraw: amount must be non-negative, "
                f"got {amount_in_notional}"
            )
        if amount_in_notional > self._internal_state.notional:
            raise EntityException(
                f"action_withdraw: requested {amount_in_notional} but only "
                f"{self._internal_state.notional} notional held"
            )
        self._internal_state.notional -= amount_in_notional
