"""Base class for instruments with a maturity (Pendle PT, Boros yield units).

Time never reaches an entity directly: :meth:`BaseEntity.update_state`
receives only the ``GlobalState``, so the observation builder computes
``seconds_to_expiry = expiry - observation.timestamp`` and puts it on the
state. The entity treats ``<= 0`` as matured and never mutates the
value it is given.

The default ``seconds_to_expiry`` is ``0.0`` on purpose: an entity fed a
state that forgot the field is matured and every trade raises, instead
of silently accruing on a phantom term.
"""
import math
from dataclasses import dataclass
from typing import Optional, Type

from fractal.core.base.entity import BaseEntity, EntityException, GlobalState
from fractal.core.base.time import SECONDS_PER_YEAR

_TERM_TOLERANCE_SECONDS = 1e-6


@dataclass
class BaseFixedTermGlobalState(GlobalState):
    """Global state carrying the time left on a fixed-term instrument.

    Attributes:
        seconds_to_expiry (float): ``expiry - now`` in seconds as computed
            by the observation builder. ``<= 0`` means matured. Default
            ``0.0`` (matured) so a forgotten field fails loudly.
    """
    seconds_to_expiry: float = 0.0


class BaseFixedTermEntity(BaseEntity):
    """Mixin-style base for entities that mature.

    Provides the term readouts, the input-validation helper that every
    ``update_state`` must call first, the guard trading actions call, and
    the ``_check_maturity`` hook — the maturity counterpart of the
    ``_check_liquidation`` convention used by lending and perp entities.

    Subclasses set ``_exception_cls`` to their own
    :class:`EntityException` subclass so guard failures carry the
    protocol's exception type.
    """
    _exception_cls: Type[EntityException] = EntityException

    def __init__(self, *args, **kwargs) -> None:
        self._last_seconds_to_expiry: Optional[float] = None
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------- readouts
    @property
    def seconds_to_expiry(self) -> float:
        """Seconds left on the term per the last applied state (``0.0`` before any)."""
        state = getattr(self, "_global_state", None)
        return float(getattr(state, "seconds_to_expiry", 0.0))

    @property
    def years_to_expiry(self) -> float:
        """``max(seconds_to_expiry, 0) / SECONDS_PER_YEAR`` (ACT/365)."""
        return max(self.seconds_to_expiry, 0.0) / SECONDS_PER_YEAR

    @property
    def is_matured(self) -> bool:
        """``True`` once ``seconds_to_expiry <= 0``."""
        return self.seconds_to_expiry <= 0.0

    # ------------------------------------------------------------- guards
    def _validate_term(self, state: BaseFixedTermGlobalState) -> None:
        """Reject a non-finite or increasing ``seconds_to_expiry``.

        Time to expiry can only shrink between observations; an increase
        means the builder fed unsorted data or forgot to compute the
        field for a later bar. Equal values (repeated bars) are fine.
        Records the value so the next call can compare against it.
        """
        seconds = getattr(state, "seconds_to_expiry", None)
        if seconds is None or not math.isfinite(seconds):
            raise self._exception_cls(
                f"seconds_to_expiry must be a finite number, got {seconds!r}"
            )
        last = self._last_seconds_to_expiry
        if last is not None and seconds > last + _TERM_TOLERANCE_SECONDS:
            raise self._exception_cls(
                f"seconds_to_expiry increased from {last} to {seconds}; "
                f"observations must be fed in chronological order"
            )
        self._last_seconds_to_expiry = float(seconds)

    def _require_not_matured(self, action: str) -> None:
        """Raise when ``action`` is attempted on a matured instrument."""
        if self.is_matured:
            raise self._exception_cls(
                f"{action} is not available after expiry "
                f"(seconds_to_expiry={self.seconds_to_expiry})"
            )

    def _check_maturity(self) -> None:
        """Hook called at the end of ``update_state`` once the term is applied.

        Default: no-op. Subclasses override to settle a matured position
        (e.g. a Boros yield unit is worth zero at maturity). Pendle PT
        does nothing here on purpose — the protocol has no auto-redeem.
        """
