"""Protocol-neutral cumulative execution ledger (issue #68).

Entities record **successful, fee-bearing trades** through
:meth:`BaseEntity.record_execution`; the strategy owns one
:class:`ExecutionLedger` per run and stamps each record with the current
observation timestamp and entity name.

Conventions:

* records are append-only and cumulative across the run — no inferred
  fees from balance changes, no future data;
* ``traded_notional`` is the notional value actually swapped/traded
  (e.g. only the swapped half of an LP zap-in), in the portfolio
  accounting unit;
* ``fee_paid`` is the fee charged on that trade in the same unit;
* deposits, withdrawals, borrowing, repayment, and internal transfers
  are **not** recorded (they are capital movements, not trades).
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, List, Optional


@dataclass
class ExecutionRecord:
    """One recorded trade execution."""

    timestamp: Optional[datetime]  # observation timestamp; None outside a run
    entity: str                    # registry name of the entity
    action: str                    # action name, e.g. ``buy`` / ``open_position``
    traded_notional: float         # notional value traded (accounting unit)
    fee_paid: float                # fee charged on the trade (accounting unit)


class ExecutionLedger:
    """Cumulative ledger of :class:`ExecutionRecord` rows for one run."""

    def __init__(self) -> None:
        self.records: List[ExecutionRecord] = []
        self.total_traded_notional: float = 0.0
        self.total_fees_paid: float = 0.0
        # Context set by the strategy before dispatching actions.
        self._current_timestamp: Optional[datetime] = None
        self._current_entity: Optional[str] = None

    def reset(self) -> None:
        """Start a fresh run: drop all records and totals.

        Replaces (not mutates) the record list so a previously returned
        :class:`StrategyResult` keeps its own snapshot.
        """
        self.records = []
        self.total_traded_notional = 0.0
        self.total_fees_paid = 0.0
        self._current_timestamp = None
        self._current_entity = None

    def set_context(self, timestamp: Optional[datetime] = None,
                    entity_name: Optional[str] = None) -> None:
        """Set default timestamp/entity applied to subsequent ``record`` calls."""
        if timestamp is not None:
            self._current_timestamp = timestamp
        if entity_name is not None:
            self._current_entity = entity_name

    def record(self, action: str, traded_notional: float, fee_paid: float,
               entity: Optional[str] = None) -> None:
        """Append one record and update cumulative totals.

        ``entity`` overrides ``_current_entity`` when given (strategies
        pass the registered entity name explicitly).
        """
        record = ExecutionRecord(
            timestamp=self._current_timestamp,
            entity=entity if entity is not None else (self._current_entity or ""),
            action=action,
            traded_notional=float(traded_notional),
            fee_paid=float(fee_paid),
        )
        self.records.append(record)
        self.total_traded_notional += record.traded_notional
        self.total_fees_paid += record.fee_paid


#: Recorder signature handed to entities: ``(action, traded_notional, fee_paid)``.
ExecutionRecorder = Callable[[str, float, float], None]
