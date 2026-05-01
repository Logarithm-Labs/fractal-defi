"""Runtime smoke test against the installed wheel.

Runs the same closed-form-verifiable lending strategy used by
``examples/quick_start.py``: deposit $10k at 5% APY for one year of
hourly observations, verify the final balance matches
``P · (1 + r/n)^n`` to floating-point precision.

Failure here means the wheel got installed but something at runtime
is broken — a missing data file, a regressed entity ``update_state``,
a serialization issue, etc. Lighter than full pytest but smoke-tests
the live framework loop end-to-end.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import List

from fractal.core.base import (
    Action,
    ActionToTake,
    BaseStrategy,
    BaseStrategyParams,
    NamedEntity,
    Observation,
)
from fractal.core.entities import SimpleLendingEntity, SimpleLendingGlobalState


@dataclass
class _Params(BaseStrategyParams):
    INITIAL_BALANCE: float = 10_000.0
    LENDING_APY: float = 0.05


class _PassiveLender(BaseStrategy[_Params]):
    def set_up(self) -> None:
        self.register_entity(NamedEntity("LENDING", SimpleLendingEntity()))
        self._funded = False

    def predict(self) -> List[ActionToTake]:
        if self._funded:
            return []
        self._funded = True
        return [ActionToTake("LENDING", Action(
            "deposit", {"amount_in_notional": self._params.INITIAL_BALANCE},
        ))]


def main() -> int:
    apy = 0.05
    days = 365
    hours = days * 24
    per_step_rate = apy / hours
    start = datetime(2024, 1, 1, tzinfo=UTC)

    observations = [
        Observation(timestamp=start + timedelta(hours=i), states={
            "LENDING": SimpleLendingGlobalState(
                collateral_price=1.0, debt_price=1.0,
                lending_rate=per_step_rate, borrowing_rate=0.0,
            ),
        })
        for i in range(hours + 1)
    ]

    s = _PassiveLender(params=_Params(INITIAL_BALANCE=10_000.0, LENDING_APY=apy))
    result = s.run(observations)

    df = result.to_dataframe()
    final = float(df["net_balance"].iloc[-1])
    closed = 10_000.0 * (1 + apy / hours) ** hours
    drift = abs(final - closed)
    metrics = result.get_default_metrics()

    print(f"  observations: {len(observations)}")
    print(f"  metrics:      {metrics}")
    print(f"  final balance: {final:,.6f}")
    print(f"  closed-form  : {closed:,.6f}")
    print(f"  drift        : {drift:.6e}")

    if drift > 1e-6:
        print(f"\nFAIL — drift {drift} exceeds 1e-6 tolerance.", file=sys.stderr)
        return 1
    if not (0.04 < metrics.apy < 0.06):
        print(f"\nFAIL — APY {metrics.apy} outside expected band.", file=sys.stderr)
        return 1

    print("\nRuntime smoke OK — strategy ran end-to-end and matches closed form.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
