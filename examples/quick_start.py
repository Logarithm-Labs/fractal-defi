"""Hello-Fractal: passive lending position with accruing APY.

The smallest end-to-end DeFi example — one lending entity, one strategy,
synthetic observations, no MLflow. Demonstrates Fractal's primitives in
~80 lines:

* a typed params dataclass (``BaseStrategyParams`` subclass)
* a strategy class with the generic argument (``BaseStrategy[Params]``)
  so ``set_params`` knows which dataclass to coerce dict-shaped grid
  cells into
* one entity registered in ``set_up`` (``SimpleLendingEntity``: deposit
  collateral, optionally borrow, interest accrues per ``update_state``)
* a ``predict`` that emits a single :class:`ActionToTake` on the first
  observation (deposit ``INITIAL_BALANCE`` as collateral) and nothing
  afterwards — collateral compounds at the configured ``LENDING_APY``
* hand-rolled hourly observations (no loader / network) so the result
  matches the closed-form compound-interest formula bit-for-bit
* a single ``strategy.run(observations)`` call returning
  :class:`StrategyResult` with ``get_default_metrics()`` /
  ``to_dataframe()``

The scenario: deposit $10,000 of USDC into a lending market that pays a
flat 5% APY, watch it compound hourly for a year, compare against the
closed-form ``P · (1 + r/n)^n``.

Run from the repo root:

    PYTHONPATH=. python examples/quick_start.py
"""
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import List

from fractal.core.base import (Action, ActionToTake, BaseStrategy,
                               BaseStrategyParams, NamedEntity, Observation)
from fractal.core.entities import SimpleLendingEntity, SimpleLendingGlobalState


@dataclass
class LendingParams(BaseStrategyParams):
    """Hyperparams for :class:`PassiveLenderStrategy`."""
    INITIAL_BALANCE: float = 10_000.0
    LENDING_APY: float = 0.05            # 5% annual on the supplied side


class PassiveLenderStrategy(BaseStrategy[LendingParams]):
    """Deposit ``INITIAL_BALANCE`` once, then sit and accrue.

    Pure compounding demo — no rebalancing, no borrowing, no liquidation
    risk. The framework drives the entity's ``update_state`` each tick,
    which applies the per-step ``lending_rate`` to ``collateral``.
    """

    def set_up(self) -> None:
        self.register_entity(NamedEntity(
            entity_name='LENDING',
            entity=SimpleLendingEntity(),
        ))
        self._funded: bool = False

    def predict(self) -> List[ActionToTake]:
        # First observation: deposit the entire initial balance as
        # collateral. Every subsequent tick is a no-op — the framework
        # still calls ``update_state`` which compounds the position.
        if self._funded:
            return []
        self._funded = True
        return [ActionToTake('LENDING', Action(
            'deposit', {'amount_in_notional': self._params.INITIAL_BALANCE},
        ))]


def build_observations(apy: float, days: int = 365) -> List[Observation]:
    """One year of hourly snapshots with a flat lending rate.

    APY is converted to a per-step rate via ``apy / hours_per_year`` —
    the simple-interest convention :class:`SimpleLendingEntity` uses on
    each ``update_state`` (compounding emerges from repeated ticks).
    Collateral price is held at 1.0 (USD-stable supply asset).
    """
    hours = days * 24
    per_step_rate = apy / hours
    start = datetime(2024, 1, 1, tzinfo=UTC)
    return [
        Observation(timestamp=start + timedelta(hours=i), states={
            'LENDING': SimpleLendingGlobalState(
                collateral_price=1.0,
                debt_price=1.0,
                lending_rate=per_step_rate,
                borrowing_rate=0.0,
            ),
        })
        for i in range(hours + 1)
    ]


if __name__ == '__main__':
    params = LendingParams(INITIAL_BALANCE=10_000.0, LENDING_APY=0.05)
    observations = build_observations(apy=params.LENDING_APY, days=365)
    print(f'simulating {len(observations)} hourly observations '
          f'(1 year @ {params.LENDING_APY:.0%} APY)')

    strategy = PassiveLenderStrategy(params=params)
    result = strategy.run(observations)
    print(f'metrics: {result.get_default_metrics()}')

    df = result.to_dataframe()
    df.to_csv('quick_start_result.csv', index=False)

    initial = params.INITIAL_BALANCE
    final = float(df['net_balance'].iloc[-1])
    closed_form = initial * (1 + params.LENDING_APY / (365 * 24)) ** (365 * 24)
    print(f'final balance: {final:,.4f}')
    print(f'closed-form  : {closed_form:,.4f}')
    print(f'mismatch     : {abs(final - closed_form):.6e}')
