"""Hand-computed tests for the execution-derived metrics (issue #68 PR3).

Definitions (user-confirmed):
* ``fees_paid`` — cumulative fee in the portfolio accounting unit;
* ``turnover`` — total traded notional / average positive NAV;
* ``fee_drag`` — fees_paid / initial portfolio NAV.
"""
from dataclasses import dataclass, fields
from datetime import datetime, timedelta, timezone

import pytest

from fractal.core.base import Action, ActionToTake, BaseStrategy, BaseStrategyParams, NamedEntity, Observation
from fractal.core.base.execution import ExecutionRecord
from fractal.core.base.strategy.result import StrategyMetrics, StrategyResult
from fractal.core.entities.simple.spot import SimpleSpotExchange, SimpleSpotExchangeGlobalState

UTC = timezone.utc


@dataclass
class P(BaseStrategyParams):
    variant: int = 0


class ScriptedStrategy(BaseStrategy[P]):
    """Registers one entity and replays a scripted action list per step."""

    def __init__(self, entity: NamedEntity, **kwargs):
        self._script: list[list[ActionToTake]] = []
        super().__init__(**kwargs)
        self.register_entity(entity)

    def set_up(self) -> None:
        pass

    def queue(self, actions: list[ActionToTake]) -> None:
        self._script.append(actions)

    def predict(self) -> list[ActionToTake]:
        return self._script.pop(0) if self._script else []


def _obs(states: dict, hours: int = 0) -> Observation:
    return Observation(
        timestamp=datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=hours),
        states=states,
    )


def _deposit(entity_name: str, amount: float) -> ActionToTake:
    return ActionToTake(entity_name=entity_name,
                        action=Action("deposit", {"amount_in_notional": amount}))


def _spot_result_with_two_buys():
    """Deposit 1000, buy 100 @ 100 (fee 0.5), buy 50 @ 100 (fee 0.25).

    Balances (cash + product valued at close=100): 1000, 999.5, 999.25.
    Traded notional 150, fees 0.75.
    """
    spot = SimpleSpotExchange(trading_fee=0.005)
    strategy = ScriptedStrategy(NamedEntity("SPOT", spot))
    strategy.queue([_deposit("SPOT", 1000.0)])
    strategy.queue([ActionToTake("SPOT", Action("buy", {"amount_in_notional": 100.0}))])
    strategy.queue([ActionToTake("SPOT", Action("buy", {"amount_in_notional": 50.0}))])
    observations = [
        _obs({"SPOT": SimpleSpotExchangeGlobalState(close=100.0)}),
        _obs({"SPOT": SimpleSpotExchangeGlobalState(close=100.0)}, 1),
        _obs({"SPOT": SimpleSpotExchangeGlobalState(close=100.0)}, 2),
    ]
    return strategy, observations


@pytest.mark.core
def test_derived_metrics_default_zero_and_construction_backward_compatible():
    m = StrategyMetrics(accumulated_return=0.1, apy=0.2, sharpe=1.0, max_drawdown=-0.3)
    assert m.fees_paid == 0.0
    assert m.turnover == 0.0
    assert m.fee_drag == 0.0


@pytest.mark.core
def test_zero_metrics_includes_derived_fields():
    r = StrategyResult(timestamps=[], internal_states=[], global_states=[], balances=[])
    m = r.get_metrics(r.to_dataframe())
    assert isinstance(m, StrategyMetrics)
    assert (m.fees_paid, m.turnover, m.fee_drag) == (0.0, 0.0, 0.0)


@pytest.mark.core
def test_no_telemetry_yields_zero_derived_metrics():
    # hand-built result without execution_records → 0.0 policy
    result = StrategyResult(
        timestamps=[datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 2, tzinfo=UTC)],
        internal_states=[{"X": None}] * 2,
        global_states=[{"X": None}] * 2,
        balances=[{"X": 100.0}, {"X": 110.0}],
    )
    m = result.get_metrics(result.to_dataframe())
    assert (m.fees_paid, m.turnover, m.fee_drag) == (0.0, 0.0, 0.0)


@pytest.mark.core
def test_turnover_fees_paid_and_fee_drag_match_hand_computed_values():
    strategy, observations = _spot_result_with_two_buys()
    result = strategy.run(observations)
    df = result.to_dataframe()
    m = result.get_metrics(df)
    # balances: 1000, 999.5, 999.25 → mean positive NAV
    mean_nav = (1000.0 + 999.5 + 999.25) / 3
    assert m.turnover == pytest.approx(150.0 / mean_nav)
    assert m.fees_paid == pytest.approx(0.75)
    assert m.fee_drag == pytest.approx(0.75 / 1000.0)
    # sanity: derived metrics flow into the MLflow logging dict
    assert {"fees_paid", "turnover", "fee_drag"} <= set(m.__dict__)


@pytest.mark.core
def test_derived_metrics_are_zero_when_initial_nav_zero():
    records = [ExecutionRecord(timestamp=None, entity="X", action="buy",
                               traded_notional=10.0, fee_paid=1.0)]
    result = StrategyResult(
        timestamps=[datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 2, tzinfo=UTC)],
        internal_states=[{"X": None}] * 2,
        global_states=[{"X": None}] * 2,
        balances=[{"X": 0.0}, {"X": 10.0}],  # zero initial NAV
        execution_records=records,
    )
    m = result.get_metrics(result.to_dataframe())
    # get_metrics short-circuits to zero metrics near zero-balance paths;
    # the ratio fields stay 0.0 there, the absolute fee total survives.
    assert m.fee_drag == 0.0
    assert m.turnover == 0.0
    assert m.fees_paid == pytest.approx(1.0)


@pytest.mark.core
def test_fees_paid_survives_a_single_bar_run():
    # One timestamp is degenerate for every NAV-based metric, but the fee
    # total has no denominator and must still be reported.
    records = [ExecutionRecord(timestamp=None, entity="X", action="buy",
                               traded_notional=100.0, fee_paid=5.0)]
    result = StrategyResult(
        timestamps=[datetime(2024, 1, 1, tzinfo=UTC)],
        internal_states=[{"X": None}],
        global_states=[{"X": None}],
        balances=[{"X": 995.0}],
        execution_records=records,
    )
    m = result.get_metrics(result.to_dataframe())
    assert m.fees_paid == pytest.approx(5.0)
    assert (m.turnover, m.fee_drag, m.apy, m.sharpe) == (0.0, 0.0, 0.0, 0.0)


@pytest.mark.core
def test_zero_metrics_sets_every_field():
    # Guards conflict resolutions between PRs that add metric fields: the
    # degenerate path must name every field explicitly.
    m = StrategyResult._zero_metrics()  # pylint: disable=protected-access
    assert {f.name for f in fields(StrategyMetrics)} <= set(m.__dict__)
    assert all(value == 0.0 for value in m.__dict__.values())


@pytest.mark.core
def test_turnover_ignores_non_positive_nav_bars():
    # explicit records with a mixed NAV path incl. a zero bar
    records = [ExecutionRecord(timestamp=None, entity="X", action="buy",
                               traded_notional=300.0, fee_paid=3.0)]
    result = StrategyResult(
        timestamps=[datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(3)],
        internal_states=[{"X": None}] * 3,
        global_states=[{"X": None}] * 3,
        balances=[{"X": 100.0}, {"X": 0.0}, {"X": 200.0}],
        execution_records=records,
    )
    m = result.get_metrics(result.to_dataframe())
    # mean positive NAV = (100 + 200) / 2 = 150 → turnover = 300/150 = 2.0
    assert m.turnover == pytest.approx(2.0)
    assert m.fees_paid == pytest.approx(3.0)
    assert m.fee_drag == pytest.approx(3.0 / 100.0)


@pytest.mark.core
def test_turnover_zero_when_no_positive_nav():
    records = [ExecutionRecord(timestamp=None, entity="X", action="buy",
                               traded_notional=10.0, fee_paid=1.0)]
    result = StrategyResult(
        timestamps=[datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 2, tzinfo=UTC)],
        internal_states=[{"X": None}] * 2,
        global_states=[{"X": None}] * 2,
        balances=[{"X": -5.0}, {"X": -5.0}],
        execution_records=records,
    )
    m = result.get_metrics(result.to_dataframe())
    assert m.turnover == 0.0


@pytest.mark.core
def test_derived_metrics_are_invariant_to_notional_price():
    """``turnover`` and ``fee_drag`` divide two accounting-unit quantities.

    ``notional_price`` re-expresses the balance series only, so it must not
    change the ratios (pre-fix they scaled by exactly that factor).
    """
    strategy, observations = _spot_result_with_two_buys()
    result = strategy.run(observations)
    df = result.to_dataframe()
    base = result.get_metrics(df)
    scaled = result.get_metrics(df, notional_price=10.0)
    assert base.turnover > 0.0 and base.fee_drag > 0.0  # non-trivial values
    assert scaled.turnover == pytest.approx(base.turnover)
    assert scaled.fee_drag == pytest.approx(base.fee_drag)
    assert scaled.fees_paid == pytest.approx(base.fees_paid)


@pytest.mark.core
def test_derived_metrics_are_invariant_to_a_per_bar_price_column():
    """Same invariance for the per-bar (string) form of ``notional_price``."""
    strategy, observations = _spot_result_with_two_buys()
    result = strategy.run(observations)
    df = result.to_dataframe()
    df["X_price"] = 2.0  # constant, so only the samples differ from the float case
    base = result.get_metrics(df)
    scaled = result.get_metrics(df, notional_price="X_price")
    assert scaled.turnover == pytest.approx(base.turnover)
    assert scaled.fee_drag == pytest.approx(base.fee_drag)
