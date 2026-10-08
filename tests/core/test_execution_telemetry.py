"""Execution-telemetry tests (issue #68 PR2).

Every test drives a real strategy ``step`` so records carry the entity
name and observation timestamp exactly as production code stamps them.
"""
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import pytest

from fractal.core.base import Action, ActionToTake, BaseStrategy, BaseStrategyParams, NamedEntity, Observation
from fractal.core.base.execution import ExecutionLedger, ExecutionRecord
from fractal.core.base.strategy.result import StrategyResult
from fractal.core.entities.protocols.boros import BorosEntity, BorosGlobalState
from fractal.core.entities.protocols.pendle_pt import PendlePTConfig, PendlePTEntity, PendlePTGlobalState
from fractal.core.entities.protocols.uniswap_v2_lp import UniswapV2LPConfig, UniswapV2LPEntity, UniswapV2LPGlobalState
from fractal.core.entities.simple.perp import SimplePerpEntity, SimplePerpGlobalState
from fractal.core.entities.simple.pool import SimplePoolEntity, SimplePoolGlobalState
from fractal.core.entities.simple.spot import SimpleSpotExchange, SimpleSpotExchangeGlobalState

UTC = timezone.utc


@dataclass
class _P(BaseStrategyParams):
    variant: int = 0


class ScriptedStrategy(BaseStrategy[_P]):
    """Registers one entity and replays a scripted action list per step."""

    def __init__(self, entity: NamedEntity, script: Optional[List[ActionToTake]] = None,
                 **kwargs):
        self._script: List[List[ActionToTake]] = script or []
        super().__init__(**kwargs)
        self.register_entity(entity)

    def set_up(self) -> None:
        pass

    def queue(self, actions: List[ActionToTake]) -> None:
        self._script.append(actions)

    def predict(self) -> List[ActionToTake]:
        return self._script.pop(0) if self._script else []


def _obs(states: dict, hours: int = 0) -> Observation:
    return Observation(
        timestamp=datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=hours),
        states=states,
    )


def _step(strategy: BaseStrategy, states: dict, hours: int = 0) -> None:
    strategy.step(_obs(states, hours))


def _deposit(entity_name: str, amount: float) -> ActionToTake:
    return ActionToTake(entity_name=entity_name,
                        action=Action("deposit", {"amount_in_notional": amount}))


# --------------------------------------------------------------------- spot
def test_spot_buy_sell_records_notional_and_fee():
    spot = SimpleSpotExchange(trading_fee=0.005)
    strategy = ScriptedStrategy(NamedEntity("SPOT", spot))
    strategy.queue([_deposit("SPOT", 1000.0)])
    strategy.queue([
        ActionToTake("SPOT", Action("buy", {"amount_in_notional": 100.0})),
    ])
    _step(strategy, {"SPOT": SimpleSpotExchangeGlobalState(close=100.0)}, hours=0)  # deposit
    _step(strategy, {"SPOT": SimpleSpotExchangeGlobalState(close=100.0)}, hours=0)  # buy
    strategy.queue([
        ActionToTake("SPOT", Action("sell", {"amount_in_product": 0.5})),
    ])
    _step(strategy, {"SPOT": SimpleSpotExchangeGlobalState(close=110.0)}, hours=1)  # sell

    records = strategy.execution_ledger.records
    assert [(r.entity, r.action) for r in records] == [
        ("SPOT", "buy"), ("SPOT", "sell"),
    ]
    buy, sell = records
    assert buy.traded_notional == pytest.approx(100.0)
    assert buy.fee_paid == pytest.approx(0.5)
    assert buy.timestamp == datetime(2024, 1, 1, tzinfo=UTC)
    # sell: gross notional 0.5 * 110 = 55, fee = 0.275
    assert sell.traded_notional == pytest.approx(55.0)
    assert sell.fee_paid == pytest.approx(0.275)
    assert sell.timestamp == datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=1)
    ledger = strategy.execution_ledger
    assert ledger.total_traded_notional == pytest.approx(155.0)
    assert ledger.total_fees_paid == pytest.approx(0.775)


def test_deposit_withdraw_and_transfer_are_not_recorded():
    spot_a = SimpleSpotExchange(trading_fee=0.005)
    spot_b = SimpleSpotExchange(trading_fee=0.005)
    strategy = ScriptedStrategy(NamedEntity("A", spot_a))
    strategy.register_entity(NamedEntity("B", spot_b))
    strategy.queue([_deposit("A", 500.0)])
    _step(strategy, {"A": SimpleSpotExchangeGlobalState(close=1.0),
                     "B": SimpleSpotExchangeGlobalState(close=1.0)})  # deposit
    strategy.queue([
        ActionToTake("A", Action("withdraw", {"amount_in_notional": 100.0})),
    ])
    _step(strategy, {"A": SimpleSpotExchangeGlobalState(close=1.0),
                     "B": SimpleSpotExchangeGlobalState(close=1.0)})  # withdraw
    # internal transfer: deposit-first, withdraw-second — never a trade
    transfer = strategy.transfer("A", "B", 50.0)
    strategy.queue(transfer)
    _step(strategy, {"A": SimpleSpotExchangeGlobalState(close=1.0),
                     "B": SimpleSpotExchangeGlobalState(close=1.0)}, hours=2)
    assert strategy.execution_ledger.records == []
    assert strategy.execution_ledger.total_traded_notional == 0.0


# --------------------------------------------------------------------- perp
def test_perp_open_records_fee_and_rejected_trade_records_nothing():
    perp = SimplePerpEntity(trading_fee=0.001, max_leverage=10)
    strategy = ScriptedStrategy(NamedEntity("PERP", perp))
    strategy.queue([_deposit("PERP", 1000.0)])
    strategy.queue([ActionToTake("PERP", Action("open_position",
                                                {"amount_in_product": 2.0}))])
    _step(strategy, {"PERP": SimplePerpGlobalState(mark_price=100.0)})  # deposit
    _step(strategy, {"PERP": SimplePerpGlobalState(mark_price=100.0)})  # open
    (record,) = strategy.execution_ledger.records
    assert record.action == "open_position"
    assert record.traded_notional == pytest.approx(200.0)
    assert record.fee_paid == pytest.approx(0.2)

    # Risk-increasing trade rejected at margin → rolled back, no record.
    strategy.queue([ActionToTake("PERP", Action("open_position",
                                                {"amount_in_product": 500.0}))])
    with pytest.raises(Exception):
        _step(strategy, {"PERP": SimplePerpGlobalState(mark_price=100.0)})
    assert len(strategy.execution_ledger.records) == 1


def test_perp_close_position_records_the_closing_leg():
    perp = SimplePerpEntity(trading_fee=0.001, max_leverage=10)
    strategy = ScriptedStrategy(NamedEntity("PERP", perp))
    strategy.queue([_deposit("PERP", 1000.0)])
    strategy.queue([ActionToTake("PERP", Action("open_position",
                                                {"amount_in_product": 2.0}))])
    _step(strategy, {"PERP": SimplePerpGlobalState(mark_price=100.0)})  # deposit
    _step(strategy, {"PERP": SimplePerpGlobalState(mark_price=100.0)})  # open
    strategy.queue([ActionToTake("PERP", Action("close_position", {}))])
    _step(strategy, {"PERP": SimplePerpGlobalState(mark_price=101.0)}, hours=1)
    records = strategy.execution_ledger.records
    # a flatten must be labelled as a close, not as the ``open_position`` call
    # it reuses internally (per-open/close telemetry breakdowns depend on it)
    assert [r.action for r in records] == ["open_position", "close_position"]
    # the closing leg executes at the CURRENT mark (101), so notional and
    # fee are computed on 2.0 × 101
    assert records[1].traded_notional == pytest.approx(202.0)
    assert records[1].fee_paid == pytest.approx(0.202)
    assert records[1].timestamp == datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=1)


# ----------------------------------------------------------------- LP pools
def test_simple_pool_open_close_records_swapped_half_only():
    pool = SimplePoolEntity(pool_fee_rate=0.01, slippage_pct=0.0)
    strategy = ScriptedStrategy(NamedEntity("POOL", pool))
    strategy.queue([_deposit("POOL", 1000.0)])
    strategy.queue([ActionToTake("POOL", Action("open_position",
                                                {"amount_in_notional": 400.0}))])
    _step(strategy, {"POOL": SimplePoolGlobalState(tvl=10000.0, liquidity=10000.0,
                                                   volume=0.0, fees=0.0, price=1.0)})  # deposit
    _step(strategy, {"POOL": SimplePoolGlobalState(tvl=10000.0, liquidity=10000.0,
                                                   volume=0.0, fees=0.0, price=1.0)})  # open
    strategy.queue([ActionToTake("POOL", Action("close_position", {}))])
    _step(strategy, {"POOL": SimplePoolGlobalState(tvl=10000.0, liquidity=10000.0,
                                                   volume=0.0, fees=0.0, price=1.0)},
          hours=1)
    records = strategy.execution_ledger.records
    assert [r.action for r in records] == ["open_position", "close_position"]
    # swapped half: 200 notional, fee 1% of it
    assert records[0].traded_notional == pytest.approx(200.0)
    assert records[0].fee_paid == pytest.approx(2.0)
    # position value at close: 400 × (1 − fee/2) = 398 → swapped half 199
    assert records[1].traded_notional == pytest.approx(199.0)
    assert records[1].fee_paid == pytest.approx(1.99)


def test_uniswap_v2_lp_open_close_records_swapped_half_only():
    lp = UniswapV2LPEntity(config=UniswapV2LPConfig(pool_fee_rate=0.003, slippage_pct=0.0))
    strategy = ScriptedStrategy(NamedEntity("LP", lp))
    strategy.queue([_deposit("LP", 1000.0)])
    strategy.queue([ActionToTake("LP", Action("open_position",
                                              {"amount_in_notional": 400.0}))])
    _step(strategy, {"LP": UniswapV2LPGlobalState(price=1.0, tvl=10000.0,
                                                  liquidity=10000.0, fees=0.0,
                                                  volume=0.0)})  # deposit
    _step(strategy, {"LP": UniswapV2LPGlobalState(price=1.0, tvl=10000.0,
                                                  liquidity=10000.0, fees=0.0,
                                                  volume=0.0)})  # open
    strategy.queue([ActionToTake("LP", Action("close_position", {}))])
    _step(strategy, {"LP": UniswapV2LPGlobalState(price=1.0, tvl=10000.0,
                                                  liquidity=10000.0, fees=0.0,
                                                  volume=0.0)}, hours=1)
    records = strategy.execution_ledger.records
    assert [r.action for r in records] == ["open_position", "close_position"]
    assert records[0].traded_notional == pytest.approx(200.0)
    assert records[0].fee_paid == pytest.approx(200.0 * 0.003)
    # volatile tokens minted: (400/2)/1 × (1 − 0.003) = 199.4; close swaps them
    assert records[1].traded_notional == pytest.approx(199.4)
    assert records[1].fee_paid == pytest.approx(199.4 * 0.003)


# -------------------------------------------------------------------- boros
def test_boros_open_records_taker_fee_times_time_to_maturity():
    boros = BorosEntity(taker_fee_rate=0.0005)
    strategy = ScriptedStrategy(NamedEntity("BOROS", boros))
    strategy.queue([_deposit("BOROS", 1_000_000.0)])
    strategy.queue([ActionToTake("BOROS", Action("open_position",
                                                 {"amount_in_product": 1000.0}))])
    boros_state = BorosGlobalState(mark_rate=0.10, underlying_price=2.0,
                                   seconds_to_expiry=365 * 24 * 3600)
    _step(strategy, {"BOROS": boros_state})  # deposit
    _step(strategy, {"BOROS": boros_state})  # open
    (record,) = strategy.execution_ledger.records
    assert record.traded_notional == pytest.approx(2000.0)
    # fee = |ΔN| · taker · TTM = 1000 · 0.0005 · 1 year = 0.5 coins,
    # recorded in notional: 0.5 × underlying_price 2.0 = 1.0
    assert record.fee_paid == pytest.approx(1.0)


# ------------------------------------------------------------------- pendle
def test_pendle_buy_sell_record_implicit_execution_cost():
    pt = PendlePTEntity(config=PendlePTConfig())
    strategy = ScriptedStrategy(NamedEntity("PT", pt))
    strategy.queue([_deposit("PT", 10_000.0)])
    strategy.queue([ActionToTake("PT", Action("buy", {"amount_in_notional": 1000.0}))])
    state = PendlePTGlobalState(
        implied_apy=0.05, asset_price=1.0, sy_exchange_rate=1.0,
        seconds_to_expiry=365 * 24 * 3600.0,
        total_pt=1_000_000.0, total_sy=1_000_000.0,
        scalar_root=1_000_000.0, ln_fee_rate_root=0.001,
    )
    _step(strategy, {"PT": state})  # deposit
    _step(strategy, {"PT": state})  # buy
    strategy.queue([ActionToTake("PT", Action("sell", {"amount_in_product": 100.0}))])
    _step(strategy, {"PT": state}, hours=1)  # sell
    records = strategy.execution_ledger.records
    assert [r.action for r in records] == ["buy", "sell"]
    buy, sell = records
    assert buy.traded_notional == pytest.approx(1000.0)
    assert buy.fee_paid >= 0.0  # blended fee + impact cost, never negative
    assert sell.traded_notional >= 0.0
    assert sell.fee_paid >= 0.0
    # round-trip must cost something positive (fee + impact)
    assert buy.fee_paid + sell.fee_paid > 0.0


# ------------------------------------------------------- result integration
def test_run_returns_execution_records_and_construction_stays_backward_compatible():
    spot = SimpleSpotExchange(trading_fee=0.005)
    strategy = ScriptedStrategy(NamedEntity("SPOT", spot))
    strategy.queue([_deposit("SPOT", 1000.0)])
    strategy.queue([ActionToTake("SPOT", Action("buy", {"amount_in_notional": 100.0}))])
    observations = [
        _obs({"SPOT": SimpleSpotExchangeGlobalState(close=100.0)}),
        _obs({"SPOT": SimpleSpotExchangeGlobalState(close=100.0)}, 1),
    ]
    result = strategy.run(observations)
    assert result.execution_records is not None
    (record,) = result.execution_records
    assert isinstance(record, ExecutionRecord)
    assert record.entity == "SPOT"
    assert record.fee_paid == pytest.approx(0.5)
    # Old 4-field construction remains valid.
    legacy = StrategyResult(timestamps=[], internal_states=[], global_states=[], balances=[])
    assert legacy.execution_records is None


def test_ledger_reset_gives_each_run_its_own_snapshot():
    spot = SimpleSpotExchange(trading_fee=0.005)
    strategy = ScriptedStrategy(NamedEntity("SPOT", spot))
    script = [_deposit("SPOT", 1000.0),
              ActionToTake("SPOT", Action("buy", {"amount_in_notional": 100.0}))]
    strategy.queue(script)
    observations = [
        _obs({"SPOT": SimpleSpotExchangeGlobalState(close=100.0)}),
        _obs({"SPOT": SimpleSpotExchangeGlobalState(close=100.0)}, 1),
    ]
    first = strategy.run(observations)
    strategy.queue(script)
    second = strategy.run(observations)
    assert len(first.execution_records) == 1  # snapshot survives the reset
    assert len(second.execution_records) == 1
    assert strategy.execution_ledger.total_fees_paid == pytest.approx(0.5)


def test_ledger_isolated_per_strategy_instance():
    spot = SimpleSpotExchange(trading_fee=0.005)
    one = ScriptedStrategy(NamedEntity("SPOT", spot))
    two = ScriptedStrategy(NamedEntity("SPOT", SimpleSpotExchange(trading_fee=0.005)))
    one.queue([_deposit("SPOT", 1000.0),
               ActionToTake("SPOT", Action("buy", {"amount_in_notional": 100.0}))])
    one.step(_obs({"SPOT": SimpleSpotExchangeGlobalState(close=100.0)}))
    one.step(_obs({"SPOT": SimpleSpotExchangeGlobalState(close=100.0)}, 1))
    assert isinstance(two.execution_ledger, ExecutionLedger)
    assert two.execution_ledger.records == []
    assert len(one.execution_ledger.records) == 1


def test_entity_recorder_off_by_default_outside_strategies():
    """Direct entity use without a strategy records nothing."""
    spot = SimpleSpotExchange(trading_fee=0.005)
    spot.action_deposit(1000.0)
    spot._global_state.close = 100.0
    spot.action_buy(100.0)
    assert spot._execution_recorder is None


def test_perp_label_returns_to_open_after_a_close():
    """The close label must not leak into the next entry on that entity."""
    perp = SimplePerpEntity(trading_fee=0.001, max_leverage=10)
    strategy = ScriptedStrategy(NamedEntity("PERP", perp))
    strategy.queue([_deposit("PERP", 1000.0)])
    strategy.queue([ActionToTake("PERP", Action("open_position", {"amount_in_product": 2.0}))])
    _step(strategy, {"PERP": SimplePerpGlobalState(mark_price=100.0)})  # deposit
    _step(strategy, {"PERP": SimplePerpGlobalState(mark_price=100.0)})  # open
    strategy.queue([ActionToTake("PERP", Action("close_position", {}))])
    _step(strategy, {"PERP": SimplePerpGlobalState(mark_price=101.0)}, hours=1)
    assert perp._closing_position is False
    strategy.queue([ActionToTake("PERP", Action("open_position", {"amount_in_product": 1.0}))])
    _step(strategy, {"PERP": SimplePerpGlobalState(mark_price=101.0)}, hours=2)
    assert [r.action for r in strategy.execution_ledger.records] == [
        "open_position", "close_position", "open_position",
    ]
