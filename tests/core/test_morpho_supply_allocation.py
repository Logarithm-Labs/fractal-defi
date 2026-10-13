"""Synthetic tests for :class:`MorphoSupplyAllocationStrategy` and :func:`water_fill`.

Pins the allocator's three promises: a common marginal rate across funded
markets, the target-rate floor leaving excess capital idle, and rebalancing
that respects the exit gate and pays only for deltas that matter.
"""
import math
from dataclasses import asdict
from datetime import datetime, timedelta, timezone

import pytest

from fractal.core.base import Observation
from fractal.core.entities.protocols.morpho_supply import MorphoSupplyGlobalState
from fractal.core.entities.simple.spot import SimpleSpotExchange, SimpleSpotExchangeGlobalState
from fractal.loaders.morpho_markets import MorphoMarketListing
from fractal.strategies.morpho_supply_allocation import (
    MorphoSupplyAllocationParams,
    MorphoSupplyAllocationStrategy,
    water_fill,
)
from fractal.strategies.morpho_supply_rotation import market_entity_name

pytestmark = pytest.mark.core

UTC = timezone.utc
START = datetime(2026, 1, 1, tzinfo=UTC)


def listing(market_id: str, lltv: float = 0.86, oracle: str = "ChainlinkOracleV2") -> MorphoMarketListing:
    return MorphoMarketListing(
        chain_id=1, market_id=market_id * 64, loan_address="0xusdc", loan_symbol="USDC", loan_decimals=6,
        collateral_address="0xcoll", collateral_symbol="PT", collateral_decimals=18,
        lltv=lltv, oracle_type=oracle, listed=True, creation_timestamp=1,
    )


def market(supply: float, utilization: float, rate_at_target: float) -> MorphoSupplyGlobalState:
    borrow = supply * utilization
    return MorphoSupplyGlobalState(
        utilization=utilization, supply_rate=0.0, supply_apy=0.0, borrow_apy=0.0, rate_at_target=rate_at_target,
        fee=0.0, supply_assets_usd=supply, borrow_assets_usd=borrow, liquidity_assets_usd=supply - borrow,
    )


def build(markets, states, capital, **overrides):
    params = MorphoSupplyAllocationParams(**{"SIGNAL_LAG_BARS": 1, "SIGNAL_WINDOW_BARS": 1, "REBALANCE_BARS": 1,
                                            "MIN_LIQUIDITY_MULTIPLE": 0.0, **overrides})
    strategy = MorphoSupplyAllocationStrategy(markets=markets, cash_entity=SimpleSpotExchange(trading_fee=0.0),
                                              params=params)
    strategy.get_entity("CASH").action_deposit(capital)
    names = [market_entity_name(m) for m in markets]
    return strategy, names, dict(zip(names, states, strict=True))


def run(strategy, states, bars: int, start: int = 0) -> None:
    for bar in range(start, start + bars):
        obs = {**states, "CASH": SimpleSpotExchangeGlobalState(close=1.0)}
        strategy.step(Observation(timestamp=START + timedelta(hours=bar), states=obs))


# ------------------------------------------------------------- water_fill
def test_water_fill_equalises_marginal_rates_and_sums_to_capital():
    # linear "take" curves: market i absorbs k_i * (ceiling_i - rate) above the rate
    take = {"a": lambda r: max(0.0, (0.10 - r) * 10e6), "b": lambda r: max(0.0, (0.07 - r) * 20e6)}
    lam, amounts = water_fill(200_000.0, take, lambda_floor=0.0, lambda_ceiling=0.10)
    assert sum(amounts.values()) == pytest.approx(200_000.0, rel=1e-9)
    assert amounts["a"] == pytest.approx(take["a"](lam)) and amounts["b"] == pytest.approx(take["b"](lam))
    assert 0.07 < lam < 0.10 and amounts["b"] == 0.0          # too little capital to reach market b
    # more capital: both funded at one marginal rate, 10e6*(0.10-l) + 20e6*(0.07-l) = 500k -> l = 0.0633
    lam, amounts = water_fill(500_000.0, take, lambda_floor=0.0, lambda_ceiling=0.10)
    assert lam == pytest.approx(0.063333, rel=1e-4)
    assert amounts["a"] == pytest.approx(take["a"](lam), rel=1e-6)
    assert amounts["b"] == pytest.approx(take["b"](lam), rel=1e-6)


def test_water_fill_splits_leftover_across_step_and_unbounded_takes():
    # exogenous-style step takes: all-or-nothing above a rate; one of them capped, one unbounded
    take = {"a": lambda r: 1.6e6 if r <= 0.08 else 0.0, "b": lambda r: float("inf") if r <= 0.05 else 0.0}
    lam, amounts = water_fill(30e6, take, lambda_floor=0.0, lambda_ceiling=0.08)
    assert amounts["a"] == pytest.approx(1.6e6) and amounts["b"] == pytest.approx(28.4e6)
    assert lam == pytest.approx(0.05, abs=1e-6)


def test_water_fill_floor_leaves_capital_idle():
    take = {"a": lambda r: max(0.0, (0.10 - r) * 10e6)}
    lam, amounts = water_fill(5_000_000.0, take, lambda_floor=0.06, lambda_ceiling=0.10)
    assert lam == 0.06
    assert amounts["a"] == pytest.approx(400_000.0)            # the rest (4.6M) stays unplaced
    assert water_fill(0.0, take, 0.06, 0.10) == (0.06, {"a": 0.0})


# -------------------------------------------------------------- allocation
def test_small_capital_goes_to_the_best_market_only():
    markets = [listing("a"), listing("b")]
    states = [market(50e6, 0.92, 0.08), market(50e6, 0.80, 0.08)]   # a pays more at every size
    strategy, names, obs = build(markets, states, 100_000.0, TARGET_APY=0.0)
    run(strategy, obs, 3)
    assert strategy.get_entity(names[0]).balance == pytest.approx(100_000.0, rel=1e-4)  # plus two bars of accrual
    assert strategy.get_entity(names[1]).balance == pytest.approx(0.0)
    assert strategy.get_entity("CASH").balance == pytest.approx(0.0, abs=1e-6)


def test_large_capital_is_split_until_marginal_rates_match():
    markets = [listing("a"), listing("b")]
    states = [market(20e6, 0.92, 0.08), market(200e6, 0.80, 0.08)]
    strategy, names, obs = build(markets, states, 30e6, TARGET_APY=0.0)
    run(strategy, obs, 3)
    a, b = (strategy.get_entity(name) for name in names)
    assert a.balance > 0.0 and b.balance > 0.0
    assert a.balance + b.balance == pytest.approx(30e6, rel=1e-4)
    assert a.effective_supply_apy == pytest.approx(b.effective_supply_apy, rel=1e-4)   # one marginal rate
    assert strategy.allocation_report()["marginal_apy"] == pytest.approx(a.effective_supply_apy, rel=1e-3)


def test_target_apy_floor_keeps_the_rest_in_cash():
    markets = [listing("a"), listing("b")]
    states = [market(20e6, 0.92, 0.08), market(200e6, 0.80, 0.08)]
    strategy, names, obs = build(markets, states, 300e6, TARGET_APY=0.06)
    run(strategy, obs, 3)
    deployed = sum(strategy.get_entity(name).balance for name in names)
    cash = strategy.get_entity("CASH").balance
    assert cash > 0.0 and deployed + cash == pytest.approx(300e6, rel=1e-4)
    for name in names:                                      # every funded market still pays the target
        entity = strategy.get_entity(name)
        if entity.balance > 1.0:
            assert entity.effective_supply_apy == pytest.approx(0.06, rel=1e-3)
    assert strategy.allocation_report()["cash_fraction"] > 0.5


def test_liquidity_cap_bounds_a_position_on_the_current_bar():
    markets = [listing("a"), listing("b")]
    states = [market(2e6, 0.95, 0.20), market(200e6, 0.80, 0.08)]      # a: 100k liquidity, juicy rate
    strategy, names, obs = build(markets, states, 10e6, TARGET_APY=0.0, MIN_LIQUIDITY_MULTIPLE=1.0, ABSORPTION=1.0)
    run(strategy, obs, 3)
    a = strategy.get_entity(names[0])
    # with full absorption our supply adds (1−u)=5% of itself to liquidity: cap = L / (1 − 0.05)
    assert a.balance <= 100_000.0 / 0.95 * (1 + 1e-6)
    assert strategy.get_entity(names[1]).balance > 9e6


def test_rebalance_respects_the_exit_gate_and_records_the_shortfall():
    markets = [listing("a"), listing("b")]
    states = [market(20e6, 0.90, 0.08), market(200e6, 0.90, 0.08)]
    strategy, names, obs = build(markets, states, 5e6, TARGET_APY=0.0, ABSORPTION=1.0)
    run(strategy, obs, 3)
    a_before = strategy.get_entity(names[0]).balance
    assert a_before > 0.0
    # market a dries up and its rate collapses: the allocator wants out, the gate allows only $50k
    dried = market(20e6, 0.9975, 0.001)                      # S - B = 50k: the gate releases ~56k per bar
    run(strategy, {**obs, names[0]: dried}, 3, start=3)
    a_after = strategy.get_entity(names[0]).balance
    assert a_after < a_before and a_after > a_before - 3 * 50_000.0 * 1.2
    assert strategy.deferrals and strategy.deferrals[0]["market"] == names[0]
    assert strategy.deferrals[0]["allowed"] < strategy.deferrals[0]["wanted"]


def test_small_deltas_are_not_traded_and_costs_flow_through_fees_paid():
    markets = [listing("a"), listing("b")]
    states = [market(20e6, 0.92, 0.08), market(200e6, 0.80, 0.08)]
    for state, apy in zip(states, (0.08, 0.05), strict=True):
        state.supply_apy = apy                              # exogenous ranking; supply_rate stays 0 -> no accrual
    strategy, _, obs = build(markets, states, 30e6, TARGET_APY=0.0, COST_BPS_PER_LEG=1.0,
                                 MIN_REBALANCE_FRACTION=0.05, OWN_IMPACT=False, MIN_LIQUIDITY_MULTIPLE=1.0)
    run(strategy, obs, 2)
    legs_after_first = len(strategy.execution_ledger.records)
    assert legs_after_first >= 2
    run(strategy, obs, 5, start=2)                          # nothing changed: no churn
    assert len(strategy.execution_ledger.records) == legs_after_first
    nav = strategy.total_balance
    assert 30e6 - nav == pytest.approx(strategy.execution_ledger.total_fees_paid, rel=1e-9)


def test_filters_exclude_markets_by_lltv_and_oracle():
    markets = [listing("a", lltv=0.965), listing("b", oracle="Unknown"), listing("c")]
    states = [market(50e6, 0.92, 0.20), market(50e6, 0.92, 0.20), market(50e6, 0.80, 0.05)]
    strategy, names, obs = build(markets, states, 1e6, TARGET_APY=0.0)
    run(strategy, obs, 3)
    assert strategy.get_entity(names[0]).balance == 0.0 and strategy.get_entity(names[1]).balance == 0.0
    assert strategy.get_entity(names[2]).balance == pytest.approx(1e6, rel=1e-4)


def test_decision_uses_the_lagged_window_not_the_current_bar():
    markets = [listing("a"), listing("b")]
    calm = [market(50e6, 0.80, 0.05), market(50e6, 0.85, 0.05)]
    strategy, names, obs = build(markets, calm, 1e6, TARGET_APY=0.0, SIGNAL_LAG_BARS=2, SIGNAL_WINDOW_BARS=1)
    run(strategy, obs, 4)
    assert strategy.get_entity(names[1]).balance == pytest.approx(1e6, rel=1e-4)
    spike = {**obs, names[0]: market(50e6, 0.99, 0.50)}       # a spikes on the bar being traded
    run(strategy, spike, 1, start=4)
    assert strategy.get_entity(names[1]).balance == pytest.approx(1e6, rel=1e-4)   # not seen yet


def test_loan_symbol_and_pool_size_filters():
    usdt = MorphoMarketListing(
        chain_id=1, market_id="u" * 64, loan_address="0xusdt", loan_symbol="USDT", loan_decimals=6,
        collateral_address="0xcoll", collateral_symbol="PT", collateral_decimals=18,
        lltv=0.86, oracle_type="ChainlinkOracleV2", listed=True, creation_timestamp=1,
    )
    markets = [usdt, listing("s"), listing("c")]
    states = [market(50e6, 0.95, 0.30), market(0.5e6, 0.95, 0.30), market(50e6, 0.80, 0.05)]
    strategy, names, obs = build(markets, states, 1e6, TARGET_APY=0.0, LOAN_SYMBOLS=("USDC",),
                                 MIN_POOL_SIZE_USD=1_000_000.0)
    run(strategy, obs, 3)
    assert strategy.get_entity(names[0]).balance == 0.0 and strategy.get_entity(names[1]).balance == 0.0
    assert strategy.get_entity(names[2]).balance == pytest.approx(1e6, rel=1e-4)


def test_defaults_are_fast_reacting_and_uncapped():
    params = MorphoSupplyAllocationParams()
    assert params.REBALANCE_BARS == 1 and params.SIGNAL_WINDOW_BARS == 1 and params.MAX_POSITIONS == 100
    assert params.OWN_IMPACT is True


# ------------------------------------------------------------ marginal mode
def hot_and_deep():
    """Synthetic small high-utilization pool and a deep lower-utilization pool."""
    return [market(10e6, 0.95, 0.08), market(200e6, 0.85, 0.06)]


def income(strategy, names) -> float:
    return sum(strategy.get_entity(n).balance * strategy.get_entity(n).effective_supply_apy for n in names)


def test_marginal_mode_maximises_income_where_average_mode_overfills_the_hot_pool():
    markets = [listing("a"), listing("b")]
    avg, names, obs = build(markets, hot_and_deep(), 1e6, TARGET_APY=0.0)
    mrg, _, _ = build(markets, hot_and_deep(), 1e6, TARGET_APY=0.0, ALLOCATION_MODE="marginal")
    run(avg, obs, 3)
    run(mrg, obs, 3)
    assert avg.get_entity(names[0]).balance > 0.95e6                 # equal average rates: all into the hot pool
    hot = mrg.get_entity(names[0]).balance
    assert 0.2e6 < hot < 0.45e6                                      # income-optimal: a small slice of the spike
    assert income(mrg, names) > income(avg, names) + 3_000.0          # ≈ +$5k/yr on $1M in the worked example


def test_marginal_floor_applies_to_the_marginal_dollar():
    """TARGET 10 %: average mode fills the hot pool until its *average* rate is 10 %, marginal mode until the
    *last dollar* earns 10 % — less capital deployed, more idle, every deployed dollar clears the target."""
    markets = [listing("a")]
    states = [market(10e6, 0.95, 0.08)]
    avg, names, obs = build(markets, states, 1e6, TARGET_APY=0.10)
    mrg, _, _ = build(markets, states, 1e6, TARGET_APY=0.10, ALLOCATION_MODE="marginal")
    run(avg, obs, 3)
    run(mrg, obs, 3)
    placed_avg, placed_mrg = avg.get_entity(names[0]).balance, mrg.get_entity(names[0]).balance
    assert 0.0 < placed_mrg < placed_avg
    assert mrg.get_entity("CASH").balance > avg.get_entity("CASH").balance
    assert mrg.allocation_report()["marginal_apy"] >= 0.10 - 0.01   # last chunk still earns about the target


def test_expected_gas_turns_off_rebalances_that_do_not_pay():
    """Two near-identical pools whose rates swap a little every bar: free trading flips, $200 gas holds."""
    markets = [listing("a"), listing("b")]
    hi, lo = market(50e6, 0.86, 0.060), market(50e6, 0.85, 0.060)
    panel = [{0: hi, 1: lo}, {0: lo, 1: hi}] * 6

    def legs(gas):
        strategy, names, _ = build(markets, [hi, lo], 2e6, TARGET_APY=0.0, ALLOCATION_MODE="marginal",
                                     GAS_USD_PER_LEG=gas, HOLD_HORIZON_BARS=24)
        for bar, states in enumerate(panel):
            run(strategy, {names[0]: states[0], names[1]: states[1]}, 1, start=bar)
        return len(strategy.execution_ledger.records), strategy
    free_legs, _ = legs(0.0)
    paid_legs, paid = legs(200.0)
    assert paid_legs <= 4 < free_legs                                # initial deployment only
    assert paid.execution_ledger.total_fees_paid == pytest.approx(200.0 * paid_legs)


def test_marginal_mode_respects_the_exit_gate():
    markets = [listing("a"), listing("b")]
    strategy, names, obs = build(markets, [market(20e6, 0.88, 0.06), market(200e6, 0.85, 0.06)], 5e6,
                                 TARGET_APY=0.0, ALLOCATION_MODE="marginal", ABSORPTION=1.0)
    run(strategy, obs, 3)
    held = strategy.get_entity(names[0]).balance
    assert held > 0.0
    dried = market(20e6, 0.9975, 0.001)        # pool a: rate gone; withdrawable = 50k + (1 − u)·P ≈ 62.5k per bar
    run(strategy, {**obs, names[0]: dried}, 3, start=3)               # the 1-bar lag sees it from the next bar
    after = strategy.get_entity(names[0]).balance
    assert held - after <= 3 * 62_500.0 * 1.05                       # cannot leave faster than the gate
    assert after < held - 50_000.0                                   # but leaves as fast as it can


def test_marginal_mode_is_validated():
    with pytest.raises(ValueError, match="ALLOCATION_MODE"):
        build([listing("a")], [market(50e6, 0.9, 0.05)], 1e6, ALLOCATION_MODE="greedy")


def test_gas_is_priced_per_chain():
    eth = listing("a")
    base = MorphoMarketListing(
        chain_id=8453, market_id="b" * 64, loan_address="0xusdc", loan_symbol="USDC", loan_decimals=6,
        collateral_address="0xcoll", collateral_symbol="PT", collateral_decimals=18,
        lltv=0.86, oracle_type="ChainlinkOracleV2", listed=True, creation_timestamp=1,
    )
    strategy, names, obs = build([eth, base], [market(50e6, 0.85, 0.05), market(50e6, 0.85, 0.05)], 1e6,
                                 TARGET_APY=0.0, GAS_USD_PER_LEG=2.0, GAS_USD_BY_CHAIN={8453: 0.02})
    assert strategy.get_entity(names[0]).leg_cost_model[0] == pytest.approx(2.0)     # Ethereum: the default
    assert strategy.get_entity(names[1]).leg_cost_model[0] == pytest.approx(0.02)    # Base: per-chain override
    run(strategy, obs, 3)
    fees = {r.entity: r.fee_paid for r in strategy.execution_ledger.records}
    assert fees[names[0]] == pytest.approx(2.0) and fees[names[1]] == pytest.approx(0.02)
    with pytest.raises(ValueError, match="gas per leg"):
        build([eth], [market(50e6, 0.85, 0.05)], 1e6, GAS_USD_BY_CHAIN={1: -1.0})


def test_rewards_mode_moves_capital_to_the_rewarded_pool():
    markets = [listing("a"), listing("b")]
    plain, rewarded = market(50e6, 0.85, 0.06), market(50e6, 0.80, 0.05)
    rewarded.reward_apy = 0.04                                         # lower base, +4 % campaign
    for mode in ("average", "marginal"):
        off, names, obs = build(markets, [plain, rewarded], 1e6, TARGET_APY=0.0, ALLOCATION_MODE=mode)
        on, _, _ = build(markets, [plain, rewarded], 1e6, TARGET_APY=0.0, ALLOCATION_MODE=mode, USE_REWARDS=True)
        run(off, obs, 3)
        run(on, obs, 3)
        assert off.get_entity(names[0]).balance > 0.9e6                # base only: the plain pool wins
        assert on.get_entity(names[1]).balance > 0.9e6                 # with rewards: the rewarded pool wins
        assert on.get_entity(names[1]).use_rewards and not off.get_entity(names[1]).use_rewards


@pytest.mark.parametrize("capital", [100_000.0, 1_000_000.0])
def test_marginal_targets_never_exceed_available_capital(capital):
    """Lock-in: DP budget slack used to allocate more notional than the portfolio owned."""
    strategy, _, obs = build(
        [listing("a"), listing("b")], hot_and_deep(), capital,
        TARGET_APY=0.0, ALLOCATION_MODE="marginal",
    )
    run(strategy, obs, 2)
    assert sum(strategy.allocation_report()["targets"].values()) <= capital * (1.0 + 1e-12)


def test_marginal_keeps_locked_off_grid_holdings_without_inventing_capital():
    """Rounding cannot force a withdrawal from fully gated, off-grid positions."""
    states = [market(20e6, 0.9, 0.05), market(20e6, 0.9, 0.05)]
    for state in states:
        state.supply_apy = 0.1
        state.liquidity_assets_usd = 0.0
    strategy, names, obs = build(
        [listing("a"), listing("b")], states, 1e6,
        OWN_IMPACT=False, TARGET_APY=0.0, ALLOCATION_MODE="marginal",
    )
    strategy.get_entity("CASH").action_withdraw(1e6)
    positions = dict(zip(names, [333_333.0, 666_667.0], strict=True))
    for name, amount in positions.items():
        strategy.get_entity(name).action_deposit(amount)
    run(strategy, obs, 3)
    assert strategy.allocation_report()["targets"] == pytest.approx(positions)
    assert strategy.get_entity("CASH").balance == 0.0
    assert [strategy.get_entity(name).balance for name in names] == pytest.approx(list(positions.values()))


@pytest.mark.parametrize("mode", ["average", "marginal"])
def test_synthetic_run_has_finite_metrics_and_conserves_net_of_fees(mode):
    """Both allocation modes run end to end without creating notional capital."""
    strategy, _, states = build(
        [listing("a"), listing("b")], hot_and_deep(), 1e6,
        TARGET_APY=0.0, ALLOCATION_MODE=mode, OWN_IMPACT=False, GAS_USD_PER_LEG=1.0,
    )
    for state in states.values():
        state.supply_apy = 0.08
    stream = [
        Observation(timestamp=START + timedelta(hours=bar),
                    states={**states, "CASH": SimpleSpotExchangeGlobalState(close=1.0)})
        for bar in range(5)
    ]
    result = strategy.run(stream)
    assert strategy.total_balance + strategy.execution_ledger.total_fees_paid == pytest.approx(1e6)
    assert all(math.isfinite(value) for value in asdict(result.get_default_metrics()).values()
               if isinstance(value, (int, float)))
