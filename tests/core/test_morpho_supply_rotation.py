"""Synthetic tests for :class:`MorphoSupplyRotationStrategy`.

These pin the four things that make the strategy honest — the lag discipline
(no lookahead), the risk filters, the exit gate with deferral accounting, and
cost telemetry — on hand-built panels where the right answer is known.
"""
from datetime import datetime, timedelta, timezone

import pytest

from fractal.core.base import Observation
from fractal.core.entities.protocols.morpho_supply import MorphoSupplyGlobalState
from fractal.core.entities.simple.spot import SimpleSpotExchange, SimpleSpotExchangeGlobalState
from fractal.loaders.morpho_markets import MorphoMarketListing
from fractal.strategies.morpho_supply_rotation import MorphoSupplyRotationParams, MorphoSupplyRotationStrategy

pytestmark = pytest.mark.core

UTC = timezone.utc
CASH = 1_000_000.0
START = datetime(2026, 1, 1, tzinfo=UTC)


def listing(market_id: str, lltv: float = 0.86, oracle: str = "ChainlinkOracleV2") -> MorphoMarketListing:
    return MorphoMarketListing(
        chain_id=1, market_id=market_id * 64, loan_address="0xusdc", loan_symbol="USDC", loan_decimals=6,
        collateral_address="0xcoll", collateral_symbol="PT", collateral_decimals=18,
        lltv=lltv, oracle_type=oracle, listed=True, creation_timestamp=1,
    )


def state(apy: float, liquidity: float = 5_000_000.0, supply: float = 20_000_000.0,
          borrow: float = 15_000_000.0, rate_at_target: float = 0.05, fee: float = 0.0) -> MorphoSupplyGlobalState:
    return MorphoSupplyGlobalState(
        utilization=borrow / supply, supply_rate=0.0, supply_apy=apy, borrow_apy=apy, rate_at_target=rate_at_target,
        fee=fee, supply_assets_usd=supply, borrow_assets_usd=borrow, liquidity_assets_usd=liquidity,
    )


def build(markets, params: MorphoSupplyRotationParams | None = None):
    """Strategy with funded cash and deterministic defaults for short tests.

    ``OWN_IMPACT=False`` by default so the accrual is exactly the panel's
    ``supply_rate`` — the synthetic panels carry a zero rate and the balance
    assertions can be exact. The own-impact path has its own test below and
    its arithmetic is pinned in ``test_morpho_supply.py``.
    """
    params = params or MorphoSupplyRotationParams(SIGNAL_LAG_BARS=1, SIGNAL_WINDOW_BARS=1, MIN_EDGE_BPS=0.0,
                                              MIN_LIQUIDITY_MULTIPLE=0.0, OWN_IMPACT=False)
    strategy = MorphoSupplyRotationStrategy(
        markets=markets, cash_entity=SimpleSpotExchange(trading_fee=0.0), params=params,
    )
    strategy.get_entity(params.CASH_ENTITY_NAME).action_deposit(CASH)
    return strategy


def step(strategy, states: dict, hours: int) -> None:
    strategy.step(Observation(timestamp=START + timedelta(hours=hours), states=states))


def run_panel(strategy, panel: list[dict], market_names: list[str], cash_name: str = "CASH"):
    """Feed one state dict per bar: ``panel[bar][market_name]``."""
    for bar, rates in enumerate(panel):
        states = {name: state(rates[name]) for name in market_names}
        states[cash_name] = SimpleSpotExchangeGlobalState(close=1.0)
        step(strategy, states, bar)
    return strategy


# ------------------------------------------------------------- selection
def test_rotates_into_the_higher_signal_market_and_leaves_the_other_empty():
    markets = [listing("a"), listing("b")]
    strategy = build(markets)
    names = [strategy._name(m) for m in markets]  # pylint: disable=protected-access
    run_panel(strategy, [dict.fromkeys(names, 0.02)] * 5, names)
    assert strategy.get_entity(names[0]).balance == pytest.approx(CASH)
    # identical signals: the first market wins the tie, no churn on every bar
    strategy2 = build(markets)
    run_panel(strategy2, [{**dict.fromkeys(names, 0.02), names[1]: 0.10}] * 5, names)
    assert strategy2.get_entity(names[1]).balance == pytest.approx(CASH)
    assert strategy2.get_entity(names[0]).balance == pytest.approx(0.0)


def test_top_k_band_avoids_churn_between_near_ties():
    markets = [listing("a"), listing("b")]
    params = MorphoSupplyRotationParams(TOP_K=2, SIGNAL_LAG_BARS=1, SIGNAL_WINDOW_BARS=1, MIN_EDGE_BPS=100.0,
                                    MIN_LIQUIDITY_MULTIPLE=0.0, OWN_IMPACT=False)
    strategy = build(markets, params)
    names = [strategy._name(m) for m in markets]  # pylint: disable=protected-access
    run_panel(strategy, [{names[0]: 0.05, names[1]: 0.045}] * 6, names)
    first = strategy.get_entity(names[0]).balance
    assert first == pytest.approx(CASH)  # 50 bp ahead: outside the band, rotated in
    # B now yields more, but by 100 bp of threshold it does not qualify, and it
    # is inside the top-2 band: paying two legs is not worth it, nothing moves.
    run_panel(strategy, [{names[0]: 0.05, names[1]: 0.055}] * 4, names)
    assert strategy.get_entity(names[0]).balance == pytest.approx(first)
    assert strategy.get_entity(names[1]).balance == pytest.approx(0.0)


# ------------------------------------------------------------- filters
def test_lltv_ceiling_excludes_a_higher_yielding_market():
    markets = [listing("a", lltv=0.945), listing("b", lltv=0.86)]
    strategy = build(markets, MorphoSupplyRotationParams(MAX_LLTV=0.915, SIGNAL_LAG_BARS=1, SIGNAL_WINDOW_BARS=1,
                                                     MIN_EDGE_BPS=0.0, MIN_LIQUIDITY_MULTIPLE=0.0,
                                                     OWN_IMPACT=False))
    names = [strategy._name(m) for m in markets]  # pylint: disable=protected-access
    run_panel(strategy, [{names[0]: 0.30, names[1]: 0.05}] * 6, names)
    assert strategy.get_entity(names[0]).balance == pytest.approx(0.0)   # 30% but LLTV too high
    assert strategy.get_entity(names[1]).balance == pytest.approx(CASH)


def test_oracle_type_allowlist_excludes_a_higher_yielding_market():
    markets = [listing("a", oracle="Unknown"), listing("b", oracle="ChainlinkOracleV2")]
    strategy = build(markets, MorphoSupplyRotationParams(ORACLE_TYPES=("ChainlinkOracleV2",), SIGNAL_LAG_BARS=1,
                                                    SIGNAL_WINDOW_BARS=1, MIN_EDGE_BPS=0.0,
                                                    MIN_LIQUIDITY_MULTIPLE=0.0, OWN_IMPACT=False))
    names = [strategy._name(m) for m in markets]  # pylint: disable=protected-access
    run_panel(strategy, [{names[0]: 0.40, names[1]: 0.04}] * 6, names)
    assert strategy.get_entity(names[1]).balance == pytest.approx(CASH)
    assert strategy.get_entity(names[0]).balance == pytest.approx(0.0)


def test_entry_is_refused_when_the_target_cannot_be_exited():
    markets = [listing("a"), listing("b")]
    params = MorphoSupplyRotationParams(SIGNAL_LAG_BARS=1, SIGNAL_WINDOW_BARS=1, MIN_EDGE_BPS=0.0,
                                    MIN_LIQUIDITY_MULTIPLE=1.0, OWN_IMPACT=False)
    strategy = build(markets, params)
    names = [strategy._name(m) for m in markets]  # pylint: disable=protected-access
    for bar in range(6):
        step(strategy, {names[0]: state(0.02, liquidity=0.0), names[1]: state(0.05, liquidity=1.0),
                        "CASH": SimpleSpotExchangeGlobalState(close=1.0)}, bar)
    # B is the best but only $1 is withdrawable, far less than the position
    assert strategy.get_entity(names[1]).balance == pytest.approx(0.0)
    assert strategy.get_entity("CASH").balance == pytest.approx(CASH)
    assert [event["kind"] for event in strategy.deferrals].count("entry") >= 1


# ------------------------------------------------------------------ gate
def test_gate_sizes_the_withdrawal_and_records_the_deferral():
    markets = [listing("a"), listing("b")]
    strategy = build(markets)
    names = [strategy._name(m) for m in markets]  # pylint: disable=protected-access
    # A yields best: all the cash lands there
    run_panel(strategy, [{names[0]: 0.05, names[1]: 0.01}] * 5, names)
    assert strategy.get_entity(names[0]).balance == pytest.approx(CASH)
    # A's liquidity collapses and B becomes the better market: the transfer is
    # sized to what A can actually pay out, and the shortfall is recorded. The
    # signal carries one bar of lag, so the collapse is only visible one bar in.
    for bar in (5, 6):
        step(strategy, {names[0]: state(0.01, liquidity=300_000.0), names[1]: state(0.09),
                        "CASH": SimpleSpotExchangeGlobalState(close=1.0)}, bar)
    assert strategy.get_entity(names[1]).balance == pytest.approx(300_000.0)
    assert strategy.get_entity(names[0]).balance == pytest.approx(CASH - 300_000.0)
    report = strategy.deferral_report()
    assert report["events"] == 1
    assert report["deferred_notional"] == pytest.approx(CASH - 300_000.0)


def test_stranded_capital_is_retried_every_bar_and_the_deferral_closes():
    markets = [listing("a"), listing("b")]
    strategy = build(markets)
    names = [strategy._name(m) for m in markets]  # pylint: disable=protected-access
    run_panel(strategy, [{names[0]: 0.05, names[1]: 0.01}] * 5, names)
    assert strategy.get_entity(names[0]).balance == pytest.approx(CASH)
    # Fully gated: nothing moves, but the block is counted (one event, not one per bar).
    for bar in range(5, 8):
        step(strategy, {names[0]: state(0.01, liquidity=0.0), names[1]: state(0.09),
                        "CASH": SimpleSpotExchangeGlobalState(close=1.0)}, bar)
        assert strategy.get_entity(names[0]).balance == pytest.approx(CASH)
    exits = [event for event in strategy.deferrals if event["kind"] == "exit"]
    assert len(exits) == 1 and "bars" not in exits[0]  # still open
    # A recovers: the retry drains it and stamps how long the exit was blocked.
    for bar in range(8, 11):
        step(strategy, {names[0]: state(0.01, liquidity=CASH), names[1]: state(0.09),
                        "CASH": SimpleSpotExchangeGlobalState(close=1.0)}, bar)
    assert strategy.get_entity(names[0]).balance == pytest.approx(0.0)
    assert strategy.get_entity(names[1]).balance == pytest.approx(CASH)
    assert strategy.deferral_report()["max_bars_blocked"] >= 1


# ------------------------------------------------------------- lookahead
def test_decision_never_uses_the_bar_it_trades_on():
    """With SIGNAL_LAG_BARS=k the signal must ignore bars t..t-k+1."""
    markets = [listing("a"), listing("b")]
    params = MorphoSupplyRotationParams(SIGNAL_LAG_BARS=2, SIGNAL_WINDOW_BARS=1, MIN_EDGE_BPS=0.0,
                                   MIN_LIQUIDITY_MULTIPLE=0.0, OWN_IMPACT=False)
    strategy = build(markets, params)
    names = [strategy._name(m) for m in markets]  # pylint: disable=protected-access
    run_panel(strategy, [{names[0]: 0.02, names[1]: 0.05}] * 8, names)
    assert strategy.get_entity(names[1]).balance == pytest.approx(CASH)
    # A spikes for exactly one bar: with 2 bars of lag the decision on that bar
    # and the next must not see it, so the position stays in B.
    step(strategy, {names[0]: state(0.90), names[1]: state(0.05), "CASH": SimpleSpotExchangeGlobalState(close=1.0)}, 8)
    assert strategy.get_entity(names[1]).balance == pytest.approx(CASH)
    step(strategy, {names[0]: state(0.01), names[1]: state(0.05), "CASH": SimpleSpotExchangeGlobalState(close=1.0)}, 9)
    assert strategy.get_entity(names[1]).balance == pytest.approx(CASH)


def test_signal_is_a_window_not_a_snapshot():
    """A one-bar spike must not move a 24-bar-average signal."""
    markets = [listing("a"), listing("b")]
    params = MorphoSupplyRotationParams(SIGNAL_LAG_BARS=1, SIGNAL_WINDOW_BARS=24, MIN_EDGE_BPS=0.0,
                                    MIN_LIQUIDITY_MULTIPLE=0.0, OWN_IMPACT=False)
    strategy = build(markets, params)
    names = [strategy._name(m) for m in markets]  # pylint: disable=protected-access
    run_panel(strategy, [{names[0]: 0.01, names[1]: 0.05}] * 30, names)
    assert strategy.get_entity(names[1]).balance == pytest.approx(CASH)
    step(strategy, {names[0]: state(0.99), names[1]: state(0.05), "CASH": SimpleSpotExchangeGlobalState(close=1.0)}, 30)
    assert strategy.get_entity(names[1]).balance == pytest.approx(CASH)  # one spike is not a level


# ------------------------------------------------------------------ costs
def test_costs_are_paid_from_the_deployed_notional_not_from_cash():
    markets = [listing("a"), listing("b")]
    params = MorphoSupplyRotationParams(SIGNAL_LAG_BARS=1, SIGNAL_WINDOW_BARS=1, MIN_EDGE_BPS=0.0,
                                    MIN_LIQUIDITY_MULTIPLE=0.0, COST_BPS_PER_LEG=5.0, OWN_IMPACT=False)
    strategy = build(markets, params)
    names = [strategy._name(m) for m in markets]  # pylint: disable=protected-access
    observations = []
    for bar, rates in enumerate([{names[0]: 0.02, names[1]: 0.01}] * 4 + [{names[0]: 0.01, names[1]: 0.09}] * 4):
        states = {name: state(rates[name]) for name in names}
        states["CASH"] = SimpleSpotExchangeGlobalState(close=1.0)
        observations.append(Observation(timestamp=START + timedelta(hours=bar), states=states))
    result = strategy.run(observations)
    assert result.execution_records
    fees = strategy.execution_ledger.total_fees_paid
    assert fees > 0.0                                   # telemetry picks the legs up
    assert strategy.get_entity("CASH").balance == pytest.approx(0.0)  # cash never funds fees
    deployed = sum(strategy.get_entity(name).balance for name in names)
    assert deployed < CASH                              # the fee came out of the notional


def test_own_impact_is_propagated_to_every_market_entity():
    markets = [listing("a"), listing("b")]
    params = MorphoSupplyRotationParams(OWN_IMPACT=True, ABSORPTION=0.5, COST_BPS_PER_LEG=3.0)
    strategy = build(markets, params)
    for market in markets:
        entity = strategy.get_entity(strategy._name(market))  # pylint: disable=protected-access
        assert entity.own_impact is True
        assert entity.absorption == pytest.approx(0.5)
        assert entity.cost_bps_per_leg == pytest.approx(3.0)


def test_hold_limit_forces_a_reselection_the_edge_threshold_would_block():
    """``MAX_HOLD_BARS`` is a deadline for re-justifying a position, nothing else.

    With ``TOP_K=2`` the held market stays inside the acceptable band and the
    edge is far below ``MIN_EDGE_BPS``, so the edge rule alone would never move
    — only the hold limit can, and it re-selects rather than forcing cash.
    """
    markets = [listing("a"), listing("b")]
    params = MorphoSupplyRotationParams(SIGNAL_LAG_BARS=1, SIGNAL_WINDOW_BARS=1, MIN_EDGE_BPS=1000.0, TOP_K=2,
                                    MIN_LIQUIDITY_MULTIPLE=0.0, MAX_HOLD_BARS=3, OWN_IMPACT=False)
    strategy = build(markets, params)
    names = [strategy._name(m) for m in markets]  # pylint: disable=protected-access
    run_panel(strategy, [{names[0]: 0.05, names[1]: 0.049}] * 4, names)
    assert strategy.get_entity(names[0]).balance == pytest.approx(CASH)
    for bar in range(4, 8):
        step(strategy, {names[0]: state(0.02), names[1]: state(0.049),
                        "CASH": SimpleSpotExchangeGlobalState(close=1.0)}, bar)
    assert strategy.get_entity(names[1]).balance == pytest.approx(CASH)  # forced re-selection
    assert strategy.get_entity(names[0]).balance == pytest.approx(0.0)


# ------------------------------------------------------- review fixes (PR #116)
def test_min_edge_applies_when_the_holder_falls_just_outside_the_band():
    """Falling to rank K+1 by less than ``MIN_EDGE_BPS`` must not trigger two legs of cost."""
    markets = [listing("a"), listing("b")]
    params = MorphoSupplyRotationParams(TOP_K=1, SIGNAL_LAG_BARS=1, SIGNAL_WINDOW_BARS=1, MIN_EDGE_BPS=50.0,
                                        MIN_LIQUIDITY_MULTIPLE=0.0, OWN_IMPACT=False)
    strategy = build(markets, params)
    names = [strategy._name(m) for m in markets]  # pylint: disable=protected-access
    run_panel(strategy, [{names[0]: 0.050, names[1]: 0.040}] * 4, names)
    assert strategy.get_entity(names[0]).balance == pytest.approx(CASH)
    # B edges ahead by 20 bp: outside the top-1 band, but below the 50 bp threshold -> hold
    run_panel(strategy, [{names[0]: 0.050, names[1]: 0.052}] * 4, names)
    assert strategy.get_entity(names[0]).balance == pytest.approx(CASH)
    # B ahead by 100 bp: now it pays -> rotate
    run_panel(strategy, [{names[0]: 0.050, names[1]: 0.060}] * 4, names)
    assert strategy.get_entity(names[1]).balance == pytest.approx(CASH)


def test_nav_drop_equals_recorded_fees_over_a_full_rotation():
    """Two legs of 5 bp each: the equity curve must lose exactly what ``fees_paid`` reports."""
    markets = [listing("a"), listing("b")]
    params = MorphoSupplyRotationParams(SIGNAL_LAG_BARS=1, SIGNAL_WINDOW_BARS=1, MIN_EDGE_BPS=0.0,
                                        MIN_LIQUIDITY_MULTIPLE=0.0, COST_BPS_PER_LEG=5.0, OWN_IMPACT=False)
    strategy = build(markets, params)
    names = [strategy._name(m) for m in markets]  # pylint: disable=protected-access
    panel = [{names[0]: 0.02, names[1]: 0.01}] * 4 + [{names[0]: 0.01, names[1]: 0.09}] * 4
    observations = []
    for bar, rates in enumerate(panel):
        states = {name: state(rates[name]) for name in names}
        states["CASH"] = SimpleSpotExchangeGlobalState(close=1.0)
        observations.append(Observation(timestamp=START + timedelta(hours=bar), states=states))
    strategy.run(observations)
    fees = strategy.execution_ledger.total_fees_paid
    nav = strategy.total_balance
    assert fees > 0.0
    assert CASH - nav == pytest.approx(fees, rel=1e-9)
    # cash -> A (1 leg), A -> B (2 legs): three fee-bearing legs, A fully drained
    assert len(strategy.execution_ledger.records) == 3
    assert strategy.get_entity(names[0]).balance == pytest.approx(0.0, abs=1e-6)


def test_entry_refusals_are_counted_as_episodes_not_bars():
    markets = [listing("a"), listing("b")]
    params = MorphoSupplyRotationParams(SIGNAL_LAG_BARS=1, SIGNAL_WINDOW_BARS=1, MIN_EDGE_BPS=0.0,
                                        MIN_LIQUIDITY_MULTIPLE=1.0, OWN_IMPACT=False)
    strategy = build(markets, params)
    names = [strategy._name(m) for m in markets]  # pylint: disable=protected-access
    for bar in range(8):
        step(strategy, {names[0]: state(0.02, liquidity=0.0), names[1]: state(0.05, liquidity=1.0),
                        "CASH": SimpleSpotExchangeGlobalState(close=1.0)}, bar)
    report = strategy.deferral_report()
    # both markets are refused every bar (the fallback walks the whole ranking):
    # one episode per market, 7 bars each — never 14 events
    assert report["entry_episodes"] == 2
    assert report["entry_blocked_bars"] == 14         # first bar has no lagged signal yet
    assert report["events"] == 0 and report["deferred_notional"] == 0.0   # exit stats untouched
    entries = [event for event in strategy.deferrals if event["kind"] == "entry"]
    assert sorted(event["market"] for event in entries) == sorted(names)
    assert all(event["bars"] == 7 and event["notional"] == pytest.approx(CASH) for event in entries)
    assert strategy.get_entity("CASH").balance == pytest.approx(CASH)


def test_ranking_on_projected_apy_prefers_the_market_our_size_does_not_crush():
    """Small market: 9% headline that $1M would dilute to ~4%; big market: 5.4% that barely moves."""
    small = state(0.09, supply=2_000_000.0, borrow=1_800_000.0, liquidity=200_000.0, rate_at_target=0.10)
    big = state(0.054, supply=200_000_000.0, borrow=180_000_000.0, liquidity=20_000_000.0, rate_at_target=0.06)
    markets = [listing("a"), listing("b")]
    base = {"SIGNAL_LAG_BARS": 1, "SIGNAL_WINDOW_BARS": 1, "MIN_EDGE_BPS": 0.0, "MIN_LIQUIDITY_MULTIPLE": 0.0,
            "OWN_IMPACT": True, "ABSORPTION": 0.0}
    projected = build(markets, MorphoSupplyRotationParams(**base, RANK_ON_PROJECTED_APY=True))
    headline = build(markets, MorphoSupplyRotationParams(**base, RANK_ON_PROJECTED_APY=False))
    names = [projected._name(m) for m in markets]  # pylint: disable=protected-access
    for bar in range(4):
        for strategy in (projected, headline):
            step(strategy, {names[0]: small, names[1]: big, "CASH": SimpleSpotExchangeGlobalState(close=1.0)}, bar)
    assert headline.get_entity(names[0]).balance > 0.0          # chases the 9% headline
    assert projected.get_entity(names[1]).balance > 0.0         # takes the rate it would actually earn
    assert projected.get_entity(names[0]).balance == pytest.approx(0.0)
    # and the projected strategy really does earn more per bar on the capital it holds
    assert (projected.get_entity(names[1]).effective_supply_apy
            > headline.get_entity(names[0]).effective_supply_apy)


def test_falls_back_to_the_next_market_when_the_best_is_too_illiquid():
    """A position too large for the top market must not sit in cash while the runner-up can hold it."""
    markets = [listing("a"), listing("b"), listing("c")]
    params = MorphoSupplyRotationParams(SIGNAL_LAG_BARS=1, SIGNAL_WINDOW_BARS=1, MIN_EDGE_BPS=0.0,
                                        MIN_LIQUIDITY_MULTIPLE=1.0, OWN_IMPACT=False)
    strategy = build(markets, params)
    names = [strategy._name(m) for m in markets]  # pylint: disable=protected-access
    for bar in range(4):
        step(strategy, {names[0]: state(0.09, liquidity=100_000.0),       # best rate, cannot hold $1M
                        names[1]: state(0.06, liquidity=50_000_000.0),    # runner-up, deep
                        names[2]: state(0.03, liquidity=50_000_000.0),
                        "CASH": SimpleSpotExchangeGlobalState(close=1.0)}, bar)
    assert strategy.get_entity(names[1]).balance == pytest.approx(CASH)
    assert strategy.get_entity("CASH").balance == pytest.approx(0.0)
    report = strategy.deferral_report()
    assert report["entry_episodes"] >= 1                                  # the skip over A is on record
    assert all(event["market"] == names[0] for event in strategy.deferrals if event["kind"] == "entry")


def test_loan_symbol_and_pool_size_filters_are_hyperparameters():
    """A USDT market and a sub-$1M market must not be selected even when they pay the most."""
    usdt = MorphoMarketListing(
        chain_id=1, market_id="u" * 64, loan_address="0xusdt", loan_symbol="USDT", loan_decimals=6,
        collateral_address="0xcoll", collateral_symbol="PT", collateral_decimals=18,
        lltv=0.86, oracle_type="ChainlinkOracleV2", listed=True, creation_timestamp=1,
    )
    markets = [usdt, listing("s"), listing("b")]
    params = MorphoSupplyRotationParams(SIGNAL_LAG_BARS=1, SIGNAL_WINDOW_BARS=1, MIN_EDGE_BPS=0.0,
                                        MIN_LIQUIDITY_MULTIPLE=0.0, OWN_IMPACT=False,
                                        LOAN_SYMBOLS=("USDC",), MIN_POOL_SIZE_USD=1_000_000.0)
    strategy = build(markets, params)
    names = [strategy._name(m) for m in markets]  # pylint: disable=protected-access
    for bar in range(4):
        step(strategy, {names[0]: state(0.20),                                  # USDT: excluded by symbol
                        names[1]: state(0.15, supply=500_000.0, borrow=400_000.0, liquidity=100_000.0),  # too small
                        names[2]: state(0.05),
                        "CASH": SimpleSpotExchangeGlobalState(close=1.0)}, bar)
    assert strategy.get_entity(names[2]).balance == pytest.approx(CASH)
    assert strategy.get_entity(names[0]).balance == 0.0 and strategy.get_entity(names[1]).balance == 0.0
    # defaults admit both stablecoins
    assert MorphoSupplyRotationParams().LOAN_SYMBOLS == ("USDC", "USDT")


def test_rotation_prices_gas_per_chain():
    base = MorphoMarketListing(
        chain_id=8453, market_id="b" * 64, loan_address="0xusdc", loan_symbol="USDC", loan_decimals=6,
        collateral_address="0xcoll", collateral_symbol="PT", collateral_decimals=18,
        lltv=0.86, oracle_type="ChainlinkOracleV2", listed=True, creation_timestamp=1,
    )
    params = MorphoSupplyRotationParams(GAS_USD_PER_LEG=1.5, GAS_USD_BY_CHAIN={8453: 0.01})
    strategy = build([listing("a"), base], params)
    gas = {m.chain_id: strategy.get_entity(strategy._name(m)).leg_cost_model[0] for m in (listing("a"), base)}  # pylint: disable=protected-access
    assert gas == {1: pytest.approx(1.5), 8453: pytest.approx(0.01)}
