"""Opt-in marginal-mode options of :class:`MorphoSupplyAllocationStrategy` and the horizon valuation.

The parity tests pin the promise that the new options are opt-in: with their
defaults, a marginal-mode backtest is bit-identical to the solver as it stood
before them (frozen verbatim in :class:`_FrozenMarginal`).
"""
import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from fractal.core.base import Observation
from fractal.core.entities.models.morpho_math import supply_apy_from_curve
from fractal.core.entities.protocols.morpho_supply import (
    MorphoSupplyEntity,
    MorphoSupplyException,
    MorphoSupplyGlobalState,
)
from fractal.core.entities.simple.spot import SimpleSpotExchange, SimpleSpotExchangeGlobalState
from fractal.loaders.morpho_markets import MorphoMarketListing
from fractal.strategies.morpho_supply_allocation import MorphoSupplyAllocationParams, MorphoSupplyAllocationStrategy
from fractal.strategies.morpho_supply_rotation import market_entity_name

pytestmark = pytest.mark.core

START = datetime(2026, 1, 1, tzinfo=timezone.utc)


def listing(market_id: str, chain_id: int = 1) -> MorphoMarketListing:
    return MorphoMarketListing(
        chain_id=chain_id, market_id=market_id * 64, loan_address="0xusdc", loan_symbol="USDC", loan_decimals=6,
        collateral_address="0xcoll", collateral_symbol="PT", collateral_decimals=18,
        lltv=0.86, oracle_type="ChainlinkOracleV2", listed=True, creation_timestamp=1,
    )


def market(supply: float, utilization: float, rate_at_target: float, fee: float = 0.0,
           reward_apy: float = 0.0, apy_tilt: float = 1.0) -> MorphoSupplyGlobalState:
    borrow = supply * utilization
    supply_apy = supply_apy_from_curve(utilization, math.log1p(rate_at_target), fee) * apy_tilt
    return MorphoSupplyGlobalState(
        utilization=utilization, supply_rate=math.log1p(supply_apy) / 8760.0, supply_apy=supply_apy,
        borrow_apy=0.0, rate_at_target=rate_at_target, fee=fee, supply_assets_usd=supply,
        borrow_assets_usd=borrow, liquidity_assets_usd=supply - borrow, reward_apy=reward_apy,
    )


# --------------------------------------------------------- frozen reference
class _FrozenMarginal(MorphoSupplyAllocationStrategy):
    """``_solve_marginal`` / ``_pool_options`` exactly as they were before the opt-in options."""

    def _solve_marginal(self, snapshots, nav):
        params = self._params
        chunks = params.ALLOC_CHUNKS
        chunk = nav / chunks
        horizon = params.HOLD_HORIZON_BARS / params.BARS_PER_YEAR
        held = {name for name in self._history if self.get_entity(name).balance > 1e-9}
        options = []
        for name in sorted(set(snapshots) | held):
            snap = snapshots.get(name) if name in self._fresh else None
            option = self._pool_options(name, snap, nav, chunk, horizon, name in held)
            if option is not None:
                options.append(option)
        capacity = chunks
        best = np.full(capacity + 1, -np.inf)
        best[0] = 0.0
        choices = []
        ks = np.arange(capacity + 1)
        for _, units, _, values in options:
            idx = ks[None, :] - units[:, None]
            table = np.where(idx >= 0, best[np.clip(idx, 0, None)], -np.inf) + values[:, None]
            choices.append(table.argmax(axis=0))
            best = table.max(axis=0)
        idle_value = horizon * params.TARGET_APY * chunk * np.clip(chunks - ks, 0, None)
        objective = best + idle_value
        k = int(np.argmax(objective))
        hold = {}
        hold_value = 0.0
        for name, _, amounts, values in options:
            nearest = int(np.argmin(np.abs(amounts - self.get_entity(name).balance)))
            hold[name] = float(amounts[nearest])
            hold_value += float(values[nearest])
        hold_value += horizon * params.TARGET_APY * max(nav - sum(hold.values()), 0.0)
        targets = {}
        if hold_value >= objective[k]:
            targets = hold
        else:
            for (name, units, amounts, _), choice in zip(reversed(options), reversed(choices), strict=True):
                c = int(choice[k])
                targets[name] = float(amounts[c])
                k -= int(units[c])
        self._last_targets = targets
        self._last_lambda = self._marginal_rate(targets, snapshots, chunk)
        return targets

    def _pool_options(self, name, snap, nav, chunk, horizon, holding):
        entity = self.get_entity(name)
        now = entity.balance
        gas, bps = entity.leg_cost_model
        stuck = max(now - min(now, entity.withdrawable), 0.0)
        eligible = snap is not None and self._selectable(name)
        if not eligible:
            if not holding:
                return None
            if snap is None:
                amounts = np.array([now])
            elif stuck < now:
                amounts = np.array([stuck])
            else:
                amounts = np.array([now])
        else:
            if not holding and entity.projected_supply_apy(0.0, snap) <= self._params.TARGET_APY:
                return None
            cap = max(min(self._liquidity_cap(name), nav), now)
            grid = np.arange(math.ceil(stuck / chunk - 1e-9), math.floor(cap / chunk + 1e-9) + 1) * chunk
            amounts = np.unique(np.concatenate([grid[(grid >= stuck - 1e-9) & (grid <= cap + 1e-9)], [now]]))
        snap = snap if snap is not None else entity.global_state
        income = amounts * np.array([entity.projected_supply_apy(float(a), snap) if a > 0 else 0.0 for a in amounts])
        moved = np.abs(amounts - now)
        cost = np.where(moved > 1e-9, gas + bps * moved / 1e4, 0.0)
        values = horizon * income - cost + np.where(moved <= 1e-9, 1e-9, 0.0)
        units = np.ceil(amounts / chunk - 1e-12).astype(int)
        return name, units, amounts, values


# ------------------------------------------------------------ scenario
MARKETS = [listing("a"), listing("b"), listing("c", chain_id=8453), listing("d")]
NAMES = [market_entity_name(m) for m in MARKETS]
BASE_PARAMS = {
    "ALLOCATION_MODE": "marginal", "SIGNAL_LAG_BARS": 1, "SIGNAL_WINDOW_BARS": 2, "REBALANCE_BARS": 2,
    "HOLD_HORIZON_BARS": 48, "ALLOC_CHUNKS": 40, "TARGET_APY": 0.03, "GAS_USD_PER_LEG": 3.0,
    "GAS_USD_BY_CHAIN": {8453: 0.2}, "COST_BPS_PER_LEG": 0.5, "ABSORPTION": 0.2, "IRM_FEEDBACK": True,
    "MIN_LIQUIDITY_MULTIPLE": 0.0,
}


def scenario_states(hour: int) -> dict:
    """Synthetic, time-varying markets: a spike pool, a deep pool, an L2 pool and one with a stale gap."""
    wave = math.sin(hour / 3.0)
    states = {
        NAMES[0]: market(4e6, min(0.93 + 0.05 * wave, 0.999), 0.06, fee=0.1),
        NAMES[1]: market(80e6, 0.86 + 0.02 * math.cos(hour / 5.0), 0.05),
        NAMES[2]: market(15e6, 0.88 + 0.03 * wave, 0.07, reward_apy=0.01),
    }
    if not 8 <= hour < 12:                       # market d goes stale for four bars
        states[NAMES[3]] = market(6e6, 0.95 - 0.04 * math.cos(hour / 2.0), 0.04, apy_tilt=1.1)
    return states


def run_scenario(cls, capital: float, bars: int = 30, **overrides):
    params = MorphoSupplyAllocationParams(**{**BASE_PARAMS, **overrides})
    strategy = cls(markets=MARKETS, cash_entity=SimpleSpotExchange(trading_fee=0.0), params=params)
    strategy.get_entity("CASH").action_deposit(capital)
    observations = [Observation(timestamp=START + timedelta(hours=hour),
                                states={**scenario_states(hour), "CASH": SimpleSpotExchangeGlobalState(close=1.0)})
                    for hour in range(bars)]
    return strategy, strategy.run(observations)


def fingerprint(strategy, result):
    return (
        result.balances,
        [(r.entity, r.action, r.traded_notional, r.fee_paid) for r in result.execution_records],
        strategy.allocation_report(),
        strategy.deferrals,
    )


# ------------------------------------------------------------- parity
@pytest.mark.parametrize("liquidity_multiple", [0.0, 1.5])
@pytest.mark.parametrize("capital", [250_000.0, 7_300_000.0, 40e6])
def test_default_options_are_bit_identical_to_the_frozen_solver(capital, liquidity_multiple):
    gate = {"MIN_LIQUIDITY_MULTIPLE": liquidity_multiple}
    frozen = fingerprint(*run_scenario(_FrozenMarginal, capital, **gate))
    current = fingerprint(*run_scenario(MorphoSupplyAllocationStrategy, capital, **gate))
    explicit = fingerprint(*run_scenario(MorphoSupplyAllocationStrategy, capital, **gate,
                                         CHUNK_USD=None, VALUATION="instant", MAX_CHUNKS=3))
    assert current == frozen
    assert explicit == frozen
    # the scenario actually trades: several pools are entered and resized
    assert len({r[0] for r in frozen[1]} - {"CASH"}) >= 2


def test_default_pool_options_are_bit_identical_on_every_resolve():
    """Below the backtest outputs: every pool's candidate amounts and values, re-solve by re-solve."""

    class Probe(MorphoSupplyAllocationStrategy):
        compared = 0

        def _solve_marginal(self, snapshots, nav):
            params = self._params
            chunk, horizon = nav / params.ALLOC_CHUNKS, params.HOLD_HORIZON_BARS / params.BARS_PER_YEAR
            held = {name for name in self._history if self.get_entity(name).balance > 1e-9}
            for name in sorted(set(snapshots) | held):
                snap = snapshots.get(name) if name in self._fresh else None
                mine = self._pool_options(name, snap, nav, chunk, horizon, name in held)
                ref = _FrozenMarginal._pool_options(self, name, snap, nav, chunk, horizon,  # pylint: disable=protected-access
                                                    name in held)
                assert (mine is None) == (ref is None)
                if mine is not None:
                    assert mine[0] == ref[0]
                    for got, want in zip(mine[1:], ref[1:], strict=True):
                        assert got.dtype == want.dtype and np.array_equal(got, want)
                    Probe.compared += 1
            targets = super()._solve_marginal(snapshots, nav)
            assert targets == _FrozenMarginal._solve_marginal(self, snapshots, nav)  # pylint: disable=protected-access
            return targets

    run_scenario(Probe, 7_300_000.0, MIN_LIQUIDITY_MULTIPLE=1.5)
    assert Probe.compared >= 30


def test_parity_scenario_exercises_the_solver():
    strategy, _ = run_scenario(MorphoSupplyAllocationStrategy, 7_300_000.0)
    log = strategy.resolve_log
    assert len(log) >= 10 and all(not row["capped"] for row in log)
    assert all(row["chunk"] == pytest.approx(row["budget"] / 40) for row in log)


def test_new_defaults():
    params = MorphoSupplyAllocationParams()
    assert (params.CHUNK_USD, params.VALUATION) == (None, "instant")
    assert isinstance(params.MAX_CHUNKS, int) and params.MAX_CHUNKS >= 1


# ---------------------------------------------------------- fixed chunk
def build_flat(capital: float, states, **overrides):
    markets = MARKETS[:len(states)]
    params = MorphoSupplyAllocationParams(**{**BASE_PARAMS, "SIGNAL_WINDOW_BARS": 1, "REBALANCE_BARS": 1,
                                             "IRM_FEEDBACK": False, "ABSORPTION": 0.0, "GAS_USD_PER_LEG": 0.0,
                                             "GAS_USD_BY_CHAIN": {}, "COST_BPS_PER_LEG": 0.0, **overrides})
    strategy = MorphoSupplyAllocationStrategy(markets=markets, cash_entity=SimpleSpotExchange(trading_fee=0.0),
                                              params=params)
    strategy.get_entity("CASH").action_deposit(capital)
    names = [market_entity_name(m) for m in markets]
    obs = {**dict(zip(names, states, strict=True)), "CASH": SimpleSpotExchangeGlobalState(close=1.0)}
    for hour in range(2):                 # the second step is the first solve, on NAV = capital exactly
        strategy.step(Observation(timestamp=START + timedelta(hours=hour), states=obs))
    return strategy, names


RICH = [market(30e6, 0.90, 0.10), market(60e6, 0.88, 0.09)]


def test_fixed_chunk_equal_to_the_relative_chunk_gives_the_same_solution():
    relative, _ = build_flat(1e6, RICH, ALLOC_CHUNKS=100)
    fixed, _ = build_flat(1e6, RICH, ALLOC_CHUNKS=7, CHUNK_USD=10_000.0)
    assert fixed._last_targets == relative._last_targets          # pylint: disable=protected-access
    row = fixed.resolve_log[0]
    assert (row["chunk"], row["capacity_units"], row["capped"]) == (10_000.0, 100, False)


def test_budget_off_the_fixed_grid_stays_idle():
    strategy, names = build_flat(1_050_000.0, RICH, TARGET_APY=0.0, CHUNK_USD=100_000.0)
    targets = strategy._last_targets                                # pylint: disable=protected-access
    assert sum(targets.values()) == pytest.approx(1e6)               # ten chunks; 50k cannot be placed
    assert all(round(amount / 100_000.0, 9).is_integer() for amount in targets.values())
    row = strategy.resolve_log[0]
    assert row["capacity_units"] == 10 and row["idle_usd"] == pytest.approx(50_000.0)
    assert sum(strategy.get_entity(n).balance for n in names) == pytest.approx(1e6, rel=1e-6)


def test_max_chunks_caps_the_grid_and_flags_the_resolve():
    strategy, _ = build_flat(1e6, RICH, TARGET_APY=0.0, CHUNK_USD=10_000.0, MAX_CHUNKS=20)
    row = strategy.resolve_log[0]
    assert (row["capacity_units"], row["capped"]) == (20, True)
    assert sum(strategy._last_targets.values()) <= 200_000.0 + 1e-6  # pylint: disable=protected-access
    assert row["idle_usd"] >= 800_000.0 - 1e-6


@pytest.mark.parametrize("budget, units, capped", [
    (200_000.0, 20, False),           # exactly MAX_CHUNKS chunks
    (205_000.0, 20, True),            # 20.5 chunks: a partial 21st chunk would be needed
    (199_999.99999999, 20, False),    # float residue below a whole count is not a partial chunk
    (195_000.0, 19, False),
    (5_000.0, 0, False),
])
def test_grid_counts_whole_chunks_and_flags_a_cap_including_a_partial_chunk(budget, units, capped):
    strategy = MorphoSupplyAllocationStrategy(
        markets=MARKETS[:1], cash_entity=SimpleSpotExchange(trading_fee=0.0),
        params=MorphoSupplyAllocationParams(ALLOCATION_MODE="marginal", CHUNK_USD=10_000.0, MAX_CHUNKS=20))
    assert strategy._grid(budget) == (10_000.0, units, capped)        # pylint: disable=protected-access


def test_capital_below_one_chunk_stays_idle():
    strategy, names = build_flat(5_000.0, RICH, TARGET_APY=0.0, CHUNK_USD=10_000.0)
    assert strategy.resolve_log[0]["capacity_units"] == 0
    assert all(strategy.get_entity(n).balance == 0.0 for n in names)


@pytest.mark.parametrize("overrides", [
    {"CHUNK_USD": 0.0}, {"CHUNK_USD": -5.0}, {"CHUNK_USD": float("nan")}, {"CHUNK_USD": True},
    {"CHUNK_USD": "1000"}, {"MAX_CHUNKS": 0}, {"MAX_CHUNKS": 2.5}, {"VALUATION": "average"},
])
def test_options_are_validated(overrides):
    with pytest.raises(ValueError):
        MorphoSupplyAllocationStrategy(markets=MARKETS[:1], cash_entity=SimpleSpotExchange(trading_fee=0.0),
                                       params=MorphoSupplyAllocationParams(**overrides))


# --------------------------------------------------- horizon valuation (entity)
def replay(entity_kwargs: dict, state: MorphoSupplyGlobalState, position: float, bars: int) -> float:
    """Realised APY of ``position`` held ``bars`` bars through ``update_state`` on a frozen state."""
    entity = MorphoSupplyEntity(**entity_kwargs)
    entity.update_state(state)
    entity.action_deposit(position)
    for _ in range(bars):
        entity.update_state(state)
    return math.expm1(math.log(entity.balance / position) * entity.bars_per_year / bars)


ENTITY_CASES = [
    {"own_impact": True},
    {"own_impact": True, "irm_feedback": True},
    {"own_impact": True, "irm_feedback": True, "absorption": 0.4},
    {"own_impact": True, "irm_feedback": True, "use_rewards": True, "reward_weight": 0.5},
    {"own_impact": False, "use_rewards": True},
]
STATES = [market(5e6, 0.95, 0.08, fee=0.1, reward_apy=0.02), market(50e6, 0.70, 0.03)]


@pytest.mark.parametrize("kwargs", ENTITY_CASES)
@pytest.mark.parametrize("state", STATES)
@pytest.mark.parametrize("bars", [1, 24, 720])
def test_horizon_apy_equals_the_bar_by_bar_backtest_replay(kwargs, state, bars):
    positions = np.array([1e4, 2e6, 3e7])
    entity = MorphoSupplyEntity(**kwargs)
    entity.update_state(state)
    valued = entity.horizon_supply_apy(positions, bars, state)
    realised = [replay(kwargs, state, float(p), bars) for p in positions]
    assert valued == pytest.approx(realised, rel=1e-9, abs=1e-13)


def test_horizon_apy_starts_from_the_entitys_current_drift():
    state = market(5e6, 0.95, 0.08)
    held = MorphoSupplyEntity(irm_feedback=True)
    held.update_state(state)
    held.action_deposit(4e6)
    for _ in range(100):
        held.update_state(state)                       # 4M on 5M pulls u below target: R* drifts down
    assert held.internal_state.rstar_log_drift < 0.0
    fresh = MorphoSupplyEntity(irm_feedback=True)
    fresh.update_state(state)
    assert held.horizon_supply_apy(0.0, 24, state) < fresh.horizon_supply_apy(0.0, 24, state)


def test_controller_drift_lowers_the_horizon_value_of_a_diluting_position():
    state = market(5e6, 0.90, 0.08)
    with_drift = MorphoSupplyEntity(irm_feedback=True)
    without = MorphoSupplyEntity(irm_feedback=False)
    for entity in (with_drift, without):
        entity.update_state(state)
    short, long_ = (with_drift.horizon_supply_apy(5e6, bars, state) for bars in (1, 720))
    assert long_ < short < without.horizon_supply_apy(5e6, 1, state) * (1 + 1e-12)
    # with nothing drifting the horizon APY tracks the instant curve (own growth is second order)
    assert without.horizon_supply_apy(5e6, 720, state) == pytest.approx(without.projected_supply_apy(5e6, state),
                                                                        rel=1e-3)


def test_horizon_apy_shape_and_validation():
    entity = MorphoSupplyEntity()
    entity.update_state(market(5e6, 0.9, 0.05))
    assert entity.horizon_supply_apy(1e6, 3).shape == ()
    assert entity.horizon_supply_apy([[1e6, 2e6]], 3).shape == (1, 2)
    for bad in (-1.0, float("nan")):
        with pytest.raises(MorphoSupplyException):
            entity.horizon_supply_apy([1e6, bad], 3)
    for bad in (0, 2.5, True):
        with pytest.raises(MorphoSupplyException):
            entity.horizon_supply_apy(1e6, bad)


# -------------------------------------------------- horizon valuation (strategy)
def test_horizon_valuation_places_less_where_the_position_would_drift_the_rate_down():
    hot = [market(10e6, 0.90, 0.08)]
    common = {"IRM_FEEDBACK": True, "TARGET_APY": 0.03, "HOLD_HORIZON_BARS": 720}
    instant, names = build_flat(10e6, hot, VALUATION="instant", **common)
    horizon, _ = build_flat(10e6, hot, VALUATION="horizon", **common)
    placed_instant = instant.get_entity(names[0]).balance
    placed_horizon = horizon.get_entity(names[0]).balance
    assert placed_instant > 0.0 and placed_horizon > 0.0
    assert placed_horizon < 0.8 * placed_instant
    assert horizon.resolve_log[0]["idle_usd"] > instant.resolve_log[0]["idle_usd"]


def test_horizon_dominance_filter_uses_the_horizon_rate_at_zero():
    state = market(10e6, 0.90, 0.08)
    strategy, names = build_flat(1e6, [state], VALUATION="horizon", IRM_FEEDBACK=True, HOLD_HORIZON_BARS=48)
    entity = strategy.get_entity(names[0])
    snap = strategy._snapshots()[names[0]]                         # pylint: disable=protected-access
    expected = float(entity.horizon_supply_apy(0.0, 48, snap))
    assert strategy._undiluted_apy(names[0], snap) == expected      # pylint: disable=protected-access
    # a target just above the first-dollar horizon rate keeps the pool out entirely
    blocked, blocked_names = build_flat(1e6, [state], VALUATION="horizon", IRM_FEEDBACK=True,
                                        HOLD_HORIZON_BARS=48, TARGET_APY=expected * (1 + 1e-9))
    assert blocked.get_entity(blocked_names[0]).balance == 0.0
