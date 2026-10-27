"""Synthetic tests for :mod:`fractal.core.entities.models.route_costs`."""
import math

import numpy as np
import pytest

from fractal.core.entities.models.route_costs import (
    CASH_COMPONENTS,
    ZERO_LEG,
    RouteCosts,
    RouteLeg,
    amortise,
    break_even,
    fast_break_even,
    latency_cost,
    venue_crossings,
)

pytestmark = pytest.mark.core

HOME = ("USDC", 1)
L2_USDC = ("USDC", 8453)
L1_USDT = ("USDT", 1)
L2_USDT = ("USDT", 8453)
LEG = RouteLeg(bridge_fixed_usd=2.0, bridge_bps=1.0, conversion_bps=3.0, gas_usd=0.5, latency_bars=4)


# ------------------------------------------------------------------ RouteLeg
def test_leg_costs_charge_bridge_only_across_chains_and_conversion_only_across_assets():
    amount = 100_000.0
    both = LEG.costs(amount)
    assert both == {"bridge": 2.0 + 10.0, "conversion": 30.0, "gas": 0.5}
    same_asset = LEG.costs(amount, cross_asset=False)
    assert same_asset == {"bridge": 12.0, "conversion": 0.0, "gas": 0.5}
    same_chain = LEG.costs(amount, cross_chain=False)
    assert same_chain == {"bridge": 0.0, "conversion": 30.0, "gas": 0.5}
    assert set(both) == set(CASH_COMPONENTS)


def test_leg_cost_is_affine_and_matches_fixed_and_bps():
    for cross_chain in (True, False):
        for cross_asset in (True, False):
            for components in (("bridge", "gas"), ("conversion",), ("bridge", "conversion", "gas", "latency"), ()):
                fixed, bps = LEG.fixed_and_bps(cross_chain=cross_chain, cross_asset=cross_asset,
                                               components=components)
                for amount in (1.0, 5_000.0, 3e6):
                    total = sum(LEG.costs(amount, cross_chain=cross_chain, cross_asset=cross_asset,
                                          components=components).values())
                    assert total == pytest.approx(fixed + bps * amount / 1e4, rel=1e-12)


def test_components_switch_off_individual_charges_and_latency():
    assert LEG.costs(1e6, components=("gas",)) == {"bridge": 0.0, "conversion": 0.0, "gas": 0.5}
    assert LEG.latency() == 4
    assert LEG.latency(components=("bridge",)) == 0
    with pytest.raises(ValueError, match="unknown cost components"):
        LEG.costs(1.0, components=("slippage",))


def test_empty_leg_costs_nothing_and_gross_for_net_inverts_the_cost():
    assert sum(LEG.costs(0.0).values()) == 0.0
    assert sum(LEG.costs(-5.0).values()) == 0.0
    assert LEG.gross_for_net(0.0) == 0.0
    for net in (10.0, 1234.5, 2e7):
        gross = LEG.gross_for_net(net)
        assert gross - sum(LEG.costs(gross).values()) == pytest.approx(net, rel=1e-12)


@pytest.mark.parametrize("field, value", [
    ("bridge_fixed_usd", -1.0), ("bridge_bps", math.nan), ("gas_usd", math.inf), ("latency_bars", 1.5),
    ("latency_bars", True), ("conversion_bps", "3"),
])
def test_leg_rejects_invalid_values(field, value):
    with pytest.raises(ValueError):
        RouteLeg(**{field: value})


def test_leg_rejects_fees_that_eat_the_whole_amount():
    with pytest.raises(ValueError, match="10000"):
        RouteLeg(bridge_bps=6e3, conversion_bps=4e3)


# ---------------------------------------------------------------- RouteCosts
def test_routes_are_home_routed_and_missing_venues_fail_loudly():
    entry_leg, exit_leg = RouteLeg(gas_usd=1.0), RouteLeg(gas_usd=2.0)
    routes = RouteCosts(entry={L2_USDC: entry_leg}, exit={L2_USDC: exit_leg})
    assert routes.leg(L2_USDC, "entry", HOME) is entry_leg
    assert routes.leg(L2_USDC, "exit", HOME) is exit_leg
    assert routes.leg(HOME, "entry", HOME) is ZERO_LEG
    with pytest.raises(KeyError, match="no exit route"):
        RouteCosts(entry={L2_USDC: entry_leg}).leg(L2_USDC, "exit", HOME)
    with pytest.raises(ValueError, match="direction"):
        routes.leg(L2_USDC, "sideways", HOME)


def test_venue_crossings():
    assert venue_crossings(L2_USDC, HOME) == (True, False)
    assert venue_crossings(L1_USDT, HOME) == (False, True)
    assert venue_crossings(L2_USDT, HOME) == (True, True)
    assert venue_crossings(HOME, HOME) == (False, False)


def test_decision_cost_prices_entries_exits_and_latency_at_the_opportunity_rate():
    entry = RouteLeg(bridge_fixed_usd=5.0, bridge_bps=2.0, gas_usd=1.0, latency_bars=24)
    exit_ = RouteLeg(bridge_fixed_usd=7.0, bridge_bps=0.0, gas_usd=1.0, latency_bars=0)
    routes = RouteCosts(entry={L2_USDC: entry}, exit={L2_USDC: exit_})
    delta = np.array([1e6, -1e6, 0.0])
    cost = routes.decision_cost(L2_USDC, HOME, delta, opportunity_apy=0.05, bars_per_year=8760.0)
    latency = 0.05 * 24 / 8760 * 1e6
    assert cost[0] == pytest.approx(6.0 + 200.0 + latency, rel=1e-12)
    assert cost[1] == pytest.approx(8.0, rel=1e-12)
    assert cost[2] == 0.0
    no_latency = routes.decision_cost(L2_USDC, HOME, delta, opportunity_apy=0.05,
                                      components=("bridge", "conversion", "gas"))
    assert no_latency[0] == pytest.approx(206.0, rel=1e-12)
    assert routes.decision_cost(HOME, HOME, delta).tolist() == [0.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="bars_per_year"):
        routes.decision_cost(L2_USDC, HOME, delta, bars_per_year=0.0)


def test_scaled_multiplies_one_component_and_rounds_latency_up():
    routes = RouteCosts(entry={L2_USDT: LEG}, exit={L2_USDT: LEG})
    triple_bridge = routes.scaled("bridge", 3.0)
    leg = triple_bridge.entry[L2_USDT]
    assert (leg.bridge_fixed_usd, leg.bridge_bps) == (6.0, 3.0)
    assert (leg.conversion_bps, leg.gas_usd, leg.latency_bars) == (3.0, 0.5, 4)
    assert routes.scaled("latency", 1.3).exit[L2_USDT].latency_bars == 6          # 5.2 bars -> 6
    assert routes.scaled("latency", 0.5).exit[L2_USDT].latency_bars == 2
    with pytest.raises(ValueError):
        routes.scaled("slippage", 2.0)
    with pytest.raises(ValueError):
        routes.scaled("gas", -1.0)


# ------------------------------------------------------------------ frictions
def test_amortise_annualises_a_round_trip_over_the_horizon():
    assert amortise(100.0, 1e6, horizon_days=36.5) == pytest.approx(1e-3, rel=1e-12)
    out = amortise(np.array([100.0, 200.0]), 1e6, horizon_days=365.0)
    assert out.tolist() == pytest.approx([1e-4, 2e-4])
    with pytest.raises(ValueError):
        amortise(1.0, 0.0, horizon_days=30.0)
    with pytest.raises(ValueError):
        amortise(1.0, 1.0, horizon_days=0.0)


def test_break_even_is_the_days_a_spread_must_persist():
    # 10 bps round trip at a 1 % spread: 0.001 / 0.01 * 365 = 36.5 days
    assert break_even(1_000.0, 1e6, 0.01) == pytest.approx(36.5, rel=1e-12)
    assert break_even(1_000.0, 1e6, 0.0) == math.inf
    out = break_even(1_000.0, 1e6, np.array([0.02, -0.01]))
    assert out[0] == pytest.approx(18.25) and out[1] == math.inf
    # consistency: amortised over T* days, the cost equals the spread
    days = break_even(500.0, 2e6, 0.03)
    assert amortise(500.0, 2e6, horizon_days=days) == pytest.approx(0.03, rel=1e-12)


def test_latency_cost_scales_with_turnover_rate_and_hours():
    # 12 round trips a year (24 one-way), 5 % rate, 24 h in transit per move
    assert latency_cost(24.0, 0.05, 24.0) == pytest.approx(24 * 0.05 * 24 / 8760, rel=1e-12)
    assert latency_cost(0.0, 0.05, 24.0) == 0.0


def test_fast_break_even():
    # fast costs 10 USD more on 1M and saves 8760/365 = 24 h: s* = 1e-5 * 8760 / 24 = 0.00365
    assert fast_break_even(1e6, 12.0, 2.0, 0.0, 24.0) == pytest.approx(0.00365, rel=1e-12)
    assert fast_break_even(1e6, 2.0, 2.0, 0.0, 24.0) == 0.0          # not dearer
    assert fast_break_even(1e6, 12.0, 2.0, 24.0, 24.0) == math.inf   # dearer, saves nothing
    with pytest.raises(ValueError):
        fast_break_even(0.0, 1.0, 0.0, 0.0, 1.0)
