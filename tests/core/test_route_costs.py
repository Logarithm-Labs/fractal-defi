"""Synthetic tests for :mod:`fractal.core.entities.models.route_costs`."""
import math

import numpy as np
import pandas as pd
import pytest

from fractal.core.entities.models.route_costs import (
    CASH_COMPONENTS,
    ZERO_LEG,
    FeeSchedule,
    RouteCosts,
    RouteLeg,
    amortise,
    amortised_cost_days,
    basis_pnl,
    break_even,
    conversion_cost,
    fast_break_even,
    fit_fee_schedule,
    friction_table,
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


def test_amortised_cost_days_counts_days_the_spread_does_not_repay():
    days = pd.date_range("2026-01-01", periods=4, freq="D")
    cost = pd.Series([0.001, 0.001, 0.002, np.nan], index=days)         # fraction of size per round trip
    spread = pd.Series([0.02, 0.005, 0.03, 0.01], index=days)
    # amortised over 36.5 days: 0.01, 0.01, 0.02 per year; the NaN day drops out
    assert amortised_cost_days(cost, spread, horizon_days=36.5) == pytest.approx(1 / 3)
    other = pd.Series([0.5], index=pd.date_range("2027-01-01", periods=1, freq="D"))
    assert math.isnan(amortised_cost_days(cost, other, horizon_days=30.0))
    with pytest.raises(ValueError):
        amortised_cost_days(cost, spread, horizon_days=0.0)


def test_conversion_cost_picks_the_cheaper_route():
    # DEX: 1M * (1 + 2) bps = 300; issuer: 1M * 1 bp + 1M * 5 % * 48 h / 8760 = 100 + 273.97
    out = conversion_cost(1e6, dex_fee_bps=1.0, basis_bps=2.0, issuer_fee_bps=1.0, issuer_latency_hours=48.0,
                          rate=0.05)
    assert out["route"] == "dex" and out["cost_usd"] == pytest.approx(300.0)
    assert out["issuer_cost_usd"] == pytest.approx(100.0 + 1e6 * 0.05 * 48 / 8760)
    # an adverse basis flips it to the issuer; a favourable one makes the DEX cheaper still
    assert conversion_cost(1e6, 1.0, 10.0, 1.0, 48.0, 0.05)["route"] == "issuer"
    assert conversion_cost(1e6, 1.0, -1.0, 1.0, 48.0, 0.05)["cost_usd"] == pytest.approx(0.0)
    # unavailable routes
    assert conversion_cost(1e6, None, 0.0, 1.0, 0.0, 0.05)["route"] == "issuer"
    assert conversion_cost(1e6, 1.0, 0.0, None, 0.0, 0.05)["route"] == "dex"
    none = conversion_cost(1e6, None, 0.0, np.nan, 0.0, 0.05)
    assert none["route"] == "none" and math.isnan(none["cost_usd"])
    vector = conversion_cost(np.array([1e5, 1e7]), 1.0, 0.0, 0.5, np.array([0.0, 24.0]), 0.05)
    assert vector["route"].tolist() == ["issuer", "dex"]
    assert vector["cost_usd"] == pytest.approx([5.0, 1000.0])


def test_fit_fee_schedule_recovers_fixed_plus_bps_and_clips_negative_parts():
    amount = np.array([1e4, 5e4, 2e5, 1e6, 3e6])
    assert fit_fee_schedule(amount, 2.0 + 1.5 * amount / 1e4) == pytest.approx((2.0, 1.5), rel=1e-9)
    assert fit_fee_schedule([1e5, 1e5, 1e5], [1.0, 3.0, 2.0]) == (2.0, 0.0)          # one size: median as fixed
    assert fit_fee_schedule(amount, 10.0 - amount / 1e6) == (pytest.approx(9.8), 0.0)  # falling fee: flat median
    fixed, bps = fit_fee_schedule(amount, -5.0 + 2.0 * amount / 1e4)                   # negative intercept
    assert fixed == 0.0 and bps == pytest.approx(np.median((-5.0 + 2.0 * amount / 1e4) / amount) * 1e4)
    assert all(math.isnan(x) for x in fit_fee_schedule([], []))
    assert fit_fee_schedule([1e4, np.nan, 2e4], [1.0, 5.0, 2.0]) == pytest.approx((0.0, 1.0), abs=1e-9)
    assert fit_fee_schedule([0.0, -1e4, 3e4], [9.0, 9.0, 4.0]) == (4.0, 0.0)          # non-positive amounts dropped
    with pytest.raises(ValueError):
        fit_fee_schedule([1.0, 2.0], [1.0])


def test_fee_schedule_evaluates_the_affine_fee():
    schedule = fit_fee_schedule((a for a in (1e4, 1e6)), iter([3.0, 102.0]))  # any iterables
    assert isinstance(schedule, FeeSchedule)
    assert schedule.fee(5e5) == pytest.approx(2.0 + 1.0 * 50.0)
    assert schedule.fee_bps_at(1e6) == pytest.approx(102.0 / 1e6 * 1e4)
    fixed, bps = schedule                                                    # still unpacks as a pair
    assert (fixed, bps) == pytest.approx((2.0, 1.0))


def _transfers() -> pd.DataFrame:
    amounts = [1e4, 1e5, 1e6, 2e6, 5e5]
    rows = [{"protocol": "bridge_a", "asset": "USDC", "src_chain": 1, "dst_chain": 8453, "mode": "std",
             "amount_usd": a, "fee_usd": 0.5 + 0.2 * a / 1e4, "latency_s": 600.0 * (i + 1),
             "gas_usd": 2.0 if a < 1e5 else 4.0, "matched": True} for i, a in enumerate(amounts)]
    rows.append({**rows[0], "matched": False, "latency_s": np.nan})
    rows.append({"protocol": "bridge_b", "asset": "USDT", "src_chain": 1, "dst_chain": 42161, "mode": "oft",
                 "amount_usd": 1e5, "fee_usd": 7.0, "latency_s": 3600.0, "gas_usd": np.nan, "matched": True})
    return pd.DataFrame(rows)


def test_friction_table_summarises_each_route():
    table = friction_table(_transfers(), sizes_usd=[1e6, 5e4], thin_below=3, gas_buckets_usd=[0.0, 1e5, np.inf])
    assert table["protocol"].tolist() == ["bridge_a", "bridge_b"]
    a, b = table.iloc[0], table.iloc[1]
    assert (a["n"], a["n_unmatched"], a["thin"], b["n"], b["thin"]) == (5, 1, False, 1, True)
    assert a["latency_p50_h"] == pytest.approx(0.5)                     # median of 10..50 minutes
    assert a["latency_p10_h"] == pytest.approx(np.quantile([600, 1200, 1800, 2400, 3000], 0.1) / 3600)
    assert (a["fee_fixed_usd"], a["fee_bps"]) == (pytest.approx(0.5), pytest.approx(0.2))
    assert a["fee_bps_at_1m"] == pytest.approx((0.5 + 20.0) / 1e6 * 1e4)
    assert a["fee_bps_at_50k"] == pytest.approx((0.5 + 1.0) / 5e4 * 1e4)
    assert (b["fee_fixed_usd"], b["fee_bps"]) == (7.0, 0.0)              # one size: flat fee
    assert a["gas_bps_0_100k"] == pytest.approx(2.0 / 1e4 * 1e4)
    assert a["gas_bps_100k_inf"] == pytest.approx(np.median([4.0 / a_ * 1e4 for a_ in (1e5, 1e6, 2e6, 5e5)]))
    assert math.isnan(b["gas_bps_0_100k"])


def test_friction_table_optional_columns_and_validation():
    plain = friction_table(_transfers().drop(columns=["matched", "gas_usd"]), sizes_usd=[1e6],
                           group_by=["protocol"], quantiles=[0.5])
    assert list(plain.columns) == ["protocol", "n", "n_unmatched", "latency_p50_h", "fee_fixed_usd", "fee_bps",
                                   "fee_bps_at_1m"]
    assert plain["n"].tolist() == [6, 1]                                 # no matched column: all count
    with pytest.raises(ValueError, match="missing columns"):
        friction_table(_transfers().drop(columns=["fee_usd"]), sizes_usd=[1e6])
    with pytest.raises(ValueError):
        friction_table(_transfers(), sizes_usd=[0.0])
    with pytest.raises(ValueError):
        friction_table(_transfers(), sizes_usd=[1e6], gas_buckets_usd=[1e5, 0.0])
    with pytest.raises(ValueError):
        friction_table(_transfers().drop(columns=["gas_usd"]), sizes_usd=[1e6], gas_buckets_usd=[0.0, 1.0])


def _basis_reference(exposure: pd.DataFrame, prices: pd.DataFrame) -> np.ndarray:
    """Row-by-row loop: last bar's exposure times this bar's relative price change."""
    out = np.zeros(len(exposure))
    marked = prices.sort_index().reindex(exposure.index, method="ffill")
    for i in range(1, len(exposure)):
        for asset, amount in exposure.iloc[i - 1].items():
            if asset in marked.columns and np.isfinite(amount) and amount:
                p0, p1 = marked[asset].iloc[i - 1], marked[asset].iloc[i]
                if np.isfinite(p0) and np.isfinite(p1) and p0 > 0:
                    out[i] += amount * (p1 / p0 - 1.0)
    return out


def test_basis_pnl_marks_last_bars_exposure_to_the_price_change():
    hours = pd.date_range("2026-01-01", periods=4, freq="h")
    exposure = pd.DataFrame({"USDT": [1e6, 1e6, 0.0, 5e5], "DAI": [0.0, 2e5, 2e5, 2e5]}, index=hours)
    prices = pd.DataFrame({"USDT": [1.0, 0.999, 1.001, 1.0], "DAI": [1.0, 1.0, 1.002, 1.002]}, index=hours)
    pnl = basis_pnl(exposure, prices)
    assert pnl.iloc[0] == 0.0
    assert pnl.iloc[1] == pytest.approx(1e6 * (0.999 - 1.0))
    assert pnl.iloc[2] == pytest.approx(1e6 * (1.001 / 0.999 - 1.0) + 2e5 * 0.002)
    assert pnl.iloc[3] == pytest.approx(0.0)                             # USDT flat at 0, DAI unchanged


def test_basis_pnl_forward_fills_sparse_prices_and_skips_bad_ones():
    hours = pd.date_range("2026-01-01", periods=6, freq="h")
    exposure = pd.DataFrame({"USDT": [1e6] * 6, "FOO": [5e5] * 6}, index=hours)
    prices = pd.DataFrame({"USDT": [1.0, 0.0, 0.998, np.nan]},
                          index=[hours[4], hours[0], hours[2], hours[5]])     # unsorted, sparse
    pnl = basis_pnl(exposure, prices)
    # prices on the bars: 0 (h0), 0, 0.998, 0.998, 1.0, nan -> only h4 has a valid previous price
    assert pnl.tolist() == pytest.approx([0.0, 0.0, 0.0, 0.0, 1e6 * (1.0 / 0.998 - 1.0), 0.0])
    assert basis_pnl(exposure[["FOO"]], prices).tolist() == [0.0] * 6     # no price column: no basis


def test_basis_pnl_matches_a_row_by_row_loop():
    rng = np.random.default_rng(7)
    hours = pd.date_range("2026-01-01", periods=200, freq="h")
    exposure = pd.DataFrame(rng.uniform(-1e6, 1e6, size=(200, 3)), index=hours, columns=["A", "B", "C"])
    exposure.iloc[rng.integers(0, 200, 20), 1] = 0.0
    price_hours = hours[np.sort(rng.choice(200, 120, replace=False))]
    prices = pd.DataFrame(1.0 + rng.normal(0.0, 1e-3, size=(120, 2)), index=price_hours, columns=["A", "B"])
    prices.iloc[rng.integers(0, 120, 5), 0] = np.nan
    assert basis_pnl(exposure, prices).to_numpy() == pytest.approx(_basis_reference(exposure, prices), abs=1e-9)
