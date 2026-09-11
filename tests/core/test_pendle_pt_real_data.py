"""Real-data replays (slow) of the two Pendle PT strategies on the reference
trajectories shipped in ``examples/pendle_pt_backtests/results``: the
observations are rebuilt from the recorded global states, so the runs
are reproducible offline."""
import math
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")

from fractal.core.base import Observation  # noqa: E402
from fractal.core.entities import (  # noqa: E402
    BorosGlobalState,
    HyperliquidGlobalState,
    MorphoGlobalState,
    PendlePTGlobalState,
)
from fractal.strategies import (  # noqa: E402
    MorphoLeveragedPT,
    MorphoLeveragedPTParams,
    PerpHedgedPT,
    PerpHedgedPTParams,
)

_RESULTS = Path(__file__).resolve().parents[2] / "examples" / "pendle_pt_backtests" / "results"


def _load(name: str) -> pd.DataFrame:
    path = _RESULTS / name
    if not path.exists():
        pytest.skip(f"fixture missing: {path}")
    return pd.read_csv(path)


def _pt_state(row) -> PendlePTGlobalState:
    return PendlePTGlobalState(
        seconds_to_expiry=float(row["PT_seconds_to_expiry"]), implied_apy=float(row["PT_implied_apy"]),
        asset_price=float(row["PT_asset_price"]), sy_exchange_rate=float(row["PT_sy_exchange_rate"]),
        total_pt=float(row["PT_total_pt"]), total_sy=float(row["PT_total_sy"]),
        scalar_root=float(row["PT_scalar_root"]), ln_fee_rate_root=float(row["PT_ln_fee_rate_root"]),
    )


def _leveraged_observations(df: pd.DataFrame):
    observations = []
    for _, row in df.iterrows():
        market = row["LENDING_collateral_market_price"]
        observations.append(Observation(timestamp=pd.Timestamp(row["timestamp"]).to_pydatetime(), states={
            "PT": _pt_state(row),
            "LENDING": MorphoGlobalState(
                collateral_price=float(row["LENDING_collateral_price"]), debt_price=float(row["LENDING_debt_price"]),
                lending_rate=float(row["LENDING_lending_rate"]), borrowing_rate=float(row["LENDING_borrowing_rate"]),
                collateral_market_price=None if pd.isna(market) else float(market),
            ),
        }))
    return observations


def _hedged_observations(df: pd.DataFrame, use_boros: bool):
    observations = []
    for _, row in df.iterrows():
        states = {
            "PT": _pt_state(row),
            "HEDGE": HyperliquidGlobalState(mark_price=float(row["HEDGE_mark_price"]),
                                            funding_rate=float(row["HEDGE_funding_rate"])),
        }
        if use_boros and not pd.isna(row.get("BOROS_mark_rate", float("nan"))):
            states["BOROS"] = BorosGlobalState(
                seconds_to_expiry=float(row["BOROS_seconds_to_expiry"]), mark_rate=float(row["BOROS_mark_rate"]),
                funding_rate=float(row["BOROS_funding_rate"]),
                funding_period_seconds=float(row["BOROS_funding_period_seconds"]),
                underlying_price=float(row["BOROS_underlying_price"]),
            )
        observations.append(Observation(timestamp=pd.Timestamp(row["timestamp"]).to_pydatetime(), states=states))
    return observations


@pytest.mark.slow
@pytest.mark.parametrize("market,bar_hours,impact", [
    ("usde_25sep2025", 24, "rate_spread"),
    ("susde_25sep2025_dai", 24, "rate_spread"),
    ("susde_26nov2026", 1, "amm"),
    ("reusd_10dec2026", 1, "rate_spread"),
])
def test_leveraged_pt_replays_the_reference_run(market, bar_hours, impact):
    df = _load(f"leveraged_{market}.csv")
    strategy = MorphoLeveragedPT(params=MorphoLeveragedPTParams(
        INITIAL_BALANCE=100_000.0, TARGET_LTV=0.80, MAX_LOOPS=8, REBALANCE_LTV_BAND=(0.70, 0.88),
        MIN_HEALTH_FACTOR=1.03, MIN_CARRY_SPREAD=-0.05, MAX_BORROW_APY=0.40,
        CARRY_GATE_LOOKBACK_BARS=7 * 24 // bar_hours,
        MIN_DAYS_TO_MATURITY_AT_ENTRY=1, BAR_HOURS=bar_hours, LLTV=0.915, PT_IMPACT_MODEL=impact,
        PT_FEE_LN_RATE=0.001, PT_IMPACT_LN_RATE_PER_SHARE=0.075,
    ))
    result = strategy.run(_leveraged_observations(df))
    out = result.to_dataframe()
    assert len(out) == len(df)
    assert out["net_balance"].iloc[-1] == pytest.approx(df["net_balance"].iloc[-1], rel=1e-9)
    assert (out["net_balance"] > 0.5 * 100_000).all()
    assert out["LENDING_liquidation_count"].max() == 0
    for value in result.get_default_metrics().__dict__.values():
        assert math.isfinite(value)
    if market in ("usde_25sep2025", "susde_25sep2025_dai"):  # held to redemption
        last = out.iloc[-1]
        assert last["LENDING_borrowed"] == 0.0 and last["LENDING_collateral"] == 0.0 and last["PT_amount"] == 0.0
        assert last["net_balance"] > 100_000
    if market == "reusd_10dec2026":  # the 2026-08-25 spike: marked down, never liquidated at 0.80 LTV
        event = out[out["PT_implied_apy"] > 0.14]
        assert len(event) >= 1 and out["net_balance"].min() < 98_000


@pytest.mark.slow
def test_unlevered_hold_to_redemption_returns_the_entry_discount():
    """``MAX_LOOPS=0`` on a full-life market must return exactly the entry
    implied APY (no debt, no band boost); the swap fee is the only cost."""
    df = _load("leveraged_usde_25sep2025.csv")
    strategy = MorphoLeveragedPT(params=MorphoLeveragedPTParams(
        INITIAL_BALANCE=100_000.0, TARGET_LTV=0.80, MAX_LOOPS=0, REBALANCE_LTV_BAND=(0.70, 0.88),
        MIN_DAYS_TO_MATURITY_AT_ENTRY=1, BAR_HOURS=24, LLTV=0.915, PT_IMPACT_MODEL="rate_spread",
        PT_FEE_LN_RATE=0.0, PT_IMPACT_LN_RATE_PER_SHARE=0.0,
    ))
    out = strategy.run(_leveraged_observations(df)).to_dataframe()
    assert (out["LENDING_borrowed"] == 0.0).all()
    first = df.iloc[0]
    years = first["PT_seconds_to_expiry"] / (365 * 86_400)
    entry_price = (1.0 + first["PT_implied_apy"]) ** (-years)
    assert out["net_balance"].iloc[-1] == pytest.approx(100_000.0 / entry_price, rel=1e-9)


@pytest.mark.slow
@pytest.mark.parametrize("tag,use_boros", [("perp", False), ("boros", True)])
def test_hedged_pt_replays_the_reference_run(tag, use_boros):
    df = _load(f"hedged_wsteth_30dec2027_{tag}.csv")
    strategy = PerpHedgedPT(params=PerpHedgedPTParams(
        INITIAL_BALANCE=100_000.0, TARGET_HEDGE_LEVERAGE=2.0, HEDGE_LEVERAGE_BAND=(1.2, 3.5),
        HEDGE_REBALANCE_THRESHOLD=0.02, USE_BOROS=use_boros, BOROS_MARGIN_SHARE=0.10,
        MIN_DAYS_TO_MATURITY_AT_ENTRY=1, PT_IMPACT_MODEL="rate_spread", PT_FEE_LN_RATE=0.001,
        PT_IMPACT_LN_RATE_PER_SHARE=0.075, PERP_TRADING_FEE=0.00035, PERP_MAX_LEVERAGE=10.0,
    ))
    result = strategy.run(_hedged_observations(df, use_boros))
    out = result.to_dataframe()
    assert out["net_balance"].iloc[-1] == pytest.approx(df["net_balance"].iloc[-1], rel=1e-9)
    spot_move = df["HEDGE_mark_price"].iloc[-1] / df["HEDGE_mark_price"].iloc[0] - 1
    equity_move = out["net_balance"].iloc[-1] / out["net_balance"].iloc[0] - 1
    assert abs(equity_move) < 0.25 * abs(spot_move)  # the hedge removes the price leg
    assert (out["net_balance"] > 0.9 * 100_000).all()
    for value in result.get_default_metrics().__dict__.values():
        assert math.isfinite(value)
