"""Real-data replays (slow) of the three sUSDe PT carry variants on the
trajectories shipped in ``examples/susde_pt_carry/results``."""
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
    SimpleSpotExchangeGlobalState,
)
from fractal.strategies import MorphoRateHedgedLeveragedPT, MorphoRateHedgedLeveragedPTParams  # noqa: E402

_RESULTS = Path(__file__).resolve().parents[2] / "examples" / "susde_pt_carry" / "results"


def _load(name: str) -> pd.DataFrame:
    path = _RESULTS / name
    if not path.exists():
        pytest.skip(f"fixture missing: {path}")
    return pd.read_csv(path)


def _observations(df: pd.DataFrame, variant: str):
    observations = []
    for _, row in df.iterrows():
        states = {
            "PT": PendlePTGlobalState(
                seconds_to_expiry=float(row["PT_seconds_to_expiry"]), implied_apy=float(row["PT_implied_apy"]),
                asset_price=float(row["PT_asset_price"]), sy_exchange_rate=float(row["PT_sy_exchange_rate"]),
                total_pt=float(row["PT_total_pt"]), total_sy=float(row["PT_total_sy"]),
                scalar_root=float(row["PT_scalar_root"]), ln_fee_rate_root=float(row["PT_ln_fee_rate_root"]),
            ),
            "LENDING": MorphoGlobalState(
                collateral_price=float(row["LENDING_collateral_price"]), debt_price=float(row["LENDING_debt_price"]),
                lending_rate=float(row["LENDING_lending_rate"]), borrowing_rate=float(row["LENDING_borrowing_rate"]),
                collateral_market_price=float(row["LENDING_collateral_market_price"]),
            ),
        }
        listed = variant == "boros" and not pd.isna(row.get("BOROS_mark_rate", float("nan")))
        if listed and float(row["BOROS_seconds_to_expiry"]) > 0:  # the YU market lists after entry
            states["BOROS"] = BorosGlobalState(
                seconds_to_expiry=float(row["BOROS_seconds_to_expiry"]), mark_rate=float(row["BOROS_mark_rate"]),
                funding_rate=float(row["BOROS_funding_rate"]),
                funding_period_seconds=float(row["BOROS_funding_period_seconds"]),
                underlying_price=float(row["BOROS_underlying_price"]),
            )
        if variant == "perp":
            price = float(row["PERP_mark_price"])
            states["SPOT"] = SimpleSpotExchangeGlobalState(open=price, high=price, low=price, close=price, volume=0.0)
            states["PERP"] = HyperliquidGlobalState(mark_price=price, funding_rate=float(row["PERP_funding_rate"]))
        observations.append(Observation(timestamp=pd.Timestamp(row["timestamp"]).to_pydatetime(), states=states))
    return observations


@pytest.mark.slow
@pytest.mark.parametrize("market,bar_hours,impact", [("susde_25sep2025", 24, "rate_spread"),
                                                     ("susde_26nov2026", 1, "amm")])
@pytest.mark.parametrize("variant", ["none", "boros", "perp"])
def test_susde_carry_replays_the_reference_run(market, bar_hours, impact, variant):
    df = _load(f"{market}_{variant}.csv")
    strategy = MorphoRateHedgedLeveragedPT(params=MorphoRateHedgedLeveragedPTParams(
        INITIAL_BALANCE=100_000.0, TARGET_LTV=0.80, MAX_LOOPS=8, REBALANCE_LTV_BAND=(0.70, 0.88),
        MIN_HEALTH_FACTOR=1.03, MIN_CARRY_SPREAD=-0.05, MAX_BORROW_APY=0.40, MIN_DAYS_TO_MATURITY_AT_ENTRY=1,
        PT_FEE_LN_RATE=0.001, PT_IMPACT_LN_RATE_PER_SHARE=0.075, HEDGE_MARGIN_SHARE=0.20, HEDGE_RATIO=1.0,
        HEDGE_REBALANCE_THRESHOLD=0.05, PERP_TARGET_LEVERAGE=2.0, RATE_HEDGE=variant,
        CARRY_GATE_LOOKBACK_BARS=7 * 24 // bar_hours, BAR_HOURS=bar_hours, LLTV=0.915, PT_IMPACT_MODEL=impact,
    ))
    result = strategy.run(_observations(df, variant))
    out = result.to_dataframe()
    assert len(out) == len(df)
    assert out["net_balance"].iloc[-1] == pytest.approx(df["net_balance"].iloc[-1], rel=1e-9)
    assert out["LENDING_liquidation_count"].max() == 0
    for value in result.get_default_metrics().__dict__.values():
        assert math.isfinite(value)
    if market == "susde_25sep2025":  # held to redemption: everything unwound
        last = out.iloc[-1]
        assert last["LENDING_borrowed"] == 0.0 and last["PT_amount"] == 0.0
        if variant == "boros":
            assert last["BOROS_size"] == 0.0
        if variant == "perp":
            assert pd.isna(last["PERP_positions_0_amount"]) or last["PERP_positions_0_amount"] == 0.0
            assert last["SPOT_amount"] == 0.0
    if variant != "none":
        assert strategy.hedge_balance() > 0
