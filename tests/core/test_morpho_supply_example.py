"""Offline, synthetic coverage of the Morpho supply example's observation adapter."""
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

from fractal.loaders.morpho_markets import MorphoMarketListing

pytestmark = pytest.mark.core


@pytest.fixture
def helpers(monkeypatch):
    """Load this example without colliding with other examples' helpers modules."""
    path = Path(__file__).resolve().parents[2] / "examples/morpho_supply_rotation/helpers.py"
    spec = importlib.util.spec_from_file_location("morpho_supply_example_helpers", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_observations_align_rewards_to_utc_without_forward_fill(helpers):
    """Sparse incentive data must not leak a later reward rate into earlier bars."""
    listing = MorphoMarketListing(
        chain_id=1, market_id="0x" + "a" * 64,
        loan_address="0x" + "b" * 40, loan_symbol="USDC", loan_decimals=6,
        collateral_address="0x" + "c" * 40, collateral_symbol="TEST", collateral_decimals=18,
        lltv=0.86, oracle_type="ChainlinkOracleV2", listed=True, creation_timestamp=0,
    )
    index = pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC")
    history = pd.DataFrame({
        "supply_apy": 0.08, "borrow_apy": 0.10, "supply_rate": 0.0,
        "utilization": 0.8, "rate_at_target": 0.05,
        "supply_assets_usd": 10_000_000.0, "borrow_assets_usd": 8_000_000.0,
        "liquidity_assets_usd": 2_000_000.0,
        "net_supply_apy": [float("nan"), 0.11, 0.07],
    }, index=index)
    panel = helpers.MarketPanel(listing, history)
    stream = helpers.observations([panel])
    assert [observation.timestamp for observation in stream] == list(index)
    assert [observation.states[panel.name].reward_apy for observation in stream] == pytest.approx([0.0, 0.03, 0.0])
    assert history["net_supply_apy"].isna().iloc[0]


def test_registry_requires_explicit_complete_schema(helpers, tmp_path):
    """An incomplete synthetic registry must fail instead of guessing missing metadata."""
    (tmp_path / "1.csv").write_text("chain_id,market_id\n1,synthetic\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing"):
        helpers.load_registry(registry_dir=str(tmp_path))
