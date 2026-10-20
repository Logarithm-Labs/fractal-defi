"""Offline, synthetic coverage of the Morpho supply example: observation adapter and script wiring."""
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
    path = Path(__file__).resolve().parents[2] / "examples/morpho_supply/helpers.py"
    if not path.is_file():
        pytest.skip("Morpho supply example is not included in the installed wheel")
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


def _load_script(monkeypatch, helpers, name):
    """Load an example script that imports ``helpers`` by its plain module name."""
    monkeypatch.setitem(sys.modules, "helpers", helpers)
    path = Path(helpers.__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"morpho_supply_example_{name}", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def _listing(market_id="a", collateral="TEST", liquidity=2_000_000.0, chain_id=1, loan="USDC",
             oracle="ChainlinkOracleV2"):
    return MorphoMarketListing(
        chain_id=chain_id, market_id="0x" + market_id * 64,
        loan_address="0x" + "b" * 40, loan_symbol=loan, loan_decimals=6,
        collateral_address="0x" + "c" * 40, collateral_symbol=collateral, collateral_decimals=18,
        lltv=0.86, oracle_type=oracle, listed=True, creation_timestamp=0, liquidity_assets_usd=liquidity,
    )


def _panel(helpers, listing, supply_apy, bars=72):
    index = pd.date_range("2026-01-01", periods=bars, freq="h", tz="UTC")
    history = pd.DataFrame({
        "supply_apy": supply_apy, "borrow_apy": supply_apy / 0.8, "supply_rate": 0.0,
        "utilization": 0.8, "rate_at_target": 0.04,
        "supply_assets_usd": 10_000_000.0, "borrow_assets_usd": 8_000_000.0,
        "liquidity_assets_usd": 2_000_000.0, "net_supply_apy": float("nan"),
    }, index=index)
    return helpers.MarketPanel(listing, history)


def test_backtest_defaults_keep_the_rotation_strategy(monkeypatch, helpers):
    """Existing backtest commands still build the rotation strategy with its documented defaults."""
    backtest = _load_script(monkeypatch, helpers, "backtest")
    strategy = backtest.build_strategy(backtest.parse_args([]), [_listing()])
    assert type(strategy).__name__ == "MorphoSupplyRotationStrategy"
    params = strategy.params
    assert (params["TOP_K"], params["SIGNAL_WINDOW_BARS"]) == (3, 24)


def test_backtest_default_costs_are_per_chain_gas_without_proportional_fees(monkeypatch, helpers):
    """Morpho supply/withdraw has no protocol fee: the default cost is the per-chain gas table, zero bps."""
    backtest = _load_script(monkeypatch, helpers, "backtest")
    for strategy in ("rotation", "allocation"):
        params = backtest.build_strategy(backtest.parse_args(["--strategy", strategy]), [_listing()]).params
        assert params["COST_BPS_PER_LEG"] == 0.0
        assert params["GAS_USD_BY_CHAIN"] == helpers.GAS_USD_BY_CHAIN
        assert params["GAS_USD_PER_LEG"] == helpers.DEFAULT_GAS_USD_PER_LEG


def test_backtest_builds_the_allocator_from_flags(monkeypatch, helpers):
    """--strategy allocation maps every allocator flag and keeps the allocator's own signal window."""
    backtest = _load_script(monkeypatch, helpers, "backtest")
    args = backtest.parse_args(["--strategy", "allocation", "--mode", "marginal", "--target-apy", "0.04",
                                "--gas-usd-per-leg", "0.5", "--hold-horizon-bars", "24", "--rebalance-bars", "2"])
    params = backtest.build_strategy(args, [_listing()]).params
    assert (params["ALLOCATION_MODE"], params["TARGET_APY"], params["GAS_USD_PER_LEG"]) == ("marginal", 0.04, 0.5)
    assert (params["HOLD_HORIZON_BARS"], params["REBALANCE_BARS"], params["SIGNAL_WINDOW_BARS"]) == (24, 2, 1)
    assert params["GAS_USD_BY_CHAIN"] == {}  # one explicit figure replaces the per-chain table


def test_baude_paper_markets_are_distinct_ethereum_usdc_listings(monkeypatch, helpers):
    """The ten mapped paper markets carry unique ids and the protocols differ only where the paper does."""
    baude = _load_script(monkeypatch, helpers, "baude_replication")
    listings = baude.paper_listings()
    assert len({listing.market_id for listing in listings}) == len(listings) == 10
    assert {(listing.chain_id, listing.loan_symbol) for listing in listings} == {(1, "USDC")}
    daily = baude.allocation_params("marginal", daily=True, gas=False, paper_filters=True)
    hourly = baude.allocation_params("marginal", daily=False, gas=True)
    assert (daily.REBALANCE_BARS, daily.SIGNAL_WINDOW_BARS, daily.GAS_USD_PER_LEG, daily.MAX_LLTV) == (24, 24, 0.0, 1.0)
    assert (hourly.REBALANCE_BARS, hourly.GAS_USD_BY_CHAIN[1], hourly.TARGET_APY) == (1, 0.5, 0.0)


def test_baude_ladder_universe_filters_pt_oracles_and_ranks_by_liquidity(monkeypatch, helpers):
    """The ladder's universe keeps Chainlink-priced loan-asset markets, optionally without PT collateral."""
    baude = _load_script(monkeypatch, helpers, "baude_replication")
    listings = [_listing("1", "WBTC", 1e6), _listing("2", "PT-sUSDE", 5e6), _listing("3", "wstETH", 3e6),
                _listing("4", "cbBTC", 9e6, loan="USDT"), _listing("5", "LBTC", 8e6, oracle="Other")]
    keep = baude.ladder_universe(listings, {"USDC"}, top=10, no_pt=True)
    assert [listing.collateral_symbol for listing in keep] == ["wstETH", "WBTC"]
    with_pt = baude.ladder_universe(listings, {"USDC"}, top=2, no_pt=False)
    assert [listing.collateral_symbol for listing in with_pt] == ["PT-sUSDE", "wstETH"]


def test_baude_run_allocation_on_synthetic_panels(monkeypatch, helpers):
    """One offline allocator run under the paper protocol earns between the two synthetic market rates."""
    baude = _load_script(monkeypatch, helpers, "baude_replication")
    panels = [_panel(helpers, _listing("a", "LOW"), 0.03), _panel(helpers, _listing("d", "HIGH"), 0.06)]
    apy, fees = baude.run_allocation(panels, helpers.observations(panels), 100_000.0,
                                     baude.allocation_params("average", daily=True, gas=False, paper_filters=True))
    assert 0.0 < apy < 0.06
    assert fees == pytest.approx(0.0)


def test_baude_ladder_skips_variants_without_markets(monkeypatch, helpers):
    """A registry without Ethereum USDC markets skips those ladder steps instead of aborting the run."""
    baude = _load_script(monkeypatch, helpers, "baude_replication")
    registry = [_listing("a", "WBTC", 5e6, chain_id=8453, loan="USDT"), _listing("d", "wstETH", 4e6, chain_id=8453)]
    monkeypatch.setattr(baude, "load_registry", lambda chain_ids: registry)
    loaded = []

    def fake_load_panels(listings, start, end, fetch_fee=True):
        loaded.append([listing.market_id for listing in listings])
        return [_panel(helpers, listing, 0.05) for listing in listings]

    monkeypatch.setattr(baude, "load_panels", fake_load_panels)
    end = pd.Timestamp("2026-01-04", tz="UTC").to_pydatetime()
    table = baude.run_ladder(days=3, end=end)
    assert set(table["variant"].str[0]) == {"5", "6"}           # Ethereum-only variants skipped
    assert all(loaded)                                           # load_panels never called with an empty universe


def test_load_panels_skips_the_fee_call_by_default(monkeypatch, helpers):
    """The fee does not move the anchored rate, so the default makes no extra API call per market."""
    index = pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC")
    history = _panel(helpers, _listing(), 0.05, bars=3).history.reset_index(drop=True).set_index(index)
    monkeypatch.setattr(helpers, "_load_history", lambda listing, start, end: history)

    def no_fee_call(listing):
        raise AssertionError("fee fetched by default")

    monkeypatch.setattr(helpers, "_current_fee", no_fee_call)
    panels = helpers.load_panels([_listing()], index[0], index[-1])
    assert len(panels) == 1 and "fee" not in panels[0].history.attrs
