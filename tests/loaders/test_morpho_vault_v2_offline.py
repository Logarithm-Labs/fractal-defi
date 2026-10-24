"""Offline synthetic tests for the Morpho Vault V2 activity loader."""
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from fractal.loaders import LoaderType, MorphoVaultV2History, MorphoVaultV2Loader
from fractal.loaders.morpho import MORPHO_GRAPHQL_URL

pytestmark = pytest.mark.core

CHAIN_ID = 1
START = datetime(2026, 1, 1, tzinfo=timezone.utc)
END = START + timedelta(hours=1)
VAULT = "0x" + "11" * 20
ASSET = "0x" + "22" * 20
ADAPTER = "0x" + "33" * 20
SENDER = "0x" + "44" * 20
MARKET_A = "0x" + "a" * 64
MARKET_B = "0x" + "b" * 64


def _vault() -> dict:
    return {
        "address": VAULT,
        "chain": {"id": CHAIN_ID},
        "asset": {"address": ASSET, "symbol": "SYN", "decimals": 6},
        "totalAssetsUsd": "2000000",
        "liquidityAdapter": {"address": ADAPTER},
        "allocators": [{"allocator": {"address": SENDER}}],
    }


def _allocation(event_type: str = "Allocate", assets: str = "1500000", log_index: int = 1) -> dict:
    return {
        "txHash": "0x" + f"{log_index:064x}",
        "logIndex": log_index,
        "timestamp": int(START.timestamp()) + log_index,
        "type": event_type,
        "adapter": ADAPTER,
        "sender": SENDER,
        "assets": assets,
        "change": assets,
        "ids": ["0x" + "55" * 32],
    }


def _transaction(event_type: str, assets: str, log_index: int) -> dict:
    return {
        "txHash": "0x" + f"{100 + log_index:064x}",
        "logIndex": log_index,
        "timestamp": int(START.timestamp()) + 100 + log_index,
        "type": event_type,
        "assets": assets,
        "vault": {"address": VAULT, "chain": {"id": CHAIN_ID}},
    }


class _FakeVaultV2Http:
    def __init__(
        self,
        *,
        vault_pages: dict[int, list[dict]] | None = None,
        allocation_pages: dict[int, list[dict]] | None = None,
        transaction_pages: dict[int, list[dict]] | None = None,
        positions: tuple[str, ...] = (MARKET_A,),
    ) -> None:
        self.vault_pages = vault_pages or {0: []}
        self.allocation_pages = allocation_pages or {0: []}
        self.transaction_pages = transaction_pages or {0: []}
        self.positions = positions
        self.calls: list[dict] = []

    def post(self, url, json=None, timeout=None, headers=None):
        assert url == MORPHO_GRAPHQL_URL
        self.calls.append(json)
        query = json["query"]
        variables = json["variables"]
        if "VaultV2Vaults" in query:
            return {"data": {"vaultV2s": {"items": self.vault_pages.get(variables["skip"], [])}}}
        if "VaultV2Adapters" in query:
            positions = [{"market": {"marketId": market}} for market in self.positions]
            return {
                "data": {
                    "vaultV2ByAddress": {
                        "adapters": {
                            "items": [
                                {
                                    "address": ADAPTER,
                                    "type": "MorphoMarketV1",
                                    "positions": {"items": positions},
                                }
                            ]
                        }
                    }
                }
            }
        if "VaultV2Allocations" in query:
            page = self.allocation_pages.get(variables["skip"], [])
            return {"data": {"vaultV2AllocationTransactions": {"items": page}}}
        if "VaultV2Transactions" in query:
            page = self.transaction_pages.get(variables["skip"], [])
            return {"data": {"vaultV2transactions": {"items": page}}}
        raise AssertionError("unexpected GraphQL query")


def _loader(http: _FakeVaultV2Http, **kwargs) -> MorphoVaultV2Loader:
    return MorphoVaultV2Loader(CHAIN_ID, START, END, http=http, **kwargs)


def test_empty_api_page_returns_well_shaped_empty_history():
    history = _loader(_FakeVaultV2Http()).read(with_run=True)
    assert isinstance(history, MorphoVaultV2History)
    assert history.vaults == ()
    assert history.allocations.empty and history.transactions.empty
    assert list(history.allocations.columns) == [
        "time", "chain_id", "vault_address", "asset_address", "asset_symbol", "type", "adapter_address",
        "sender_address", "assets_native", "assets_delta_native", "change_native", "market_id", "map_method",
        "ids", "tx_hash", "log_index",
    ]
    assert history.coverage.total_allocations == 0
    assert history.coverage.mapped_fraction is None


def test_single_vault_maps_unique_adapter_position_and_preserves_metadata():
    http = _FakeVaultV2Http(
        vault_pages={0: [_vault()]},
        allocation_pages={0: [_allocation()]},
        transaction_pages={0: [_transaction("Deposit", "2000000", 1)]},
    )
    history = _loader(http).read(with_run=True)
    assert len(history.vaults) == 1
    listing = history.vaults[0]
    assert listing.vault_address == VAULT and listing.asset_symbol == "SYN"
    assert listing.total_assets_usd == pytest.approx(2_000_000.0)
    assert listing.allocators == (SENDER,)
    row = history.allocations.iloc[0]
    assert row["market_id"] == MARKET_A and row["map_method"] == "adapter_position"
    assert row["time"] == pd.Timestamp(START + timedelta(seconds=1))
    assert history.coverage.mapped_allocations == 1
    vault_call = next(call for call in http.calls if "VaultV2Vaults" in call["query"])
    assert vault_call["variables"]["where"] == {
        "chainId_in": [CHAIN_ID],
        "totalAssetsUsd_gte": 1_000_000.0,
    }
    allocation_call = next(call for call in http.calls if "VaultV2Allocations" in call["query"])
    assert allocation_call["variables"]["where"] == {
        "timestamp_gte": int(START.timestamp()),
        "timestamp_lte": int(END.timestamp()),
    }


def test_unmappable_allocation_is_retained_and_counted_in_coverage():
    http = _FakeVaultV2Http(
        vault_pages={0: [_vault()]},
        allocation_pages={0: [_allocation()]},
        positions=(MARKET_A, MARKET_B),
    )
    history = _loader(http).read(with_run=True)
    assert len(history.allocations) == 1
    assert history.allocations.iloc[0]["market_id"] is None
    assert history.allocations.iloc[0]["map_method"] == "none"
    assert history.coverage.total_allocations == 1
    assert history.coverage.unmapped_allocations == 1
    assert history.coverage.mapped_fraction == 0.0
    assert history.coverage.mapped_assets_fraction == 0.0


def test_repeated_full_pages_stop_pagination_without_duplicate_rows():
    event = _allocation()
    http = _FakeVaultV2Http(
        vault_pages={0: [_vault()], 1: [_vault()]},
        allocation_pages={0: [event], 1: [event]},
    )
    history = _loader(http, vault_page_size=1, event_page_size=1).read(with_run=True)
    vault_calls = [call for call in http.calls if "VaultV2Vaults" in call["query"]]
    allocation_calls = [call for call in http.calls if "VaultV2Allocations" in call["query"]]
    assert len(vault_calls) == 2
    assert len(allocation_calls) == 2
    assert len(history.vaults) == 1 and len(history.allocations) == 1


def test_native_units_and_flow_signs_are_locked_in():
    http = _FakeVaultV2Http(
        vault_pages={0: [_vault()]},
        allocation_pages={
            0: [
                _allocation("Allocate", "1500000", 1),
                _allocation("Deallocate", "250000", 2),
            ]
        },
        transaction_pages={
            0: [
                _transaction("Deposit", "2000000", 1),
                _transaction("Withdraw", "500000", 2),
                _transaction("Transfer", "750000", 3),
            ]
        },
    )
    history = _loader(http).read(with_run=True)
    assert history.allocations["assets_native"].tolist() == pytest.approx([1.5, 0.25])
    assert history.allocations["assets_delta_native"].tolist() == pytest.approx([1.5, -0.25])
    assert history.transactions["assets_native"].tolist() == pytest.approx([2.0, 0.5, 0.75])
    assert history.transactions["assets_delta_native"].tolist() == pytest.approx([2.0, -0.5, 0.0])
    assert history.coverage.total_assets_native == pytest.approx(1.75)


def test_cache_key_is_deterministic_and_covers_every_input():
    first = _loader(_FakeVaultV2Http())
    same = _loader(_FakeVaultV2Http())
    assert first._cache_key() == same._cache_key()
    variants = [
        MorphoVaultV2Loader(8453, START, END, http=_FakeVaultV2Http()),
        MorphoVaultV2Loader(CHAIN_ID, START + timedelta(seconds=1), END, http=_FakeVaultV2Http()),
        MorphoVaultV2Loader(CHAIN_ID, START, END + timedelta(seconds=1), http=_FakeVaultV2Http()),
        _loader(_FakeVaultV2Http(), tvl_floor_usd=0),
        _loader(_FakeVaultV2Http(), vault_page_size=10),
        _loader(_FakeVaultV2Http(), event_page_size=10),
    ]
    assert all(loader._cache_key() != first._cache_key() for loader in variants)
    assert first.loader_type is LoaderType.PICKLE


def test_pickle_cache_round_trip(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_PATH", str(tmp_path))
    http = _FakeVaultV2Http(vault_pages={0: [_vault()]}, allocation_pages={0: [_allocation()]})
    fresh = _loader(http).read(with_run=True)
    cached = _loader(_FakeVaultV2Http()).read(with_run=False)
    assert cached.vaults == fresh.vaults and cached.coverage == fresh.coverage
    pd.testing.assert_frame_equal(cached.allocations, fresh.allocations)


@pytest.mark.integration
def test_live_vault_v2_api_smoke(request):
    if "integration" not in request.config.option.markexpr:
        pytest.skip("live API smoke runs only under the integration marker")
    end = datetime.now(timezone.utc)
    history = MorphoVaultV2Loader(
        CHAIN_ID,
        end - timedelta(minutes=5),
        end,
        tvl_floor_usd=100_000_000,
    ).read(with_run=True)
    assert isinstance(history, MorphoVaultV2History)
    assert all(vault.chain_id == CHAIN_ID for vault in history.vaults)
