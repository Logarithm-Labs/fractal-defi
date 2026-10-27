"""Offline synthetic tests for the Morpho Vault V2 activity loader."""
import warnings
from collections.abc import Callable
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from fractal.loaders import LoaderType, MorphoVaultV2History, MorphoVaultV2Loader
from fractal.loaders.morpho import MORPHO_GRAPHQL_URL, MorphoLoaderException
from fractal.loaders.morpho_vault_v2 import (
    MorphoMarketParams,
    _collateral_id,
    attribute_supply_changes,
    classify_allocations,
    fetch_market_params,
    hour_end,
    hourly_supply_changes,
    morpho_market_id,
    vault_v2_market_params_id,
)

pytestmark = pytest.mark.core

CHAIN_ID = 1
START = datetime(2026, 1, 1, tzinfo=timezone.utc)
END = START + timedelta(hours=1)
VAULT = "0x" + "11" * 20
VAULT_2 = "0x" + "12" * 20
VAULT_3 = "0x" + "13" * 20
ASSET = "0x" + "22" * 20
ADAPTER = "0x" + "33" * 20
SENDER = "0x" + "44" * 20
COLLATERAL = "0x" + "66" * 20
ORACLE = "0x" + "77" * 20
IRM = "0x" + "88" * 20
LLTV = 860_000_000_000_000_000
MARKET_A = "0x" + "a" * 64
MARKET_B = "0x" + "b" * 64
ADAPTER_ID = "0x6baa0f3b766dec34ef84d9c601b23fecf043715129a51b9bf2ab043df54e618e"

Handler = Callable[[dict], tuple[list[dict], int | None]]


def _vault(address: str = VAULT, decimals: int | None = 6) -> dict:
    return {
        "address": address,
        "chain": {"id": CHAIN_ID},
        "asset": {"address": ASSET, "symbol": "SYN", "decimals": decimals},
        "totalAssetsUsd": "2000000",
        "liquidityAdapter": {"address": ADAPTER},
        "allocators": [{"allocator": {"address": SENDER}}],
    }


def _market(market_id: str = MARKET_A, lltv: int = LLTV) -> dict:
    return {
        "marketId": market_id,
        "lltv": str(lltv),
        "irmAddress": IRM,
        "oracleAddress": ORACLE,
        "loanAsset": {"address": ASSET},
        "collateralAsset": {"address": COLLATERAL},
    }


def _ids(*lltvs: int) -> list[str]:
    """Ids a MorphoMarketV1 adapter emits for the synthetic market(s) with these LLTVs."""
    return [
        ADAPTER_ID,
        _collateral_id(COLLATERAL),
        *(vault_v2_market_params_id(ADAPTER, ASSET, COLLATERAL, ORACLE, IRM, lltv) for lltv in lltvs),
    ]


def _allocation(
    event_type: str = "Allocate",
    assets: str = "1500000",
    log_index: int = 1,
    ids: list[str] | None = None,
) -> dict:
    return {
        "txHash": "0x" + f"{log_index:064x}",
        "logIndex": log_index,
        "timestamp": int(START.timestamp()) + log_index,
        "type": event_type,
        "adapter": ADAPTER,
        "sender": SENDER,
        "assets": assets,
        "change": assets,
        "ids": _ids(LLTV) if ids is None else ids,
    }


def _transaction(event_type: str, assets: str, log_index: int, vault: str = VAULT) -> dict:
    return {
        "txHash": "0x" + f"{100 + log_index:064x}",
        "logIndex": log_index,
        "timestamp": int(START.timestamp()) + 100 + log_index,
        "type": event_type,
        "assets": assets,
        "vault": {"address": vault, "chain": {"id": CHAIN_ID}},
    }


def _by_skip(pages: dict[int, list[dict]]) -> Handler:
    return lambda variables: (pages.get(variables["skip"], []), None)


class _FakeVaultV2Http:
    def __init__(
        self,
        *,
        vault_pages: dict[int, list[dict]] | None = None,
        allocations: Handler | None = None,
        transactions: Handler | None = None,
        markets: list[dict] | None = None,
    ) -> None:
        self.handlers: dict[str, tuple[str, Handler]] = {
            "VaultV2Vaults": ("vaultV2s", _by_skip(vault_pages or {})),
            "VaultV2Allocations": ("vaultV2AllocationTransactions", allocations or _by_skip({})),
            "VaultV2Transactions": ("vaultV2transactions", transactions or _by_skip({})),
            "VaultV2Markets": ("markets", _by_skip({0: [_market()] if markets is None else markets})),
        }
        self.calls: list[dict] = []

    def post(self, url, json=None, timeout=None, headers=None):
        assert url == MORPHO_GRAPHQL_URL
        self.calls.append(json)
        for name, (root, handler) in self.handlers.items():
            if f"query {name}(" in json["query"]:
                items, total = handler(json["variables"])  # pylint: disable=not-callable
                block = {"items": items}
                if total is not None:
                    block["pageInfo"] = {"countTotal": total}
                return {"data": {root: block}}
        raise AssertionError("unexpected GraphQL query")

    def variables(self, name: str) -> list[dict]:
        return [call["variables"] for call in self.calls if f"query {name}(" in call["query"]]


def _loader(http: _FakeVaultV2Http, **kwargs) -> MorphoVaultV2Loader:
    kwargs.setdefault("chain_ids", [CHAIN_ID])
    return MorphoVaultV2Loader(START, END, http=http, **kwargs)


def test_market_params_id_matches_reference_abi_encoding():
    # Reference values computed with eth_abi.encode + keccak256 for the synthetic inputs.
    assert vault_v2_market_params_id(ADAPTER, ASSET, COLLATERAL, ORACLE, IRM, LLTV) == (
        "0x6a13eb6bbb1bb326dba5521a12165d33591ee82fad9573e5131c47653761e3fe"
    )
    zero = "0x" + "00" * 20
    assert vault_v2_market_params_id(ADAPTER, ASSET, zero, zero, zero, 0) == (
        "0xb5ac995e0fc657a35f90ce542b5ae0f97539364515c30d124c8414fbdd078f4a"
    )
    assert _collateral_id(COLLATERAL) == "0x3e7eb50ee20e57e63cdfa816040473776d6d1bd5e39739432c0d3e04ea18a521"


def test_morpho_market_id_matches_reference_abi_encoding():
    # Reference values computed with eth_abi.encode + keccak256 for the synthetic inputs.
    assert morpho_market_id(ASSET, COLLATERAL, ORACLE, IRM, LLTV) == (
        "0x0e1dbc97eb802ed1bd06fbcbe5d76ea2d11a8257e5aef2b3da7a7b6c8441618f"
    )
    zero = "0x" + "00" * 20
    assert morpho_market_id(ASSET, zero, zero, zero, 0) == (
        "0x687231f3f9933a17929f9b1c101ad76167bf2d67e6f270d08b88a2c3b732975b"
    )
    upper = ["0x" + address[2:].upper() for address in (ASSET, COLLATERAL, ORACLE, IRM)]
    assert morpho_market_id(*upper, LLTV) == morpho_market_id(ASSET, COLLATERAL, ORACLE, IRM, LLTV)
    assert morpho_market_id(ASSET, COLLATERAL, ORACLE, IRM, LLTV + 1) != morpho_market_id(
        ASSET, COLLATERAL, ORACLE, IRM, LLTV
    )


def test_empty_api_page_returns_well_shaped_typed_empty_history():
    http = _FakeVaultV2Http()
    history = _loader(http).read(with_run=True)
    assert isinstance(history, MorphoVaultV2History)
    assert history.vaults == ()
    assert history.allocations.empty and history.transactions.empty
    assert list(history.allocations.columns) == [
        "time", "chain_id", "vault_address", "asset_address", "asset_symbol", "type", "adapter_address",
        "sender_address", "assets_native", "assets_delta_native", "change_native", "market_id", "map_method",
        "ids", "tx_hash", "log_index",
    ]
    for frame in (history.allocations, history.transactions):
        assert isinstance(frame["time"].dtype, pd.DatetimeTZDtype)
        assert frame["chain_id"].dtype == "int64" and frame["log_index"].dtype == "int64"
    assert history.coverage.total_allocations == 0
    assert history.coverage.mapped_fraction is None
    assert not http.variables("VaultV2Markets")


def test_allocation_maps_through_market_params_id_and_preserves_metadata():
    http = _FakeVaultV2Http(
        vault_pages={0: [_vault()]},
        allocations=_by_skip({0: [_allocation()]}),
        transactions=_by_skip({0: [_transaction("Deposit", "2000000", 1)]}),
        markets=[_market(MARKET_A, LLTV), _market(MARKET_B, LLTV - 1)],
    )
    history = _loader(http).read(with_run=True)
    assert len(history.vaults) == 1
    listing = history.vaults[0]
    assert listing.vault_address == VAULT and listing.asset_symbol == "SYN"
    assert listing.total_assets_usd == pytest.approx(2_000_000.0)
    assert listing.allocators == (SENDER,)
    row = history.allocations.iloc[0]
    assert row["market_id"] == MARKET_A and row["map_method"] == "id"
    assert row["time"] == pd.Timestamp(START + timedelta(seconds=1))
    assert history.coverage.mapped_allocations == 1
    assert http.variables("VaultV2Vaults")[0]["where"] == {
        "chainId_in": [CHAIN_ID],
        "totalAssetsUsd_gte": 1_000_000.0,
    }
    assert http.variables("VaultV2Allocations")[0]["where"] == {
        "timestamp_gte": int(START.timestamp()),
        "timestamp_lte": int(END.timestamp()),
    }
    assert http.variables("VaultV2Markets")[0]["where"] == {"chainId_in": [CHAIN_ID], "loanAssetAddress_in": [ASSET]}


@pytest.mark.parametrize(
    ("ids", "case"),
    [
        (_ids(LLTV + 1), "market-params id of an unknown market"),
        ([ADAPTER_ID, _collateral_id(COLLATERAL)], "no market-params id (non-market adapter)"),
        (_ids(LLTV, LLTV - 1), "two candidate market-params ids"),
        ([], "no ids"),
    ],
)
def test_unresolvable_allocation_is_retained_and_counted_in_coverage(ids, case):
    http = _FakeVaultV2Http(
        vault_pages={0: [_vault()]},
        allocations=_by_skip({0: [_allocation(ids=ids)]}),
        markets=[_market(MARKET_A, LLTV), _market(MARKET_B, LLTV - 1)],
    )
    history = _loader(http).read(with_run=True)
    assert len(history.allocations) == 1, case
    assert history.allocations.iloc[0]["market_id"] is None, case
    assert history.allocations.iloc[0]["map_method"] == "none"
    assert history.coverage.unmapped_allocations == 1
    assert history.coverage.mapped_fraction == 0.0
    assert history.coverage.mapped_assets_fraction == 0.0


def test_repeated_full_pages_stop_pagination_without_duplicate_rows():
    event = _allocation()
    http = _FakeVaultV2Http(
        vault_pages={0: [_vault()], 1: [_vault()]},
        allocations=_by_skip({0: [event], 1: [event]}),
    )
    history = _loader(http, vault_page_size=1, event_page_size=1).read(with_run=True)
    assert len(http.variables("VaultV2Vaults")) == 2
    assert len(http.variables("VaultV2Allocations")) == 2
    assert len(history.vaults) == 1 and len(history.allocations) == 1


def test_allocation_window_bisects_only_while_count_exceeds_the_page():
    late = {**_allocation(log_index=3), "timestamp": int(END.timestamp()) - 1}
    events = [_allocation(log_index=1), _allocation(log_index=2), late]

    def handler(variables):
        lo, hi = variables["where"]["timestamp_gte"], variables["where"]["timestamp_lte"]
        rows = [event for event in events if lo <= event["timestamp"] <= hi]
        return rows[:variables["first"]], len(rows)

    http = _FakeVaultV2Http(vault_pages={0: [_vault()]}, allocations=handler)
    history = _loader(http, event_page_size=2).read(with_run=True)
    assert history.allocations["log_index"].tolist() == [1, 2, 3]
    calls = http.variables("VaultV2Allocations")
    windows = [(v["where"]["timestamp_gte"], v["where"]["timestamp_lte"]) for v in calls]
    assert windows[0] == (int(START.timestamp()), int(END.timestamp()))
    assert len(windows) == 3  # one split; a page holding exactly countTotal rows is final
    assert all(v["skip"] == 0 for v in calls)


def test_single_second_overflow_paginates_and_warns_on_a_short_result():
    second = int(START.timestamp())
    events = [{**_allocation(log_index=i), "timestamp": second} for i in (1, 2, 3)]

    def handler(variables):
        skip, first = variables["skip"], variables["first"]
        return events[skip:skip + first], len(events)

    http = _FakeVaultV2Http(vault_pages={0: [_vault()]}, allocations=handler)
    one_second = MorphoVaultV2Loader(START, START, chain_ids=[CHAIN_ID], event_page_size=2, http=http)
    assert one_second.read(with_run=True).allocations["log_index"].tolist() == [1, 2, 3]

    stuck = _FakeVaultV2Http(vault_pages={0: [_vault()]}, allocations=lambda v: (events[:2], len(events)))
    with pytest.warns(UserWarning, match="2 distinct events of countTotal 3"):
        history = MorphoVaultV2Loader(START, START, chain_ids=[CHAIN_ID], event_page_size=2, http=stuck).read(
            with_run=True
        )
    assert history.allocations["log_index"].tolist() == [1, 2]


def test_transactions_survive_short_skip_pages_by_bisecting_the_window():
    base = int(START.timestamp())
    rows = [
        {**_transaction("Deposit", "1000000", i), "timestamp": base + i, "txHash": "0x" + f"{1000 + i:064x}"}
        for i in range(1, 51)
    ]

    def lossy(variables):
        """Windowed source whose full pages come back three rows short."""
        where, skip, first = variables["where"], variables["skip"], variables["first"]
        window = [row for row in rows if where["timestamp_gte"] <= row["timestamp"] <= where["timestamp_lte"]]
        page = window[skip:skip + first]
        return (page[:-3] if len(page) == first else page), len(window)

    http = _FakeVaultV2Http(vault_pages={0: [_vault()]}, transactions=lossy)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        history = _loader(http, event_page_size=10).read(with_run=True)
    assert len(history.transactions) == len(rows)
    assert all(v["skip"] == 0 for v in http.variables("VaultV2Transactions"))


def test_transaction_count_short_of_count_total_warns():
    second = int(START.timestamp())
    rows = [{**_transaction("Deposit", "1", i), "timestamp": second} for i in (1, 2, 3)]
    http = _FakeVaultV2Http(vault_pages={0: [_vault()]}, transactions=lambda v: (rows[:2], 3))
    loader = MorphoVaultV2Loader(START, START, chain_ids=[CHAIN_ID], event_page_size=2, http=http)
    with pytest.warns(UserWarning, match="countTotal 3"):
        assert len(loader.read(with_run=True).transactions) == 2


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("1500000000000000000000", 1500.0), ("1.5e+21", 1500.0), (1.5e21, 1500.0), ("15E20", 1500.0), (7, 7e-18)],
)
def test_bigint_amounts_parse_exactly_including_scientific_notation(raw, expected):
    vault = {**_vault(), "asset": {"address": ASSET, "symbol": "SYN", "decimals": 18}}
    http = _FakeVaultV2Http(vault_pages={0: [vault]}, allocations=_by_skip({0: [{**_allocation(), "assets": raw}]}))
    assert _loader(http).read(with_run=True).allocations["assets_native"].iloc[0] == pytest.approx(expected)


@pytest.mark.parametrize("raw", ["abc", "1.5", "1e-3", "nan", "inf", True])
def test_malformed_bigint_amount_raises_a_named_error(raw):
    http = _FakeVaultV2Http(vault_pages={0: [_vault()]}, allocations=_by_skip({0: [{**_allocation(), "assets": raw}]}))
    with pytest.raises(MorphoLoaderException, match="integer amount"):
        _loader(http).read(with_run=True)


def test_fetch_market_params_by_id_batches_and_by_loan_token():
    market_c = "0x" + "c" * 64

    def markets(variables):
        where = variables["where"]
        catalogue = [_market(MARKET_A, LLTV), _market(MARKET_B, LLTV - 1), {**_market(market_c, 0),
                     "collateralAsset": None, "oracleAddress": None, "irmAddress": None}]
        if "uniqueKey_in" in where:
            items = [item for item in catalogue if item["marketId"] in where["uniqueKey_in"]]
        else:
            items = [item for item in catalogue if item["loanAsset"]["address"] in where["loanAssetAddress_in"]]
        return items[variables["skip"]:variables["skip"] + variables["first"]], len(items)

    http = _FakeVaultV2Http()
    http.handlers["VaultV2Markets"] = ("markets", markets)
    by_id = fetch_market_params(CHAIN_ID, market_ids=[MARKET_A.upper().replace("0X", "0x"), market_c], http=http,
                                batch_size=1)
    assert by_id == {
        MARKET_A: MorphoMarketParams(ASSET, COLLATERAL, ORACLE, IRM, LLTV),
        market_c: MorphoMarketParams(ASSET, "0x" + "00" * 20, "0x" + "00" * 20, "0x" + "00" * 20, 0),
    }
    assert [v["where"]["uniqueKey_in"] for v in http.variables("VaultV2Markets")] == [[MARKET_A], [market_c]]
    assert morpho_market_id(*by_id[MARKET_A]) == "0x0e1dbc97eb802ed1bd06fbcbe5d76ea2d11a8257e5aef2b3da7a7b6c8441618f"
    by_loan = fetch_market_params(CHAIN_ID, loan_addresses=[ASSET], http=http)
    assert set(by_loan) == {MARKET_A, MARKET_B, market_c}
    assert fetch_market_params(CHAIN_ID, http=http) == {}


def test_transactions_are_batched_per_chain_and_attributed_to_their_vault():
    vaults = [_vault(VAULT), _vault(VAULT_2), _vault(VAULT_3)]
    rows = {
        VAULT: _transaction("Deposit", "1000000", 1, VAULT),
        VAULT_3: _transaction("Withdraw", "2000000", 2, VAULT_3),
    }

    def handler(variables):
        return [rows[v] for v in variables["where"]["vaultAddress_in"] if v in rows], None

    http = _FakeVaultV2Http(vault_pages={0: vaults}, transactions=handler)
    history = _loader(http, vault_batch_size=2).read(with_run=True)
    calls = http.variables("VaultV2Transactions")
    assert [v["where"]["vaultAddress_in"] for v in calls] == [[VAULT, VAULT_2], [VAULT_3]]
    assert all(v["where"]["type_in"] == ["Deposit", "Withdraw", "Transfer"] for v in calls)
    assert history.transactions["vault_address"].tolist() == [VAULT, VAULT_3]
    assert history.transactions["assets_delta_native"].tolist() == pytest.approx([1.0, -2.0])


def test_vault_filters_are_optional_and_normalised():
    http = _FakeVaultV2Http(vault_pages={0: [_vault(VAULT_2), _vault(VAULT), _vault(VAULT_2)]})
    mixed_case = "0x" + ASSET[2:].upper()
    loader = MorphoVaultV2Loader(START, END, asset_addresses=[mixed_case], tvl_floor_usd=0, http=http)
    listings = loader.list_vaults()
    assert [listing.vault_address for listing in listings] == [VAULT, VAULT_2]
    assert http.variables("VaultV2Vaults")[0]["where"] == {"assetAddress_in": [ASSET]}
    assert not http.variables("VaultV2Allocations")


def test_native_units_and_flow_signs_are_locked_in():
    http = _FakeVaultV2Http(
        vault_pages={0: [_vault()]},
        allocations=_by_skip(
            {0: [_allocation("Allocate", "1500000", 1), _allocation("Deallocate", "250000", 2)]}
        ),
        transactions=_by_skip(
            {
                0: [
                    _transaction("Deposit", "2000000", 1),
                    _transaction("Withdraw", "500000", 2),
                    _transaction("Transfer", "750000", 3),
                ]
            }
        ),
    )
    history = _loader(http).read(with_run=True)
    assert history.allocations["assets_native"].tolist() == pytest.approx([1.5, 0.25])
    assert history.allocations["assets_delta_native"].tolist() == pytest.approx([1.5, -0.25])
    assert history.transactions["assets_native"].tolist() == pytest.approx([2.0, 0.5, 0.75])
    assert history.transactions["assets_delta_native"].tolist() == pytest.approx([2.0, -0.5, 0.0])
    assert history.coverage.total_assets_native == pytest.approx(1.75)


def test_missing_asset_decimals_keeps_rows_with_null_amounts():
    http = _FakeVaultV2Http(vault_pages={0: [_vault(decimals=None)]}, allocations=_by_skip({0: [_allocation()]}))
    history = _loader(http).read(with_run=True)
    assert history.vaults[0].asset_decimals is None
    assert history.allocations["assets_native"].isna().all()
    assert history.allocations.iloc[0]["market_id"] == MARKET_A
    assert history.coverage.mapped_assets_fraction is None


def test_unknown_allocation_type_is_rejected():
    http = _FakeVaultV2Http(vault_pages={0: [_vault()]}, allocations=_by_skip({0: [_allocation("Reallocate")]}))
    with pytest.raises(MorphoLoaderException, match="unsupported Vault V2 allocation type"):
        _loader(http).read(with_run=True)


def test_classify_allocations_uses_same_tx_join_then_current_allocators():
    other_sender = "0x" + "45" * 20
    shared_tx = _transaction("Withdraw", "500000", 7)
    depositor = {**_allocation("Deallocate", "500000", 5, ids=[]), "txHash": shared_tx["txHash"],
                 "sender": other_sender}
    in_other_vault_tx = {**_allocation("Allocate", "1", 6, ids=[]), "txHash": shared_tx["txHash"],
                         "sender": other_sender}
    transfer_tx = _transaction("Transfer", "1", 8)
    transfer_only = {**_allocation("Allocate", "1", 8, ids=[]), "txHash": transfer_tx["txHash"],
                     "sender": other_sender}
    http = _FakeVaultV2Http(
        vault_pages={0: [_vault(), _vault(VAULT_2)]},
        allocations=lambda v: (
            ([_allocation(log_index=1), depositor, transfer_only] if v["vaultAddress"] == VAULT
             else [in_other_vault_tx]) if v["skip"] == 0 else [],
            None,
        ),
        transactions=_by_skip({0: [shared_tx, transfer_tx, _transaction("Deposit", "1", 9, VAULT_2)]}),
    )
    history = _loader(http).read(with_run=True)
    labels = classify_allocations(history.allocations, history.transactions, history.vaults)
    assert labels.name == "class" and labels.index.equals(history.allocations.index)
    by_log = dict(zip(history.allocations["vault_address"] + ":" + history.allocations["log_index"].astype(str),
                      labels, strict=True))
    assert by_log == {
        f"{VAULT}:1": "allocator",  # sent by a current allocator, no depositor tx
        f"{VAULT}:5": "depositor_driven",  # same tx as a Withdraw of the same vault
        f"{VAULT}:8": "other",  # same tx only as a share Transfer
        f"{VAULT_2}:6": "other",  # same tx hash, but the Withdraw belongs to another vault
    }
    without_vaults = classify_allocations(history.allocations, history.transactions)
    assert without_vaults.tolist().count("allocator") == 0
    empty = classify_allocations(history.allocations.iloc[:0], history.transactions.iloc[:0], history.vaults)
    assert empty.empty and empty.name == "class"


def _hour(hours: float) -> pd.Timestamp:
    return pd.Timestamp(START) + pd.Timedelta(hours=hours)


def test_hour_end_labels_the_closing_snapshot():
    times = pd.Series([_hour(0), _hour(0) + pd.Timedelta(seconds=1), _hour(1) - pd.Timedelta(seconds=1)])
    assert hour_end(times).tolist() == [_hour(0), _hour(1), _hour(1)]


def test_hourly_supply_changes_pairs_consecutive_snapshots_and_drops_gaps():
    supply = pd.DataFrame(
        {
            "chain_id": [1, 1, 1, 1, 1],
            "market_id": [MARKET_A, MARKET_A, MARKET_A, MARKET_B, MARKET_B],
            "time": [_hour(0), _hour(1), _hour(3), _hour(0), _hour(1)],
            "supply_assets": [100.0, 110.0, 50.0, 0.0, 5.0],
        }
    )
    changes = hourly_supply_changes(supply)
    assert changes[["market_id", "time"]].values.tolist() == [[MARKET_A, _hour(1)], [MARKET_B, _hour(1)]]
    assert changes["net_change"].tolist() == [10.0, 5.0]
    assert changes["rel_change"].iloc[0] == pytest.approx(0.1)
    assert pd.isna(changes["rel_change"].iloc[1])  # previous supply of zero


def test_attribute_supply_changes_assigns_the_dominant_same_sign_source():
    changes = pd.DataFrame(
        {
            "chain_id": [1, 1, 1, 1, 1],
            "market_id": [MARKET_A, MARKET_A, MARKET_A, MARKET_A, MARKET_B],
            "time": [_hour(1), _hour(2), _hour(3), _hour(4), _hour(1)],
            "net_change": [10.0, -10.0, 10.0, 0.0, 10.0],
        }
    )
    flows = pd.DataFrame(
        {
            "chain_id": 1,
            "market_id": [MARKET_A, MARKET_A, MARKET_A, MARKET_A, MARKET_A, MARKET_A, None],
            "time": [
                _hour(0.5), _hour(0.6),  # hour 1: allocator +8 beats depositor +1
                _hour(1.5), _hour(2.0),  # hour 2 (2.0 is still hour 2): depositor -9, allocator +2
                _hour(2.5),  # hour 3: +3 only, below the share threshold
                _hour(3.5),  # hour 4: no net change
                _hour(0.5),  # unmapped row is ignored
            ],
            "source": ["allocator", "depositor_driven", "depositor_driven", "allocator", "allocator", "allocator",
                       "allocator"],
            "assets_delta_native": [8.0, 1.0, -9.0, 2.0, 3.0, 5.0, 100.0],
        }
    )
    out = attribute_supply_changes(changes, flows, min_share=0.5)
    assert out["source"].tolist() == ["allocator", "depositor_driven", None, None, None]
    assert out["source_share"].iloc[:2].tolist() == pytest.approx([0.8, 0.9])
    assert out["source_share"].iloc[2:].isna().all()
    assert out["flow_allocator"].tolist() == [8.0, 2.0, 3.0, 5.0, 0.0]
    assert out["flow_depositor_driven"].tolist() == [1.0, -9.0, 0.0, 0.0, 0.0]
    assert attribute_supply_changes(changes, flows, min_share=0.25)["source"].iloc[2] == "allocator"
    against = attribute_supply_changes(changes.assign(net_change=-changes["net_change"]), flows, min_share=0.01)
    assert against["source"].iloc[0] is None  # flows opposite to the change never explain it


def test_attribute_supply_changes_without_flows_and_validation():
    changes = pd.DataFrame({"chain_id": [1], "market_id": [MARKET_A], "time": [_hour(1)], "net_change": [1.0]})
    empty = pd.DataFrame(columns=["chain_id", "market_id", "time", "source", "assets_delta_native"])
    out = attribute_supply_changes(changes, empty, min_share=0.5)
    assert out["source"].tolist() == [None] and out["source_share"].isna().all()
    with pytest.raises(ValueError, match="min_share"):
        attribute_supply_changes(changes, empty, min_share=0)


def test_classified_allocations_feed_supply_attribution_end_to_end():
    http = _FakeVaultV2Http(vault_pages={0: [_vault()]}, allocations=_by_skip({0: [_allocation()]}))
    history = _loader(http).read(with_run=True)
    flows = history.allocations.assign(
        source=classify_allocations(history.allocations, history.transactions, history.vaults)
    )
    changes = pd.DataFrame({"chain_id": [1], "market_id": [MARKET_A], "time": [_hour(1)], "net_change": [1.5]})
    out = attribute_supply_changes(changes, flows, min_share=0.5)
    assert out["source"].tolist() == ["allocator"] and out["source_share"].iloc[0] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tvl_floor_usd": -1},
        {"vault_page_size": 0},
        {"event_page_size": 1001},
        {"vault_batch_size": 0},
        {"chain_ids": []},
        {"asset_addresses": []},
        {"loader_type": LoaderType.CSV},
    ],
)
def test_invalid_arguments_are_rejected(kwargs):
    with pytest.raises(ValueError):
        MorphoVaultV2Loader(START, END, http=_FakeVaultV2Http(), **kwargs)


def test_reversed_window_is_rejected():
    with pytest.raises(ValueError, match="precedes"):
        MorphoVaultV2Loader(END, START, http=_FakeVaultV2Http())


def test_cache_key_is_deterministic_and_covers_every_input():
    first = _loader(_FakeVaultV2Http())
    assert first._cache_key() == _loader(_FakeVaultV2Http(), chain_ids=[CHAIN_ID, CHAIN_ID])._cache_key()
    variants = [
        _loader(_FakeVaultV2Http(), chain_ids=[8453]),
        _loader(_FakeVaultV2Http(), chain_ids=None),
        _loader(_FakeVaultV2Http(), asset_addresses=[ASSET]),
        MorphoVaultV2Loader(START + timedelta(seconds=1), END, chain_ids=[CHAIN_ID], http=_FakeVaultV2Http()),
        MorphoVaultV2Loader(START, END + timedelta(seconds=1), chain_ids=[CHAIN_ID], http=_FakeVaultV2Http()),
        _loader(_FakeVaultV2Http(), tvl_floor_usd=0),
        _loader(_FakeVaultV2Http(), vault_page_size=10),
        _loader(_FakeVaultV2Http(), event_page_size=10),
    ]
    keys = {loader._cache_key() for loader in variants}
    assert len(keys) == len(variants) and first._cache_key() not in keys
    reordered = _loader(_FakeVaultV2Http(), asset_addresses=["0x" + VAULT[2:].upper(), ASSET])
    assert _loader(_FakeVaultV2Http(), asset_addresses=[ASSET, VAULT])._cache_key() == reordered._cache_key()
    assert first.loader_type is LoaderType.PICKLE


def test_pickle_cache_round_trip(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_PATH", str(tmp_path))
    http = _FakeVaultV2Http(vault_pages={0: [_vault()]}, allocations=_by_skip({0: [_allocation()]}))
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
        end - timedelta(hours=6),
        end,
        chain_ids=[CHAIN_ID],
        tvl_floor_usd=100_000_000,
    ).read(with_run=True)
    assert isinstance(history, MorphoVaultV2History)
    assert all(vault.chain_id == CHAIN_ID for vault in history.vaults)
    assert set(history.allocations["map_method"]) <= {"id", "none"}
