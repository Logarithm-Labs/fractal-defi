"""Offline tests for the MetaMorpho (V1) vault history loader."""
import warnings
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

import fractal.loaders as loaders_package
from fractal.loaders import MorphoVaultLoader
from fractal.loaders.morpho import MORPHO_GRAPHQL_URL, MorphoLoaderException
from fractal.loaders.morpho_vaults import MorphoVaultLoader as ModuleMorphoVaultLoader

pytestmark = pytest.mark.core

UTC = timezone.utc
CHAIN_ID = 999
DAY = 86_400
START = datetime(2026, 1, 1, tzinfo=UTC)
END = START + timedelta(days=2)
T0 = int(START.timestamp())
ASSET = "0x" + "aa" * 20
COLUMNS = [
    "timestamp",
    "chain_id",
    "vault_address",
    "vault_symbol",
    "asset_address",
    "asset_symbol",
    "total_assets",
    "total_assets_usd",
]


def _vault_address(index: int) -> str:
    return "0x" + f"{index:02x}" * 20


def _vault(index: int, raw=None, usd=None, *, decimals: int | None = 6) -> dict:
    """Synthetic vault with points newest first, as the API returns them."""
    raw = raw if raw is not None else {T0: 1_000_000 * (index + 1), T0 + DAY: str(2_000_000 * (index + 1))}
    usd = usd if usd is not None else {t: float(index + 1) for t in raw}
    return {
        "address": _vault_address(index).upper(),
        "symbol": f"V{index}",
        "chain": {"id": CHAIN_ID},
        "asset": {"address": ASSET.upper(), "symbol": "TKN", "decimals": decimals},
        "historicalState": {
            "totalAssets": [{"x": t, "y": y} for t, y in sorted(raw.items(), reverse=True)],
            "totalAssetsUsd": [{"x": t, "y": y} for t, y in sorted(usd.items(), reverse=True)],
        },
    }


class _VaultsHttp:
    """Serves a vault list by ``skip``/``first``; ``short_pages`` drops one vault per non-final page."""

    def __init__(self, vaults: list[dict], *, total: int | None = -1, short_pages: bool = False,
                 repeat_first: bool = False):
        self.vaults = vaults
        self.total = len(vaults) if total == -1 else total
        self.short_pages = short_pages
        self.repeat_first = repeat_first
        self.calls: list[dict] = []

    def post(self, url, json=None, timeout=None, headers=None):
        assert url == MORPHO_GRAPHQL_URL
        self.calls.append(json)
        variables = json["variables"]
        skip = 0 if self.repeat_first else variables["skip"]
        first = variables["first"]
        more_follow = skip + first < len(self.vaults)
        page = self.vaults[skip:skip + first - (1 if self.short_pages and more_follow else 0)]
        info = {"count": len(page)}
        if self.total is not None:
            info["countTotal"] = self.total
        return {"data": {"vaults": {"items": page, "pageInfo": info}}}


@pytest.fixture(autouse=True)
def _isolated_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_PATH", str(tmp_path))


def _loader(http=None, **kwargs) -> MorphoVaultLoader:
    kwargs.setdefault("start_time", START)
    kwargs.setdefault("end_time", END)
    return MorphoVaultLoader(CHAIN_ID, http=http or _VaultsHttp([]), **kwargs)


def test_empty_listing_is_a_typed_empty_frame():
    frame = _loader().read(with_run=True)
    assert frame.empty
    assert list(frame.columns) == COLUMNS == list(MorphoVaultLoader.COLUMNS)
    assert frame["timestamp"].dtype == "int64"
    assert frame["total_assets"].dtype == "float64"


def test_rows_are_ascending_scaled_exactly_and_lowercased():
    big = 123456789012345678901234567
    vault = _vault(1, raw={T0: 1_132_797, T0 + DAY: str(big)}, usd={T0: 1.13, T0 + DAY: 2.0}, decimals=18)
    http = _VaultsHttp([vault])

    frame = _loader(http=http).read(with_run=True)

    assert frame["timestamp"].tolist() == [T0, T0 + DAY]
    assert frame["total_assets"].tolist() == [1_132_797 / 10**18, 123456789.01234567]
    assert frame["total_assets_usd"].tolist() == [1.13, 2.0]
    assert frame["vault_address"].unique().tolist() == [_vault_address(1)]
    assert frame["asset_address"].unique().tolist() == [ASSET]
    assert frame["vault_symbol"].unique().tolist() == ["V1"]
    variables = http.calls[0]["variables"]
    assert variables["o"] == {"startTimestamp": T0, "endTimestamp": T0 + 2 * DAY, "interval": "DAY"}
    assert variables["where"] == {"chainId_in": [CHAIN_ID]}
    assert "orderBy: Address" in http.calls[0]["query"]


def test_live_point_and_points_outside_the_window_are_dropped():
    raw = {T0 - DAY: 1, T0: 2, T0 + 2 * DAY: 3, T0 + 2 * DAY + 1234: 4, T0 + 3 * DAY: 5}
    live_end = END + timedelta(seconds=5000)
    frame = _loader(http=_VaultsHttp([_vault(0, raw=raw)]), end_time=live_end).read(with_run=True)
    assert frame["timestamp"].tolist() == [T0, T0 + 2 * DAY]


def test_missing_decimals_or_point_give_nan_not_an_error():
    vault = _vault(0, raw={T0: 5}, usd={T0: 1.0, T0 + DAY: 2.0}, decimals=None)
    frame = _loader(http=_VaultsHttp([vault])).read(with_run=True)
    assert frame["timestamp"].tolist() == [T0, T0 + DAY]
    assert frame["total_assets"].isna().all()
    assert frame["total_assets_usd"].tolist() == [1.0, 2.0]


def test_filters_are_validated_lowercased_and_sent():
    http = _VaultsHttp([])
    _loader(http=http, vault_addresses=[_vault_address(2).upper(), _vault_address(1)],
            asset_addresses=[ASSET.upper()]).read(with_run=True)
    assert http.calls[0]["variables"]["where"] == {
        "chainId_in": [CHAIN_ID],
        "address_in": [_vault_address(1), _vault_address(2)],
        "assetAddress_in": [ASSET],
    }


@pytest.mark.parametrize(("kwargs", "error"), [
    ({"interval": "WEEK"}, ValueError),
    ({"page_size": 0}, ValueError),
    ({"end_time": START - timedelta(seconds=1)}, ValueError),
    ({"vault_addresses": ["0x12"]}, ValueError),
    ({"vault_addresses": []}, ValueError),
    ({"asset_addresses": ASSET}, TypeError),
])
def test_invalid_arguments_raise(kwargs, error):
    with pytest.raises(error):
        _loader(**kwargs)


def test_short_pages_do_not_end_the_listing_when_count_total_is_known():
    vaults = [_vault(i) for i in range(7)]
    http = _VaultsHttp(vaults, short_pages=True)

    with pytest.warns(UserWarning, match="5 distinct vaults of countTotal 7"):
        frame = _loader(http=http, page_size=3).read(with_run=True)

    assert [call["variables"]["skip"] for call in http.calls] == [0, 3, 6]
    assert frame["vault_address"].nunique() == 5


def test_full_pages_page_through_every_vault_without_warning():
    vaults = [_vault(i) for i in range(7)]
    http = _VaultsHttp(vaults)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        frame = _loader(http=http, page_size=3).read(with_run=True)

    assert frame["vault_address"].unique().tolist() == [_vault_address(i) for i in range(7)]
    assert len(frame) == 14


def test_repeated_page_stops_the_listing():
    http = _VaultsHttp([_vault(0), _vault(1)], total=None, repeat_first=True)
    frame = _loader(http=http, page_size=2).read(with_run=True)
    assert len(http.calls) == 2
    assert frame["vault_address"].nunique() == 2


def test_missing_block_raises():
    class _NoBlock:
        def post(self, url, json=None, timeout=None, headers=None):
            return {"data": {}}

    with pytest.raises(MorphoLoaderException, match="'vaults' missing"):
        _loader(http=_NoBlock()).run()


def test_cache_key_covers_inputs_and_ignores_filter_order():
    base = _loader(vault_addresses=[_vault_address(1), _vault_address(2)])
    assert base._cache_key() == _loader(vault_addresses=[_vault_address(2), _vault_address(1)])._cache_key()
    assert base._cache_key() != _loader()._cache_key()
    assert base._cache_key() != _loader(vault_addresses=[_vault_address(1)])._cache_key()
    assert _loader()._cache_key() != _loader(asset_addresses=[ASSET])._cache_key()
    assert _loader()._cache_key() != _loader(interval="HOUR")._cache_key()
    assert _loader()._cache_key() != _loader(end_time=END + timedelta(days=1))._cache_key()


def test_csv_cache_round_trip_preserves_the_tidy_frame():
    vault = _vault(0)
    vault["symbol"] = None
    fresh = _loader(http=_VaultsHttp([vault, _vault(1, decimals=None)])).read(with_run=True)
    cached = _loader().read(with_run=False)
    pd.testing.assert_frame_equal(fresh, cached)
    assert fresh["vault_symbol"].tolist()[:2] == ["", ""]


def test_package_export():
    assert loaders_package.MorphoVaultLoader is ModuleMorphoVaultLoader
