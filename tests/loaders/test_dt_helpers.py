"""Offline tests for the shared loader helpers added with the fixed-term work:
``annualise_funding``, ``require_no_nan`` and ``Loader._utc_index``."""
import numpy as np
import pandas as pd
import pytest

from fractal.loaders._dt import SECONDS_PER_YEAR, annualise_funding, require_no_nan
from fractal.loaders.base_loader import Loader, LoaderType


class _StubLoader(Loader):
    """Concrete ``Loader`` with no I/O — only the inherited helpers are exercised."""

    def extract(self) -> None:
        pass

    def transform(self) -> None:
        pass

    def read(self, with_run: bool = False):
        return None


@pytest.mark.core
def test_annualise_funding_matches_boros_convention():
    """Binance 8h funding × 1095 and Hyperliquid 1h × 8760 — how Boros reports settlementApr."""
    assert annualise_funding(0.0001, 8 * 3600) == pytest.approx(0.0001 * 1095)
    assert annualise_funding(0.0001, 3600) == pytest.approx(0.0001 * 8760)
    assert annualise_funding(-0.00003143, 8 * 3600) == pytest.approx(-0.00003143 * SECONDS_PER_YEAR / 28_800)
    with pytest.raises(ValueError):
        annualise_funding(0.1, 0)


@pytest.mark.core
def test_require_no_nan_names_column_and_count():
    df = pd.DataFrame({"a": [1.0, np.nan, np.nan], "b": [1.0, 2.0, 3.0]})
    require_no_nan(df, ["b"])
    with pytest.raises(ValueError, match=r"\{'a': 2\}"):
        require_no_nan(df, ["a", "b"])
    with pytest.raises(ValueError, match="missing"):
        require_no_nan(df, ["c"])


@pytest.mark.core
def test_utc_index_from_epoch_seconds_survives_csv_and_json_round_trip(tmp_path):
    """An int64 ``time`` column comes back identical from both cache formats."""
    loader = _StubLoader(loader_type=LoaderType.CSV)
    loader._data = pd.DataFrame({"time": [1_700_000_000, 1_700_003_600], "x": [1.0, 2.0]})
    idx = loader._utc_index()
    assert idx.name == "time"
    assert str(idx.tz) == "UTC"
    assert idx[1] - idx[0] == pd.Timedelta(hours=1)

    csv_path = tmp_path / "cache.csv"
    loader._data.to_csv(csv_path, index=False)
    json_path = tmp_path / "cache.json"
    loader._data.to_json(json_path, orient="records")

    for path, reader in ((csv_path, pd.read_csv), (json_path, lambda p: pd.read_json(p, orient="records"))):
        loader._data = reader(path)
        assert loader._utc_index().equals(idx), path


@pytest.mark.core
def test_utc_index_requires_the_column():
    loader = _StubLoader(loader_type=LoaderType.CSV)
    loader._data = pd.DataFrame({"x": [1.0]})
    with pytest.raises(ValueError, match="missing"):
        loader._utc_index()
