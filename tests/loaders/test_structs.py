"""Lock-ins for the loader structs: strict time index, per-bar
``LendingHistory`` with optional columns, the Pendle and Boros shapes."""
import numpy as np
import pandas as pd
import pytest

from fractal.loaders import BorosMarketHistory, LendingHistory, PendleMarketHistory, PriceHistory

TIMES = pd.to_datetime([1_700_000_000, 1_700_003_600], unit="s", utc=True)


@pytest.mark.core
def test_integer_epochs_are_rejected_not_guessed():
    with pytest.raises(TypeError, match="unit"):
        PriceHistory(prices=[1.0, 2.0], time=np.array([1_700_000_000, 1_700_003_600]))
    history = PriceHistory(prices=[1.0, 2.0], time=TIMES)
    assert history.index.name == "time" and str(history.index.tz) == "UTC"


@pytest.mark.core
def test_lending_history_optional_columns_only_when_passed():
    plain = LendingHistory(lending_rates=[0.0, 0.0], borrowing_rates=[1e-5, 1e-5], time=TIMES)
    assert list(plain.columns) == ["lending_rate", "borrowing_rate"]
    rich = LendingHistory(lending_rates=[0.0, 0.0], borrowing_rates=[1e-5, 1e-5], time=TIMES,
                          utilization=[0.8, 0.9], borrow_apy=[0.09, 0.1])
    assert list(rich.columns) == ["lending_rate", "borrowing_rate", "utilization", "borrow_apy"]


@pytest.mark.core
def test_pendle_and_boros_histories_expose_their_column_contracts():
    pendle = PendleMarketHistory(time=TIMES, **{c: [1.0, 1.0] for c in PendleMarketHistory.COLUMNS})
    assert list(pendle.columns) == list(PendleMarketHistory.COLUMNS)
    boros = BorosMarketHistory(time=[], **{c: [] for c in BorosMarketHistory.COLUMNS})
    assert boros.empty and list(boros.columns) == list(BorosMarketHistory.COLUMNS)
