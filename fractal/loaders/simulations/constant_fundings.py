"""Synthetic loader: emit a constant funding-rate series for a given period."""
from datetime import datetime

import pandas as pd

from fractal.loaders._dt import to_utc
from fractal.loaders.base_loader import Loader, LoaderType
from fractal.loaders.structs import FundingHistory


class ConstantFundingsLoader(Loader):
    """Returns a :class:`FundingHistory` with ``rate`` constant over the
    specified period.
    """

    def __init__(
        self,
        rate: float = -0.001,
        freq: str = "D",
        start: str | datetime = "2020-01-01",
        end: str | datetime = "2025-01-01",
        loader_type: LoaderType = LoaderType.CSV,
    ) -> None:
        super().__init__(loader_type=loader_type)
        self.rate: float = float(rate)
        self.freq: str = freq
        self.start = start if isinstance(start, datetime) else pd.Timestamp(start, tz="UTC").to_pydatetime()
        self.end = end if isinstance(end, datetime) else pd.Timestamp(end, tz="UTC").to_pydatetime()
        self.start = to_utc(self.start)
        self.end = to_utc(self.end)

    def extract(self) -> None:
        pass

    def transform(self) -> None:
        pass

    def load(self) -> None:
        # No-op: this loader is fully synthetic and reproducible from params.
        pass

    def read(self, with_run: bool = False) -> FundingHistory:
        index = pd.date_range(start=self.start, end=self.end, freq=self.freq, tz="UTC")
        return FundingHistory(time=index.values, rates=[self.rate] * len(index))
