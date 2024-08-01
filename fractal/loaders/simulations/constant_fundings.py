import pandas as pd

from fractal.loaders.base_loader import Loader, LoaderType
from fractal.loaders.structs import FundingHistory


class ConstantFundingsLoader(Loader):
    """
    Generates a constant funding rate for a given period.
    """
    def __init__(self, rate: float = -0.001, freq: str = "D",
                 start: str = '2020-01-01', end: str = '2025-01-01') -> None:
        """
        Generates a constant funding rate for a given period.
        """
        super().__init__(loader_type=LoaderType.CSV)
        self.rate = rate
        self.freq = freq
        self.start = start
        self.end = end

    def extract(self):
        pass

    def transform(self):
        pass

    def load(self):
        pass

    def read(self, with_run: bool = False):
        return FundingHistory(time=pd.date_range(start=self.start, end=self.end, freq=self.freq), rates=self.rate)
