from typing import Optional

import numpy as np
import pandas as pd


class PriceHistory(pd.DataFrame):
    """
    Price history data structure.
    """
    def __init__(self, prices: np.array, time: np.array):
        super().__init__(data=prices, index=time, columns=['price'])


class FundingHistory(pd.DataFrame):
    """
    Funding history data structure.
    """
    def __init__(self, rates: np.array, time: np.array):
        super().__init__(data=rates,
                         index=time, columns=['rate'])


class RateHistory(pd.DataFrame):
    """
    Staking rate history data structure.
    """
    def __init__(self, rates: np.array, time: np.array):
        super().__init__(data=rates,
                         index=time, columns=['rate'])


class LendingHistory(pd.DataFrame):
    """
    Lending/borrowing history data structure.
    """
    def __init__(self, lending_rates: np.array, borrowing_rates: np.array, time: np.array):
        super().__init__(data=np.array([lending_rates, borrowing_rates]).T,
                         index=time, columns=['lending_rate', 'borrowing_rate'])


class PoolHistory(pd.DataFrame):
    """
    Pool data structure.
    """
    def __init__(self, tvls: np.array, volumes: np.array, fees: np.array,
                 liquidity: np.array, time: np.array, prices: Optional[np.array] = None):
        super().__init__(
            data=np.array([tvls, volumes, fees, liquidity]).T,
            index=time,
            columns=['tvl', 'volume', 'fees', 'liquidity']
        )
