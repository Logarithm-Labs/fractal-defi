import random
from pathlib import Path
from typing import Dict, List
from uuid import uuid4

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from scipy import stats

from fractal.loaders.loader import Loader, LoaderType
from fractal.loaders.structs import PoolHistory, PriceHistory


class LPSimulatedStates(pd.DataFrame):
    """
    Pool data structure.
    """
    def __init__(self, tvls: np.array, volumes: np.array, fees: np.array,
                 liquidity: np.array, time: np.array, prices: np.array):
        super().__init__(
            data=np.array([tvls, volumes, fees, liquidity, prices]).T,
            index=time,
            columns=['tvl', 'volume', 'fees', 'liquidity', 'price']
        )


class LPMLSimulatedStatesLoader(Loader):
    """
    A loader class for generating simulated states
        using machine learning models.
    Generates pool stated (with price_history) for LP strategies.
    Prices generates as GBM process with drift and volatility,
        which are calculated from the historical data
        in rolling windows.
    The pool data is generated using CatBoostRegressor models,
        using the price data, log returns, realized volatility,
        momentum, and moving average difference as predictors.
    After prediction, the residuals are added to the predictions,
        as samples from the residuals distribution.
    The residuals distribution is estimated using the histogram.
    Also, pool data is smoothed using a rolling mean.

    Args:
        price_history (PriceHistory): The price history data.
        pool_history (PoolHistory): The pool data.
        trajectories_number (int, optional): The number of trajectories
            to generate. Defaults to 100.
        loader_type (LoaderType, optional): The type of loader.
            Defaults to LoaderType.PICKLE.
        seed (int, optional): The seed for random number generation.
            Defaults to 420.
        random_filename (bool, optional): Whether to use a random filename
            for saving the data. Defaults to True.

    Attributes:
        pool_history (PoolHistory): The pool data.
        price_history (PriceHistory): The price history data.
        _data (pd.DataFrame): The merged data of price_history and pool data.
        _simulated_data (List[pd.DataFrame]): The list of simulated data.
        trajectories_number (int): The number of trajectories to generate.
        _dump_filename (str): The filename for saving the data.
        _random (random.Random): The random number generator.
        _np_random (numpy.random.Generator): The numpy random number generator.
        _seed (int): The seed for random number generation.

    """

    def __init__(
        self,
        price_history: PriceHistory,
        pool_history: PoolHistory,
        trajectories_number: int = 100,
        loader_type: LoaderType = LoaderType.PICKLE,
        seed: int = 420,
        random_filename: bool = True,
    ) -> None:
        """
        Initializes the LPMLSimulatedStatesLoader.

        Args:
            price_history (PriceHistory): The price history data.
            pool_history (PoolHistory): The pool data.
            trajectories_number (int, optional): The number of trajectories
            to generate. Defaults to 100.
            loader_type (LoaderType, optional): The type of loader.
                Defaults to LoaderType.PICKLE.
            seed (int, optional): The seed for random number generation.
                Defaults to 420.
            random_filename (bool, optional): Whether to use a random
                filename for saving data. Defaults to True.

        """
        super().__init__(loader_type)
        self.pool_history = pool_history
        self.price_history = price_history
        self._data = None
        self._simulated_data = []
        self.trajectories_number = trajectories_number
        if self.loader_type == LoaderType.PICKLE:
            if random_filename:
                self._file_id = str(uuid4())
            else:
                self._file_id = "ml_simulated"
        else:
            raise NotImplementedError
        # Seed all random number generators
        self._random = random.Random()
        self._random.seed(seed)
        self._np_random = np.random.default_rng(seed)
        self._seed = seed
        np.random.seed(seed)
        random.seed(seed)

    def extract(self):
        self._data = pd.merge(self.price_history, self.pool_history,
                              on="date", how="inner")
        self._data["date"] = self._data.index
        self._data.reset_index(drop=True, inplace=True)
        self._data["logreturns"] = np.log(self._data["price"].pct_change() + 1)
        self._data["rv"] = self._data["logreturns"].rolling(24).std()
        self._data["momentum"] = self._data["price"].pct_change(48)
        self._data["madiff"] = (self._data["price"].rolling(24).mean() -
                                self._data["price"].rolling(24 * 4).mean())
        self._data = self._data.bfill()

    def transform(self):

        N = self.trajectories_number
        predictors = ["price", "rv", "momentum", "madiff"]
        columns_to_predict = ["tvl", "fees", "rate", "liquidity"]
        models: Dict[CatBoostRegressor] = {}
        residuals = {}
        residuals_distribution = {}

        for column in columns_to_predict:
            model = CatBoostRegressor(iterations=500,
                                      random_seed=self._seed, verbose=False)
            model = model.fit(self._data[predictors],
                              self._data[column], verbose=False)
            res = model.predict(self._data[predictors]) - self._data[column]
            res_dist = stats.rv_histogram(np.histogram(res, bins=100))
            models[column] = model
            residuals[column] = res
            residuals_distribution[column] = res_dist

        for _ in range(1, N + 1):
            df = self._data.copy()

            df["mu"] = df["logreturns"].rolling(24).mean()
            df["sigma"] = df["logreturns"].rolling(24).std()
            df = df.bfill()
            df = df.reset_index(drop=True)
            price = np.zeros(len(df))
            price[0] = df["price"][0]
            mu = df["mu"].values
            sigma = df["sigma"].values

            for j in range(1, len(df)):
                price[j] = price[j - 1] * np.exp(mu[j] + sigma[j] *
                                                 self._np_random.normal())

            df["price"] = price
            df["logreturns"] = np.log(df["price"].pct_change() + 1)
            df["rv"] = df["logreturns"].rolling(24).std()
            df["momentum"] = df["price"].pct_change(48)
            df = df.bfill()

            for column in columns_to_predict:
                predicted = (models[column].predict(df[predictors]) +
                             residuals_distribution[column].rvs(size=len(df)))
                predicted = np.maximum(predicted, 0)
                df[column] = predicted
                df[column] = df[column].rolling(48).mean()

            df = df.bfill()

            self._simulated_data.append(df[["date", "tvl", "fees",
                                            "price", "rate",
                                            "liquidity"]].copy())
        self._data = [LPSimulatedStates(
            tvls=df["tvl"].values,
            volumes=df["fees"].values,
            fees=df["rate"].values,
            liquidity=df["liquidity"].values,
            time=df["date"].values,
            prices=df["price"].values
        ) for df in self._simulated_data]

    def load(self):
        self._load(self._file_id)

    def read(self, with_run: bool = False) -> List[LPSimulatedStates]:
        if with_run:
            self.run()
        return self._read(self._file_id)

    def delete_dump_file(self):
        Path(self.file_path(self._file_id)).unlink(missing_ok=True)
