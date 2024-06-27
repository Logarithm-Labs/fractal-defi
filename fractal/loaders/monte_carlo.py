import random
from pathlib import Path
from typing import List
from uuid import uuid4

import numpy as np

from fractal.loaders.loader import Loader, LoaderType
from fractal.loaders.structs import PriceHistory


class MonteCarloHourPriceLoader(Loader):
    """
    A class that represents a Monte Carlo hour price loader.

    This loader performs Monte Carlo simulation on the price data,
    obtained from a base loader.

    Generates prices using a GBM process with volatility,
        and without drift.
    Length of the simulation is equal to the length of the base loader data.

    Attributes:
        price_history (PriceHistory): The price history data.
        trajectories_number (int): The number of trajectories to simulate.
        loader_type (LoaderType): The type of loader used to save and load data.
        seed (int): The seed value used for random number generation.

    Methods:
        extract(): Extracts the price data from the base loader.
        transform(): Performs Monte Carlo simulation on the price data.
        load(): Saves the simulated price data using the specified loader type.
        read(with_run: bool = False): Reads the simulated
            price data from the saved file.
        run(): Executes the entire process of extracting,
            transforming, and loading the data.
    """

    def __init__(
        self,
        price_history: PriceHistory,
        trajectories_number: int = 100,
        loader_type: LoaderType = LoaderType.PICKLE,
        seed: int = 420,
    ) -> None:
        """
        Initializes a new instance of the MonteCarloHourPriceLoader class.

        Args:
            price_history (PriceHistory): The price history data.
            trajectories_number (int, optional): The number of
                trajectories to simulate. Defaults to 100.
            loader_type (LoaderType, optional): The type of loader used to
                save and load data. Defaults to LoaderType.PICKLE.
            seed (int, optional): The seed value used for random number
                generation. Defaults to 420.
        """
        super().__init__(loader_type)
        self._data = None
        self.trajectories_number = trajectories_number
        self.price_history = price_history
        self._file_id = str(uuid4())
        self._random = random.Random()
        self._random.seed(seed)

    def extract(self):
        self._data = self.price_history

    def transform(self):
        # Monte Carlo Simulation
        std = self._data["price"].pct_change().std()
        price_0 = self._data.iloc[0]["price"]
        prices_simulations: List[PriceHistory] = []
        for _ in range(self.trajectories_number):
            prices = []
            price = price_0
            for _ in range(len(self._data)):
                price *= 1 + self._random.normalvariate(mu=0, sigma=std)
                prices.append(price)
            self._data["price"] = np.array(prices)
            prices_simulations.append(self._data.copy())
        self._data = prices_simulations

    def load(self):
        self._load(self._file_id)

    def read(self, with_run: bool = False) -> List[PriceHistory]:
        if with_run:
            self.run()
        return self._read(self._file_id)

    def delete_dump_file(self):
        Path(self.file_path(self._file_id)).unlink(missing_ok=True)
