import os
from abc import ABC, abstractmethod
from enum import Enum

import pandas as pd


class LoaderType(Enum):
    CSV = 1
    JSON = 2
    SQL = 3
    PICKLE = 4


class Loader(ABC):
    """
    Abstract base class for loaders that handle data extraction, transformation, and loading.
    This class provides a common interface for different data loaders, allowing for easy
    integration and extension.

    It follows ETL (Extract, Transform, Load) principles, where each loader is responsible for
    its own data source and format. The loader type can be specified to determine the format
    in which the data will be saved or read.
    """
    def __init__(self, loader_type: LoaderType = LoaderType.CSV, *args, **kwargs) -> None:
        if not isinstance(loader_type, LoaderType):
            raise ValueError(f"Loader type {loader_type} not supported")
        self.loader_type: LoaderType = loader_type
        self._data: pd.DataFrame = None

        # Determine base path from environment or current working directory
        base_path: str = os.getenv('DATA_PATH') or os.getenv('PYTHONPATH') or os.getcwd()
        self._base_path: str = os.path.join(base_path, 'fractal_data')

    @abstractmethod
    def extract(self):
        pass

    @abstractmethod
    def transform(self):
        pass

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def read(self, with_run: bool = False) -> pd.DataFrame:
        pass

    def file_path(self, *args: str) -> str:
        """Build a file path using the base path, loader name, and given arguments."""
        file_name = '_'.join(args)
        return os.path.join(self._base_path, self.__class__.__name__.lower(), file_name)

    def _load(self, *args: str) -> None:
        """
        Save data to the specified file format based on LoaderType.
        """
        if self._data is None:
            raise ValueError("No data to save. Please ensure data is loaded and transformed.")

        file_name = '_'.join(args)
        directory = os.path.join(self._base_path, self.__class__.__name__.lower())
        os.makedirs(directory, exist_ok=True)
        path_name = os.path.join(directory, file_name)

        if self.loader_type == LoaderType.CSV:
            self._data.to_csv(f'{path_name}.csv')
        elif self.loader_type == LoaderType.JSON:
            self._data.to_json(f'{path_name}.json', orient='records')
        elif self.loader_type == LoaderType.SQL:
            raise NotImplementedError("SQL loader not implemented")
        elif self.loader_type == LoaderType.PICKLE:
            self._data.to_pickle(f'{path_name}.pkl')
        else:
            raise ValueError(f"Loader type {self.loader_type} not supported")

    def _read(self, *args) -> None:
        """
        Read data from the specified file format based on LoaderType.
        """
        file_path: str = self.file_path(*args)
        if self.loader_type == LoaderType.CSV:
            self._data = pd.read_csv(f'{file_path}.csv')
        elif self.loader_type == LoaderType.JSON:
            self._data = pd.read_json(f'{file_path}.json', orient='records')
        elif self.loader_type == LoaderType.SQL:
            raise NotImplementedError("SQL loader not implemented")
        elif self.loader_type == LoaderType.PICKLE:
            self._data = pd.read_pickle(f'{file_path}.pkl')
        else:
            raise ValueError(f"Loader type {self.loader_type} not supported")

    def run(self) -> None:
        """
        Execute the full loading process: extract, transform, and save the data.
        """
        self.extract()
        self.transform()
        self.load()
