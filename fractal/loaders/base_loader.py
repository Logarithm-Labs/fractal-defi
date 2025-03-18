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

    def __init__(self, loader_type: LoaderType, *args, **kwargs) -> None:
        if loader_type not in LoaderType:
            raise ValueError(f"Loader type {loader_type} not supported")
        self.loader_type: LoaderType = loader_type
        self._data: pd.DataFrame = None
        base_path: str = os.getenv('DATA_PATH', '')
        if base_path == '':
            base_path = os.getenv('PYTHONPATH', '')
            if base_path == '':
                base_path = os.getcwd()
        self.__base_path: str = base_path + 'fractal_data/'

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

    def file_path(self, *args):
        file_name = '_'.join(args)
        return self.__base_path + self.__class__.__name__.lower() + '/' + file_name

    def _load(self, *args):
        file_name = '_'.join(args)
        directory = f'{self.__base_path}/{self.__class__.__name__.lower()}'
        if not os.path.exists(directory):
            os.makedirs(directory)
        path_name = f'{directory}/{file_name}'
        if self.loader_type == LoaderType.CSV:
            self._data.to_csv(f'{path_name}.csv')
            return
        if self.loader_type == LoaderType.JSON:
            self._data.to_json(f'{path_name}.json', orient='records')
            return
        if self.loader_type == LoaderType.SQL:
            raise NotImplementedError("SQL loader not implemented")
        if self.loader_type == LoaderType.PICKLE:
            self._data.to_pickle(f'{path_name}.pkl')
            return
        raise ValueError(f"Loader type {self.loader_type} not supported")

    def _read(self, *args):
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

    def run(self):
        self.extract()
        self.transform()
        self.load()
