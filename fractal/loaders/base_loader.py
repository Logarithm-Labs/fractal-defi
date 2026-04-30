"""Base ``Loader`` class — every concrete loader inherits from this.

Lifecycle: ``extract → transform → load`` via ``run()``. ``read(with_run=True)``
runs the pipeline; ``read()`` reads the on-disk cache. Cache files live under
``<DATA_PATH or PYTHONPATH or cwd>/fractal_data/<loader_class>/<key>.<ext>``.

Subclasses MUST override:
  - ``extract``, ``transform`` — populate ``self._data``;
  - ``read(with_run)`` — return one of the typed structs from ``structs.py``.

Subclasses SHOULD override ``_cache_key`` so the on-disk filename reflects
all parameters that affect the dump (ticker, interval, time window, …).
The default implementation uses the empty string and is only safe for
single-shot loaders such as the constant-funding stub.
"""
import os
import pickle
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

import pandas as pd


class LoaderType(Enum):
    CSV = 1
    JSON = 2
    SQL = 3
    PICKLE = 4


class Loader(ABC):
    def __init__(self, *args, loader_type: LoaderType = LoaderType.CSV, **kwargs) -> None:
        if not isinstance(loader_type, LoaderType):
            raise ValueError(f"Loader type {loader_type} not supported")
        self.loader_type: LoaderType = loader_type
        self._data: Any = None

        base_path: str = os.getenv("DATA_PATH") or os.getenv("PYTHONPATH") or os.getcwd()
        self._base_path: str = os.path.join(base_path, "fractal_data")

    # ------------------------------------------------------------------ ABC
    @abstractmethod
    def extract(self) -> None:
        ...

    @abstractmethod
    def transform(self) -> None:
        ...

    def load(self) -> None:
        """Default: write ``self._data`` to disk under the configured loader type."""
        self._load(self._cache_key())

    @abstractmethod
    def read(self, with_run: bool = False):
        ...

    def _cache_key(self) -> str:
        """Filename stem used for cache. Override in subclasses."""
        return "data"

    # --------------------------------------------------------------- helpers
    def file_path(self, *args: str) -> str:
        file_name = "_".join(args)
        return os.path.join(self._base_path, self.__class__.__name__.lower(), file_name)

    def _load(self, *args: str) -> None:
        if self._data is None:
            raise ValueError("No data to save. Please ensure data is loaded and transformed.")

        file_name = "_".join(args)
        directory = os.path.join(self._base_path, self.__class__.__name__.lower())
        os.makedirs(directory, exist_ok=True)
        path_name = os.path.join(directory, file_name)

        if self.loader_type == LoaderType.CSV:
            if not isinstance(self._data, pd.DataFrame):
                raise TypeError(
                    f"CSV loader requires self._data to be a DataFrame, got {type(self._data)}"
                )
            self._data.to_csv(f"{path_name}.csv")
        elif self.loader_type == LoaderType.JSON:
            if not isinstance(self._data, pd.DataFrame):
                raise TypeError(
                    f"JSON loader requires self._data to be a DataFrame, got {type(self._data)}"
                )
            self._data.to_json(f"{path_name}.json", orient="records")
        elif self.loader_type == LoaderType.PICKLE:
            # pickle handles arbitrary Python objects, including list-of-DataFrame
            with open(f"{path_name}.pkl", "wb") as fh:
                pickle.dump(self._data, fh)
        elif self.loader_type == LoaderType.SQL:
            raise NotImplementedError("SQL loader not implemented")
        else:
            raise ValueError(f"Loader type {self.loader_type} not supported")

    def _read(self, *args) -> None:
        file_path: str = self.file_path(*args)
        if self.loader_type == LoaderType.CSV:
            self._data = pd.read_csv(f"{file_path}.csv")
        elif self.loader_type == LoaderType.JSON:
            self._data = pd.read_json(f"{file_path}.json", orient="records")
        elif self.loader_type == LoaderType.PICKLE:
            with open(f"{file_path}.pkl", "rb") as fh:
                self._data = pickle.load(fh)
        elif self.loader_type == LoaderType.SQL:
            raise NotImplementedError("SQL loader not implemented")
        else:
            raise ValueError(f"Loader type {self.loader_type} not supported")

    def run(self) -> None:
        """Execute the full pipeline: extract → transform → load."""
        self.extract()
        self.transform()
        self.load()
