import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
from uuid import uuid4

from loguru import logger


class BaseLogger(ABC):

    def __init__(self, base_artifacts_path: Optional[str] = None, class_name: str = None):
        self._id: str = str(uuid4())
        if class_name is None:
            class_name = self.__class__.__name__
        self._init_base_path(base_artifacts_path=base_artifacts_path, class_name=class_name)
        self._setup_logger()

    @abstractmethod
    def _init_base_path(self, *args, base_artifacts_path: Optional[str] = None, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _setup_logger(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def debug(self, message: str):
        raise NotImplementedError

    @property
    @abstractmethod
    def base_artifacts_path(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def logs_path(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def datasets_path(self) -> str:
        raise NotImplementedError


class DefaultLogger(BaseLogger):

    def _init_base_path(self, base_artifacts_path: Optional[str] = None, class_name: str = None):
        if base_artifacts_path is None:
            # Honor an explicit run-output root if set; otherwise fall back
            # to cwd. ``PYTHONPATH`` is a colon-separated import list,
            # not a directory, so it is intentionally not consulted here.
            base_path: str = os.getenv("FRACTAL_RUNS_PATH") or os.getcwd()
            path = Path(f'{base_path}/runs/{class_name}/{self._id}')
            path.mkdir(parents=True, exist_ok=True)
            base_artifacts_path = path.absolute().as_posix()
        self._base_artifacts_path = base_artifacts_path

    def _setup_logger(self):
        """Add a per-instance loguru sink that writes only this run's logs.

        Earlier versions called the process-wide ``logger.remove()`` here,
        which wiped every other sink in the interpreter (parallel runs,
        host-app sinks). Now we keep the global logger untouched and store
        the handler id so :meth:`close` can drop only our own sink.
        """
        logs_base_path: str = self.logs_path
        loggformat = "{time:YYYY-MM-DD HH:mm:ss} | {message}"
        run_id = self._id
        self._handler_id: int = logger.add(
            f'{logs_base_path}/logs.log',
            format=loggformat,
            level="DEBUG",
            filter=lambda record: record["extra"].get("strategy_run_id") == run_id,
        )
        self._logger = logger.bind(strategy_run_id=self._id)

    def close(self) -> None:
        """Remove only this instance's sink. Idempotent."""
        hid = getattr(self, "_handler_id", None)
        if hid is None:
            return
        try:
            logger.remove(hid)
        except ValueError:
            # Already removed (e.g. by a prior close()).
            pass
        self._handler_id = None

    @property
    def base_artifacts_path(self) -> str:
        return self._base_artifacts_path

    @property
    def logs_path(self) -> str:
        return f'{self.base_artifacts_path}/logs'

    @property
    def datasets_path(self) -> str:
        return f'{self.base_artifacts_path}/datasets'

    def debug(self, message: str):
        """
        Log a debug message.
        """
        self._logger.debug(message)
