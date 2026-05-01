"""Tests for :class:`DefaultLogger`."""
import os

import pytest

from fractal.core.base.strategy.logger import DefaultLogger


@pytest.mark.core
def test_default_logger_creates_paths(tmp_path):
    logger = DefaultLogger(base_artifacts_path=str(tmp_path), class_name="TestStrategy")
    assert os.path.isdir(logger.base_artifacts_path)
    # logs path is derived under base
    assert logger.logs_path.startswith(logger.base_artifacts_path)
    assert logger.datasets_path.startswith(logger.base_artifacts_path)


@pytest.mark.core
def test_default_logger_base_path_falls_back_to_cwd(monkeypatch, tmp_path):
    """When ``base_artifacts_path=None`` and ``PYTHONPATH`` is unset the
    logger should anchor under cwd, not under the empty string."""
    monkeypatch.delenv("PYTHONPATH", raising=False)
    monkeypatch.chdir(tmp_path)
    logger = DefaultLogger(class_name="UnderCwd")
    assert os.path.isdir(logger.base_artifacts_path)
    # Path must be under tmp_path (cwd), not empty/relative-to-system-root.
    assert str(tmp_path) in logger.base_artifacts_path


@pytest.mark.core
def test_default_logger_debug_writes_log_file(tmp_path):
    logger = DefaultLogger(base_artifacts_path=str(tmp_path), class_name="WriteTest")
    logger.debug("hello-from-test")
    # loguru flushes synchronously; file should contain the message.
    log_files = [p for p in os.listdir(logger.logs_path) if p.endswith(".log")]
    assert log_files, "no log file written"
    log_file_path = os.path.join(logger.logs_path, log_files[0])
    with open(log_file_path) as fh:
        content = fh.read()
    assert "hello-from-test" in content
