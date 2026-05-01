"""Fractal — open-source Python research library for DeFi strategies.

The package version exposed here is read from the installed pip
metadata via :mod:`importlib.metadata`, so there's a single source of
truth (``setup.py::version``) and no string drift between
``setup.py``, ``CHANGELOG.md``, and ``fractal.__version__``.

Idiomatic — matches the convention used by numpy / pandas / mlflow:

    >>> import fractal
    >>> fractal.__version__
    'x.y.z'
"""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("fractal-defi")
except PackageNotFoundError:  # pragma: no cover
    # Source checkout that wasn't ``pip install``-ed (no metadata yet).
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
