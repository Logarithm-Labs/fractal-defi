"""Monte-Carlo price-trajectory simulator (GBM + bootstrap).

Two simulation modes:

* ``gbm`` — geometric Brownian motion in exact log form,
  ``S_{t+1} = S_t · exp((μ − σ²/2) + σ · Z)``, where ``Z ~ N(0, 1)``.
  Trajectories are strictly positive (cannot cross zero) and form a
  martingale when ``μ = 0``.

* ``bootstrap`` — resamples log-returns from the historical empirical
  distribution with replacement. Preserves fat tails / skew that the
  Gaussian σ can't capture, which matters for crypto.

Calibration: ``σ`` is the std of historical log-returns
(``np.diff(np.log(price))``) by default. Pass ``sigma`` explicitly to
override. ``μ`` is the per-step drift (default ``0`` — drift-free, the
standard stress-testing convention).

Output: a :data:`TrajectoryBundle` (``List[PriceHistory]``) of length
``trajectories_number``, each with the same length and ``DatetimeIndex``
as the (optionally sliced) input history. The first value of every
trajectory equals the historical first price (``S_0``); subsequent
values are the simulation steps.

Loader contract notes:

* Cache key is **deterministic** in ``(history hash, mode, μ, σ, seed,
  n_traj, window)`` — two instances with identical inputs share the
  same on-disk pickle, just like other loaders.
* ``start_time`` / ``end_time`` slice the *input* history before
  calibration and simulation. This mirrors the windowing semantics of
  HTTP-based loaders.
* ``LoaderType.PICKLE`` is required because the payload is a list of
  DataFrames and won't round-trip through CSV / JSON.
"""
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

import numpy as np

from fractal.loaders._dt import to_seconds, to_utc
from fractal.loaders.base_loader import Loader, LoaderType
from fractal.loaders.structs import PriceHistory, TrajectoryBundle

Mode = Literal["gbm", "bootstrap"]


class MonteCarloPriceLoader(Loader):
    """Reproducible Monte-Carlo simulator over a historical price series."""

    def __init__(
        self,
        price_history: PriceHistory,
        trajectories_number: int = 100,
        mu: float = 0.0,
        sigma: Optional[float] = None,
        mode: Mode = "gbm",
        loader_type: LoaderType = LoaderType.PICKLE,
        seed: int = 420,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> None:
        # ``loader_type`` is keyword-only on the ABC (after ``*args``) — must
        # pass it by name, not positionally, otherwise it lands in ``*args``
        # and ``self.loader_type`` silently falls back to ``LoaderType.CSV``.
        super().__init__(loader_type=loader_type)
        if loader_type != LoaderType.PICKLE:
            raise ValueError(
                "MonteCarloPriceLoader requires LoaderType.PICKLE because it "
                "dumps a list of DataFrames."
            )
        if mode not in ("gbm", "bootstrap"):
            raise ValueError(f"Unknown mode {mode!r}; expected 'gbm' or 'bootstrap'.")
        self.price_history: PriceHistory = price_history
        self.trajectories_number: int = int(trajectories_number)
        self.mu: float = float(mu)
        self._sigma_override: Optional[float] = None if sigma is None else float(sigma)
        self.mode: Mode = mode
        self.seed: int = int(seed)
        self.start_time: Optional[datetime] = to_utc(start_time)
        self.end_time: Optional[datetime] = to_utc(end_time)
        self._rng = np.random.default_rng(self.seed)
        self._trajectories: TrajectoryBundle = []
        self._sigma: float = 0.0  # populated by transform()

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _log_returns(prices: np.ndarray) -> np.ndarray:
        prices = np.asarray(prices, dtype=float)
        if prices.size < 2:
            return np.array([], dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_ret = np.diff(np.log(prices))
        return log_ret[np.isfinite(log_ret)]

    @classmethod
    def _calibrate_sigma(cls, prices: np.ndarray) -> float:
        log_ret = cls._log_returns(prices)
        if log_ret.size < 2:
            return 0.0
        sigma = float(np.std(log_ret, ddof=1))
        return sigma if np.isfinite(sigma) else 0.0

    def _windowed_history(self) -> PriceHistory:
        """Slice ``self.price_history`` to the requested window, if any."""
        h = self.price_history
        if h is None or h.empty:
            return h
        idx = h.index
        mask = np.ones(len(h), dtype=bool)
        if self.start_time is not None:
            mask &= idx >= self.start_time
        if self.end_time is not None:
            mask &= idx <= self.end_time
        if mask.all():
            return h
        return h.loc[mask]

    def _gbm_paths(self, s0: float, sigma: float, n: int, m: int) -> np.ndarray:
        """Vectorized GBM trajectories of shape ``(m, n)``; column 0 is ``s0``."""
        if n <= 0:
            return np.empty((m, 0), dtype=float)
        if n == 1:
            return np.full((m, 1), s0, dtype=float)
        Z = self._rng.standard_normal(size=(m, n - 1))
        log_inc = (self.mu - 0.5 * sigma * sigma) + sigma * Z
        log_paths = np.concatenate(
            [np.zeros((m, 1)), np.cumsum(log_inc, axis=1)], axis=1
        )
        return s0 * np.exp(log_paths)

    def _bootstrap_paths(
        self, s0: float, log_returns: np.ndarray, n: int, m: int
    ) -> np.ndarray:
        if n <= 0:
            return np.empty((m, 0), dtype=float)
        if n == 1 or log_returns.size == 0:
            return np.full((m, n), s0, dtype=float)
        samples = self._rng.choice(log_returns, size=(m, n - 1), replace=True)
        log_paths = np.concatenate(
            [np.zeros((m, 1)), np.cumsum(samples, axis=1)], axis=1
        )
        return s0 * np.exp(log_paths)

    # ------------------------------------------------------- cache key
    def _history_fingerprint(self) -> str:
        h = self.price_history
        if h is None or h.empty:
            return "empty"
        # Hash the prices and the index timestamps so two histories that
        # differ in shape OR values produce different cache files.
        prices = np.ascontiguousarray(h["price"].astype(float).values)
        ts = np.ascontiguousarray(h.index.asi8)  # int64 ns since epoch
        digest = hashlib.sha1()
        digest.update(prices.tobytes())
        digest.update(ts.tobytes())
        return digest.hexdigest()[:12]

    def _cache_key(self) -> str:
        sigma_part = "auto" if self._sigma_override is None else f"{self._sigma_override:.10g}"
        s = to_seconds(self.start_time) if self.start_time is not None else "open"
        e = to_seconds(self.end_time) if self.end_time is not None else "close"
        return (
            f"{self._history_fingerprint()}-{self.mode}-mu{self.mu:.10g}-"
            f"sigma{sigma_part}-seed{self.seed}-n{self.trajectories_number}-"
            f"win{s}-{e}"
        )

    # ------------------------------------------------------------ lifecycle
    def extract(self) -> None:
        # Source data is the user-provided history; nothing to fetch.
        pass

    def transform(self) -> None:
        h = self._windowed_history()
        if h is None or h.empty:
            self._trajectories = []
            self._data = []
            return
        prices = h["price"].astype(float).values
        s0 = float(prices[0])
        n = len(h)
        m = self.trajectories_number
        index = h.index

        log_ret = self._log_returns(prices)
        if self._sigma_override is not None:
            self._sigma = self._sigma_override
        else:
            self._sigma = self._calibrate_sigma(prices)

        if self.mode == "gbm":
            paths = self._gbm_paths(s0, self._sigma, n, m)
        else:  # bootstrap
            paths = self._bootstrap_paths(s0, log_ret, n, m)

        self._trajectories = [
            PriceHistory(prices=paths[k], time=index.values) for k in range(m)
        ]
        self._data = self._trajectories

    def load(self) -> None:
        if self._data is None:
            self._data = self._trajectories
        self._load(self._cache_key())

    def read(self, with_run: bool = False) -> TrajectoryBundle:
        if with_run:
            self.run()
        else:
            self._read(self._cache_key())
            self._trajectories = self._data
        return self._trajectories

    def delete_dump_file(self) -> None:
        Path(self.file_path(self._cache_key()) + ".pkl").unlink(missing_ok=True)

    # ----------------------------------------------------------- properties
    @property
    def calibrated_sigma(self) -> float:
        """The σ actually used for the last :meth:`transform` call."""
        return self._sigma
