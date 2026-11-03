"""Moving-block bootstrap confidence intervals for daily series.

Daily strategy P&L, returns and rate spreads are autocorrelated, so an i.i.d.
bootstrap understates their sampling variance. The moving-block bootstrap
(Künsch, 1989) resamples whole runs of consecutive observations instead:
each replicate is built from ``ceil(n / L)`` blocks of length ``L`` whose
starts are drawn uniformly from ``0 .. n - L`` and is truncated to ``n``
observations. Serial dependence shorter than ``L`` survives the resampling.

The right ``L`` is rarely known, so :func:`block_bootstrap` evaluates several
block lengths and reports the **widest** of their percentile intervals as the
headline CI (a conservative choice; the per-length intervals are kept). One
uniform draw matrix drives every block length (common random numbers): with
the same seed, a given series length and block length always yields the same
indices, whichever other block lengths are requested. Several aligned series
(the columns of a ``DataFrame`` or of a 2-D array) are resampled **jointly**
with the same day indices, which preserves their cross-correlation.

Example — a CI on the annualised mean daily return of a backtest::

    df = result.to_dataframe().set_index("timestamp")
    daily = df["net_balance"].resample("1D").last().pct_change().dropna()
    ci = block_bootstrap(daily, statistic=lambda r: 365 * r.mean(), seed=7)
    ci.lower, ci.upper, ci.block_length

Intended for daily-scale series: the draw matrix holds
``n_resamples * n`` floats.
"""
import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from numbers import Integral

import numpy as np
import pandas as pd

__all__ = [
    "DEFAULT_BLOCK_LENGTHS",
    "DEFAULT_N_RESAMPLES",
    "BootstrapCI",
    "block_bootstrap",
    "block_bootstrap_indices",
    "percentile_ci",
    "widest_block_ci",
]

DEFAULT_BLOCK_LENGTHS: tuple[int, ...] = (3, 7, 14)
DEFAULT_N_RESAMPLES = 2000

SeedLike = int | np.random.Generator | np.random.SeedSequence | None
Statistic = Callable[[np.ndarray], float]


@dataclass(frozen=True, eq=False)
class BootstrapCI:
    """Percentile CIs per block length and the widest of them.

    Attributes:
        estimate: The statistic on the full sample (``nan`` when unknown).
        lower: Lower bound of the widest per-block interval.
        upper: Upper bound of the widest per-block interval.
        block_length: The requested block length whose interval is widest
            (ties go to the first one requested).
        level: Two-sided confidence level, e.g. ``0.95``.
        n_resamples: Largest number of replicates of any block length.
        per_block: ``{block_length: (lower, upper)}`` for every length.
        replicates: ``{block_length: replicate array}`` (non-finite
            replicates included, as produced).
    """
    estimate: float
    lower: float
    upper: float
    block_length: int
    level: float
    n_resamples: int
    per_block: dict[int, tuple[float, float]]
    replicates: dict[int, np.ndarray] = field(default_factory=dict, repr=False)

    @property
    def width(self) -> float:
        """``upper - lower`` (``nan`` when a bound is undefined)."""
        return self.upper - self.lower

    def contains(self, value: float) -> bool:
        """Whether ``value`` lies in ``[lower, upper]``; ``False`` for an undefined interval."""
        return bool(self.lower <= value <= self.upper)


def _check_level(level: float) -> float:
    level = float(level)
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1); got {level}")
    return level


def _block_lengths(block_lengths: int | Iterable[int]) -> tuple[int, ...]:
    """Validate block lengths; a single int is allowed; duplicates are dropped (first kept)."""
    raw = (block_lengths,) if isinstance(block_lengths, Integral) else tuple(block_lengths)
    if not raw:
        raise ValueError("block_lengths must not be empty")
    out: list[int] = []
    for block in raw:
        if isinstance(block, bool) or not isinstance(block, Integral):
            raise TypeError(f"block lengths must be integers; got {block!r}")
        if block < 1:
            raise ValueError(f"block lengths must be >= 1; got {block}")
        if int(block) not in out:
            out.append(int(block))
    return tuple(out)


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer; got {value!r}")
    return int(value)


def _indices_from_uniform(uniform: np.ndarray, n: int, block: int) -> np.ndarray:
    """Moving-block row indices for one block length from the shared uniform draws."""
    length = min(block, n)
    n_blocks = math.ceil(n / length)
    starts = np.floor(uniform[:, :n_blocks] * (n - length + 1)).astype(np.int64)
    rows = starts[:, :, None] + np.arange(length, dtype=np.int64)[None, None, :]
    return rows.reshape(uniform.shape[0], -1)[:, :n]


def block_bootstrap_indices(
    n: int,
    block_lengths: int | Iterable[int] = DEFAULT_BLOCK_LENGTHS,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    seed: SeedLike = None,
) -> dict[int, np.ndarray]:
    """Moving-block resampling indices, one ``(n_resamples, n)`` array per block length.

    One uniform matrix ``U`` of shape ``(n_resamples, n)`` is drawn from
    ``seed``; for block length ``L`` the block starts of replicate ``r`` are
    ``floor(U[r, :ceil(n / L)] * (n - L + 1))`` and each block covers
    ``start .. start + L - 1``. Rows are truncated to ``n`` indices.

    Edge cases: ``L = 1`` is the i.i.d. bootstrap; ``L >= n`` is clamped to
    ``n``, where the only block is the whole series, so every replicate is
    the original order (a zero-width interval — the series is too short for
    that block length). Duplicate lengths are dropped.

    Args:
        n: Number of observations (>= 1).
        block_lengths: One block length or several (each >= 1).
        n_resamples: Replicates per block length (>= 1).
        seed: ``int``, ``numpy.random.Generator``, ``SeedSequence`` or
            ``None`` (fresh entropy). A ``Generator`` is advanced.

    Returns:
        ``{block_length: int64 array of shape (n_resamples, n)}`` keyed by the
        requested (unclamped) lengths, in request order.
    """
    n = _positive_int(n, "n")
    n_resamples = _positive_int(n_resamples, "n_resamples")
    blocks = _block_lengths(block_lengths)
    uniform = np.random.default_rng(seed).random((n_resamples, n))
    return {block: _indices_from_uniform(uniform, n, block) for block in blocks}


def percentile_ci(replicates: Iterable[float] | np.ndarray, level: float = 0.95) -> tuple[float, float]:
    """Two-sided percentile interval of bootstrap replicates.

    Non-finite replicates (an undefined ratio, say) are dropped; with none
    left the interval is ``(nan, nan)``.
    """
    alpha = (1.0 - _check_level(level)) / 2.0
    draws = np.asarray(replicates, dtype=float).ravel()
    draws = draws[np.isfinite(draws)]
    if draws.size == 0:
        return math.nan, math.nan
    return float(np.quantile(draws, alpha)), float(np.quantile(draws, 1.0 - alpha))


def widest_block_ci(
    replicates_by_block: Mapping[int, Iterable[float] | np.ndarray],
    level: float = 0.95,
    estimate: float = math.nan,
) -> BootstrapCI:
    """Percentile CI per block length and the widest of them.

    Useful when the replicates come from a custom resampling loop driven by
    :func:`block_bootstrap_indices`. An undefined interval (no finite
    replicate) counts as infinitely wide, so it is selected and its ``nan``
    bounds surface instead of a falsely narrow interval. Ties go to the first
    block length in mapping order.
    """
    level = _check_level(level)
    if not replicates_by_block:
        raise ValueError("replicates_by_block must not be empty")
    replicates = {int(b): np.asarray(d, dtype=float).ravel() for b, d in replicates_by_block.items()}
    per_block = {b: percentile_ci(d, level) for b, d in replicates.items()}

    def width(block: int) -> float:
        lower, upper = per_block[block]
        span = upper - lower
        return span if math.isfinite(span) else math.inf

    widest = max(per_block, key=width)
    lower, upper = per_block[widest]
    return BootstrapCI(
        estimate=float(estimate),
        lower=lower,
        upper=upper,
        block_length=widest,
        level=level,
        n_resamples=max(d.size for d in replicates.values()),
        per_block=per_block,
        replicates=replicates,
    )


def _as_values(data: pd.Series | pd.DataFrame | np.ndarray | Iterable[float]) -> np.ndarray:
    """Time-ordered float values; rows with any NaN are dropped (jointly across columns)."""
    if isinstance(data, (pd.Series, pd.DataFrame)):
        frame = data.sort_index()
        values = frame.to_numpy(dtype=float)
    else:
        values = np.asarray(data, dtype=float)
    if values.ndim not in (1, 2):
        raise ValueError(f"data must be 1-D or 2-D; got {values.ndim}-D")
    keep = ~np.isnan(values) if values.ndim == 1 else ~np.isnan(values).any(axis=1)
    values = values[keep]
    if values.shape[0] == 0:
        raise ValueError("data has no complete (non-NaN) observations")
    return values


def _scalar(value, where: str) -> float:
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 0:
        raise ValueError(f"statistic must return a scalar; got shape {arr.shape} on {where}")
    return float(arr)


def block_bootstrap(
    data: pd.Series | pd.DataFrame | np.ndarray | Iterable[float],
    statistic: Statistic | None = None,
    block_lengths: int | Iterable[int] = DEFAULT_BLOCK_LENGTHS,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    level: float = 0.95,
    seed: SeedLike = None,
) -> BootstrapCI:
    """Moving-block bootstrap CI of a statistic of one or several aligned series.

    Args:
        data: A ``Series`` (sorted by its index), a ``DataFrame`` whose columns
            are resampled jointly (same rows in every replicate), or a 1-D /
            2-D array whose rows are observations in time order. Rows with a
            NaN in any column are dropped before resampling.
        statistic: Maps the resampled values — a 1-D array, or an
            ``(n, k)`` array with the columns in input order — to a float.
            ``None`` means the mean of a 1-D series (vectorised); 2-D data
            needs an explicit statistic.
        block_lengths: Block lengths to evaluate (see
            :func:`block_bootstrap_indices` for the edge cases).
        n_resamples: Replicates per block length.
        level: Two-sided confidence level in ``(0, 1)``.
        seed: Seed or ``numpy.random.Generator``; the same seed gives the same
            indices for every series of the same length, so separate calls on
            aligned series stay jointly resampled.

    Returns:
        :class:`BootstrapCI` with the full-sample estimate, the widest
        per-block percentile interval and the per-block intervals. Replicates
        where the statistic is non-finite are left out of the quantiles.
    """
    level = _check_level(level)
    values = _as_values(data)
    if statistic is None and values.ndim != 1:
        raise ValueError("2-D data needs an explicit statistic")
    indices = block_bootstrap_indices(values.shape[0], block_lengths, n_resamples, seed)
    replicates: dict[int, np.ndarray] = {}
    for block, rows in indices.items():
        if statistic is None:
            replicates[block] = values[rows].mean(axis=1)
        else:
            replicates[block] = np.array(
                [_scalar(statistic(values[row]), f"block length {block}") for row in rows], dtype=float
            )
    estimate = float(values.mean()) if statistic is None else _scalar(statistic(values), "the full sample")
    return widest_block_ci(replicates, level, estimate)
