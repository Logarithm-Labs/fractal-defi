"""Pure helpers for realised bridge fees.

``fee_bps`` turns one transfer's input and output amounts into a realised
fee, ``fee_stats`` summarises many of them, and ``fit_fee_schedule`` fits the
affine per-route schedule ``fee = fixed + bps * size / 1e4`` to observed
transfers.
"""

import math
import statistics
from collections.abc import Iterable
from typing import NamedTuple

import numpy as np

_DECIMAL_RESCALES = (1.0, 1e12, 1e-12)


def comparable_input(input_amount: float, output_amount: float) -> float | None:
    """``input_amount`` in output-token units when both legs carry the same asset.

    Origin tokens may use 18 decimals against 6 on the destination (or the
    reverse), so a ratio within ``[0.5, 2]`` after a 1, 1e12 or 1e-12 rescale
    is accepted. Anything else (a swap, or a non-positive amount) returns
    ``None``.
    """
    if input_amount <= 0 or output_amount <= 0:
        return None
    ratio = input_amount / output_amount
    for scale in _DECIMAL_RESCALES:
        if 0.5 <= ratio / scale <= 2.0:
            return input_amount / scale
    return None


def fee_bps(input_amount: float, output_amount: float, max_abs_bps: float = 500.0) -> float | None:
    """Realised fee ``(input - output) / input`` in basis points.

    Returns ``None`` when the legs are not comparable (see
    :func:`comparable_input`) or the absolute fee exceeds ``max_abs_bps``.
    A negative value is a rebate.
    """
    comparable = comparable_input(input_amount, output_amount)
    if comparable is None:
        return None
    bps = (comparable - output_amount) / comparable * 1e4
    return bps if abs(bps) <= max_abs_bps else None


def _quantile(sorted_values: list[float], q: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = q * (len(sorted_values) - 1)
    low = math.floor(position)
    high = min(low + 1, len(sorted_values) - 1)
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (position - low)


def fee_stats(pairs: Iterable[tuple[float | None, float]]) -> dict[str, float | int | None]:
    """Median, quartiles and volume-weighted mean of ``(fee_bps, volume)`` pairs.

    Pairs whose fee is ``None`` are skipped. Quartiles use linear
    interpolation. ``vw_mean_bps`` is ``None`` when the total volume is not
    positive.
    """
    kept = [(fee, volume) for fee, volume in pairs if fee is not None]
    if not kept:
        return {"n": 0, "median_bps": None, "p25_bps": None, "p75_bps": None, "vw_mean_bps": None}
    fees = sorted(fee for fee, _ in kept)
    total_volume = sum(volume for _, volume in kept)
    return {
        "n": len(kept),
        "median_bps": statistics.median(fees),
        "p25_bps": _quantile(fees, 0.25),
        "p75_bps": _quantile(fees, 0.75),
        "vw_mean_bps": sum(fee * volume for fee, volume in kept) / total_volume if total_volume > 0 else None,
    }


class FeeSchedule(NamedTuple):
    """Affine fee schedule: ``fee = fixed + bps * size / 1e4`` (same units as ``size``)."""

    fixed: float
    bps: float

    def fee(self, size: float) -> float:
        """Fee for a transfer of ``size``."""
        return self.fixed + self.bps * size / 1e4

    def fee_bps_at(self, size: float) -> float:
        """All-in fee for a transfer of ``size``, in basis points of ``size``."""
        return self.fee(size) / size * 1e4


def fit_fee_schedule(amounts: Iterable[float], fees: Iterable[float]) -> FeeSchedule:
    """Least-squares ``fee = fixed + bps * amount / 1e4`` with both parts ``>= 0``.

    Pairs with a non-finite value or a non-positive amount are dropped.
    With no pairs the result is NaN. With a single distinct amount the
    proportional part is not identified, so the median fee is the fixed part.
    If the unconstrained fit gives a negative rate, the fit is the median fee
    as a fixed charge; if it gives a negative fixed part, it is the median
    proportional rate with no fixed charge.
    """
    amount = np.asarray(list(amounts), dtype=float)
    fee = np.asarray(list(fees), dtype=float)
    if amount.shape != fee.shape:
        raise ValueError(f"amounts and fees must have the same length, got {amount.size} and {fee.size}")
    keep = np.isfinite(amount) & np.isfinite(fee) & (amount > 0)
    amount, fee = amount[keep], fee[keep]
    if amount.size == 0:
        return FeeSchedule(math.nan, math.nan)
    if np.unique(amount).size < 2:
        return FeeSchedule(max(float(np.median(fee)), 0.0), 0.0)
    design = np.column_stack([np.ones_like(amount), amount / 1e4])
    fixed, bps = np.linalg.lstsq(design, fee, rcond=None)[0]
    if bps < 0:
        return FeeSchedule(max(float(np.median(fee)), 0.0), 0.0)
    if fixed < 0:
        return FeeSchedule(0.0, max(float(np.median(fee / amount)) * 1e4, 0.0))
    return FeeSchedule(float(fixed), float(bps))
