"""Pure helpers for realised bridge fees.

``fee_bps`` turns one transfer's input and output amounts into a realised
fee and ``fee_stats`` summarises many of them. The affine per-route fit
``fee = fixed + bps * size`` lives in core
(``fractal.core.entities.models.route_costs.fit_fee_schedule``).
"""

import math
import statistics
from collections.abc import Iterable

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
