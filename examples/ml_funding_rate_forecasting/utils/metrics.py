import numpy as np


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error.

    Returns 0 for samples where both ``y_true`` and ``y_pred`` are
    zero (defining 0/0 as a perfect prediction) instead of NaN.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true) + np.abs(y_pred)
    # ``denom > 0`` excludes BOTH zero-denominator and NaN entries
    # (NumPy comparisons against NaN are always False). Zero-denom
    # entries are 0/0, which we treat as a perfect match (contributes
    # nothing to the mean); NaN entries come from rolling/shift gaps
    # in the inputs and would otherwise poison the mean.
    mask = denom > 0
    if not mask.any():
        return 0.0
    return float(np.mean(2 * np.abs(y_pred[mask] - y_true[mask]) / denom[mask]))
