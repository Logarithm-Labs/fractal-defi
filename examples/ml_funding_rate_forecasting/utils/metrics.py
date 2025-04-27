import numpy as np

def smape(y_true: np.array, y_pred: np.array) -> np.array:
    return np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))