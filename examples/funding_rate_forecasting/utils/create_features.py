import numpy as np
import pandas as pd
from scipy.stats import linregress
from statsmodels.tsa.stattools import acf, pacf


def extract_time_series_features(funding_rate_series: pd.Series) -> dict:
    series = funding_rate_series.dropna()
    mean_rate = series.abs().mean()
    features = {}

    norm = lambda x: x / mean_rate

    # 1. Волатильность за весь период
    features["std"] = norm(series.std())
    features["mean_abs_change"] = norm(series.diff().abs().mean())

    # 2. Автокорреляции (до 3 лага)
    acf_values = acf(series, nlags=3, fft=False)
    for i in range(1, 4):
        features[f"acf_lag{i}"] = acf_values[i]

    # 3. Тренд — линейная регрессия
    x = np.arange(len(series))
    slope, intercept, r_value, p_value, std_err = linregress(x, series)
    features["trend_slope"] = norm(slope)
    features["trend_r2"] = r_value**2

    # 4. Скользящие std и mean
    features["rolling_std_24"] = norm(series.rolling(24).std().mean())
    features["rolling_mean_24"] = norm(series.rolling(24).mean().mean())

    # 5. Кол-во смен знака (индикатор смены настроений)
    signs = np.sign(series)
    features["sign_changes"] = (signs != signs.shift(1)).sum() / series.shape[0]

    # 6. Пики/всплески (аномалии)
    zscores = (series - series.mean()) / series.std()
    features["spike_count_z3"] = (np.abs(zscores) > 3).sum() / series.shape[0]

    # 7. Коэффициент асимметрии и эксцесса
    features["skew"] = series.skew()
    features["kurtosis"] = series.kurt()

    # 8. Возврат к среднему (mean reversion): autocorr на 1 шаг вперёд
    features["mean_reversion_strength"] = -acf_values[
        1
    ]  # сильная отриц. автокорреляция → возврат

    return features
