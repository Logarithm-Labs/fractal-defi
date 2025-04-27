import numpy as np
import pandas as pd
from scipy.stats import linregress
from statsmodels.tsa.stattools import acf, pacf


def extract_time_series_features(funding_rate_series: pd.Series) -> dict:
    series = funding_rate_series.dropna()
    mean_rate = series.abs().mean()
    features = {}

    norm = lambda x: x / mean_rate

    # 1. Volatility over the entire period
    features["std"] = norm(series.std())
    features["mean_abs_change"] = norm(series.diff().abs().mean())

    # 2. Autocorrelation (up to 3 lags)
    acf_values = acf(series, nlags=3, fft=False)
    for i in range(1, 4):
        features[f"acf_lag{i}"] = acf_values[i]

    # 3. Trend — linear regression
    x = np.arange(len(series))
    slope, intercept, r_value, p_value, std_err = linregress(x, series)
    features["trend_slope"] = norm(slope)
    features["trend_r2"] = r_value**2

    # 4. Sliding std and mean
    features["rolling_std_24"] = norm(series.rolling(24).std().mean())
    features["rolling_mean_24"] = norm(series.rolling(24).mean().mean())

    # 5. Number of sign changes (mood change indicator)
    signs = np.sign(series)
    features["sign_changes"] = (signs != signs.shift(1)).sum() / series.shape[0]

    # 6. Peaks/spikes (anomalies)
    zscores = (series - series.mean()) / series.std()
    features["spike_count_z3"] = (np.abs(zscores) > 3).sum() / series.shape[0]

    # 7. Coefficient of skewness and kurtosis
    features["skew"] = series.skew()
    features["kurtosis"] = series.kurt()

    # 8. Return to the mean (mean reversal): autocorr 1 step ahead
    features["mean_reversion_strength"] = -acf_values[
        1
    ]  # strong negativity. autocorrelation → return

    return features


def extract_features(df: pd.DataFrame, current_idx: int) -> pd.Series:
    t = int(
        df.loc[current_idx, "hour_index"] * 8
        + int(df.loc[current_idx, "open_time"][14:16])
    )
    window = df.iloc[current_idx : current_idx + t]

    features = {
        "log_return": np.log(
            window["close_spot"].iloc[-1] / window["close_spot"].iloc[0]
        ),
        "volatility": window["close_spot"].pct_change().std(),
        "avg_volume_futr": window["volume_futr"].mean(),
        "avg_volume_spot": window["volume_spot"].mean(),
        "basis_mean": (window["close_futr"] - window["close_spot"]).mean(),
        "basis_std": (window["close_futr"] - window["close_spot"]).std(),
    }

    return pd.Series(features)


def rsi(data: pd.DataFrame, period: int = 14) -> pd.Series:
    delta = data["close_futr"].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return 100 - (100 / (1 + rs))


def mfi(data: pd.DataFrame, period: int = 14) -> pd.Series:
    typical_price = (data["high_futr"] + data["low_futr"] + data["close_futr"]) / 3
    money_flow = typical_price * data["volume_futr"]
    positive_flow = money_flow.where(typical_price.diff() > 0, 0)
    negative_flow = money_flow.where(typical_price.diff() < 0, 0)
    pos_flow_sum = positive_flow.rolling(window=period).sum()
    neg_flow_sum = negative_flow.rolling(window=period).sum()
    mfi = 100 - (100 / (1 + (pos_flow_sum / neg_flow_sum)))
    return mfi


def ema(data: pd.DataFrame, period: int = 14) -> pd.Series:
    alpha = 2 / (period + 1)
    ema_values = [data["close_futr"][0]]
    for price in data["close_futr"][1:]:
        ema_values.append((price * alpha) + (ema_values[-1] * (1 - alpha)))
    return pd.Series(ema_values, index=data.index)
