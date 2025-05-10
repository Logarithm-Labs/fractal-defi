from typing import Dict, List, Tuple

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class UniswapFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for Uniswap V3 environment.
    Extracts technical indicators and market features from raw observations.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        features_dim: int = 22,
        price_history_length: int = 168,
        alpha: float = 0.05,
    ):
        super().__init__(observation_space, features_dim)
        self.price_history_length = price_history_length
        self.alpha = alpha

        # Initialize history buffers
        self.price_history = []
        self.high_price_history = []
        self.low_price_history = []
        self.close_price_history = []

        # Initialize feature calculation methods
        self._init_feature_methods()

    def _init_feature_methods(self):
        """Initialize all feature calculation methods."""
        self.feature_methods = {
            "dema": self._calculate_dema,
            "psar": self._calculate_parabolic_sar,
            "apo": self._calculate_apo,
            "aroon": self._calculate_aroon_oscillator,
            "bop": self._calculate_bop,
            "cci": self._calculate_cci,
            "cmo": self._calculate_cmo,
            "dx": self._calculate_dx,
            "momentum": self._calculate_momentum,
            "trix": self._calculate_trix,
            "ultimate": self._calculate_ultimate_oscillator,
            "ewma_volatility": self._calculate_ewma_volatility,
            "moving_averages": self._calculate_moving_averages,
            "bollinger_bands": self._calculate_bollinger_bands,
            "adxr": self._calculate_adxr,
            "stochastic": self._calculate_stochastic_momentum,
            "natr": self._calculate_natr,
            "true_range": self._calculate_true_range,
            "hilbert": self._calculate_hilbert_transform,
        }

    def _update_history(self, observations: Dict[str, np.ndarray]):
        """Update price history with new observations."""
        self.price_history.append(observations["price"][0])
        self.high_price_history.append(observations["high_price"][0])
        self.low_price_history.append(observations["low_price"][0])
        self.close_price_history.append(observations["close_price"][0])

        # Keep history at fixed length
        if len(self.price_history) > self.price_history_length:
            self.price_history = []
            self.high_price_history = []
            self.low_price_history = []
            self.close_price_history = []

    def _calculate_price_ratios(
        self, observations: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """Calculate various price ratios from raw price data."""
        open_price = observations["open_price"][0]
        if open_price == 0:
            return {
                "high_to_open": 1.0,
                "low_to_open": 1.0,
                "close_to_open": 1.0,
            }
        return {
            "high_to_open": observations["high_price"][0] / open_price,
            "low_to_open": observations["low_price"][0] / open_price,
            "close_to_open": observations["close_price"][0] / open_price,
        }

    def _calculate_ewma_volatility(self, prices: List[float]) -> float:
        """Calculate exponentially weighted moving average volatility."""
        if len(prices) < 2:
            return 0.0
        try:
            returns = np.diff(np.log(prices))
            ewma = np.zeros_like(returns)
            ewma[0] = returns[0] ** 2
            for i in range(1, len(returns)):
                ewma[i] = self.alpha * returns[i] ** 2 + (1 - self.alpha) * ewma[i - 1]
            return np.sqrt(ewma[-1])
        except (ZeroDivisionError, ValueError):
            return 0.0

    def _calculate_moving_averages(self, prices: List[float]) -> Tuple[float, float]:
        """Calculate 24 and 168 window moving averages."""
        if len(prices) < 24:
            return 0.0, 0.0
        try:
            ma24 = np.mean(prices[-24:])
            ma168 = np.mean(prices[-168:])
            return ma24, ma168
        except (ZeroDivisionError, ValueError):
            return 0.0, 0.0

    def _calculate_bollinger_bands(
        self, prices: List[float], window: int = 12
    ) -> Tuple[float, float, float]:
        """Calculate Bollinger Bands."""
        if len(prices) < window:
            return 0.0, 0.0, 0.0
        try:
            ma = np.mean(prices[-window:])
            std = np.std(prices[-window:])
            upper_band = ma + 2 * std
            lower_band = ma - 2 * std
            return upper_band, ma, lower_band
        except (ZeroDivisionError, ValueError):
            return 0.0, 0.0, 0.0

    def _calculate_adxr(
        self,
        high_history: List[float],
        low_history: List[float],
        close_history: List[float],
        window: int = 12,
    ) -> float:
        """Calculate Average Directional Movement Index Rating."""
        if len(high_history) or len(low_history) or len(close_history) < window * 2:
            return 0.0
        try:
            high = np.array([p for p in high_history])
            low = np.array([p for p in low_history])
            close = np.array([p for p in close_history])

            # Calculate +DM and -DM
            plus_dm = np.zeros_like(high)
            minus_dm = np.zeros_like(high)
            for i in range(1, len(high)):
                up_move = high[i] - high[i - 1]
                down_move = low[i - 1] - low[i]
                if up_move > down_move and up_move > 0:
                    plus_dm[i] = up_move
                if down_move > up_move and down_move > 0:
                    minus_dm[i] = down_move

            # Calculate True Range
            tr = np.zeros_like(high)
            for i in range(1, len(high)):
                tr[i] = max(
                    high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i] - close[i - 1]),
                )

            # Calculate smoothed values
            tr_mean = np.mean(tr[-window:])
            if tr_mean == 0:
                return 0.0

            plus_di = 100 * np.mean(plus_dm[-window:]) / tr_mean
            minus_di = 100 * np.mean(minus_dm[-window:]) / tr_mean

            # Calculate ADX
            if plus_di + minus_di == 0:
                return 0.0

            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = np.mean(dx[-window:])

            return adx
        except (ZeroDivisionError, ValueError):
            return 0.0

    def _calculate_bop(
        self, prices: List[float], observations: Dict[str, np.ndarray]
    ) -> float:
        """Calculate Balance of Power."""
        if len(prices) < 2:
            return 0.0
        try:
            high = observations["high_price"][0]
            low = observations["low_price"][0]
            close = observations["close_price"][0]
            open_price = observations["open_price"][0]

            if high - low == 0:
                return 0.0

            bop = (close - open_price) / (high - low)
            return bop
        except (ZeroDivisionError, ValueError):
            return 0.0

    def _calculate_dx(
        self,
        high_history: List[float],
        low_history: List[float],
        close_history: List[float],
        window: int = 12,
    ) -> float:
        """Calculate Directional Movement Index."""
        if len(high_history) or len(low_history) or len(close_history) < window * 2:
            return 0.0
        try:
            high = np.array([p for p in high_history])
            low = np.array([p for p in low_history])
            close = np.array([p for p in close_history])

            # Calculate +DM and -DM
            plus_dm = np.zeros_like(high)
            minus_dm = np.zeros_like(high)
            for i in range(1, len(high)):
                up_move = high[i] - high[i - 1]
                down_move = low[i - 1] - low[i]
                if up_move > down_move and up_move > 0:
                    plus_dm[i] = up_move
                if down_move > up_move and down_move > 0:
                    minus_dm[i] = down_move

            # Calculate True Range
            tr = np.zeros_like(high)
            for i in range(1, len(high)):
                tr[i] = max(
                    high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i] - close[i - 1]),
                )

            # Calculate smoothed values
            tr_mean = np.mean(tr[-window:])
            if tr_mean == 0:
                return 0.0

            plus_di = 100 * np.mean(plus_dm[-window:]) / tr_mean
            minus_di = 100 * np.mean(minus_dm[-window:]) / tr_mean

            # Calculate DX
            if plus_di + minus_di == 0:
                return 0.0

            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            return dx
        except (ZeroDivisionError, ValueError):
            return 0.0

    def _calculate_dema(self, prices: List[float], window: int = 12) -> float:
        """Calculate Double Exponential Moving Average."""
        if len(prices) < window:
            return 0.0
        try:
            prices_array = np.array(prices)
            ema1 = np.zeros_like(prices_array)
            ema2 = np.zeros_like(prices_array)
            alpha = 2 / (window + 1)

            # Calculate first EMA
            ema1[0] = prices_array[0]
            for i in range(1, len(prices_array)):
                ema1[i] = alpha * prices_array[i] + (1 - alpha) * ema1[i - 1]

            # Calculate second EMA
            ema2[0] = ema1[0]
            for i in range(1, len(ema1)):
                ema2[i] = alpha * ema1[i] + (1 - alpha) * ema2[i - 1]

            # Calculate DEMA
            dema = 2 * ema1[-1] - ema2[-1]
            return dema
        except (ZeroDivisionError, ValueError):
            return 0.0

    def _calculate_parabolic_sar(
        self,
        high_history: List[float],
        low_history: List[float],
        acceleration: float = 0.02,
        max_acceleration: float = 0.2,
    ) -> float:
        """Calculate Parabolic SAR."""
        if len(high_history) < 2:
            return 0.0
        try:
            high = np.array(high_history)
            low = np.array(low_history)

            # Initialize
            trend = 1  # 1 for uptrend, -1 for downtrend
            sar = low[0]
            extreme_point = high[0]
            current_acceleration = acceleration

            # Calculate SAR for the last point
            for i in range(1, len(high)):
                if trend == 1:
                    if low[i] < sar:
                        trend = -1
                        sar = extreme_point
                        extreme_point = low[i]
                        current_acceleration = acceleration
                    else:
                        if high[i] > extreme_point:
                            extreme_point = high[i]
                            current_acceleration = min(
                                current_acceleration + acceleration, max_acceleration
                            )
                        sar = sar + current_acceleration * (extreme_point - sar)
                else:
                    if high[i] > sar:
                        trend = 1
                        sar = extreme_point
                        extreme_point = high[i]
                        current_acceleration = acceleration
                    else:
                        if low[i] < extreme_point:
                            extreme_point = low[i]
                            current_acceleration = min(
                                current_acceleration + acceleration, max_acceleration
                            )
                        sar = sar + current_acceleration * (extreme_point - sar)

            return sar
        except (ZeroDivisionError, ValueError):
            return 0.0

    def _calculate_apo(
        self, prices: List[float], fast_period: int = 12, slow_period: int = 26
    ) -> float:
        """Calculate Absolute Price Oscillator."""
        if len(prices) < max(fast_period, slow_period):
            return 0.0
        try:
            prices_array = np.array(prices)
            fast_alpha = 2 / (fast_period + 1)
            slow_alpha = 2 / (slow_period + 1)

            # Calculate EMAs
            fast_ema = np.zeros_like(prices_array)
            slow_ema = np.zeros_like(prices_array)

            fast_ema[0] = prices_array[0]
            slow_ema[0] = prices_array[0]

            for i in range(1, len(prices_array)):
                fast_ema[i] = (
                    fast_alpha * prices_array[i] + (1 - fast_alpha) * fast_ema[i - 1]
                )
                slow_ema[i] = (
                    slow_alpha * prices_array[i] + (1 - slow_alpha) * slow_ema[i - 1]
                )

            return fast_ema[-1] - slow_ema[-1]
        except (ZeroDivisionError, ValueError):
            return 0.0

    def _calculate_aroon_oscillator(
        self, high_history: List[float], low_history: List[float], period: int = 14
    ) -> float:
        """Calculate Aroon Oscillator."""
        if len(high_history) < period or len(low_history) < period:
            return 0.0
        try:
            # Get last 'period' values
            high = np.array(high_history[-period:])
            low = np.array(low_history[-period:])

            # Find the number of periods since the highest high and lowest low
            high_days = period - np.argmax(high) - 1
            low_days = period - np.argmin(low) - 1

            # Calculate Aroon Up and Down
            aroon_up = (period - high_days) / period * 100
            aroon_down = (period - low_days) / period * 100

            # Calculate Aroon Oscillator
            return aroon_up - aroon_down
        except (ZeroDivisionError, ValueError):
            return 0.0

    def _calculate_cci(
        self,
        high_history: List[float],
        low_history: List[float],
        close_history: List[float],
        period: int = 20,
    ) -> float:
        """Calculate Commodity Channel Index."""
        if (
            len(high_history) < period
            or len(low_history) < period
            or len(close_history) < period
        ):
            return 0.0
        try:
            # Get last 'period' values
            high = np.array(high_history[-period:])
            low = np.array(low_history[-period:])
            close = np.array(close_history[-period:])

            # Calculate typical price
            tp = (high + low + close) / 3

            # Calculate SMA of typical price
            sma_tp = np.mean(tp)

            # Calculate Mean Deviation
            mean_deviation = np.mean(np.abs(tp - sma_tp))

            if mean_deviation == 0:
                return 0.0

            # Calculate CCI
            cci = (tp[-1] - sma_tp) / (0.015 * mean_deviation)

            return cci
        except (ZeroDivisionError, ValueError):
            return 0.0

    def _calculate_cmo(self, prices: List[float], period: int = 14) -> float:
        """Calculate Chande Momentum Oscillator."""
        if len(prices) < period + 1:
            return 0.0
        try:
            # Calculate price changes
            changes = np.diff(prices[-period - 1:])

            # Separate positive and negative changes
            positive_sum = np.sum([change for change in changes if change > 0])
            negative_sum = abs(np.sum([change for change in changes if change < 0]))

            if positive_sum + negative_sum == 0:
                return 0.0

            # Calculate CMO
            cmo = 100 * (positive_sum - negative_sum) / (positive_sum + negative_sum)

            return cmo
        except (ZeroDivisionError, ValueError):
            return 0.0

    def _calculate_momentum(self, prices: List[float], period: int = 10) -> float:
        """Calculate Momentum."""
        if len(prices) <= period:
            return 0.0
        try:
            return prices[-1] - prices[-period - 1]
        except (ZeroDivisionError, ValueError):
            return 0.0

    def _calculate_trix(self, prices: List[float], period: int = 15) -> float:
        """Calculate TRIX indicator."""
        if len(prices) < period * 3:
            return 0.0
        try:
            prices_array = np.array(prices)
            alpha = 2 / (period + 1)

            # Calculate triple EMA
            ema1 = np.zeros_like(prices_array)
            ema2 = np.zeros_like(prices_array)
            ema3 = np.zeros_like(prices_array)

            # First EMA
            ema1[0] = prices_array[0]
            for i in range(1, len(prices_array)):
                ema1[i] = alpha * prices_array[i] + (1 - alpha) * ema1[i - 1]

            # Second EMA
            ema2[0] = ema1[0]
            for i in range(1, len(ema1)):
                ema2[i] = alpha * ema1[i] + (1 - alpha) * ema2[i - 1]

            # Third EMA
            ema3[0] = ema2[0]
            for i in range(1, len(ema2)):
                ema3[i] = alpha * ema2[i] + (1 - alpha) * ema3[i - 1]

            # Calculate TRIX
            if ema3[-2] == 0:
                return 0.0
            trix = (ema3[-1] - ema3[-2]) / ema3[-2] * 100

            return trix
        except (ZeroDivisionError, ValueError):
            return 0.0

    def _calculate_ultimate_oscillator(
        self,
        high_history: List[float],
        low_history: List[float],
        close_history: List[float],
        period1: int = 7,
        period2: int = 14,
        period3: int = 28,
    ) -> float:
        """Calculate Ultimate Oscillator."""
        if len(high_history) < max(period1, period2, period3):
            return 0.0
        try:
            high = np.array(high_history)
            low = np.array(low_history)
            close = np.array(close_history)

            # Calculate buying pressure (BP) and true range (TR)
            bp = close - np.minimum(low, np.roll(close, 1))
            tr = np.maximum(high, np.roll(close, 1)) - np.minimum(
                low, np.roll(close, 1)
            )

            # Calculate averages for different periods
            avg1 = np.sum(bp[-period1:]) / np.sum(tr[-period1:])
            avg2 = np.sum(bp[-period2:]) / np.sum(tr[-period2:])
            avg3 = np.sum(bp[-period3:]) / np.sum(tr[-period3:])

            # Calculate Ultimate Oscillator
            uo = 100 * (4 * avg1 + 2 * avg2 + avg3) / 7

            return uo
        except (ZeroDivisionError, ValueError):
            return 0.0

    def _calculate_stochastic_momentum(
        self,
        high_history: List[float],
        low_history: List[float],
        close_history: List[float],
        k_period: int = 14,
        d_period: int = 3,
    ) -> Tuple[float, float, float]:
        """Calculate Stochastic Momentum Indicator."""
        if (
            len(high_history) < k_period
            or len(low_history) < k_period
            or len(close_history) < k_period
        ):
            return 0.0, 0.0, 0.0
        try:
            # Get relevant price ranges
            high = np.array(high_history[-k_period:])
            low = np.array(low_history[-k_period:])
            close = np.array(close_history[-k_period:])

            # Calculate %K
            lowest_low = np.min(low)
            highest_high = np.max(high)

            if highest_high - lowest_low == 0:
                return 0.0, 0.0, 0.0

            k = 100 * (close[-1] - lowest_low) / (highest_high - lowest_low)

            # Calculate %D (3-period SMA of %K)
            if len(close_history) < k_period + d_period:
                return k, 0.0, 0.0

            d = np.mean([k for _ in range(d_period)])

            # Calculate Slow Stochastic
            slow_d = np.mean([d for _ in range(d_period)])

            return k, d, slow_d
        except (ZeroDivisionError, ValueError):
            return 0.0, 0.0, 0.0

    def _calculate_natr(
        self,
        high_history: List[float],
        low_history: List[float],
        close_history: List[float],
        period: int = 14,
    ) -> float:
        """Calculate Normalized Average True Range."""
        if (
            len(high_history) < period
            or len(low_history) < period
            or len(close_history) < period
        ):
            return 0.0
        try:
            high = np.array(high_history)
            low = np.array(low_history)
            close = np.array(close_history)

            # Calculate True Range
            tr = np.zeros(len(high))
            for i in range(1, len(high)):
                tr[i] = max(
                    high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i] - close[i - 1]),
                )

            # Calculate ATR
            atr = np.mean(tr[-period:])

            # Normalize ATR
            if close[-1] == 0:
                return 0.0

            natr = 100 * atr / close[-1]

            return natr
        except (ZeroDivisionError, ValueError):
            return 0.0

    def _calculate_true_range(
        self,
        high_history: List[float],
        low_history: List[float],
        close_history: List[float],
    ) -> float:
        """Calculate True Range."""
        if len(high_history) < 2 or len(low_history) < 2 or len(close_history) < 2:
            return 0.0
        try:
            high = high_history[-1]
            low = low_history[-1]
            prev_close = close_history[-2]

            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))

            return tr
        except (ZeroDivisionError, ValueError):
            return 0.0

    def _calculate_hilbert_transform(
        self, prices: List[float], period: int = 14
    ) -> Tuple[float, float]:
        """Calculate Hilbert Transform - Dominant Cycle Period and Phase."""
        if len(prices) < period * 2:
            return 0.0, 0.0
        try:
            prices_array = np.array(prices[-period * 2:])

            # Smooth prices
            alpha = 0.0962
            smooth = np.zeros_like(prices_array)
            smooth[0] = prices_array[0]
            for i in range(1, len(prices_array)):
                smooth[i] = (1 - alpha) * smooth[i - 1] + alpha * prices_array[i]

            # Calculate quadrature
            quad = np.zeros_like(smooth)
            for i in range(period, len(smooth)):
                quad[i] = 1.25 * (smooth[i - 7] - smooth[i - 5]) - 0.25 * (
                    smooth[i - 3] - smooth[i - 1]
                )

            # Calculate dominant cycle period using autocorrelation
            corr = np.correlate(smooth, smooth, mode="full")
            corr = corr[len(corr) // 2:]
            peaks = np.where((corr[1:-1] > corr[:-2]) & (corr[1:-1] > corr[2:]))[0] + 1
            if len(peaks) > 0:
                dominant_cycle = peaks[0]
            else:
                dominant_cycle = period

            # Calculate dominant cycle phase
            if np.sum(quad[-period:] ** 2) == 0:
                phase = 0.0
            else:
                phase = (
                    np.arctan2(
                        np.sum(quad[-period:] * smooth[-period:]),
                        np.sum(quad[-period:] ** 2),
                    )
                    * 180
                    / np.pi
                )

            return float(dominant_cycle), float(phase)
        except (ZeroDivisionError, ValueError):
            return 0.0, 0.0

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the feature extractor.
        Takes raw observations and returns processed features.
        """
        batch_size = observations["price"].shape[0]

        # Convert observations to numpy for feature calculation
        observations_np = {k: v.cpu().numpy() for k, v in observations.items()}

        # Initialize batch features array
        batch_features = np.zeros((batch_size, self.features_dim), dtype=np.float32)

        # Process each observation in the batch
        for i in range(batch_size):
            # Update history with new observations
            self._update_history({k: v[i: i + 1] for k, v in observations_np.items()})

            # Calculate price ratios
            price_ratios = self._calculate_price_ratios(
                {k: v[i: i + 1] for k, v in observations_np.items()}
            )

            # Basic features
            batch_features[i, 0:8] = [
                float(observations_np["balance"][i]),
                float(observations_np["is_position"][i]),
                float(observations_np["price"][i]),
                float(observations_np["centralized_price"][i]),
                float(price_ratios["high_to_open"]),
                float(price_ratios["low_to_open"]),
                float(price_ratios["close_to_open"]),
                float(observations_np["volume"][i]),
            ]

            # Calculate technical indicators
            dema = self._calculate_dema(self.price_history)
            dema_ratio = float(
                dema / observations_np["price"][i]
                if observations_np["price"][i] != 0
                else 1.0
            )
            batch_features[i, 8] = dema_ratio

            psar = self._calculate_parabolic_sar(
                self.high_price_history, self.low_price_history
            )
            psar_ratio = float(
                psar / observations_np["price"][i]
                if observations_np["price"][i] != 0
                else 1.0
            )
            batch_features[i, 9] = psar_ratio

            # Calculate remaining indicators
            indicators = [
                float(
                    self._calculate_dx(
                        self.high_price_history,
                        self.low_price_history,
                        self.close_price_history,
                    )
                ),
                float(self._calculate_apo(self.price_history)),
                float(
                    self._calculate_aroon_oscillator(
                        self.high_price_history, self.low_price_history
                    )
                ),
                float(
                    self._calculate_bop(
                        self.price_history,
                        {k: v[i: i + 1] for k, v in observations_np.items()},
                    )
                ),
                float(
                    self._calculate_cci(
                        self.high_price_history,
                        self.low_price_history,
                        self.close_price_history,
                    )
                ),
                float(self._calculate_cmo(self.price_history)),
                float(self._calculate_momentum(self.price_history)),
                float(self._calculate_trix(self.price_history)),
                float(
                    self._calculate_ultimate_oscillator(
                        self.high_price_history,
                        self.low_price_history,
                        self.close_price_history,
                    )
                ),
            ]
            batch_features[i, 10:19] = indicators

            # Add stochastic indicators
            k, d, slow_d = self._calculate_stochastic_momentum(
                self.high_price_history,
                self.low_price_history,
                self.close_price_history,
            )
            batch_features[i, 19:22] = [float(k), float(d), float(slow_d)]

        # Convert to tensor
        features_tensor = torch.from_numpy(batch_features).to(
            device=observations["price"].device
        )

        return features_tensor
