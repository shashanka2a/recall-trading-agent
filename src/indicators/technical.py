"""
Technical indicators for trading strategies
"""

import statistics
from typing import List, Tuple, Optional
import numpy as np


class TechnicalIndicators:
    """Collection of technical analysis indicators"""
    
    @staticmethod
    def sma(prices: List[float], period: int) -> float:
        """Simple Moving Average"""
        if len(prices) < period:
            return prices[-1] if prices else 0.0
        return sum(prices[-period:]) / period
    
    @staticmethod
    def ema(prices: List[float], period: int) -> float:
        """Exponential Moving Average"""
        if len(prices) < period:
            return prices[-1] if prices else 0.0
        
        multiplier = 2 / (period + 1)
        ema_value = prices[0]
        
        for price in prices[1:]:
            ema_value = (price * multiplier) + (ema_value * (1 - multiplier))
        
        return ema_value
    
    @staticmethod
    def rsi(prices: List[float], period: int = 14) -> float:
        """Relative Strength Index"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [delta if delta > 0 else 0 for delta in deltas]
        losses = [-delta if delta < 0 else 0 for delta in deltas]
        
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def bollinger_bands(
        prices: List[float], 
        period: int = 20, 
        std_dev: float = 2.0
    ) -> Tuple[float, float, float]:
        """Bollinger Bands (upper, middle, lower)"""
        if len(prices) < period:
            price = prices[-1] if prices else 0.0
            return price, price, price
        
        sma = sum(prices[-period:]) / period
        variance = sum((p - sma) ** 2 for p in prices[-period:]) / period
        std = variance ** 0.5
        
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        
        return upper, sma, lower
    
    @staticmethod
    def macd(
        prices: List[float], 
        fast: int = 12, 
        slow: int = 26, 
        signal: int = 9
    ) -> Tuple[float, float, float]:
        """MACD (macd_line, signal_line, histogram)"""
        if len(prices) < slow:
            return 0.0, 0.0, 0.0
        
        fast_ema = TechnicalIndicators.ema(prices, fast)
        slow_ema = TechnicalIndicators.ema(prices, slow)
        macd_line = fast_ema - slow_ema
        
        # Calculate signal line (EMA of MACD)
        if len(prices) >= slow + signal:
            # Get MACD values for signal calculation
            macd_values = []
            for i in range(len(prices) - signal + 1, len(prices) + 1):
                if i >= slow:
                    f_ema = TechnicalIndicators.ema(prices[:i], fast)
                    s_ema = TechnicalIndicators.ema(prices[:i], slow)
                    macd_values.append(f_ema - s_ema)
            
            signal_line = TechnicalIndicators.ema(macd_values, signal) if macd_values else macd_line
        else:
            signal_line = macd_line
        
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def stochastic(
        highs: List[float], 
        lows: List[float], 
        closes: List[float], 
        k_period: int = 14,
        d_period: int = 3
    ) -> Tuple[float, float]:
        """Stochastic Oscillator (%K, %D)"""
        if len(closes) < k_period:
            return 50.0, 50.0
        
        # Calculate %K
        lowest_low = min(lows[-k_period:])
        highest_high = max(highs[-k_period:])
        current_close = closes[-1]
        
        if highest_high - lowest_low == 0:
            k_percent = 50.0
        else:
            k_percent = ((current_close - lowest_low) / (highest_high - lowest_low)) * 100
        
        # Calculate %D (SMA of %K)
        if len(closes) >= k_period + d_period - 1:
            k_values = []
            for i in range(len(closes) - d_period + 1, len(closes) + 1):
                if i >= k_period:
                    ll = min(lows[i-k_period:i])
                    hh = max(highs[i-k_period:i])
                    cc = closes[i-1]
                    if hh - ll != 0:
                        k_val = ((cc - ll) / (hh - ll)) * 100
                    else:
                        k_val = 50.0
                    k_values.append(k_val)
            
            d_percent = sum(k_values) / len(k_values) if k_values else k_percent
        else:
            d_percent = k_percent
        
        return k_percent, d_percent
    
    @staticmethod
    def williams_r(
        highs: List[float], 
        lows: List[float], 
        closes: List[float], 
        period: int = 14
    ) -> float:
        """Williams %R"""
        if len(closes) < period:
            return -50.0
        
        highest_high = max(highs[-period:])
        lowest_low = min(lows[-period:])
        current_close = closes[-1]
        
        if highest_high - lowest_low == 0:
            return -50.0
        
        williams_r = ((highest_high - current_close) / (highest_high - lowest_low)) * -100
        return williams_r
    
    @staticmethod
    def atr(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> float:
        """Average True Range"""
        if len(closes) < 2:
            return 0.0
        
        true_ranges = []
        for i in range(1, len(closes)):
            high_low = highs[i] - lows[i]
            high_close_prev = abs(highs[i] - closes[i-1])
            low_close_prev = abs(lows[i] - closes[i-1])
            
            true_range = max(high_low, high_close_prev, low_close_prev)
            true_ranges.append(true_range)
        
        if len(true_ranges) < period:
            return statistics.mean(true_ranges) if true_ranges else 0.0
        
        return statistics.mean(true_ranges[-period:])
    
    @staticmethod
    def obv(closes: List[float], volumes: List[float]) -> float:
        """On-Balance Volume"""
        if len(closes) != len(volumes) or len(closes) < 2:
            return 0.0
        
        obv_value = 0.0
        for i in range(1, len(closes)):
            if closes[i] > closes[i-1]:
                obv_value += volumes[i]
            elif closes[i] < closes[i-1]:
                obv_value -= volumes[i]
            # If equal, OBV stays the same
        
        return obv_value
    
    @staticmethod
    def cci(highs: List[float], lows: List[float], closes: List[float], period: int = 20) -> float:
        """Commodity Channel Index"""
        if len(closes) < period:
            return 0.0
        
        # Calculate typical prices
        typical_prices = []
        for i in range(len(closes)):
            typical_price = (highs[i] + lows[i] + closes[i]) / 3
            typical_prices.append(typical_price)
        
        # Calculate SMA of typical prices
        sma_tp = sum(typical_prices[-period:]) / period
        
        # Calculate mean deviation
        mean_deviation = sum(abs(tp - sma_tp) for tp in typical_prices[-period:]) / period
        
        if mean_deviation == 0:
            return 0.0
        
        # Calculate CCI
        current_tp = typical_prices[-1]
        cci = (current_tp - sma_tp) / (0.015 * mean_deviation)
        
        return cci
    
    @staticmethod
    def momentum(prices: List[float], period: int = 10) -> float:
        """Price Momentum"""
        if len(prices) < period + 1:
            return 0.0
        
        current_price = prices[-1]
        past_price = prices[-(period + 1)]
        
        if past_price == 0:
            return 0.0
        
        return ((current_price - past_price) / past_price) * 100
    
    @staticmethod
    def roc(prices: List[float], period: int = 10) -> float:
        """Rate of Change"""
        return TechnicalIndicators.momentum(prices, period)
    
    @staticmethod
    def trix(prices: List[float], period: int = 14) -> float:
        """TRIX - Triple Exponential Average"""
        if len(prices) < period * 3:
            return 0.0
        
        # First EMA
        ema1_values = []
        for i in range(period, len(prices) + 1):
            ema1 = TechnicalIndicators.ema(prices[:i], period)
            ema1_values.append(ema1)
        
        if len(ema1_values) < period:
            return 0.0
        
        # Second EMA
        ema2_values = []
        for i in range(period, len(ema1_values) + 1):
            ema2 = TechnicalIndicators.ema(ema1_values[:i], period)
            ema2_values.append(ema2)
        
        if len(ema2_values) < period:
            return 0.0
        
        # Third EMA
        ema3_values = []
        for i in range(period, len(ema2_values) + 1):
            ema3 = TechnicalIndicators.ema(ema2_values[:i], period)
            ema3_values.append(ema3)
        
        if len(ema3_values) < 2:
            return 0.0
        
        # TRIX calculation
        current_ema3 = ema3_values[-1]
        previous_ema3 = ema3_values[-2]
        
        if previous_ema3 == 0:
            return 0.0
        
        trix = ((current_ema3 - previous_ema3) / previous_ema3) * 10000
        return trix
    
    @staticmethod
    def adx(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> float:
        """Average Directional Index"""
        if len(closes) < period + 1:
            return 0.0
        
        # Calculate True Range and Directional Movements
        plus_dm = []
        minus_dm = []
        true_ranges = []
        
        for i in range(1, len(closes)):
            high_diff = highs[i] - highs[i-1]
            low_diff = lows[i-1] - lows[i]
            
            plus_dm.append(high_diff if high_diff > low_diff and high_diff > 0 else 0)
            minus_dm.append(low_diff if low_diff > high_diff and low_diff > 0 else 0)
            
            # True Range
            high_low = highs[i] - lows[i]
            high_close = abs(highs[i] - closes[i-1])
            low_close = abs(lows[i] - closes[i-1])
            true_ranges.append(max(high_low, high_close, low_close))
        
        if len(true_ranges) < period:
            return 0.0
        
        # Calculate smoothed averages
        smoothed_plus_dm = sum(plus_dm[-period:]) / period
        smoothed_minus_dm = sum(minus_dm[-period:]) / period
        smoothed_tr = sum(true_ranges[-period:]) / period
        
        if smoothed_tr == 0:
            return 0.0
        
        # Calculate Directional Indicators
        plus_di = 100 * (smoothed_plus_dm / smoothed_tr)
        minus_di = 100 * (smoothed_minus_dm / smoothed_tr)
        
        # Calculate DX
        di_sum = plus_di + minus_di
        if di_sum == 0:
            return 0.0
        
        dx = 100 * abs(plus_di - minus_di) / di_sum
        
        # ADX is typically a smoothed version of DX
        # For simplicity, returning DX as ADX approximation
        return dx
    
    @staticmethod
    def vwap(prices: List[float], volumes: List[float]) -> float:
        """Volume Weighted Average Price"""
        if len(prices) != len(volumes) or not prices:
            return 0.0
        
        total_volume = sum(volumes)
        if total_volume == 0:
            return statistics.mean(prices)
        
        weighted_sum = sum(price * volume for price, volume in zip(prices, volumes))
        return weighted_sum / total_volume
    
    @staticmethod
    def z_score(prices: List[float], period: int = 20) -> float:
        """Z-Score (standardized price)"""
        if len(prices) < period:
            return 0.0
        
        recent_prices = prices[-period:]
        mean_price = statistics.mean(recent_prices)
        std_price = statistics.stdev(recent_prices) if len(recent_prices) > 1 else 0
        
        if std_price == 0:
            return 0.0
        
        current_price = prices[-1]
        return (current_price - mean_price) / std_price


class PatternRecognition:
    """Pattern recognition utilities"""
    
    @staticmethod
    def is_doji(open_price: float, high: float, low: float, close: float, threshold: float = 0.1) -> bool:
        """Detect Doji candlestick pattern"""
        body_size = abs(close - open_price)
        total_range = high - low
        
        if total_range == 0:
            return True
        
        return (body_size / total_range) < threshold
    
    @staticmethod
    def is_hammer(open_price: float, high: float, low: float, close: float) -> bool:
        """Detect Hammer candlestick pattern"""
        body_size = abs(close - open_price)
        lower_shadow = min(open_price, close) - low
        upper_shadow = high - max(open_price, close)
        
        return (
            lower_shadow >= 2 * body_size and
            upper_shadow <= body_size * 0.1 and
            body_size > 0
        )
    
    @staticmethod
    def detect_support_resistance(
        prices: List[float], 
        window: int = 5, 
        min_touches: int = 2
    ) -> Tuple[List[float], List[float]]:
        """Detect support and resistance levels"""
        if len(prices) < window * 2:
            return [], []
        
        # Find local minima (support) and maxima (resistance)
        supports = []
        resistances = []
        
        for i in range(window, len(prices) - window):
            # Check for local minimum (support)
            is_support = all(prices[i] <= prices[j] for j in range(i - window, i + window + 1) if j != i)
            if is_support:
                supports.append(prices[i])
            
            # Check for local maximum (resistance)
            is_resistance = all(prices[i] >= prices[j] for j in range(i - window, i + window + 1) if j != i)
            if is_resistance:
                resistances.append(prices[i])
        
        return supports, resistances