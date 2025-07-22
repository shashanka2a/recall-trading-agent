
"""Breakout trading strategy implementation."""

import logging
import statistics
from typing import List, Optional, Tuple

from .base import BaseStrategy
from config.settings import TradingConfig
from src.core.models import MarketData, Portfolio, TradingSignal
from src.core.enums import StrategyType, SignalType
from src.indicators.technical import TechnicalIndicators, PatternRecognition

logger = logging.getLogger(__name__)


class BreakoutStrategy(BaseStrategy):
    def __init__(self, config: TradingConfig):
        super().__init__(config, StrategyType.BREAKOUT)
        self.lookback_period = 20
        self.breakout_threshold = 0.015
        self.volume_spike_threshold = 1.8
        self.consolidation_periods = 5
        self.max_breakout_age = 3
        self.factor_weights = {
            'breakout_strength': 0.30,
            'volume_confirmation': 0.25,
            'consolidation_quality': 0.20,
            'momentum_alignment': 0.15,
            'pattern_confirmation': 0.10
        }

    async def generate_signal(self, symbol: str, market_data: List[MarketData], portfolio: Portfolio) -> Optional[TradingSignal]:
        if len(market_data) < self.config.lookback_periods["long"]:
            logger.debug(f"Insufficient data for {symbol}: {len(market_data)} periods")
            return None
        prices = [data.price for data in market_data]
        volumes = [data.volume for data in market_data]
        highs = [data.high_24h or data.price for data in market_data]
        lows = [data.low_24h or data.price for data in market_data]
        current_price = prices[-1]
        support_levels, resistance_levels = self._calculate_support_resistance(prices, highs, lows)
        if not support_levels and not resistance_levels:
            logger.debug(f"No clear support/resistance levels for {symbol}")
            return None
        factors = self._calculate_breakout_factors(prices, volumes, highs, lows, support_levels, resistance_levels)
        signal_strength = self.calculate_signal_strength_score(factors, self.factor_weights)
        signal_type = self._determine_signal_type(signal_strength, factors)
        if signal_type == SignalType.HOLD:
            return None
        additional_factors = {
            'volume_confirmation': factors.get('volume_confirmation', 1.0),
            'trend_alignment': factors.get('momentum_alignment', 1.0),
            'market_conditions': self._assess_market_conditions(market_data)
        }
        confidence = self.calculate_confidence(signal_strength, additional_factors)
        allocation_multiplier = min(abs(signal_strength), 1.0)
        if factors.get('volume_confirmation', 1.0) > 1.5:
            allocation_multiplier *= 1.2
        target_allocation = allocation_multiplier * self.config.max_position_size
        reasoning_components = self._build_reasoning(factors, signal_strength, support_levels, resistance_levels)
        reasoning = self.format_reasoning(reasoning_components)
        stop_loss, take_profit = self._calculate_dynamic_levels(current_price, signal_type, support_levels, resistance_levels)
        return TradingSignal(
            symbol=symbol,
            signal=signal_type,
            confidence=confidence,
            strategy=self.strategy_type,
            target_allocation=target_allocation,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reasoning=reasoning,
            metadata={
                'signal_strength': signal_strength,
                'factors': factors,
                'current_price': current_price,
                'support_levels': support_levels,
                'resistance_levels': resistance_levels,
                'breakout_level': factors.get('breakout_level')
            }
        )

    def _calculate_support_resistance(self, prices, highs, lows):
        support_levels, resistance_levels = PatternRecognition.detect_support_resistance(prices, window=5, min_touches=2)
        lookback = min(self.lookback_period, len(prices))
        recent_prices = prices[-lookback:]
        recent_highs = highs[-lookback:]
        recent_lows = lows[-lookback:]
        significant_highs = []
        significant_lows = []
        for i in range(2, len(recent_prices) - 2):
            if (recent_highs[i] > recent_highs[i - 1] and recent_highs[i] > recent_highs[i - 2] and
                recent_highs[i] > recent_highs[i + 1] and recent_highs[i] > recent_highs[i + 2]):
                significant_highs.append(recent_highs[i])
            if (recent_lows[i] < recent_lows[i - 1] and recent_lows[i] < recent_lows[i - 2] and
                recent_lows[i] < recent_lows[i + 1] and recent_lows[i] < recent_lows[i + 2]):
                significant_lows.append(recent_lows[i])
        all_resistance = list(set(resistance_levels + significant_highs))
        all_support = list(set(support_levels + significant_lows))
        current_price = prices[-1]
        filtered_resistance = [level for level in all_resistance if current_price * 0.99 <= level <= current_price * 1.1]
        filtered_support = [level for level in all_support if current_price * 0.9 <= level <= current_price * 1.01]
        return sorted(filtered_support, reverse=True), sorted(filtered_resistance)

    def _calculate_breakout_factors(self, prices, volumes, highs, lows, support_levels, resistance_levels):
        factors = {}
        current_price = prices[-1]
        breakout_info = self._detect_breakout(current_price, support_levels, resistance_levels)
        factors.update(breakout_info)
        factors['volume_confirmation'] = self._calculate_volume_confirmation(volumes)
        factors['consolidation_quality'] = 1.0
        factors['momentum_alignment'] = 1.0
        factors['pattern_confirmation'] = 0.5
        factors['range_analysis'] = 0.5
        factors['breakout_age'] = 1.0
        return factors

    def _detect_breakout(self, current_price, support_levels, resistance_levels):
        result = {
            'breakout_direction': 0,
            'breakout_strength': 0.0,
            'breakout_level': None,
            'breakout_distance': 0.0
        }
        if resistance_levels:
            nearest_resistance = min(resistance_levels, key=lambda x: abs(x - current_price))
            if current_price > nearest_resistance:
                distance = (current_price - nearest_resistance) / nearest_resistance
                if distance >= self.breakout_threshold:
                    result['breakout_direction'] = 1
                    result['breakout_level'] = nearest_resistance
                    result['breakout_distance'] = distance
                    result['breakout_strength'] = min(1.0, distance / self.breakout_threshold)
        if support_levels and result['breakout_direction'] == 0:
            nearest_support = max(support_levels, key=lambda x: abs(x - current_price))
            if current_price < nearest_support:
                distance = (nearest_support - current_price) / nearest_support
                if distance >= self.breakout_threshold:
                    result['breakout_direction'] = -1
                    result['breakout_level'] = nearest_support
                    result['breakout_distance'] = distance
                    result['breakout_strength'] = min(1.0, distance / self.breakout_threshold)
        return result

    def _calculate_volume_confirmation(self, volumes: List[float]) -> float:
        if len(volumes) < 10:
            return 1.0
        current_volume = volumes[-1]
        avg_volume = statistics.mean(volumes[-10:])
        if avg_volume > 0:
            volume_ratio = current_volume / avg_volume
            if volume_ratio >= self.volume_spike_threshold:
                return min(2.0, 1.0 + (volume_ratio - self.volume_spike_threshold) * 0.5)
            elif volume_ratio >= 1.2:
                return 1.0 + (volume_ratio - 1.0) * 0.5
            else:
                return max(0.5, volume_ratio)
        return 1.0

    def _determine_signal_type(self, signal_strength: float, factors: dict) -> SignalType:
        breakout_direction = factors.get('breakout_direction', 0)
        breakout_strength = factors.get('breakout_strength', 0)
        volume_confirmation = factors.get('volume_confirmation', 1.0)
        if breakout_direction == 0:
            return SignalType.HOLD
        if signal_strength > 0.7 and breakout_strength > 0.8 and volume_confirmation > 1.5:
            return SignalType.STRONG_BUY if breakout_direction > 0 else SignalType.STRONG_SELL
        elif signal_strength > 0.5 and breakout_strength > 0.5:
            return SignalType.BUY if breakout_direction > 0 else SignalType.SELL
        else:
            return SignalType.HOLD

    def _assess_market_conditions(self, market_data: List[MarketData]) -> float:
        return 1.0

    def _build_reasoning(self, factors, signal_strength, support_levels, resistance_levels) -> List[str]:
        return ["Breakout detected."]

    def _calculate_dynamic_levels(self, current_price, signal_type, support_levels, resistance_levels) -> Tuple[float, float]:
        return current_price * 0.95, current_price * 1.10


class VolumeBreakoutStrategy(BreakoutStrategy):
    def __init__(self, config: TradingConfig):
        super().__init__(config)
        self.name = "VolumeBreakoutStrategy"
        self.volume_ma_period = 20
        self.volume_spike_levels = [1.5, 2.0, 3.0]
        self.accumulation_detection = True

    def _calculate_volume_confirmation(self, volumes: List[float]) -> float:
        if len(volumes) < self.volume_ma_period:
            return super()._calculate_volume_confirmation(volumes)
        current_volume = volumes[-1]
        sma_volume = statistics.mean(volumes[-self.volume_ma_period:])
        recent_avg = statistics.mean(volumes[-5:])
        volume_trend = self._calculate_volume_trend(volumes)
        volume_spike_score = 0
        if sma_volume > 0:
            spike_ratio = current_volume / sma_volume
            for i, threshold in enumerate(self.volume_spike_levels):
                if spike_ratio >= threshold:
                    volume_spike_score = 1.0 + i * 0.3
        trend_score = max(0.5, 1.0 + volume_trend)
        if sma_volume > 0:
            acceleration = (recent_avg - sma_volume) / sma_volume
            acceleration_score = max(0.7, 1.0 + acceleration)
        else:
            acceleration_score = 1.0
        final_score = (volume_spike_score * 0.5 + trend_score * 0.3 + acceleration_score * 0.2)
        return min(2.5, max(0.3, final_score))

    def _calculate_volume_trend(self, volumes: List[float]) -> float:
        if len(volumes) < 10:
            return 0.0
        recent_volumes = volumes[-10:]
        trend_slope = self._calculate_trend_slope(recent_volumes)
        mean_volume = statistics.mean(recent_volumes)
        normalized_trend = (trend_slope * 10) / mean_volume if mean_volume > 0 else 0
        return max(-1.0, min(1.0, normalized_trend))
