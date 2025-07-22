"""
Mean reversion trading strategy implementation
"""

import logging
import statistics
from typing import List, Optional

from .base import BaseStrategy
from config.settings import TradingConfig
from src.core.models import MarketData, Portfolio, TradingSignal
from src.core.enums import StrategyType, SignalType
from src.indicators.technical import TechnicalIndicators

logger = logging.getLogger(__name__)


class MeanReversionStrategy(BaseStrategy):
    """
    Mean reversion strategy that identifies oversold/overbought conditions.
    
    Signals:
    - BUY when price is significantly below mean (oversold)
    - SELL when price is significantly above mean (overbought)
    - Uses Bollinger Bands, Z-scores, and RSI for confirmation
    """
    
    def __init__(self, config: TradingConfig):
        super().__init__(config, StrategyType.MEAN_REVERSION)
        
        # Strategy-specific parameters
        self.bb_period = 20             # Bollinger Bands period
        self.bb_std_dev = 2.0           # Bollinger Bands standard deviations
        self.z_score_threshold = 2.0    # Z-score threshold for extreme values
        self.rsi_oversold = 30          # RSI oversold threshold
        self.rsi_overbought = 70        # RSI overbought threshold
        self.min_reversion_periods = 3  # Minimum periods for mean reversion
        
        # Scoring weights
        self.factor_weights = {
            'bollinger_position': 0.35,
            'z_score': 0.25,
            'rsi_divergence': 0.20,
            'price_distance': 0.15,
            'volume_confirmation': 0.05
        }
    
    async def generate_signal(
        self, 
        symbol: str, 
        market_data: List[MarketData], 
        portfolio: Portfolio
    ) -> Optional[TradingSignal]:
        """Generate mean reversion trading signal"""
        
        if len(market_data) < self.config.lookback_periods["medium"]:
            logger.debug(f"Insufficient data for {symbol}: {len(market_data)} periods")
            return None
        
        # Extract price and volume data
        prices = [data.price for data in market_data]
        volumes = [data.volume for data in market_data]
        current_price = prices[-1]
        
        # Calculate mean reversion factors
        factors = self._calculate_mean_reversion_factors(prices, volumes)
        
        # Calculate overall signal strength
        signal_strength = self.calculate_signal_strength_score(factors, self.factor_weights)
        
        # Determine signal type
        signal_type = self._determine_signal_type(signal_strength, factors)
        
        if signal_type == SignalType.HOLD:
            return None
        
        # Calculate confidence with additional factors
        additional_factors = {
            'volume_confirmation': factors.get('volume_confirmation', 1.0),
            'trend_alignment': self._assess_trend_strength(prices),
            'market_conditions': self._assess_market_conditions(market_data)
        }
        
        confidence = self.calculate_confidence(signal_strength, additional_factors)
        
        # Calculate target allocation (smaller for mean reversion)
        allocation_multiplier = min(abs(signal_strength), 1.0) * 0.8  # More conservative
        target_allocation = allocation_multiplier * self.config.max_position_size
        
        # Generate reasoning
        reasoning_components = self._build_reasoning(factors, signal_strength)
        reasoning = self.format_reasoning(reasoning_components)
        
        # Create signal
        signal = TradingSignal(
            symbol=symbol,
            signal=signal_type,
            confidence=confidence,
            strategy=self.strategy_type,
            target_allocation=target_allocation,
            stop_loss=current_price * (1 - self.config.stop_loss_percent),
            take_profit=current_price * (1 + self.config.take_profit_percent * 0.8),  # Conservative
            reasoning=reasoning,
            metadata={
                'signal_strength': signal_strength,
                'factors': factors,
                'current_price': current_price,
                'mean_price': factors.get('mean_price', current_price)
            }
        )
        
        return signal
    
    def _calculate_mean_reversion_factors(self, prices: List[float], volumes: List[float]) -> dict:
        """Calculate various mean reversion factors"""
        factors = {}
        current_price = prices[-1]
        
        # Bollinger Bands analysis
        upper, middle, lower = TechnicalIndicators.bollinger_bands(
            prices, self.bb_period, self.bb_std_dev
        )
        
        if upper > lower:
            bb_position = (current_price - lower) / (upper - lower)
            factors['bollinger_position'] = self._normalize_bb_position(bb_position)
        else:
            factors['bollinger_position'] = 0.0
        
        factors['bollinger_upper'] = upper
        factors['bollinger_middle'] = middle
        factors['bollinger_lower'] = lower
        
        # Z-score calculation
        lookback = min(self.bb_period, len(prices))
        recent_prices = prices[-lookback:]
        mean_price = statistics.mean(recent_prices)
        std_price = statistics.stdev(recent_prices) if len(recent_prices) > 1 else 0
        
        factors['mean_price'] = mean_price
        
        if std_price > 0:
            z_score = (current_price - mean_price) / std_price
            factors['z_score'] = max(-3.0, min(3.0, z_score))  # Clamp extreme values
        else:
            factors['z_score'] = 0.0
        
        # RSI analysis for momentum divergence
        rsi = TechnicalIndicators.rsi(prices)
        factors['rsi'] = rsi
        factors['rsi_divergence'] = self._calculate_rsi_divergence(rsi)
        
        # Price distance from moving averages
        sma_short = TechnicalIndicators.sma(prices, self.config.lookback_periods["short"])
        sma_long = TechnicalIndicators.sma(prices, self.config.lookback_periods["medium"])
        
        if sma_long > 0:
            short_distance = (current_price - sma_short) / sma_short
            long_distance = (current_price - sma_long) / sma_long
            factors['price_distance'] = (short_distance + long_distance) / 2
        else:
            factors['price_distance'] = 0.0
        
        # Volume confirmation
        factors['volume_confirmation'] = self._calculate_volume_factor(volumes)
        
        # Mean reversion strength (how long has price been away from mean)
        factors['reversion_strength'] = self._calculate_reversion_strength(prices, mean_price)
        
        return factors
    
    def _normalize_bb_position(self, bb_position: float) -> float:
        """Normalize Bollinger Band position to signal strength"""
        # Convert BB position to mean reversion signal
        # 0 = at lower band (oversold, buy signal)
        # 1 = at upper band (overbought, sell signal)
        # 0.5 = at middle (neutral)
        
        if bb_position <= 0.2:  # Near lower band
            return min(0.8, (0.2 - bb_position) * 4)  # Strong buy signal
        elif bb_position >= 0.8:  # Near upper band
            return max(-0.8, (0.8 - bb_position) * 4)  # Strong sell signal
        else:
            # Linear scaling around middle
            return (0.5 - bb_position) * 2
    
    def _calculate_rsi_divergence(self, rsi: float) -> float:
        """Calculate RSI divergence signal"""
        if rsi <= self.rsi_oversold:
            # Oversold condition - buy signal
            return min(1.0, (self.rsi_oversold - rsi) / 10)
        elif rsi >= self.rsi_overbought:
            # Overbought condition - sell signal
            return max(-1.0, (self.rsi_overbought - rsi) / 10)
        else:
            # Neutral zone
            return 0.0
    
    def _calculate_volume_factor(self, volumes: List[float]) -> float:
        """Calculate volume confirmation factor"""
        if len(volumes) < 5:
            return 1.0
        
        recent_volume = volumes[-1]
        avg_volume = statistics.mean(volumes[-10:]) if len(volumes) >= 10 else recent_volume
        
        if avg_volume > 0:
            volume_ratio = recent_volume / avg_volume
            # For mean reversion, we prefer moderate volume
            if 0.7 <= volume_ratio <= 1.5:
                return 1.0  # Optimal volume
            elif volume_ratio > 2.0:
                return 0.8  # High volume might indicate trend continuation
            else:
                return 0.9  # Low volume is acceptable for mean reversion
        
        return 1.0
    
    def _calculate_reversion_strength(self, prices: List[float], mean_price: float) -> float:
        """Calculate how long price has been away from mean"""
        if len(prices) < self.min_reversion_periods:
            return 0.0
        
        # Count consecutive periods away from mean
        consecutive_periods = 0
        threshold = 0.02  # 2% threshold
        
        for i in range(len(prices) - 1, -1, -1):
            if abs(prices[i] - mean_price) / mean_price > threshold:
                consecutive_periods += 1
            else:
                break
        
        # More periods away from mean = stronger reversion signal
        return min(1.0, consecutive_periods / 10)
    
    def _assess_trend_strength(self, prices: List[float]) -> float:
        """Assess overall trend strength (mean reversion works better in ranging markets)"""
        if len(prices) < 20:
            return 0.5
        
        # Calculate trend using linear regression slope
        n = len(prices[-20:])
        x_values = list(range(n))
        y_values = prices[-20:]
        
        # Simple linear regression
        sum_x = sum(x_values)
        sum_y = sum(y_values)
        sum_xy = sum(x * y for x, y in zip(x_values, y_values))
        sum_x2 = sum(x * x for x in x_values)
        
        if n * sum_x2 - sum_x * sum_x != 0:
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            mean_price = sum_y / n
            trend_strength = abs(slope) / mean_price if mean_price > 0 else 0
            
            # Mean reversion works better when trend is weak
            return max(0.3, 1.0 - min(trend_strength * 1000, 0.7))
        
        return 0.5
    
    def _determine_signal_type(self, signal_strength: float, factors: dict) -> SignalType:
        """Determine signal type based on strength and factors"""
        
        # Require multiple confirmations for strong signals
        z_score = factors.get('z_score', 0)
        rsi_divergence = factors.get('rsi_divergence', 0)
        bb_position = factors.get('bollinger_position', 0)
        
        # Strong buy conditions (oversold)
        if (signal_strength > 0.6 and 
            z_score < -1.5 and 
            rsi_divergence > 0.3 and
            bb_position > 0.3):
            return SignalType.STRONG_BUY
        
        # Regular buy conditions
        elif signal_strength > 0.4 and z_score < -1.0:
            return SignalType.BUY
        
        # Strong sell conditions (overbought)
        elif (signal_strength < -0.6 and 
              z_score > 1.5 and 
              rsi_divergence < -0.3 and
              bb_position < -0.3):
            return SignalType.STRONG_SELL
        
        # Regular sell conditions
        elif signal_strength < -0.4 and z_score > 1.0:
            return SignalType.SELL
        
        else:
            return SignalType.HOLD
    
    def _assess_market_conditions(self, market_data: List[MarketData]) -> float:
        """Assess market conditions for mean reversion trading"""
        conditions = self.check_market_conditions(market_data)
        
        # Mean reversion works better in:
        # - Moderate volatility (not too high, not too low)
        # - Ranging markets (low trend)
        # - Normal volume conditions
        
        volatility_score = 1.0 - abs(conditions['volatility'] - 0.4) * 2  # Prefer moderate volatility
        trend_score = 1.0 - abs(conditions['trend'])  # Prefer ranging markets
        volume_score = conditions['volume']  # Normal volume preference
        
        return (volatility_score * 0.5 + trend_score * 0.4 + volume_score * 0.1)
    
    def _build_reasoning(self, factors: dict, signal_strength: float) -> List[str]:
        """Build reasoning components for the signal"""
        components = []
        
        # Bollinger Bands
        bb_pos = factors.get('bollinger_position', 0)
        if bb_pos > 0.4:
            components.append(f"oversold (BB position: {bb_pos:.2f})")
        elif bb_pos < -0.4:
            components.append(f"overbought (BB position: {bb_pos:.2f})")
        
        # Z-score
        z_score = factors.get('z_score', 0)
        if abs(z_score) > 1.5:
            condition = "oversold" if z_score < 0 else "overbought"
            components.append(f"{condition} Z-score ({z_score:.2f})")
        
        # RSI
        rsi = factors.get('rsi', 50)
        if rsi <= 30:
            components.append(f"RSI oversold ({rsi:.1f})")
        elif rsi >= 70:
            components.append(f"RSI overbought ({rsi:.1f})")
        
        # Price distance from mean
        price_distance = factors.get('price_distance', 0)
        if abs(price_distance) > 0.05:
            direction = "below" if price_distance < 0 else "above"
            components.append(f"price {abs(price_distance):.2%} {direction} mean")
        
        # Reversion strength
        reversion_strength = factors.get('reversion_strength', 0)
        if reversion_strength > 0.5:
            components.append(f"extended move ({reversion_strength:.2f})")
        
        if not components:
            components.append(f"mean reversion signal ({signal_strength:.2f})")
        
        return components


class StatisticalMeanReversionStrategy(MeanReversionStrategy):
    """
    Advanced statistical mean reversion strategy with additional features:
    - Multiple timeframe analysis
    - Cointegration testing
    - Adaptive thresholds based on volatility regime
    """
    
    def __init__(self, config: TradingConfig):
        super().__init__(config)
        self.name = "StatisticalMeanReversionStrategy"
        
        # Additional parameters
        self.volatility_lookback = 30
        self.adaptive_thresholds = True
        self.cointegration_test = True
        self.multi_timeframe = True
    
    async def generate_signal(
        self, 
        symbol: str, 
        market_data: List[MarketData], 
        portfolio: Portfolio
    ) -> Optional[TradingSignal]:
        """Generate advanced mean reversion signal with statistical tests"""
        
        # Get base signal
        base_signal = await super().generate_signal(symbol, market_data, portfolio)
        
        if not base_signal:
            return None
        
        # Apply advanced filters
        if self.adaptive_thresholds:
            base_signal = self._apply_adaptive_thresholds(base_signal, market_data)
        
        if self.multi_timeframe:
            mtf_confirmation = self._multi_timeframe_confirmation(market_data)
            base_signal.confidence *= mtf_confirmation
            
            if mtf_confirmation < 0.7:
                base_signal.metadata['mtf_warning'] = True
        
        # Volatility regime adjustment
        vol_adjustment = self._volatility_regime_adjustment(market_data)
        base_signal.target_allocation *= vol_adjustment
        base_signal.metadata['volatility_regime'] = vol_adjustment
        
        return base_signal
    
    def _apply_adaptive_thresholds(self, signal: TradingSignal, market_data: List[MarketData]) -> TradingSignal:
        """Apply adaptive thresholds based on current volatility"""
        
        if len(market_data) < self.volatility_lookback:
            return signal
        
        prices = [data.price for data in market_data[-self.volatility_lookback:]]
        returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
        current_volatility = statistics.stdev(returns) if len(returns) > 1 else 0
        
        # Adjust thresholds based on volatility
        # Higher volatility = higher thresholds (more conservative)
        vol_multiplier = max(0.5, min(2.0, current_volatility * 50))
        
        # Adjust confidence based on volatility appropriateness
        optimal_volatility = 0.02  # 2% daily volatility is optimal for mean reversion
        vol_penalty = abs(current_volatility - optimal_volatility) / optimal_volatility
        signal.confidence *= max(0.5, 1.0 - vol_penalty * 0.5)
        
        signal.metadata['adaptive_threshold'] = vol_multiplier
        signal.metadata['current_volatility'] = current_volatility
        
        return signal
    
    def _multi_timeframe_confirmation(self, market_data: List[MarketData]) -> float:
        """Check for mean reversion signals across multiple timeframes"""
        
        if len(market_data) < 50:
            return 1.0
        
        # Check different timeframes
        timeframes = [5, 10, 20]  # Short, medium, long
        confirmations = []
        
        for tf in timeframes:
            if len(market_data) >= tf * 2:
                prices = [data.price for data in market_data[-tf*2:]]
                
                # Calculate mean reversion score for this timeframe
                recent_prices = prices[-tf:]
                older_prices = prices[:tf]
                
                recent_mean = statistics.mean(recent_prices)
                older_mean = statistics.mean(older_prices)
                overall_mean = statistics.mean(prices)
                
                # Check if price is reverting to mean
                current_price = prices[-1]
                mean_distance_recent = abs(current_price - recent_mean) / recent_mean
                mean_distance_overall = abs(current_price - overall_mean) / overall_mean
                
                # Reversion signal: current price closer to overall mean than recent mean
                if mean_distance_overall < mean_distance_recent:
                    confirmations.append(1.0)
                else:
                    confirmations.append(0.5)
        
        return statistics.mean(confirmations) if confirmations else 1.0
    
    def _volatility_regime_adjustment(self, market_data: List[MarketData]) -> float:
        """Adjust position size based on volatility regime"""
        
        if len(market_data) < 20:
            return 1.0
        
        prices = [data.price for data in market_data[-20:]]
        returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
        
        current_vol = statistics.stdev(returns) if len(returns) > 1 else 0
        
        # Historical volatility comparison
        if len(market_data) >= 60:
            historical_prices = [data.price for data in market_data[-60:-20]]
            historical_returns = [(historical_prices[i] - historical_prices[i-1]) / historical_prices[i-1] 
                                for i in range(1, len(historical_prices))]
            historical_vol = statistics.stdev(historical_returns) if len(historical_returns) > 1 else current_vol
            
            vol_ratio = current_vol / historical_vol if historical_vol > 0 else 1.0
            
            # Reduce position size in high volatility regimes
            if vol_ratio > 1.5:
                return 0.7  # Reduce position by 30%
            elif vol_ratio > 1.2:
                return 0.85  # Reduce position by 15%
            else:
                return 1.0  # Normal position size
        
        return 1.0