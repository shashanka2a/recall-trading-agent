"""
Momentum trading strategy implementation
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


class MomentumStrategy(BaseStrategy):
    """
    Momentum strategy that identifies and follows price trends.
    
    Signals:
    - BUY when price momentum is positive with volume confirmation
    - SELL when price momentum turns negative
    - Uses multiple timeframes for confirmation
    """
    
    def __init__(self, config: TradingConfig):
        super().__init__(config, StrategyType.MOMENTUM)
        
        # Strategy-specific parameters
        self.momentum_threshold = 0.03  # 3% momentum threshold
        self.volume_threshold = 1.5     # 1.5x average volume for confirmation
        self.trend_confirmation_periods = 3  # Periods for trend confirmation
        
        # Scoring weights
        self.factor_weights = {
            'price_momentum': 0.4,
            'sma_crossover': 0.3,
            'volume_confirmation': 0.2,
            'rsi_filter': 0.1
        }
    
    async def generate_signal(
        self, 
        symbol: str, 
        market_data: List[MarketData], 
        portfolio: Portfolio
    ) -> Optional[TradingSignal]:
        """Generate momentum-based trading signal"""
        
        if len(market_data) < self.config.lookback_periods["medium"]:
            logger.debug(f"Insufficient data for {symbol}: {len(market_data)} periods")
            return None
        
        # Extract price and volume data
        prices = [data.price for data in market_data]
        volumes = [data.volume for data in market_data]
        current_price = prices[-1]
        
        # Calculate momentum factors
        factors = self._calculate_momentum_factors(prices, volumes)
        
        # Calculate overall signal strength
        signal_strength = self.calculate_signal_strength_score(factors, self.factor_weights)
        
        # Determine signal type
        signal_type = self._determine_signal_type(signal_strength, factors)
        
        if signal_type == SignalType.HOLD:
            return None
        
        # Calculate confidence with additional factors
        additional_factors = {
            'volume_confirmation': factors.get('volume_confirmation', 1.0),
            'trend_alignment': abs(factors.get('price_momentum', 0)),
            'market_conditions': self._assess_market_conditions(market_data)
        }
        
        confidence = self.calculate_confidence(signal_strength, additional_factors)
        
        # Calculate target allocation
        allocation_multiplier = min(abs(signal_strength), 1.0)
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
            take_profit=current_price * (1 + self.config.take_profit_percent),
            reasoning=reasoning,
            metadata={
                'signal_strength': signal_strength,
                'factors': factors,
                'current_price': current_price
            }
        )
        
        return signal
    
    def _calculate_momentum_factors(self, prices: List[float], volumes: List[float]) -> dict:
        """Calculate various momentum factors"""
        factors = {}
        
        # Price momentum (multiple timeframes)
        short_momentum = self._calculate_price_momentum(prices, self.config.lookback_periods["short"])
        medium_momentum = self._calculate_price_momentum(prices, self.config.lookback_periods["medium"])
        
        # Weighted average momentum
        factors['price_momentum'] = (short_momentum * 0.6 + medium_momentum * 0.4)
        
        # Moving average crossover
        sma_short = TechnicalIndicators.sma(prices, self.config.lookback_periods["short"])
        sma_long = TechnicalIndicators.sma(prices, self.config.lookback_periods["medium"])
        
        if sma_long > 0:
            sma_ratio = (sma_short - sma_long) / sma_long
            factors['sma_crossover'] = max(-1.0, min(1.0, sma_ratio * 20))  # Normalize
        else:
            factors['sma_crossover'] = 0.0
        
        # Volume confirmation
        factors['volume_confirmation'] = self._calculate_volume_factor(volumes)
        
        # RSI filter (contrarian for overbought/oversold)
        rsi = TechnicalIndicators.rsi(prices)
        if rsi > 70:
            factors['rsi_filter'] = -0.5  # Reduce buy signals when overbought
        elif rsi < 30:
            factors['rsi_filter'] = 0.5   # Enhance buy signals when oversold
        else:
            factors['rsi_filter'] = 0.0
        
        # Acceleration (momentum of momentum)
        if len(prices) >= 10:
            recent_momentum = self._calculate_price_momentum(prices[-10:], 5)
            older_momentum = self._calculate_price_momentum(prices[-15:-5], 5)
            factors['acceleration'] = recent_momentum - older_momentum
        else:
            factors['acceleration'] = 0.0
        
        return factors
    
    def _calculate_price_momentum(self, prices: List[float], period: int) -> float:
        """Calculate price momentum over specified period"""
        if len(prices) < period + 1:
            return 0.0
        
        start_price = prices[-(period + 1)]
        end_price = prices[-1]
        
        if start_price > 0:
            return (end_price - start_price) / start_price
        return 0.0
    
    def _calculate_volume_factor(self, volumes: List[float]) -> float:
        """Calculate volume confirmation factor"""
        if len(volumes) < 10:
            return 1.0
        
        recent_volume = volumes[-1]
        avg_volume = statistics.mean(volumes[-10:])
        
        if avg_volume > 0:
            volume_ratio = recent_volume / avg_volume
            
            # Convert to factor between 0.5 and 1.5
            if volume_ratio > self.volume_threshold:
                return min(1.5, 1.0 + (volume_ratio - 1.0) * 0.3)
            else:
                return max(0.5, volume_ratio / self.volume_threshold)
        
        return 1.0
    
    def _determine_signal_type(self, signal_strength: float, factors: dict) -> SignalType:
        """Determine signal type based on strength and factors"""
        
        # Strong signals require multiple confirmations
        if signal_strength > 0.7 and factors.get('volume_confirmation', 1.0) > 1.2:
            return SignalType.STRONG_BUY
        elif signal_strength > 0.4:
            return SignalType.BUY
        elif signal_strength < -0.7 and factors.get('volume_confirmation', 1.0) > 1.2:
            return SignalType.STRONG_SELL
        elif signal_strength < -0.4:
            return SignalType.SELL
        else:
            return SignalType.HOLD
    
    def _assess_market_conditions(self, market_data: List[MarketData]) -> float:
        """Assess overall market conditions for momentum trading"""
        conditions = self.check_market_conditions(market_data)
        
        # Momentum works better in trending markets with moderate volatility
        volatility_score = 1.0 - abs(conditions['volatility'] - 0.5) * 2  # Prefer moderate volatility
        trend_score = abs(conditions['trend'])  # Prefer trending markets
        volume_score = conditions['volume']     # Prefer active markets
        
        return (volatility_score * 0.4 + trend_score * 0.4 + volume_score * 0.2)
    
    def _build_reasoning(self, factors: dict, signal_strength: float) -> List[str]:
        """Build reasoning components for the signal"""
        components = []
        
        # Price momentum
        momentum = factors.get('price_momentum', 0)
        if abs(momentum) > 0.02:
            direction = "positive" if momentum > 0 else "negative"
            components.append(f"{direction} momentum ({momentum:.2%})")
        
        # SMA crossover
        sma_cross = factors.get('sma_crossover', 0)
        if abs(sma_cross) > 0.1:
            direction = "bullish" if sma_cross > 0 else "bearish"
            components.append(f"SMA {direction} crossover")
        
        # Volume confirmation
        volume_factor = factors.get('volume_confirmation', 1.0)
        if volume_factor > 1.3:
            components.append(f"high volume confirmation ({volume_factor:.1f}x)")
        elif volume_factor < 0.7:
            components.append(f"low volume warning ({volume_factor:.1f}x)")
        
        # RSI filter
        rsi_filter = factors.get('rsi_filter', 0)
        if abs(rsi_filter) > 0.3:
            condition = "oversold" if rsi_filter > 0 else "overbought"
            components.append(f"RSI {condition} filter")
        
        # Acceleration
        acceleration = factors.get('acceleration', 0)
        if abs(acceleration) > 0.01:
            direction = "accelerating" if acceleration > 0 else "decelerating"
            components.append(f"momentum {direction}")
        
        if not components:
            components.append(f"signal strength: {signal_strength:.2f}")
        
        return components


class AdvancedMomentumStrategy(MomentumStrategy):
    """
    Advanced momentum strategy with additional features:
    - Multiple timeframe analysis
    - Momentum divergence detection
    - Adaptive thresholds based on volatility
    """
    
    def __init__(self, config: TradingConfig):
        super().__init__(config)
        self.name = "AdvancedMomentumStrategy"
        
        # Additional parameters
        self.volatility_adjustment = True
        self.divergence_detection = True
        self.min_trend_strength = 0.02
    
    async def generate_signal(
        self, 
        symbol: str, 
        market_data: List[MarketData], 
        portfolio: Portfolio
    ) -> Optional[TradingSignal]:
        """Generate advanced momentum signal with additional checks"""
        
        # Get base signal
        base_signal = await super().generate_signal(symbol, market_data, portfolio)
        
        if not base_signal:
            return None
        
        # Apply advanced filters
        if self.volatility_adjustment:
            base_signal = self._apply_volatility_adjustment(base_signal, market_data)
        
        if self.divergence_detection:
            divergence_factor = self._detect_price_volume_divergence(market_data)
            base_signal.confidence *= divergence_factor
            
            if divergence_factor < 0.8:
                base_signal.metadata['divergence_warning'] = True
        
        # Check minimum trend strength
        trend_strength = abs(base_signal.metadata.get('factors', {}).get('price_momentum', 0))
        if trend_strength < self.min_trend_strength:
            logger.debug(f"Trend strength {trend_strength:.3f} below minimum {self.min_trend_strength}")
            return None
        
        return base_signal
    
    def _apply_volatility_adjustment(self, signal: TradingSignal, market_data: List[MarketData]) -> TradingSignal:
        """Adjust signal based on current volatility"""
        if len(market_data) < 20:
            return signal
        
        prices = [data.price for data in market_data]
        returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
        volatility = statistics.stdev(returns) if len(returns) > 1 else 0
        
        # Adjust target allocation based on volatility
        # Higher volatility = smaller positions
        volatility_factor = max(0.5, min(1.5, 1.0 - volatility * 10))
        signal.target_allocation *= volatility_factor
        
        # Adjust confidence based on volatility regime
        # Moderate volatility is preferred for momentum
        optimal_volatility = 0.02  # 2% daily volatility
        volatility_penalty = abs(volatility - optimal_volatility) / optimal_volatility
        signal.confidence *= max(0.7, 1.0 - volatility_penalty * 0.5)
        
        signal.metadata['volatility_adjustment'] = volatility_factor
        signal.metadata['current_volatility'] = volatility
        
        return signal
    
    def _detect_price_volume_divergence(self, market_data: List[MarketData]) -> float:
        """Detect price-volume divergence which can weaken momentum signals"""
        if len(market_data) < 10:
            return 1.0
        
        prices = [data.price for data in market_data[-10:]]
        volumes = [data.volume for data in market_data[-10:]]
        
        # Calculate price and volume trends
        price_trend = (prices[-1] - prices[0]) / prices[0] if prices[0] > 0 else 0
        
        volume_trend = 0
        if len(volumes) > 5:
            recent_vol = statistics.mean(volumes[-3:])
            older_vol = statistics.mean(volumes[:3])
            volume_trend = (recent_vol - older_vol) / older_vol if older_vol > 0 else 0
        
        # Check for divergence
        if price_trend > 0.01 and volume_trend < -0.2:
            # Price up, volume down - bearish divergence
            return 0.7
        elif price_trend < -0.01 and volume_trend > 0.2:
            # Price down, volume up - could be selling climax
            return 0.8
        else:
            # No significant divergence
            return 1.0