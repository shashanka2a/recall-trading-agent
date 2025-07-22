"""
Volatility-based trading strategy implementation
"""

import logging
import statistics
import math
from typing import List, Optional, Dict

from .base import BaseStrategy
from config.settings import TradingConfig
from src.core.models import MarketData, Portfolio, TradingSignal
from src.core.enums import StrategyType, SignalType
from src.indicators.technical import TechnicalIndicators

logger = logging.getLogger(__name__)


class VolatilityStrategy(BaseStrategy):
    """
    Volatility strategy that trades based on volatility regime changes.
    
    Signals:
    - BUY during volatility expansion with trend confirmation
    - SELL during volatility contraction in downtrends
    - Uses ATR, volatility ratios, and regime detection
    """
    
    def __init__(self, config: TradingConfig):
        super().__init__(config, StrategyType.VOLATILITY)
        
        # Strategy-specific parameters
        self.volatility_lookback = 20       # Period for volatility calculation
        self.regime_threshold = 1.5         # Volatility regime change threshold
        self.low_vol_threshold = 0.8        # Low volatility threshold
        self.high_vol_threshold = 1.5       # High volatility threshold
        self.atr_period = 14                # ATR calculation period
        self.vol_ma_period = 10             # Volatility moving average period
        
        # Scoring weights
        self.factor_weights = {
            'volatility_regime': 0.35,
            'volatility_trend': 0.25,
            'price_momentum': 0.20,
            'regime_persistence': 0.15,
            'mean_reversion_signal': 0.05
        }
    
    async def generate_signal(
        self, 
        symbol: str, 
        market_data: List[MarketData], 
        portfolio: Portfolio
    ) -> Optional[TradingSignal]:
        """Generate volatility-based trading signal"""
        
        if len(market_data) < self.config.lookback_periods["long"]:
            logger.debug(f"Insufficient data for {symbol}: {len(market_data)} periods")
            return None
        
        # Extract price data
        prices = [data.price for data in market_data]
        highs = [data.high_24h or data.price for data in market_data]
        lows = [data.low_24h or data.price for data in market_data]
        volumes = [data.volume for data in market_data]
        current_price = prices[-1]
        
        # Calculate volatility factors
        factors = self._calculate_volatility_factors(prices, highs, lows, volumes)
        
        # Calculate overall signal strength
        signal_strength = self.calculate_signal_strength_score(factors, self.factor_weights)
        
        # Determine signal type
        signal_type = self._determine_signal_type(signal_strength, factors)
        
        if signal_type == SignalType.HOLD:
            return None
        
        # Calculate confidence with additional factors
        additional_factors = {
            'regime_strength': factors.get('volatility_regime', 0),
            'trend_alignment': factors.get('price_momentum', 0),
            'market_conditions': self._assess_market_conditions(market_data)
        }
        
        confidence = self.calculate_confidence(signal_strength, additional_factors)
        
        # Calculate target allocation (smaller positions for volatility strategies)
        allocation_multiplier = min(abs(signal_strength), 1.0) * 0.6  # More conservative
        target_allocation = allocation_multiplier * self.config.max_position_size
        
        # Adjust for volatility regime
        vol_regime = factors.get('current_volatility_regime', 'normal')
        if vol_regime == 'high':
            target_allocation *= 0.7  # Reduce position in high volatility
        elif vol_regime == 'low':
            target_allocation *= 1.2  # Increase position in low volatility
        
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
            take_profit=current_price * (1 + self.config.take_profit_percent * 0.7),  # Conservative
            reasoning=reasoning,
            metadata={
                'signal_strength': signal_strength,
                'factors': factors,
                'current_price': current_price,
                'volatility_regime': vol_regime,
                'current_volatility': factors.get('current_volatility', 0)
            }
        )
        
        return signal
    
    def _calculate_volatility_factors(
        self, 
        prices: List[float], 
        highs: List[float], 
        lows: List[float],
        volumes: List[float]
    ) -> dict:
        """Calculate various volatility factors"""
        factors = {}
        
        # Calculate returns for volatility metrics
        returns = self._calculate_returns(prices)
        
        # Current volatility
        current_volatility = self._calculate_realized_volatility(returns)
        factors['current_volatility'] = current_volatility
        
        # Historical volatility comparison
        factors.update(self._calculate_volatility_regime(returns))
        
        # ATR-based volatility
        atr = TechnicalIndicators.atr(highs, lows, prices, self.atr_period)
        factors['atr'] = atr
        factors['atr_ratio'] = self._calculate_atr_ratio(highs, lows, prices)
        
        # Volatility trend
        factors['volatility_trend'] = self._calculate_volatility_trend(returns)
        
        # Price momentum in context of volatility
        factors['price_momentum'] = self._calculate_volatility_adjusted_momentum(prices, returns)
        
        # Regime persistence
        factors['regime_persistence'] = self._calculate_regime_persistence(returns)
        
        # Mean reversion signals in high volatility
        factors['mean_reversion_signal'] = self._calculate_vol_mean_reversion(prices, current_volatility)
        
        # Volatility clustering
        factors['volatility_clustering'] = self._calculate_volatility_clustering(returns)
        
        # Volume-volatility relationship
        factors['volume_volatility_relationship'] = self._calculate_volume_volatility_relationship(
            volumes, returns
        )
        
        return factors
    
    def _calculate_returns(self, prices: List[float]) -> List[float]:
        """Calculate price returns"""
        if len(prices) < 2:
            return []
        
        returns = []
        for i in range(1, len(prices)):
            if prices[i-1] > 0:
                ret = (prices[i] - prices[i-1]) / prices[i-1]
                returns.append(ret)
        
        return returns
    
    def _calculate_realized_volatility(self, returns: List[float], periods: int = 20) -> float:
        """Calculate realized volatility"""
        if len(returns) < periods:
            return statistics.stdev(returns) if len(returns) > 1 else 0.0
        
        recent_returns = returns[-periods:]
        return statistics.stdev(recent_returns) if len(recent_returns) > 1 else 0.0
    
    def _calculate_volatility_regime(self, returns: List[float]) -> Dict[str, float]:
        """Determine current volatility regime"""
        result = {
            'volatility_regime': 0.0,
            'current_volatility_regime': 'normal',
            'volatility_percentile': 0.5
        }
        
        if len(returns) < self.volatility_lookback * 2:
            return result
        
        # Current volatility
        current_vol = self._calculate_realized_volatility(returns, self.volatility_lookback)
        
        # Historical volatility distribution
        historical_vols = []
        for i in range(self.volatility_lookback, len(returns)):
            if i >= self.volatility_lookback:
                window_returns = returns[i-self.volatility_lookback:i]
                vol = statistics.stdev(window_returns) if len(window_returns) > 1 else 0
                historical_vols.append(vol)
        
        if not historical_vols:
            return result
        
        # Calculate percentile
        sorted_vols = sorted(historical_vols)
        if current_vol <= sorted_vols[0]:
            percentile = 0.0
        elif current_vol >= sorted_vols[-1]:
            percentile = 1.0
        else:
            # Find position in sorted list
            position = 0
            for vol in sorted_vols:
                if current_vol > vol:
                    position += 1
                else:
                    break
            percentile = position / len(sorted_vols)
        
        result['volatility_percentile'] = percentile
        
        # Determine regime
        if percentile <= 0.2:
            result['current_volatility_regime'] = 'low'
            result['volatility_regime'] = 0.8  # Positive signal for low vol
        elif percentile >= 0.8:
            result['current_volatility_regime'] = 'high'
            result['volatility_regime'] = -0.6  # Negative signal for high vol
        else:
            result['current_volatility_regime'] = 'normal'
            result['volatility_regime'] = 0.0
        
        return result
    
    def _calculate_atr_ratio(self, highs: List[float], lows: List[float], prices: List[float]) -> float:
        """Calculate ATR ratio (current ATR vs historical average)"""
        if len(prices) < self.atr_period * 2:
            return 1.0
        
        current_atr = TechnicalIndicators.atr(highs, lows, prices, self.atr_period)
        
        # Historical ATR average
        historical_atrs = []
        for i in range(self.atr_period, len(prices) - self.atr_period):
            atr = TechnicalIndicators.atr(
                highs[i:i+self.atr_period], 
                lows[i:i+self.atr_period], 
                prices[i:i+self.atr_period], 
                self.atr_period
            )
            historical_atrs.append(atr)
        
        if historical_atrs:
            avg_atr = statistics.mean(historical_atrs)
            return current_atr / avg_atr if avg_atr > 0 else 1.0
        
        return 1.0
    
    def _calculate_volatility_trend(self, returns: List[float]) -> float:
        """Calculate trend in volatility"""
        if len(returns) < self.volatility_lookback * 2:
            return 0.0
        
        # Calculate rolling volatilities
        volatilities = []
        for i in range(self.volatility_lookback, len(returns)):
            window_returns = returns[i-self.volatility_lookback:i]
            vol = statistics.stdev(window_returns) if len(window_returns) > 1 else 0
            volatilities.append(vol)
        
        if len(volatilities) < self.vol_ma_period:
            return 0.0
        
        # Calculate trend in volatility
        recent_vol = statistics.mean(volatilities[-5:]) if len(volatilities) >= 5 else volatilities[-1]
        older_vol = statistics.mean(volatilities[-15:-10]) if len(volatilities) >= 15 else volatilities[0]
        
        if older_vol > 0:
            vol_trend = (recent_vol - older_vol) / older_vol
            return max(-1.0, min(1.0, vol_trend * 5))  # Normalize
        
        return 0.0
    
    def _calculate_volatility_adjusted_momentum(self, prices: List[float], returns: List[float]) -> float:
        """Calculate momentum adjusted for volatility"""
        if len(prices) < 10 or len(returns) < 10:
            return 0.0
        
        # Price momentum
        price_momentum = (prices[-1] - prices[-10]) / prices[-10] if prices[-10] > 0 else 0
        
        # Current volatility
        current_vol = self._calculate_realized_volatility(returns[-10:])
        
        # Volatility-adjusted momentum
        if current_vol > 0:
            adjusted_momentum = price_momentum / (current_vol * math.sqrt(10))  # Adjust for time
            return max(-1.0, min(1.0, adjusted_momentum))
        
        return 0.0
    
    def _calculate_regime_persistence(self, returns: List[float]) -> float:
        """Calculate how persistent the current volatility regime is"""
        if len(returns) < self.volatility_lookback * 3:
            return 0.5
        
        # Calculate recent volatility regimes
        recent_vols = []
        for i in range(len(returns) - self.volatility_lookback * 2, len(returns), 5):
            if i >= self.volatility_lookback:
                window_returns = returns[i-self.volatility_lookback:i]
                vol = statistics.stdev(window_returns) if len(window_returns) > 1 else 0
                recent_vols.append(vol)
        
        if len(recent_vols) < 3:
            return 0.5
        
        # Check consistency of volatility regime
        vol_mean = statistics.mean(recent_vols)
        vol_std = statistics.stdev(recent_vols) if len(recent_vols) > 1 else 0
        
        # Low standard deviation = high persistence
        if vol_mean > 0:
            persistence = 1.0 - min(1.0, (vol_std / vol_mean) * 2)
            return max(0.0, persistence)
        
        return 0.5
    
    def _calculate_vol_mean_reversion(self, prices: List[float], current_volatility: float) -> float:
        """Calculate mean reversion signals during high volatility periods"""
        if len(prices) < 20:
            return 0.0
        
        # Only generate mean reversion signals in high volatility
        if current_volatility < 0.02:  # 2% daily volatility threshold
            return 0.0
        
        # Calculate distance from moving average
        sma_20 = TechnicalIndicators.sma(prices, 20)
        current_price = prices[-1]
        
        if sma_20 > 0:
            distance = (current_price - sma_20) / sma_20
            
            # In high volatility, look for mean reversion opportunities
            if distance > 0.05:  # 5% above mean
                return -0.5  # Sell signal
            elif distance < -0.05:  # 5% below mean
                return 0.5   # Buy signal
        
        return 0.0
    
    def _calculate_volatility_clustering(self, returns: List[float]) -> float:
        """Detect volatility clustering patterns"""
        if len(returns) < 20:
            return 0.5
        
        # Calculate squared returns (proxy for volatility)
        squared_returns = [r**2 for r in returns[-20:]]
        
        # Check for clustering using autocorrelation
        mean_sq_returns = statistics.mean(squared_returns)
        
        # Simple clustering detection
        recent_volatility = statistics.mean(squared_returns[-5:])
        historical_volatility = statistics.mean(squared_returns[:-5])
        
        if historical_volatility > 0:
            clustering_ratio = recent_volatility / historical_volatility
            
            if clustering_ratio > 1.5:
                return 0.8  # High volatility cluster
            elif clustering_ratio < 0.7:
                return 0.3  # Low volatility cluster
        
        return 0.5  # Normal clustering
    
    def _calculate_volume_volatility_relationship(self, volumes: List[float], returns: List[float]) -> float:
        """Analyze relationship between volume and volatility"""
        if len(volumes) < 20 or len(returns) < 20:
            return 0.5
        
        # Calculate correlation between volume and absolute returns
        recent_volumes = volumes[-20:]
        recent_abs_returns = [abs(r) for r in returns[-20:]]
        
        if len(recent_volumes) == len(recent_abs_returns):
            # Simple correlation calculation
            vol_mean = statistics.mean(recent_volumes)
            ret_mean = statistics.mean(recent_abs_returns)
            
            covariance = sum((v - vol_mean) * (r - ret_mean) 
                           for v, r in zip(recent_volumes, recent_abs_returns)) / len(recent_volumes)
            
            vol_std = statistics.stdev(recent_volumes) if len(recent_volumes) > 1 else 0
            ret_std = statistics.stdev(recent_abs_returns) if len(recent_abs_returns) > 1 else 0
            
            if vol_std > 0 and ret_std > 0:
                correlation = covariance / (vol_std * ret_std)
                return max(0.0, min(1.0, (correlation + 1) / 2))  # Normalize to 0-1
        
        return 0.5
    
    def _determine_signal_type(self, signal_strength: float, factors: dict) -> SignalType:
        """Determine signal type based on volatility analysis"""
        
        vol_regime = factors.get('current_volatility_regime', 'normal')
        vol_trend = factors.get('volatility_trend', 0)
        price_momentum = factors.get('price_momentum', 0)
        mean_reversion = factors.get('mean_reversion_signal', 0)
        
        # Low volatility regime - trend following
        if vol_regime == 'low':
            if signal_strength > 0.6 and price_momentum > 0.3:
                return SignalType.BUY
            elif signal_strength < -0.6 and price_momentum < -0.3:
                return SignalType.SELL
        
        # High volatility regime - mean reversion
        elif vol_regime == 'high':
            if mean_reversion > 0.3:
                return SignalType.BUY
            elif mean_reversion < -0.3:
                return SignalType.SELL
        
        # Normal volatility - moderate signals
        else:
            if signal_strength > 0.5:
                return SignalType.BUY if price_momentum > 0 else SignalType.HOLD
            elif signal_strength < -0.5:
                return SignalType.SELL if price_momentum < 0 else SignalType.HOLD
        
        return SignalType.HOLD
    
    def _assess_market_conditions(self, market_data: List[MarketData]) -> float:
        """Assess market conditions for volatility trading"""
        conditions = self.check_market_conditions(market_data)
        
        # Volatility strategies work in all market conditions but prefer:
        # - Moderate to high volatility
        # - Active volume
        # - Clear regime definition
        
        volatility_score = min(1.0, conditions['volatility'] * 1.5)
        volume_score = conditions['volume']
        trend_score = 0.8  # Volatility strategies are trend-agnostic
        
        return (volatility_score * 0.5 + volume_score * 0.3 + trend_score * 0.2)
    
    def _build_reasoning(self, factors: dict, signal_strength: float) -> List[str]:
        """Build reasoning components for the signal"""
        components = []
        
        # Volatility regime
        vol_regime = factors.get('current_volatility_regime', 'normal')
        vol_percentile = factors.get('volatility_percentile', 0.5)
        components.append(f"{vol_regime} volatility regime ({vol_percentile:.1%} percentile)")
        
        # Volatility trend
        vol_trend = factors.get('volatility_trend', 0)
        if abs(vol_trend) > 0.3:
            direction = "increasing" if vol_trend > 0 else "decreasing"
            components.append(f"volatility {direction}")
        
        # ATR analysis
        atr_ratio = factors.get('atr_ratio', 1.0)
        if atr_ratio > 1.3:
            components.append(f"high ATR ({atr_ratio:.2f}x)")
        elif atr_ratio < 0.7:
            components.append(f"low ATR ({atr_ratio:.2f}x)")
        
        # Price momentum
        momentum = factors.get('price_momentum', 0)
        if abs(momentum) > 0.3:
            direction = "positive" if momentum > 0 else "negative"
            components.append(f"{direction} vol-adjusted momentum")
        
        # Mean reversion signal
        mean_rev = factors.get('mean_reversion_signal', 0)
        if abs(mean_rev) > 0.3:
            direction = "oversold" if mean_rev > 0 else "overbought"
            components.append(f"high-vol {direction}")
        
        # Regime persistence
        persistence = factors.get('regime_persistence', 0.5)
        if persistence > 0.7:
            components.append("persistent regime")
        elif persistence < 0.3:
            components.append("changing regime")
        
        return components


class AdaptiveVolatilityStrategy(VolatilityStrategy):
    """
    Advanced volatility strategy with adaptive parameters
    """
    
    def __init__(self, config: TradingConfig):
        super().__init__(config)
        self.name = "AdaptiveVolatilityStrategy"
        
        # Adaptive parameters
        self.regime_history = []
        self.performance_tracking = {}
        self.adaptive_thresholds = True
    
    async def generate_signal(
        self, 
        symbol: str, 
        market_data: List[MarketData], 
        portfolio: Portfolio
    ) -> Optional[TradingSignal]:
        """Generate adaptive volatility signal"""
        
        # Get base signal
        base_signal = await super().generate_signal(symbol, market_data, portfolio)
        
        if not base_signal:
            return None
        
        # Apply adaptive adjustments
        if self.adaptive_thresholds:
            base_signal = self._apply_adaptive_adjustments(base_signal, market_data, symbol)
        
        # Update regime history
        self._update_regime_history(base_signal.metadata.get('volatility_regime', 'normal'))
        
        return base_signal
    
    def _apply_adaptive_adjustments(self, signal: TradingSignal, market_data: List[MarketData], symbol: str) -> TradingSignal:
        """Apply adaptive adjustments based on recent performance"""
        
        # Adjust confidence based on recent regime stability
        regime_stability = self._calculate_regime_stability()
        signal.confidence *= regime_stability
        
        # Adjust position size based on recent volatility performance
        vol_performance = self.performance_tracking.get(symbol, {}).get('volatility_performance', 1.0)
        signal.target_allocation *= vol_performance
        
        # Adaptive stop loss based on current volatility
        current_vol = signal.metadata.get('current_volatility', 0.02)
        vol_multiplier = max(0.5, min(2.0, current_vol * 50))  # Scale with volatility
        
        if signal.stop_loss:
            current_price = signal.metadata.get('current_price', 0)
            if current_price > 0:
                stop_distance = abs(signal.stop_loss - current_price) / current_price
                adaptive_stop_distance = stop_distance * vol_multiplier
                
                if signal.signal in [SignalType.BUY, SignalType.STRONG_BUY]:
                    signal.stop_loss = current_price * (1 - adaptive_stop_distance)
                else:
                    signal.stop_loss = current_price * (1 + adaptive_stop_distance)
        
        signal.metadata['adaptive_adjustments'] = {
            'regime_stability': regime_stability,
            'vol_performance': vol_performance,
            'vol_multiplier': vol_multiplier
        }
        
        return signal
    
    def _update_regime_history(self, regime: str) -> None:
        """Update volatility regime history"""
        self.regime_history.append(regime)
        
        # Keep only recent history
        if len(self.regime_history) > 50:
            self.regime_history = self.regime_history[-50:]
    
    def _calculate_regime_stability(self) -> float:
        """Calculate stability of recent volatility regimes"""
        if len(self.regime_history) < 10:
            return 1.0
        
        recent_regimes = self.regime_history[-10:]
        
        # Count regime changes
        changes = 0
        for i in range(1, len(recent_regimes)):
            if recent_regimes[i] != recent_regimes[i-1]:
                changes += 1
        
        # Stability = 1 - (change_rate / max_possible_changes)
        max_changes = len(recent_regimes) - 1
        stability = 1.0 - (changes / max_changes) if max_changes > 0 else 1.0
        
        return max(0.5, stability)  # Minimum 50% confidence


class VolumeVolatilityStrategy(VolatilityStrategy):
    """
    Volatility strategy enhanced with volume analysis
    """
    
    def __init__(self, config: TradingConfig):
        super().__init__(config)
        self.name = "VolumeVolatilityStrategy"
        
        # Volume-volatility parameters
        self.volume_vol_correlation_threshold = 0.6
        self.volume_spike_vol_threshold = 2.0
    
    def _calculate_volume_volatility_relationship(self, volumes: List[float], returns: List[float]) -> float:
        """Enhanced volume-volatility relationship analysis"""
        base_score = super()._calculate_volume_volatility_relationship(volumes, returns)
        
        if len(volumes) < 10 or len(returns) < 10:
            return base_score
        
        # Additional volume-volatility metrics
        recent_volumes = volumes[-10:]
        recent_returns = [abs(r) for r in returns[-10:]]
        
        # Check for volume spikes during volatility spikes
        vol_spikes = [i for i, r in enumerate(recent_returns) if r > statistics.mean(recent_returns) * 1.5]
        vol_spike_volumes = [recent_volumes[i] for i in vol_spikes]
        
        if vol_spike_volumes:
            avg_spike_volume = statistics.mean(vol_spike_volumes)
            avg_normal_volume = statistics.mean(recent_volumes)
            
            if avg_normal_volume > 0:
                volume_amplification = avg_spike_volume / avg_normal_volume
                
                if volume_amplification > self.volume_spike_vol_threshold:
                    return min(1.0, base_score * 1.3)  # Bonus for volume confirmation
        
        return base_score