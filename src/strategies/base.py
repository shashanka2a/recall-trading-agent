"""
Base strategy class and common strategy utilities
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any

from config.settings import TradingConfig
from src.core.models import MarketData, Portfolio, TradingSignal
from src.core.enums import StrategyType, SignalType

logger = logging.getLogger(__name__)


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies"""
    
    def __init__(self, config: TradingConfig, strategy_type: StrategyType):
        self.config = config
        self.strategy_type = strategy_type
        self.name = self.__class__.__name__
        self._signal_history: List[TradingSignal] = []
        
    @abstractmethod
    async def generate_signal(
        self, 
        symbol: str, 
        market_data: List[MarketData], 
        portfolio: Portfolio
    ) -> Optional[TradingSignal]:
        """Generate trading signal for symbol"""
        pass
    
    def calculate_confidence(self, signal_strength: float, additional_factors: Optional[Dict] = None) -> float:
        """Calculate confidence score based on signal strength and additional factors"""
        base_confidence = min(max(abs(signal_strength), 0.0), 1.0)
        
        if additional_factors:
            # Volume confirmation
            volume_factor = additional_factors.get('volume_confirmation', 1.0)
            base_confidence *= volume_factor
            
            # Trend alignment
            trend_factor = additional_factors.get('trend_alignment', 1.0)
            base_confidence *= trend_factor
            
            # Market conditions
            market_factor = additional_factors.get('market_conditions', 1.0)
            base_confidence *= market_factor
        
        return min(base_confidence, 1.0)
    
    def validate_signal(self, signal: TradingSignal) -> bool:
        """Validate generated signal"""
        if not signal:
            return False
        
        # Check confidence threshold
        if signal.confidence < self.config.min_confidence:
            logger.debug(f"{self.name}: Signal confidence {signal.confidence:.2f} below threshold")
            return False
        
        # Check target allocation
        if signal.target_allocation > self.config.max_position_size:
            logger.warning(f"{self.name}: Target allocation {signal.target_allocation:.2f} exceeds max")
            signal.target_allocation = self.config.max_position_size
        
        return True
    
    def add_signal_to_history(self, signal: TradingSignal) -> None:
        """Add signal to history for tracking"""
        self._signal_history.append(signal)
        
        # Keep only recent signals (last 100)
        if len(self._signal_history) > 100:
            self._signal_history = self._signal_history[-100:]
    
    def get_recent_signals(self, symbol: str, count: int = 10) -> List[TradingSignal]:
        """Get recent signals for symbol"""
        symbol_signals = [s for s in self._signal_history if s.symbol == symbol]
        return symbol_signals[-count:] if symbol_signals else []
    
    def calculate_signal_strength_score(self, factors: Dict[str, float], weights: Dict[str, float]) -> float:
        """Calculate weighted signal strength score"""
        score = 0.0
        total_weight = 0.0
        
        for factor, value in factors.items():
            weight = weights.get(factor, 0.0)
            if weight > 0:
                score += value * weight
                total_weight += weight
        
        return score / total_weight if total_weight > 0 else 0.0
    
    def check_market_conditions(self, market_data: List[MarketData]) -> Dict[str, float]:
        """Analyze overall market conditions"""
        if len(market_data) < 10:
            return {"volatility": 0.5, "trend": 0.0, "volume": 0.5}
        
        prices = [data.price for data in market_data]
        volumes = [data.volume for data in market_data]
        
        # Calculate volatility (coefficient of variation)
        price_mean = sum(prices) / len(prices)
        price_variance = sum((p - price_mean) ** 2 for p in prices) / len(prices)
        volatility = (price_variance ** 0.5) / price_mean if price_mean > 0 else 0
        
        # Calculate trend (linear regression slope approximation)
        n = len(prices)
        sum_x = sum(range(n))
        sum_y = sum(prices)
        sum_xy = sum(i * prices[i] for i in range(n))
        sum_x2 = sum(i * i for i in range(n))
        
        if n * sum_x2 - sum_x * sum_x != 0:
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            trend = slope / price_mean if price_mean > 0 else 0
        else:
            trend = 0
        
        # Calculate volume trend
        volume_mean = sum(volumes) / len(volumes)
        recent_volume = sum(volumes[-5:]) / min(5, len(volumes))
        volume_trend = (recent_volume - volume_mean) / volume_mean if volume_mean > 0 else 0
        
        return {
            "volatility": min(volatility * 10, 1.0),  # Normalize
            "trend": max(-1.0, min(trend * 1000, 1.0)),  # Normalize and clamp
            "volume": max(0.0, min(volume_trend + 0.5, 1.0))  # Normalize
        }
    
    def format_reasoning(self, components: List[str]) -> str:
        """Format reasoning string from components"""
        if not components:
            return f"{self.strategy_type.value} strategy signal"
        
        return f"{self.strategy_type.value}: {'; '.join(components)}"


class StrategyManager:
    """Manages multiple trading strategies"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.strategies: List[BaseStrategy] = []
        self.strategy_weights = config.strategy_weights.copy()
        
    def add_strategy(self, strategy: BaseStrategy) -> None:
        """Add strategy to manager"""
        self.strategies.append(strategy)
        logger.info(f"Added strategy: {strategy.name}")
    
    def remove_strategy(self, strategy_type: StrategyType) -> None:
        """Remove strategy by type"""
        self.strategies = [s for s in self.strategies if s.strategy_type != strategy_type]
        if strategy_type in self.strategy_weights:
            del self.strategy_weights[strategy_type]
        logger.info(f"Removed strategy: {strategy_type.value}")
    
    async def generate_all_signals(
        self, 
        symbol: str, 
        market_data: List[MarketData], 
        portfolio: Portfolio
    ) -> List[TradingSignal]:
        """Generate signals from all strategies"""
        signals = []
        
        for strategy in self.strategies:
            try:
                signal = await strategy.generate_signal(symbol, market_data, portfolio)
                
                if signal and strategy.validate_signal(signal):
                    strategy.add_signal_to_history(signal)
                    signals.append(signal)
                    logger.debug(f"{strategy.name} generated signal: {signal.signal.name}")
                
            except Exception as e:
                logger.error(f"Strategy {strategy.name} failed for {symbol}: {e}")
        
        return signals
    
    def aggregate_signals(self, signals: List[TradingSignal]) -> Optional[TradingSignal]:
        """Aggregate multiple signals using weighted voting"""
        if not signals:
            return None
        
        # Group signals by type
        signal_groups = {}
        for signal in signals:
            signal_type = signal.signal
            if signal_type not in signal_groups:
                signal_groups[signal_type] = []
            signal_groups[signal_type].append(signal)
        
        # Calculate weighted scores for each signal type
        signal_scores = {}
        for signal_type, group_signals in signal_groups.items():
            total_score = 0.0
            for signal in group_signals:
                strategy_weight = self.strategy_weights.get(signal.strategy, 0.2)
                weighted_confidence = signal.confidence * strategy_weight
                total_score += weighted_confidence
            
            signal_scores[signal_type] = total_score
        
        # Find best signal
        if not signal_scores:
            return None
        
        best_signal_type = max(signal_scores.keys(), key=lambda x: signal_scores[x])
        best_score = signal_scores[best_signal_type]
        
        # Check if signal is strong enough
        if best_score < self.config.min_confidence:
            return None
        
        # Create aggregated signal
        relevant_signals = signal_groups[best_signal_type]
        symbol = signals[0].symbol
        
        # Average target allocation from contributing signals
        avg_target = sum(s.target_allocation for s in relevant_signals) / len(relevant_signals)
        
        # Combine reasoning
        reasoning_parts = []
        for signal in relevant_signals:
            reasoning_parts.append(f"{signal.strategy.value}: {signal.reasoning}")
        
        return TradingSignal(
            symbol=symbol,
            signal=best_signal_type,
            confidence=min(best_score, 1.0),
            strategy=StrategyType.ML_ENSEMBLE,  # Mark as ensemble
            target_allocation=min(avg_target, self.config.max_position_size),
            reasoning=f"Ensemble ({len(relevant_signals)} strategies): {'; '.join(reasoning_parts)}",
            metadata={
                "contributing_strategies": len(relevant_signals),
                "total_score": best_score,
                "signal_distribution": {st.name: len(sigs) for st, sigs in signal_groups.items()}
            }
        )
    
    def update_strategy_weights(self, performance_data: Dict[StrategyType, float]) -> None:
        """Update strategy weights based on performance"""
        if not performance_data:
            return
        
        # Normalize performance scores
        total_performance = sum(abs(score) for score in performance_data.values())
        if total_performance == 0:
            return
        
        # Update weights based on relative performance
        for strategy_type, performance in performance_data.items():
            if strategy_type in self.strategy_weights:
                # Increase weight for better performing strategies
                weight_adjustment = (performance / total_performance) * 0.1  # 10% max adjustment
                new_weight = self.strategy_weights[strategy_type] + weight_adjustment
                self.strategy_weights[strategy_type] = max(0.05, min(0.5, new_weight))
        
        # Renormalize weights
        total_weight = sum(self.strategy_weights.values())
        if total_weight > 0:
            self.strategy_weights = {
                k: v / total_weight 
                for k, v in self.strategy_weights.items()
            }
        
        logger.info(f"Updated strategy weights: {self.strategy_weights}")
    
    def get_strategy_performance(self, lookback_days: int = 7) -> Dict[StrategyType, float]:
        """Calculate recent performance by strategy"""
        # This would typically analyze historical signals vs actual outcomes
        # For now, return placeholder data
        return {strategy.strategy_type: 0.0 for strategy in self.strategies}
    
    def get_active_strategies(self) -> List[str]:
        """Get list of active strategy names"""
        return [strategy.name for strategy in self.strategies]