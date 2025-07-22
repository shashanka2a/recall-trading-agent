"""
Machine Learning Ensemble trading strategy implementation
"""

import logging
import statistics
import math
from typing import List, Optional, Dict, Tuple
import numpy as np

from .base import BaseStrategy
from config.settings import TradingConfig
from src.core.models import MarketData, Portfolio, TradingSignal
from src.core.enums import StrategyType, SignalType
from src.indicators.technical import TechnicalIndicators

logger = logging.getLogger(__name__)


class MLEnsembleStrategy(BaseStrategy):
    """
    Machine Learning Ensemble strategy that combines multiple features and models.
    
    Uses feature engineering and ensemble methods to generate trading signals:
    - Technical indicators
    - Price patterns
    - Volume analysis
    - Market microstructure
    - Sentiment proxies
    """
    
    def __init__(self, config: TradingConfig):
        super().__init__(config, StrategyType.ML_ENSEMBLE)
        
        # Feature engineering parameters
        self.feature_window = 30        # Lookback window for features
        self.prediction_horizon = 5     # Forward-looking prediction periods
        self.min_training_samples = 50  # Minimum samples for model training
        
        # Ensemble model weights
        self.model_weights = {
            'technical_model': 0.30,
            'momentum_model': 0.25,
            'mean_reversion_model': 0.20,
            'volatility_model': 0.15,
            'pattern_model': 0.10
        }
        
        # Feature importance weights
        self.feature_weights = {
            'price_features': 0.25,
            'momentum_features': 0.20,
            'volatility_features': 0.15,
            'volume_features': 0.15,
            'technical_features': 0.15,
            'pattern_features': 0.10
        }
        
        # Model state
        self.feature_history = []
        self.model_predictions = {}
        self.feature_importance = {}
    
    async def generate_signal(
        self, 
        symbol: str, 
        market_data: List[MarketData], 
        portfolio: Portfolio
    ) -> Optional[TradingSignal]:
        """Generate ML ensemble trading signal"""
        
        if len(market_data) < self.config.lookback_periods["long"]:
            logger.debug(f"Insufficient data for {symbol}: {len(market_data)} periods")
            return None
        
        # Extract features from market data
        features = self._extract_features(market_data)
        
        if not features:
            logger.debug(f"Failed to extract features for {symbol}")
            return None
        
        # Generate predictions from ensemble models
        predictions = self._generate_ensemble_predictions(features, market_data)
        
        # Combine predictions
        final_prediction = self._combine_predictions(predictions)
        
        # Convert prediction to trading signal
        signal_type, confidence = self._prediction_to_signal(final_prediction, features)
        
        if signal_type == SignalType.HOLD:
            return None
        
        # Calculate target allocation
        target_allocation = self._calculate_ml_allocation(final_prediction, confidence, features)
        
        # Generate reasoning
        reasoning = self._build_ml_reasoning(predictions, features, final_prediction)
        
        current_price = market_data[-1].price
        
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
                'ml_prediction': final_prediction,
                'model_predictions': predictions,
                'feature_summary': self._summarize_features(features),
                'confidence_breakdown': self._calculate_confidence_breakdown(predictions)
            }
        )
        
        return signal
    
    def _extract_features(self, market_data: List[MarketData]) -> Optional[Dict]:
        """Extract comprehensive feature set from market data"""
        
        if len(market_data) < self.feature_window:
            return None
        
        # Extract basic data
        prices = [data.price for data in market_data]
        volumes = [data.volume for data in market_data]
        highs = [data.high_24h or data.price for data in market_data]
        lows = [data.low_24h or data.price for data in market_data]
        
        features = {}
        
        # Price-based features
        features.update(self._extract_price_features(prices))
        
        # Momentum features
        features.update(self._extract_momentum_features(prices))
        
        # Volatility features
        features.update(self._extract_volatility_features(prices, highs, lows))
        
        # Volume features
        features.update(self._extract_volume_features(volumes, prices))
        
        # Technical indicator features
        features.update(self._extract_technical_features(prices, highs, lows, volumes))
        
        # Pattern features
        features.update(self._extract_pattern_features(prices, volumes))
        
        # Market microstructure features
        features.update(self._extract_microstructure_features(market_data))
        
        return features
    
    def _extract_price_features(self, prices: List[float]) -> Dict:
        """Extract price-based features"""
        features = {}
        current_price = prices[-1]
        
        # Price levels relative to recent history
        for period in [5, 10, 20]:
            if len(prices) >= period:
                sma = TechnicalIndicators.sma(prices, period)
                features[f'price_vs_sma_{period}'] = (current_price - sma) / sma if sma > 0 else 0
        
        # Price percentiles
        for window in [10, 20, 30]:
            if len(prices) >= window:
                recent_prices = prices[-window:]
                sorted_prices = sorted(recent_prices)
                position = sum(1 for p in sorted_prices if p <= current_price)
                features[f'price_percentile_{window}'] = position / len(sorted_prices)
        
        # Price acceleration
        if len(prices) >= 15:
            recent_slope = self._calculate_slope(prices[-5:])
            older_slope = self._calculate_slope(prices[-15:-10])
            features['price_acceleration'] = recent_slope - older_slope
        
        return features
    
    def _extract_momentum_features(self, prices: List[float]) -> Dict:
        """Extract momentum-based features"""
        features = {}
        
        # Rate of change over different periods
        for period in [3, 5, 10, 20]:
            if len(prices) > period:
                roc = (prices[-1] - prices[-period-1]) / prices[-period-1] if prices[-period-1] > 0 else 0
                features[f'roc_{period}'] = roc
        
        # MACD features
        if len(prices) >= 26:
            macd_line, signal_line, histogram = TechnicalIndicators.macd(prices)
            features['macd_line'] = macd_line
            features['macd_signal'] = signal_line
            features['macd_histogram'] = histogram
            features['macd_crossover'] = 1 if macd_line > signal_line else -1
        
        # Momentum oscillator
        if len(prices) >= 20:
            momentum = TechnicalIndicators.momentum(prices, 10)
            features['momentum_10'] = momentum
            
            # Momentum divergence
            price_change = (prices[-1] - prices[-10]) / prices[-10] if prices[-10] > 0 else 0
            features['momentum_divergence'] = momentum - price_change
        
        return features
    
    def _extract_volatility_features(self, prices: List[float], highs: List[float], lows: List[float]) -> Dict:
        """Extract volatility-based features"""
        features = {}
        
        # Historical volatility
        returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices)) if prices[i-1] > 0]
        
        for window in [5, 10, 20]:
            if len(returns) >= window:
                recent_returns = returns[-window:]
                vol = statistics.stdev(recent_returns) if len(recent_returns) > 1 else 0
                features[f'volatility_{window}'] = vol
        
        # ATR features
        if len(prices) >= 14:
            atr = TechnicalIndicators.atr(highs, lows, prices, 14)
            features['atr_14'] = atr
            
            # ATR relative to price
            features['atr_ratio'] = atr / prices[-1] if prices[-1] > 0 else 0
        
        # Volatility regime
        if len(returns) >= 20:
            short_vol = statistics.stdev(returns[-5:]) if len(returns) >= 5 else 0
            long_vol = statistics.stdev(returns[-20:]) if len(returns) >= 20 else 0
            features['volatility_regime'] = short_vol / long_vol if long_vol > 0 else 1
        
        # GARCH-like volatility clustering
        if len(returns) >= 10:
            squared_returns = [r**2 for r in returns[-10:]]
            features['volatility_clustering'] = statistics.stdev(squared_returns) if len(squared_returns) > 1 else 0
        
        return features
    
    def _extract_volume_features(self, volumes: List[float], prices: List[float]) -> Dict:
        """Extract volume-based features"""
        features = {}
        
        if not volumes or len(volumes) < 10:
            return features
        
        current_volume = volumes[-1]
        
        # Volume ratios
        for period in [5, 10, 20]:
            if len(volumes) >= period:
                avg_volume = statistics.mean(volumes[-period:])
                features[f'volume_ratio_{period}'] = current_volume / avg_volume if avg_volume > 0 else 1
        
        # Volume trend
        if len(volumes) >= 10:
            recent_vol = statistics.mean(volumes[-5:])
            older_vol = statistics.mean(volumes[-10:-5])
            features['volume_trend'] = (recent_vol - older_vol) / older_vol if older_vol > 0 else 0
        
        # On-Balance Volume
        if len(volumes) == len(prices) and len(prices) >= 10:
            obv = TechnicalIndicators.obv(prices, volumes)
            features['obv'] = obv
            
            # OBV trend
            if len(prices) >= 20:
                obv_older = TechnicalIndicators.obv(prices[-20:-10], volumes[-20:-10])
                features['obv_trend'] = (obv - obv_older) / abs(obv_older) if obv_older != 0 else 0
        
        # Volume-price trend
        if len(volumes) >= 10 and len(prices) >= 10:
            price_changes = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, min(len(prices), 10))]
            volume_changes = [(volumes[i] - volumes[i-1]) / volumes[i-1] for i in range(1, min(len(volumes), 10)) if volumes[i-1] > 0]
            
            if len(price_changes) == len(volume_changes) and len(price_changes) > 1:
                # Simple correlation
                correlation = self._calculate_correlation(price_changes, volume_changes)
                features['volume_price_correlation'] = correlation
        
        return features
    
    def _extract_technical_features(self, prices: List[float], highs: List[float], lows: List[float], volumes: List[float]) -> Dict:
        """Extract technical indicator features"""
        features = {}
        
        # RSI
        if len(prices) >= 14:
            rsi = TechnicalIndicators.rsi(prices, 14)
            features['rsi_14'] = rsi
            features['rsi_overbought'] = 1 if rsi > 70 else 0
            features['rsi_oversold'] = 1 if rsi < 30 else 0
        
        # Bollinger Bands
        if len(prices) >= 20:
            upper, middle, lower = TechnicalIndicators.bollinger_bands(prices, 20, 2)
            current_price = prices[-1]
            
            if upper > lower:
                bb_position = (current_price - lower) / (upper - lower)
                features['bb_position'] = bb_position
                features['bb_squeeze'] = (upper - lower) / middle if middle > 0 else 0
        
        # Stochastic Oscillator
        if len(prices) >= 14 and len(highs) >= 14 and len(lows) >= 14:
            k_percent, d_percent = TechnicalIndicators.stochastic(highs, lows, prices, 14, 3)
            features['stoch_k'] = k_percent
            features['stoch_d'] = d_percent
            features['stoch_crossover'] = 1 if k_percent > d_percent else -1
        
        # Williams %R
        if len(prices) >= 14:
            williams_r = TechnicalIndicators.williams_r(highs, lows, prices, 14)
            features['williams_r'] = williams_r
        
        # CCI
        if len(prices) >= 20:
            cci = TechnicalIndicators.cci(highs, lows, prices, 20)
            features['cci_20'] = cci
        
        return features
    
    def _extract_pattern_features(self, prices: List[float], volumes: List[float]) -> Dict:
        """Extract pattern-based features"""
        features = {}
        
        # Support and resistance
        if len(prices) >= 20:
            support_levels, resistance_levels = self._find_support_resistance(prices)
            current_price = prices[-1]
            
            # Distance to nearest support/resistance
            if support_levels:
                nearest_support = max(support_levels)
                features['support_distance'] = (current_price - nearest_support) / current_price
            
            if resistance_levels:
                nearest_resistance = min(resistance_levels)
                features['resistance_distance'] = (nearest_resistance - current_price) / current_price
        
        # Trend patterns
        if len(prices) >= 20:
            # Higher highs, higher lows pattern
            recent_highs = [max(prices[i:i+5]) for i in range(len(prices)-15, len(prices)-5, 5)]
            recent_lows = [min(prices[i:i+5]) for i in range(len(prices)-15, len(prices)-5, 5)]
            
            if len(recent_highs) >= 2:
                features['higher_highs'] = 1 if recent_highs[-1] > recent_highs[-2] else 0
            if len(recent_lows) >= 2:
                features['higher_lows'] = 1 if recent_lows[-1] > recent_lows[-2] else 0
        
        # Gap analysis
        if len(prices) >= 5:
            gaps = []
            for i in range(1, min(5, len(prices))):
                gap = (prices[-i] - prices[-i-1]) / prices[-i-1] if prices[-i-1] > 0 else 0
                gaps.append(abs(gap))
            
            features['avg_gap_size'] = statistics.mean(gaps) if gaps else 0
            features['max_recent_gap'] = max(gaps) if gaps else 0
        
        return features
    
    def _extract_microstructure_features(self, market_data: List[MarketData]) -> Dict:
        """Extract market microstructure features"""
        features = {}
        
        # Bid-ask spread analysis (if available)
        spreads = []
        for data in market_data[-10:]:
            if data.bid and data.ask and data.bid > 0:
                spread = (data.ask - data.bid) / data.price if data.price > 0 else 0
                spreads.append(spread)
        
        if spreads:
            features['avg_spread'] = statistics.mean(spreads)
            features['spread_volatility'] = statistics.stdev(spreads) if len(spreads) > 1 else 0
        
        # Price efficiency measures
        prices = [data.price for data in market_data[-20:]]
        if len(prices) >= 10:
            # Variance ratio test (simplified)
            returns_1 = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices)) if prices[i-1] > 0]
            
            if len(returns_1) >= 5:
                returns_5 = [(prices[i] - prices[i-5]) / prices[i-5] for i in range(5, len(prices)) if prices[i-5] > 0]
                
                if returns_5:
                    var_1 = statistics.variance(returns_1) if len(returns_1) > 1 else 0
                    var_5 = statistics.variance(returns_5) if len(returns_5) > 1 else 0
                    
                    features['variance_ratio'] = (var_5 / 5) / var_1 if var_1 > 0 else 1
        
        # Time-based features
        if len(market_data) >= 10:
            timestamps = [data.timestamp for data in market_data[-10:]]
            time_intervals = [(timestamps[i] - timestamps[i-1]).total_seconds() for i in range(1, len(timestamps))]
            
            if time_intervals:
                features['avg_time_interval'] = statistics.mean(time_intervals)
                features['time_irregularity'] = statistics.stdev(time_intervals) if len(time_intervals) > 1 else 0
        
        return features
    
    def _generate_ensemble_predictions(self, features: Dict, market_data: List[MarketData]) -> Dict:
        """Generate predictions from ensemble of models"""
        predictions = {}
        
        # Technical model
        predictions['technical_model'] = self._technical_model_predict(features)
        
        # Momentum model
        predictions['momentum_model'] = self._momentum_model_predict(features)
        
        # Mean reversion model
        predictions['mean_reversion_model'] = self._mean_reversion_model_predict(features)
        
        # Volatility model
        predictions['volatility_model'] = self._volatility_model_predict(features)
        
        # Pattern recognition model
        predictions['pattern_model'] = self._pattern_model_predict(features)
        
        return predictions
    
    def _technical_model_predict(self, features: Dict) -> float:
        """Technical indicator based model"""
        score = 0.0
        weight_sum = 0.0
        
        # RSI signal
        if 'rsi_14' in features:
            rsi = features['rsi_14']
            if rsi < 30:
                score += 0.8
            elif rsi > 70:
                score -= 0.8
            else:
                score += (50 - rsi) / 50 * 0.3
            weight_sum += 1.0
        
        # Bollinger Bands signal
        if 'bb_position' in features:
            bb_pos = features['bb_position']
            if bb_pos < 0.2:
                score += 0.6  # Near lower band - buy
            elif bb_pos > 0.8:
                score -= 0.6  # Near upper band - sell
            else:
                score += (0.5 - bb_pos) * 0.4
            weight_sum += 0.8
        
        # MACD signal
        if 'macd_crossover' in features:
            score += features['macd_crossover'] * 0.5
            weight_sum += 0.7
        
        # Stochastic signal
        if 'stoch_k' in features and 'stoch_d' in features:
            k = features['stoch_k']
            d = features['stoch_d']
            if k < 20 and k > d:
                score += 0.4
            elif k > 80 and k < d:
                score -= 0.4
            weight_sum += 0.5
        
        return score / weight_sum if weight_sum > 0 else 0.0
    
    def _momentum_model_predict(self, features: Dict) -> float:
        """Momentum-based model"""
        score = 0.0
        weight_sum = 0.0
        
        # Rate of change signals
        for period in [3, 5, 10]:
            feature_name = f'roc_{period}'
            if feature_name in features:
                roc = features[feature_name]
                # Weight longer periods less
                weight = 1.0 / period
                score += np.tanh(roc * 20) * weight  # Normalize ROC
                weight_sum += weight
        
        # Price vs SMA signals
        for period in [5, 10, 20]:
            feature_name = f'price_vs_sma_{period}'
            if feature_name in features:
                price_vs_sma = features[feature_name]
                weight = 1.0 / math.sqrt(period)
                score += np.tanh(price_vs_sma * 10) * weight
                weight_sum += weight
        
        # Momentum oscillator
        if 'momentum_10' in features:
            momentum = features['momentum_10']
            score += np.tanh(momentum / 5) * 0.8
            weight_sum += 0.8
        
        return score / weight_sum if weight_sum > 0 else 0.0
    
    def _mean_reversion_model_predict(self, features: Dict) -> float:
        """Mean reversion model"""
        score = 0.0
        weight_sum = 0.0
        
        # Price percentile signals
        for window in [10, 20]:
            feature_name = f'price_percentile_{window}'
            if feature_name in features:
                percentile = features[feature_name]
                # Mean reversion signal: buy low percentiles, sell high percentiles
                if percentile < 0.2:
                    score += 0.6
                elif percentile > 0.8:
                    score -= 0.6
                else:
                    score += (0.5 - percentile) * 0.8
                weight_sum += 1.0
        
        # Support/resistance distance
        if 'support_distance' in features:
            support_dist = features['support_distance']
            if support_dist < 0.02:  # Close to support
                score += 0.5
            weight_sum += 0.5
        
        if 'resistance_distance' in features:
            resistance_dist = features['resistance_distance']
            if resistance_dist < 0.02:  # Close to resistance
                score -= 0.5
            weight_sum += 0.5
        
        # RSI mean reversion
        if 'rsi_14' in features:
            rsi = features['rsi_14']
            if rsi < 25:
                score += 0.8
            elif rsi > 75:
                score -= 0.8
            weight_sum += 0.8
        
        return score / weight_sum if weight_sum > 0 else 0.0
    
    def _volatility_model_predict(self, features: Dict) -> float:
        """Volatility-based model"""
        score = 0.0
        weight_sum = 0.0
        
        # Volatility regime
        if 'volatility_regime' in features:
            vol_regime = features['volatility_regime']
            if vol_regime < 0.8:  # Low volatility regime
                # Look for momentum signals
                if 'roc_5' in features:
                    score += np.tanh(features['roc_5'] * 15) * 0.6
                    weight_sum += 0.6
            elif vol_regime > 1.2:  # High volatility regime
                # Look for mean reversion
                if 'price_percentile_10' in features:
                    percentile = features['price_percentile_10']
                    score += (0.5 - percentile) * 0.8
                    weight_sum += 0.8
            weight_sum += 0.5
        
        # ATR signals
        if 'atr_ratio' in features:
            atr_ratio = features['atr_ratio']
            if atr_ratio > 0.03:  # High ATR - reduce position confidence
                score *= 0.8
            weight_sum += 0.3
        
        # Volatility clustering
        if 'volatility_clustering' in features:
            clustering = features['volatility_clustering']
            # High clustering suggests continued volatility
            if clustering > 0.02:
                score *= 0.9  # Reduce confidence
            weight_sum += 0.2
        
        return score / weight_sum if weight_sum > 0 else 0.0
    
    def _pattern_model_predict(self, features: Dict) -> float:
        """Pattern recognition model"""
        score = 0.0
        weight_sum = 0.0
        
        # Trend pattern recognition
        if 'higher_highs' in features and 'higher_lows' in features:
            higher_highs = features['higher_highs']
            higher_lows = features['higher_lows']
            
            if higher_highs and higher_lows:
                score += 0.6  # Uptrend pattern
            elif not higher_highs and not higher_lows:
                score -= 0.6  # Downtrend pattern
            weight_sum += 1.0
        
        # Volume-price relationship
        if 'volume_price_correlation' in features:
            correlation = features['volume_price_correlation']
            # Positive correlation suggests trend continuation
            score += correlation * 0.4
            weight_sum += 0.4
        
        # Gap analysis
        if 'max_recent_gap' in features:
            max_gap = features['max_recent_gap']
            if max_gap > 0.02:  # Significant gap
                # Gaps often get filled - contrarian signal
                if 'roc_3' in features:
                    recent_roc = features['roc_3']
                    score -= np.sign(recent_roc) * 0.3  # Contrarian to recent move
            weight_sum += 0.3
        
        # OBV trend
        if 'obv_trend' in features:
            obv_trend = features['obv_trend']
            score += np.tanh(obv_trend * 5) * 0.5
            weight_sum += 0.5
        
        return score / weight_sum if weight_sum > 0 else 0.0
    
    def _combine_predictions(self, predictions: Dict) -> float:
        """Combine predictions using weighted ensemble"""
        combined_score = 0.0
        total_weight = 0.0
        
        for model_name, prediction in predictions.items():
            weight = self.model_weights.get(model_name, 0.2)
            combined_score += prediction * weight
            total_weight += weight
        
        return combined_score / total_weight if total_weight > 0 else 0.0
    
    def _prediction_to_signal(self, prediction: float, features: Dict) -> Tuple[SignalType, float]:
        """Convert ML prediction to trading signal"""
        
        # Calculate confidence based on prediction strength and feature quality
        base_confidence = min(abs(prediction), 1.0)
        
        # Adjust confidence based on feature availability
        feature_completeness = len(features) / 30  # Assume 30 ideal features
        confidence_adjustment = min(1.0, feature_completeness)
        
        confidence = base_confidence * confidence_adjustment
        
        # Determine signal type
        if prediction > 0.6:
            return SignalType.STRONG_BUY, confidence
        elif prediction > 0.3:
            return SignalType.BUY, confidence
        elif prediction < -0.6:
            return SignalType.STRONG_SELL, confidence
        elif prediction < -0.3:
            return SignalType.SELL, confidence
        else:
            return SignalType.HOLD, confidence
    
    def _calculate_ml_allocation(self, prediction: float, confidence: float, features: Dict) -> float:
        """Calculate target allocation using ML prediction"""
        
        # Base allocation from prediction strength
        base_allocation = abs(prediction) * self.config.max_position_size
        
        # Adjust for confidence
        confidence_adjusted = base_allocation * confidence
        
        # Adjust for volatility (reduce allocation in high volatility)
        if 'volatility_20' in features:
            vol_adjustment = max(0.5, 1.0 - features['volatility_20'] * 20)
            confidence_adjusted *= vol_adjustment
        
        # Adjust for feature quality
        feature_quality = self._assess_feature_quality(features)
        final_allocation = confidence_adjusted * feature_quality
        
        return min(final_allocation, self.config.max_position_size)
    
    def _assess_feature_quality(self, features: Dict) -> float:
        """Assess the quality of extracted features"""
        
        quality_score = 0.0
        checks = 0
        
        # Check for key technical features
        key_technical = ['rsi_14', 'bb_position', 'macd_line']
        technical_count = sum(1 for feature in key_technical if feature in features)
        quality_score += (technical_count / len(key_technical)) * 0.3
        checks += 0.3
        
        # Check for momentum features
        momentum_features = [f'roc_{p}' for p in [3, 5, 10]]
        momentum_count = sum(1 for feature in momentum_features if feature in features)
        quality_score += (momentum_count / len(momentum_features)) * 0.25
        checks += 0.25
        
        # Check for volatility features
        vol_features = ['volatility_20', 'atr_14', 'volatility_regime']
        vol_count = sum(1 for feature in vol_features if feature in features)
        quality_score += (vol_count / len(vol_features)) * 0.2
        checks += 0.2
        
        # Check for volume features
        volume_features = ['volume_ratio_10', 'obv', 'volume_trend']
        volume_count = sum(1 for feature in volume_features if feature in features)
        quality_score += (volume_count / len(volume_features)) * 0.15
        checks += 0.15
        
        # Check for pattern features
        pattern_features = ['higher_highs', 'support_distance', 'resistance_distance']
        pattern_count = sum(1 for feature in pattern_features if feature in features)
        quality_score += (pattern_count / len(pattern_features)) * 0.1
        checks += 0.1
        
        return max(0.5, quality_score / checks) if checks > 0 else 0.5
    
    def _build_ml_reasoning(self, predictions: Dict, features: Dict, final_prediction: float) -> str:
        """Build reasoning string from ML predictions"""
        
        components = []
        
        # Overall prediction
        strength = "strong" if abs(final_prediction) > 0.6 else "moderate" if abs(final_prediction) > 0.3 else "weak"
        direction = "bullish" if final_prediction > 0 else "bearish"
        components.append(f"{strength} {direction} ensemble prediction ({final_prediction:.3f})")
        
        # Top contributing models
        sorted_predictions = sorted(predictions.items(), key=lambda x: abs(x[1]), reverse=True)
        for model_name, pred in sorted_predictions[:2]:
            if abs(pred) > 0.2:
                model_direction = "bullish" if pred > 0 else "bearish"
                components.append(f"{model_name.replace('_', ' ')}: {model_direction}")
        
        # Key feature highlights
        feature_highlights = []
        
        if 'rsi_14' in features:
            rsi = features['rsi_14']
            if rsi < 30:
                feature_highlights.append("RSI oversold")
            elif rsi > 70:
                feature_highlights.append("RSI overbought")
        
        if 'bb_position' in features:
            bb_pos = features['bb_position']
            if bb_pos < 0.2:
                feature_highlights.append("near BB lower band")
            elif bb_pos > 0.8:
                feature_highlights.append("near BB upper band")
        
        if 'volatility_regime' in features:
            vol_regime = features['volatility_regime']
            if vol_regime < 0.8:
                feature_highlights.append("low volatility regime")
            elif vol_regime > 1.2:
                feature_highlights.append("high volatility regime")
        
        if feature_highlights:
            components.append(f"key factors: {', '.join(feature_highlights)}")
        
        return "; ".join(components)
    
    def _summarize_features(self, features: Dict) -> Dict:
        """Create a summary of key features for metadata"""
        summary = {}
        
        # Technical summary
        if 'rsi_14' in features:
            summary['rsi'] = features['rsi_14']
        if 'bb_position' in features:
            summary['bollinger_position'] = features['bb_position']
        
        # Momentum summary
        roc_features = {k: v for k, v in features.items() if k.startswith('roc_')}
        if roc_features:
            summary['avg_momentum'] = statistics.mean(roc_features.values())
        
        # Volatility summary
        vol_features = {k: v for k, v in features.items() if 'volatility' in k}
        if vol_features:
            summary['avg_volatility'] = statistics.mean(vol_features.values())
        
        # Volume summary
        volume_features = {k: v for k, v in features.items() if 'volume' in k}
        if volume_features:
            summary['volume_score'] = statistics.mean(volume_features.values())
        
        return summary
    
    def _calculate_confidence_breakdown(self, predictions: Dict) -> Dict:
        """Calculate confidence breakdown by model"""
        breakdown = {}
        
        for model_name, prediction in predictions.items():
            confidence = min(abs(prediction), 1.0)
            weight = self.model_weights.get(model_name, 0.2)
            breakdown[model_name] = {
                'prediction': prediction,
                'confidence': confidence,
                'weight': weight,
                'contribution': prediction * weight
            }
        
        return breakdown
    
    # Helper methods
    def _calculate_slope(self, values: List[float]) -> float:
        """Calculate slope of values using linear regression"""
        if len(values) < 2:
            return 0.0
        
        n = len(values)
        x_values = list(range(n))
        
        sum_x = sum(x_values)
        sum_y = sum(values)
        sum_xy = sum(x * y for x, y in zip(x_values, values))
        sum_x2 = sum(x * x for x in x_values)
        
        denominator = n * sum_x2 - sum_x * sum_x
        if denominator == 0:
            return 0.0
        
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        return slope / statistics.mean(values) if statistics.mean(values) > 0 else 0
    
    def _find_support_resistance(self, prices: List[float]) -> Tuple[List[float], List[float]]:
        """Find support and resistance levels"""
        if len(prices) < 10:
            return [], []
        
        # Simple pivot point detection
        support_levels = []
        resistance_levels = []
        
        for i in range(2, len(prices) - 2):
            # Local minimum (support)
            if (prices[i] < prices[i-1] and prices[i] < prices[i-2] and
                prices[i] < prices[i+1] and prices[i] < prices[i+2]):
                support_levels.append(prices[i])
            
            # Local maximum (resistance)
            if (prices[i] > prices[i-1] and prices[i] > prices[i-2] and
                prices[i] > prices[i+1] and prices[i] > prices[i+2]):
                resistance_levels.append(prices[i])
        
        return support_levels, resistance_levels
    
    def _calculate_correlation(self, x: List[float], y: List[float]) -> float:
        """Calculate correlation between two series"""
        if len(x) != len(y) or len(x) < 2:
            return 0.0
        
        mean_x = statistics.mean(x)
        mean_y = statistics.mean(y)
        
        numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(len(x)))
        
        sum_sq_x = sum((x[i] - mean_x) ** 2 for i in range(len(x)))
        sum_sq_y = sum((y[i] - mean_y) ** 2 for i in range(len(y)))
        
        denominator = math.sqrt(sum_sq_x * sum_sq_y)
        
        return numerator / denominator if denominator > 0 else 0.0