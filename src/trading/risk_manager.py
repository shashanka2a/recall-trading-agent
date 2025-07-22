# src/trading/risk_manager.py

from typing import Optional
from src.core.models import TradingSignal, Portfolio
from config.settings import TradingConfig


class RiskManager:
    """
    Risk management logic for position sizing
    """

    def __init__(self, config: TradingConfig):
        self.config = config

    def calculate_kelly_position_size(self, signal: TradingSignal, portfolio: Portfolio, current_price: float) -> float:
        """Calculate position size using the Kelly Criterion"""
        win_prob = 0.5 + (signal.confidence - 0.5) * 0.4
        expected_win = signal.metadata.get("expected_return", self.config.take_profit_percent)
        expected_loss = self.config.stop_loss_percent

        if win_prob > 0 and expected_loss > 0:
            kelly_fraction = (win_prob * expected_win - (1 - win_prob) * expected_loss) / expected_win
            kelly_fraction = max(0, min(kelly_fraction, self.config.kelly_fraction))
        else:
            kelly_fraction = self.config.kelly_fraction * 0.5

        target_value = portfolio.total_value * kelly_fraction * signal.target_allocation
        position_size = target_value / current_price

        max_position_value = portfolio.total_value * self.config.max_position_size
        max_position_size = max_position_value / current_price

        return min(position_size, max_position_size)
