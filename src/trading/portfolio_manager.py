# src/trading/portfolio_manager.py

import asyncio
import statistics
from datetime import datetime
from typing import List, Dict
from collections import defaultdict, deque

from config.settings import TradingConfig
from src.core.models import Portfolio, MarketData, TradeOrder
from src.core.enums import StrategyType
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PortfolioManager:
    """
    Manages portfolio data and market history
    """

    def __init__(self, config: TradingConfig):
        self.config = config
        self.price_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.current_portfolio: Portfolio = Portfolio(cash=0.0)

    def update_portfolio(self, portfolio: Portfolio) -> None:
        self.current_portfolio = portfolio
        self.current_portfolio.update_total_value()

    async def get_market_data_history(self, symbol: str) -> List[MarketData]:
        # Normally this would fetch from an API or cache, here we assume it's filled externally
        return list(self.price_history[symbol])

    def add_market_data(self, data: MarketData) -> None:
        self.price_history[data.symbol].append(data)

    def get_current_price(self, symbol: str) -> float:
        history = self.price_history[symbol]
        return history[-1].price if history else 0.0

    def calculate_rebalancing_orders(self, portfolio: Portfolio) -> List[TradeOrder]:
        """Generate trade orders to rebalance portfolio to match target allocations"""
        orders = []
        total_value = portfolio.total_value

        for symbol, target_alloc in portfolio.target_allocations.items():
            current_alloc = portfolio.get_allocation(symbol)
            allocation_diff = target_alloc - current_alloc

            if abs(allocation_diff) < self.config.rebalance_threshold:
                continue

            target_value = total_value * target_alloc
            current_value = portfolio.positions.get(symbol).market_value if symbol in portfolio.positions else 0
            diff_value = target_value - current_value

            price = self.get_current_price(symbol)
            if price <= 0:
                continue

            if diff_value > self.config.min_trade_amount:
                # Buy order
                amount = diff_value / price
                orders.append(TradeOrder(from_token="USDC", to_token=symbol, amount=amount, max_slippage=self.config.max_slippage))
            elif diff_value < -self.config.min_trade_amount:
                # Sell order
                amount = -diff_value / price
                orders.append(TradeOrder(from_token=symbol, to_token="USDC", amount=amount, max_slippage=self.config.max_slippage))

        return orders