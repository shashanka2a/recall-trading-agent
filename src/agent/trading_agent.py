"""
Main trading agent that orchestrates all components
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from collections import defaultdict, deque

from config.settings import TradingConfig
from src.core.models import Portfolio, TradingSignal, TradeOrder, PerformanceMetrics
from src.core.enums import SignalType, StrategyType
from src.trading.client import RecallClient
from src.trading.portfolio_manager import PortfolioManager
from src.strategies.base import StrategyManager
from src.strategies.momentum import MomentumStrategy, AdvancedMomentumStrategy
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.breakout_strategy import BreakoutStrategy
from src.strategies.volatility import VolatilityStrategy
from src.strategies.ml_ensemble import MLEnsembleStrategy
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TradingAgent:
    """
    Main trading agent that coordinates all trading activities
    """
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.client = RecallClient(config)
        self.portfolio_manager = PortfolioManager(config)
        self.strategy_manager = StrategyManager(config)
        
        # Initialize strategies
        self._initialize_strategies()
        
        # Performance tracking
        self.performance_history: List[Dict] = []
        self.trade_history: List[Dict] = []
        self.cycle_count = 0
        self.trade_count = 0
        self.start_time = datetime.now()
        
        # State management
        self.running = False
        self.last_portfolio_value = 0.0
        self.daily_pnl = 0.0
        
        logger.info(f"🤖 Trading Agent initialized")
        logger.info(f"Environment: {config.environment}")
        logger.info(f"Strategies: {self.strategy_manager.get_active_strategies()}")
    
    def _initialize_strategies(self) -> None:
        """Initialize all trading strategies"""
        
        # Add strategies based on configuration
        if StrategyType.MOMENTUM in self.config.strategy_weights:
            if self.config.strategy_weights[StrategyType.MOMENTUM] > 0.3:
                # Use advanced momentum for higher weights
                self.strategy_manager.add_strategy(AdvancedMomentumStrategy(self.config))
            else:
                self.strategy_manager.add_strategy(MomentumStrategy(self.config))
        
        if StrategyType.MEAN_REVERSION in self.config.strategy_weights:
            self.strategy_manager.add_strategy(MeanReversionStrategy(self.config))
        
        if StrategyType.BREAKOUT in self.config.strategy_weights:
            self.strategy_manager.add_strategy(BreakoutStrategy(self.config))
        
        if StrategyType.VOLATILITY in self.config.strategy_weights:
            self.strategy_manager.add_strategy(VolatilityStrategy(self.config))
        
        if StrategyType.ML_ENSEMBLE in self.config.strategy_weights:
            self.strategy_manager.add_strategy(MLEnsembleStrategy(self.config))
    
    async def execute_trading_cycle(self) -> None:
        """Execute one complete trading cycle"""
        
        self.cycle_count += 1
        cycle_start = time.time()
        
        logger.info(f"🔄 Starting cycle #{self.cycle_count}")
        
        try:
            async with self.client:
                # Health check
                if not await self.client.health_check():
                    logger.error("❌ API health check failed, skipping cycle")
                    return
                
                # Get current portfolio
                portfolio = await self.client.get_portfolio()
                if portfolio.total_value == 0:
                    logger.error("❌ Failed to fetch portfolio, skipping cycle")
                    return
                
                # Update portfolio manager
                self.portfolio_manager.update_portfolio(portfolio)
                
                # Log portfolio status
                self._log_portfolio_status(portfolio)
                
                # Record performance
                self._record_performance(portfolio)
                
                # Check for rebalancing needs
                await self._handle_rebalancing(portfolio)
                
                # Generate and execute trading signals
                await self._process_trading_signals(portfolio)
                
                # Update strategy performance
                self._update_strategy_performance()
                
        except Exception as e:
            logger.error(f"❌ Error in trading cycle: {e}")
        finally:
            cycle_duration = time.time() - cycle_start
            logger.info(f"✅ Cycle #{self.cycle_count} completed in {cycle_duration:.1f}s")
    
    async def _handle_rebalancing(self, portfolio: Portfolio) -> None:
        """Handle portfolio rebalancing if needed"""
        
        if not portfolio.needs_rebalancing(self.config.rebalance_threshold):
            return
        
        logger.info("🔄 Portfolio needs rebalancing")
        
        rebalancing_orders = self.portfolio_manager.calculate_rebalancing_orders(portfolio)
        
        for order in rebalancing_orders:
            try:
                result = await self.client.execute_trade(order)
                if result.success:
                    logger.info(f"✅ Rebalancing trade: {order.amount:.4f} {order.from_token} → {order.to_token}")
                    self._record_trade(order, result)
                else:
                    logger.warning(f"❌ Rebalancing failed: {result.error_message}")
                
                # Rate limiting
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.error(f"❌ Rebalancing error: {e}")
    
    async def _process_trading_signals(self, portfolio: Portfolio) -> None:
        """Process trading signals for all target assets"""
        
        symbols = [symbol for symbol in self.config.target_allocations.keys() if symbol != "USDC"]
        
        for symbol in symbols:
            try:
                await self._process_symbol_signals(symbol, portfolio)
                await asyncio.sleep(1)  # Rate limiting between symbols
            except Exception as e:
                logger.error(f"❌ Error processing {symbol}: {e}")
    
    async def _process_symbol_signals(self, symbol: str, portfolio: Portfolio) -> None:
        """Process trading signals for a specific symbol"""
        
        logger.debug(f"🔍 Analyzing {symbol}")
        
        # Get market data
        market_data = await self.portfolio_manager.get_market_data_history(symbol)
        if not market_data:
            logger.warning(f"No market data for {symbol}")
            return
        
        # Generate signals from all strategies
        signals = await self.strategy_manager.generate_all_signals(symbol, market_data, portfolio)
        
        if not signals:
            logger.debug(f"💤 No signals generated for {symbol}")
            return
        
        # Aggregate signals
        final_signal = self.strategy_manager.aggregate_signals(signals)
        
        if not final_signal:
            logger.debug(f"💤 No actionable signal for {symbol}")
            return
        
        logger.info(f"📊 {symbol} Signal: {final_signal.signal.name} "
                   f"(confidence: {final_signal.confidence:.2f})")
        logger.info(f"💭 Reasoning: {final_signal.reasoning}")
        
        # Execute trade if signal is strong enough
        if final_signal.is_actionable(self.config.min_confidence):
            await self._execute_signal(final_signal, portfolio)
    
    async def _execute_signal(self, signal: TradingSignal, portfolio: Portfolio) -> None:
        """Execute a trading signal"""
        
        # Create trade order
        order = self._create_trade_order(signal, portfolio)
        if not order:
            return
        
        # Execute trade
        try:
            result = await self.client.execute_trade(order)
            
            if result.success:
                self.trade_count += 1
                logger.info(f"✅ Trade #{self.trade_count} executed: "
                           f"{order.amount:.4f} {order.from_token} → {order.to_token}")
                self._record_trade(order, result, signal)
            else:
                logger.warning(f"❌ Trade failed: {result.error_message}")
                
        except Exception as e:
            logger.error(f"❌ Trade execution error: {e}")
    
    def _create_trade_order(self, signal: TradingSignal, portfolio: Portfolio) -> Optional[TradeOrder]:
        """Create trade order from signal"""
        
        current_price = self.portfolio_manager.get_current_price(signal.symbol)
        if current_price <= 0:
            logger.error(f"Invalid price for {signal.symbol}")
            return None
        
        # Calculate position size
        position_size = self._calculate_position_size(signal, portfolio, current_price)
        if position_size <= 0:
            return None
        
        # Determine order direction
        if signal.signal in [SignalType.BUY, SignalType.STRONG_BUY]:
            # Buy order
            order_value = position_size * current_price
            if order_value > portfolio.cash * 0.95:  # 5% buffer
                position_size = portfolio.cash * 0.95 / current_price
            
            if position_size * current_price < self.config.min_trade_amount:
                logger.debug(f"Trade value too small for {signal.symbol}")
                return None
            
            return TradeOrder(
                from_token="USDC",
                to_token=signal.symbol,
                amount=position_size,
                max_slippage=self.config.max_slippage
            )
        
        elif signal.signal in [SignalType.SELL, SignalType.STRONG_SELL]:
            # Sell order
            current_position = portfolio.positions.get(signal.symbol)
            if not current_position or current_position.quantity <= 0:
                logger.debug(f"No position to sell for {signal.symbol}")
                return None
            
            sell_amount = min(position_size, current_position.quantity)
            
            if sell_amount * current_price < self.config.min_trade_amount:
                logger.debug(f"Sell value too small for {signal.symbol}")
                return None
            
            return TradeOrder(
                from_token=signal.symbol,
                to_token="USDC",
                amount=sell_amount,
                max_slippage=self.config.max_slippage
            )
        
        return None
    
    def _calculate_position_size(self, signal: TradingSignal, portfolio: Portfolio, current_price: float) -> float:
        """Calculate position size using Kelly criterion and risk management"""
        
        # Base position size from signal
        base_size = signal.target_allocation * portfolio.total_value / current_price
        
        # Apply Kelly criterion
        kelly_size = self._calculate_kelly_size(signal, portfolio, current_price)
        
        # Use smaller of the two
        position_size = min(base_size, kelly_size)
        
        # Apply maximum position limits
        max_position_value = portfolio.total_value * self.config.max_position_size
        max_position_size = max_position_value / current_price
        
        return min(position_size, max_position_size)
    
    def _calculate_kelly_size(self, signal: TradingSignal, portfolio: Portfolio, current_price: float) -> float:
        """Calculate position size using Kelly criterion"""
        
        # Estimate win probability from confidence
        win_prob = 0.5 + (signal.confidence - 0.5) * 0.4  # Scale confidence to probability
        
        # Estimate expected returns
        expected_win = signal.metadata.get('expected_return', self.config.take_profit_percent)
        expected_loss = self.config.stop_loss_percent
        
        # Kelly fraction
        if win_prob > 0 and expected_loss > 0:
            kelly_fraction = (win_prob * expected_win - (1 - win_prob) * expected_loss) / expected_win
            kelly_fraction = max(0, min(kelly_fraction, self.config.kelly_fraction))
        else:
            kelly_fraction = self.config.kelly_fraction * 0.5
        
        # Apply Kelly sizing
        kelly_value = portfolio.total_value * kelly_fraction
        return kelly_value / current_price
    
    def _log_portfolio_status(self, portfolio: Portfolio) -> None:
        """Log current portfolio status"""
        
        logger.info(f"💼 Portfolio Value: ${portfolio.total_value:.2f}")
        logger.info(f"💰 Cash: ${portfolio.cash:.2f} ({portfolio.cash/portfolio.total_value:.1%})")
        
        for symbol, position in portfolio.positions.items():
            if position.quantity > 0:
                allocation = position.market_value / portfolio.total_value
                pnl_pct = position.pnl_percent
                logger.info(f"   📊 {symbol}: {position.quantity:.4f} "
                           f"(${position.market_value:.2f}, {allocation:.1%}, "
                           f"PnL: {pnl_pct:+.2%})")
    
    def _record_performance(self, portfolio: Portfolio) -> None:
        """Record portfolio performance"""
        
        current_time = datetime.now()
        
        # Calculate daily P&L
        if self.last_portfolio_value > 0:
            daily_return = (portfolio.total_value - self.last_portfolio_value) / self.last_portfolio_value
        else:
            daily_return = 0.0
        
        # Record performance point
        performance_point = {
            'timestamp': current_time,
            'total_value': portfolio.total_value,
            'cash': portfolio.cash,
            'daily_return': daily_return,
            'total_pnl': portfolio.get_total_pnl(),
            'unrealized_pnl': portfolio.get_unrealized_pnl(),
            'realized_pnl': portfolio.get_realized_pnl(),
            'cycle': self.cycle_count,
            'trade_count': self.trade_count
        }
        
        self.performance_history.append(performance_point)
        
        # Keep only recent history (last 1000 points)
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
        
        self.last_portfolio_value = portfolio.total_value
        
        # Log performance summary
        if len(self.performance_history) > 1:
            initial_value = self.performance_history[0]['total_value']
            total_return = (portfolio.total_value - initial_value) / initial_value
            logger.info(f"📈 Total Return: {total_return:+.2%} | Trades: {self.trade_count}")
    
    def _record_trade(self, order: TradeOrder, result, signal: Optional[TradingSignal] = None) -> None:
        """Record trade execution"""
        
        trade_record = {
            'timestamp': datetime.now(),
            'order': order,
            'result': result,
            'signal': signal,
            'cycle': self.cycle_count
        }
        
        self.trade_history.append(trade_record)
        
        # Keep only recent trades (last 500)
        if len(self.trade_history) > 500:
            self.trade_history = self.trade_history[-500:]
    
    def _update_strategy_performance(self) -> None:
        """Update strategy performance metrics"""
        
        # Calculate strategy performance based on recent signals vs outcomes
        # This is a simplified version - in practice would track signal accuracy
        
        performance_data = self.strategy_manager.get_strategy_performance()
        
        if performance_data and any(performance_data.values()):
            self.strategy_manager.update_strategy_weights(performance_data)
    
    async def run_competition(self, duration_hours: int = 24) -> None:
        """Run competition mode for specified duration"""
        
        logger.info(f"🏆 Starting {duration_hours}h competition mode")
        logger.info(f"Target allocations: {self.config.target_allocations}")
        logger.info(f"Strategy weights: {self.config.strategy_weights}")
        
        self.running = True
        end_time = datetime.now() + timedelta(hours=duration_hours)
        
        try:
            while self.running and datetime.now() < end_time:
                await self.execute_trading_cycle()
                
                # Dynamic sleep based on market conditions and performance
                sleep_time = self._calculate_dynamic_sleep()
                logger.info(f"😴 Sleeping for {sleep_time}s...")
                await asyncio.sleep(sleep_time)
                
        except KeyboardInterrupt:
            logger.info("👋 Competition stopped by user")
        except Exception as e:
            logger.error(f"❌ Competition error: {e}")
        finally:
            self.running = False
            await self.generate_final_report()
    
    def _calculate_dynamic_sleep(self) -> int:
        """Calculate dynamic sleep time based on conditions"""
        
        base_sleep = self.config.update_interval
        
        # Adjust based on recent performance
        if len(self.performance_history) >= 2:
            recent_return = self.performance_history[-1]['daily_return']
            
            # Sleep longer if losing money (be more conservative)
            if recent_return < -0.02:  # -2% daily loss
                base_sleep = int(base_sleep * 1.5)
            # Sleep shorter if making money (be more active)
            elif recent_return > 0.02:  # +2% daily gain
                base_sleep = int(base_sleep * 0.8)
        
        # Add some randomness to avoid predictable patterns
        import random
        jitter = random.randint(-5, 5)
        
        return max(30, base_sleep + jitter)  # Minimum 30 seconds
    
    async def show_portfolio_status(self) -> None:
        """Show current portfolio status"""
        
        async with self.client:
            portfolio = await self.client.get_portfolio()
            
            if portfolio.total_value == 0:
                logger.error("❌ Failed to fetch portfolio")
                return
            
            print("\n📊 PORTFOLIO STATUS")
            print("=" * 50)
            print(f"💰 Total Value: ${portfolio.total_value:.2f}")
            print(f"💵 Cash: ${portfolio.cash:.2f} ({portfolio.cash/portfolio.total_value:.1%})")
            print("\n📈 POSITIONS:")
            
            for symbol, position in portfolio.positions.items():
                if position.quantity > 0:
                    allocation = position.market_value / portfolio.total_value
                    target_allocation = portfolio.get_target_allocation(symbol)
                    drift = allocation - target_allocation
                    
                    print(f"  {symbol}: {position.quantity:.4f}")
                    print(f"    Value: ${position.market_value:.2f}")
                    print(f"    Allocation: {allocation:.1%} (target: {target_allocation:.1%}, drift: {drift:+.1%})")
                    print(f"    P&L: {position.pnl_percent:+.2%}")
                    print()
            
            # Show recent performance
            if self.performance_history:
                initial_value = self.performance_history[0]['total_value']
                total_return = (portfolio.total_value - initial_value) / initial_value
                print(f"📈 Total Return: {total_return:+.2%}")
                print(f"🔄 Total Trades: {self.trade_count}")
            
            print("=" * 50)
    
    async def generate_performance_report(self) -> None:
        """Generate detailed performance report"""
        
        if not self.performance_history:
            print("📊 No performance data available")
            return
        
        # Calculate metrics
        metrics = self._calculate_performance_metrics()
        
        print("\n📈 PERFORMANCE REPORT")
        print("=" * 60)
        print(f"⏱️  Period: {self.start_time.strftime('%Y-%m-%d %H:%M')} - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print(f"💰 Initial Value: ${metrics.initial_value:.2f}")
        print(f"💰 Final Value: ${metrics.current_value:.2f}")
        print(f"📈 Total Return: {metrics.total_return:+.2%}")
        print(f"📊 Annualized Return: {metrics.annualized_return:+.2%}")
        print(f"📉 Volatility: {metrics.volatility:.2%}")
        print(f"⚡ Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
        print(f"📉 Max Drawdown: {metrics.max_drawdown:.2%}")
        print(f"🎯 Win Rate: {metrics.win_rate:.1%}")
        print(f"💹 Profit Factor: {metrics.profit_factor:.2f}")
        print(f"🔄 Total Trades: {metrics.total_trades}")
        print(f"✅ Winning Trades: {metrics.winning_trades}")
        print(f"❌ Losing Trades: {metrics.losing_trades}")
        print(f"💵 Average Win: ${metrics.avg_win:.2f}")
        print(f"💸 Average Loss: ${metrics.avg_loss:.2f}")
        print("=" * 60)
        
        # Strategy performance
        print("\n🤖 STRATEGY PERFORMANCE")
        print("-" * 30)
        for strategy_type, weight in self.config.strategy_weights.items():
            print(f"{strategy_type.value}: {weight:.1%}")
        print()
        
        # Recent trades
        if self.trade_history:
            print("📋 RECENT TRADES")
            print("-" * 30)
            for trade in self.trade_history[-5:]:
                order = trade['order']
                result = trade['result']
                timestamp = trade['timestamp'].strftime('%H:%M:%S')
                success = "✅" if result.success else "❌"
                print(f"{timestamp} {success} {order.amount:.4f} {order.from_token} → {order.to_token}")
            print()
    
    def _calculate_performance_metrics(self) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        
        if not self.performance_history:
            return PerformanceMetrics(
                total_return=0, annualized_return=0, volatility=0, sharpe_ratio=0,
                max_drawdown=0, win_rate=0, profit_factor=0, total_trades=0,
                winning_trades=0, losing_trades=0, avg_win=0, avg_loss=0,
                current_value=0, initial_value=0
            )
        
        # Basic metrics
        initial_value = self.performance_history[0]['total_value']
        current_value = self.performance_history[-1]['total_value']
        total_return = (current_value - initial_value) / initial_value if initial_value > 0 else 0
        
        # Calculate daily returns
        daily_returns = []
        for i in range(1, len(self.performance_history)):
            prev_value = self.performance_history[i-1]['total_value']
            curr_value = self.performance_history[i]['total_value']
            daily_return = (curr_value - prev_value) / prev_value if prev_value > 0 else 0
            daily_returns.append(daily_return)
        
        # Risk metrics
        if daily_returns:
            volatility = (sum(r**2 for r in daily_returns) / len(daily_returns)) ** 0.5
            avg_return = sum(daily_returns) / len(daily_returns)
            sharpe_ratio = avg_return / volatility if volatility > 0 else 0
        else:
            volatility = 0
            sharpe_ratio = 0
        
        # Annualized return
        days_elapsed = (datetime.now() - self.start_time).days
        if days_elapsed > 0:
            annualized_return = (1 + total_return) ** (365 / days_elapsed) - 1
        else:
            annualized_return = 0
        
        # Maximum drawdown
        max_drawdown = 0
        peak_value = initial_value
        for point in self.performance_history:
            value = point['total_value']
            if value > peak_value:
                peak_value = value
            else:
                drawdown = (peak_value - value) / peak_value
                max_drawdown = max(max_drawdown, drawdown)
        
        # Trade statistics
        winning_trades = 0
        losing_trades = 0
        total_wins = 0
        total_losses = 0
        
        for trade in self.trade_history:
            if trade['result'].success:
                # Simplified P&L calculation - would need actual trade outcomes
                if trade['signal'] and trade['signal'].signal in [SignalType.BUY, SignalType.STRONG_BUY]:
                    winning_trades += 1
                    total_wins += 100  # Placeholder
                else:
                    losing_trades += 1
                    total_losses += 50  # Placeholder
        
        total_trades = winning_trades + losing_trades
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        avg_win = total_wins / winning_trades if winning_trades > 0 else 0
        avg_loss = total_losses / losing_trades if losing_trades > 0 else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else 0
        
        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            avg_win=avg_win,
            avg_loss=avg_loss,
            current_value=current_value,
            initial_value=initial_value
        )
    
    async def generate_final_report(self) -> None:
        """Generate final competition report"""
        
        logger.info("=" * 60)
        logger.info("🏆 FINAL COMPETITION REPORT")
        logger.info("=" * 60)
        
        if self.performance_history:
            metrics = self._calculate_performance_metrics()
            
            logger.info(f"⏱️  Duration: {datetime.now() - self.start_time}")
            logger.info(f"💰 Initial Value: ${metrics.initial_value:.2f}")
            logger.info(f"💰 Final Value: ${metrics.current_value:.2f}")
            logger.info(f"📈 Total Return: {metrics.total_return:+.2%}")
            logger.info(f"📊 Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
            logger.info(f"📉 Max Drawdown: {metrics.max_drawdown:.2%}")
            logger.info(f"🔄 Total Trades: {metrics.total_trades}")
            logger.info(f"🔄 Total Cycles: {self.cycle_count}")
            logger.info(f"🎯 Win Rate: {metrics.win_rate:.1%}")
            
            # Environment info
            logger.info(f"🌍 Environment: {self.config.environment.upper()}")
            logger.info(f"🧪 Dry Run: {self.config.dry_run}")
            
        else:
            logger.info("📊 No performance data recorded")
        
        logger.info("=" * 60)
    
    def stop(self) -> None:
        """Stop the trading agent"""
        self.running = False
        logger.info("🛑 Trading agent stop requested")