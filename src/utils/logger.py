"""
Logging utilities for the trading agent
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.core.enums import LOG_FORMAT, LOG_DATE_FORMAT


def setup_logging(
    level: str = "INFO",
    log_to_file: bool = True,
    log_dir: str = "logs"
) -> None:
    """
    Setup logging configuration for the trading agent
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_to_file: Whether to log to file
        log_dir: Directory for log files
    """
    
    # Create logs directory
    if log_to_file:
        Path(log_dir).mkdir(exist_ok=True)
    
    # Configure logging level
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create formatters
    formatter = logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT)
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler
    if log_to_file:
        log_filename = f"trading_agent_{datetime.now().strftime('%Y%m%d')}.log"
        log_filepath = Path(log_dir) / log_filename
        
        file_handler = logging.FileHandler(log_filepath)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Suppress noisy external loggers
    logging.getLogger("aiohttp").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    # Log startup message
    logger = logging.getLogger(__name__)
    logger.info("🚀 Logging system initialized")
    logger.info(f"📊 Log level: {level}")
    if log_to_file:
        logger.info(f"📁 Log file: {log_filepath}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class TradingLogger:
    """
    Specialized logger for trading operations with structured logging
    """
    
    def __init__(self, name: str):
        self.logger = get_logger(name)
        self.trade_count = 0
        self.cycle_count = 0
    
    def log_trade_execution(
        self, 
        symbol: str, 
        action: str, 
        amount: float, 
        price: float,
        success: bool,
        reasoning: Optional[str] = None
    ) -> None:
        """Log trade execution with structured data"""
        
        self.trade_count += 1
        status = "✅" if success else "❌"
        
        message = f"{status} Trade #{self.trade_count}: {action.upper()} {amount:.4f} {symbol} @ ${price:.4f}"
        
        if reasoning:
            message += f" | Reason: {reasoning}"
        
        if success:
            self.logger.info(message)
        else:
            self.logger.warning(message)
    
    def log_portfolio_update(
        self, 
        total_value: float, 
        cash: float, 
        positions: dict,
        pnl_percent: Optional[float] = None
    ) -> None:
        """Log portfolio status update"""
        
        cash_percent = (cash / total_value * 100) if total_value > 0 else 0
        
        message = f"💼 Portfolio: ${total_value:.2f} | Cash: ${cash:.2f} ({cash_percent:.1f}%)"
        
        if pnl_percent is not None:
            message += f" | P&L: {pnl_percent:+.2f}%"
        
        self.logger.info(message)
        
        # Log positions
        for symbol, position in positions.items():
            if hasattr(position, 'quantity') and position.quantity > 0:
                allocation = (position.market_value / total_value * 100) if total_value > 0 else 0
                self.logger.info(f"   📊 {symbol}: {position.quantity:.4f} ({allocation:.1f}%)")
    
    def log_signal_generation(
        self, 
        symbol: str, 
        strategy: str, 
        signal: str, 
        confidence: float,
        reasoning: str
    ) -> None:
        """Log trading signal generation"""
        
        message = f"📡 {strategy}: {symbol} {signal} (conf: {confidence:.2f}) | {reasoning}"
        self.logger.info(message)
    
    def log_cycle_start(self) -> None:
        """Log start of trading cycle"""
        self.cycle_count += 1
        self.logger.info(f"🔄 Starting cycle #{self.cycle_count}")
    
    def log_cycle_end(self, duration: float) -> None:
        """Log end of trading cycle"""
        self.logger.info(f"✅ Cycle #{self.cycle_count} completed in {duration:.1f}s")
    
    def log_performance_summary(
        self, 
        total_return: float, 
        trades: int, 
        win_rate: Optional[float] = None
    ) -> None:
        """Log performance summary"""
        
        message = f"📈 Performance: {total_return:+.2f}% return | {trades} trades"
        
        if win_rate is not None:
            message += f" | {win_rate:.1f}% win rate"
        
        self.logger.info(message)
    
    def log_error_with_context(
        self, 
        error: Exception, 
        context: str,
        symbol: Optional[str] = None
    ) -> None:
        """Log error with trading context"""
        
        message = f"❌ {context}"
        if symbol:
            message += f" ({symbol})"
        message += f": {str(error)}"
        
        self.logger.error(message, exc_info=True)


def create_performance_logger() -> TradingLogger:
    """Create a specialized performance logger"""
    return TradingLogger("performance")


def create_strategy_logger(strategy_name: str) -> TradingLogger:
    """Create a specialized strategy logger"""
    return TradingLogger(f"strategy.{strategy_name}")


def log_startup_info(config) -> None:
    """Log startup information"""
    logger = get_logger("startup")
    
    logger.info("=" * 50)
    logger.info("🤖 Recall Trading Agent Starting")
    logger.info("=" * 50)
    logger.info(f"Environment: {config.environment.upper()}")
    logger.info(f"Dry Run: {config.dry_run}")
    logger.info(f"Max Position Size: {config.max_position_size:.1%}")
    logger.info(f"Min Confidence: {config.min_confidence:.1%}")
    logger.info(f"Target Allocations: {config.target_allocations}")
    logger.info(f"Strategy Weights: {config.strategy_weights}")
    logger.info("=" * 50)


def log_shutdown_info() -> None:
    """Log shutdown information"""
    logger = get_logger("shutdown")
    
    logger.info("=" * 50)
    logger.info("🛑 Trading Agent Shutting Down")
    logger.info("=" * 50)