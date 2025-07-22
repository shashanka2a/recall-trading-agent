"""
Enums and constants for the trading agent
"""

from enum import Enum, IntEnum


class SignalType(IntEnum):
    """Trading signal types with numeric values for strength calculation"""
    STRONG_SELL = -2
    SELL = -1
    HOLD = 0
    BUY = 1
    STRONG_BUY = 2


class StrategyType(Enum):
    """Trading strategy types"""
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    VOLATILITY = "volatility"
    ML_ENSEMBLE = "ml_ensemble"


class OrderType(Enum):
    """Order types"""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"


class OrderStatus(Enum):
    """Order status"""
    PENDING = "pending"
    EXECUTED = "executed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TradingMode(Enum):
    """Trading modes"""
    LIVE = "live"
    BACKTEST = "backtest"
    DRY_RUN = "dry_run"


class Environment(Enum):
    """API environments"""
    SANDBOX = "sandbox"
    PRODUCTION = "production"


class TimeFrame(Enum):
    """Time frames for analysis"""
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAY_1 = "1d"


class RiskLevel(Enum):
    """Risk levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


# Constants
DEFAULT_TOKENS = ["USDC", "WETH", "SOL"]
BASE_CURRENCY = "USDC"
SUPPORTED_CHAINS = ["ethereum", "solana", "base"]

# Technical indicator constants
DEFAULT_RSI_PERIOD = 14
DEFAULT_SMA_SHORT = 5
DEFAULT_SMA_LONG = 20
DEFAULT_BB_PERIOD = 20
DEFAULT_BB_STD = 2.0

# Risk management constants
MAX_POSITION_SIZE = 0.30
MIN_TRADE_AMOUNT = 10.0
DEFAULT_STOP_LOSS = 0.05
DEFAULT_TAKE_PROFIT = 0.15
MAX_DAILY_TRADES = 20

# Rate limiting
API_RATE_LIMIT = 60  # requests per minute
RATE_LIMIT_BUFFER = 0.1  # 10% buffer

# Logging
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"