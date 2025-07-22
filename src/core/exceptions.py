"""
Custom exceptions for the trading agent
"""


class TradingAgentError(Exception):
    """Base exception for trading agent errors"""
    pass


class APIError(TradingAgentError):
    """Raised when API calls fail"""
    pass


class ValidationError(TradingAgentError):
    """Raised when validation fails"""
    pass


class RateLimitError(APIError):
    """Raised when rate limits are exceeded"""
    pass


class InsufficientFundsError(TradingAgentError):
    """Raised when insufficient funds for trade"""
    pass


class ConfigurationError(TradingAgentError):
    """Raised when configuration is invalid"""
    pass


class StrategyError(TradingAgentError):
    """Raised when strategy execution fails"""
    pass


class PortfolioError(TradingAgentError):
    """Raised when portfolio operations fail"""
    pass

