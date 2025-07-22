"""
Configuration management for the trading agent
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

from src.core.enums import StrategyType

logger = logging.getLogger(__name__)


@dataclass
class TradingConfig:
    """Main configuration for the trading agent"""
    
    # API Configuration
    recall_api_key: str
    recall_api_url: str
    openai_api_key: str = ""
    
    # Environment
    environment: str = "sandbox"
    dry_run: bool = False
    
    # Trading Parameters
    max_position_size: float = 0.22
    min_confidence: float = 0.65
    rebalance_threshold: float = 0.04
    min_trade_amount: float = 15.0
    max_slippage: float = 0.015
    
    # Risk Management
    stop_loss_percent: float = 0.04
    take_profit_percent: float = 0.12
    max_portfolio_volatility: float = 0.20
    kelly_fraction: float = 0.2
    
    # Strategy Configuration
    strategy_weights: Dict[StrategyType, float] = field(default_factory=lambda: {
        StrategyType.MOMENTUM: 0.35,
        StrategyType.MEAN_REVERSION: 0.25,
        StrategyType.BREAKOUT: 0.20,
        StrategyType.VOLATILITY: 0.10,
        StrategyType.ML_ENSEMBLE: 0.10
    })
    
    # Portfolio Configuration
    target_allocations: Dict[str, float] = field(default_factory=lambda: {
        "USDC": 0.25,
        "WETH": 0.40,
        "SOL": 0.35
    })
    
    # Technical Indicators
    lookback_periods: Dict[str, int] = field(default_factory=lambda: {
        "short": 5,
        "medium": 20,
        "long": 50
    })
    
    # Timing
    update_interval: int = 50
    rate_limit_delay: float = 1.0
    
    # Logging
    log_level: str = "INFO"
    log_to_file: bool = True
    
    def validate(self) -> bool:
        """Validate configuration parameters"""
        
        # Required fields
        if not self.recall_api_key:
            logger.error("recall_api_key is required")
            return False
        
        if not self.recall_api_url:
            logger.error("recall_api_url is required")
            return False
        
        # Validate percentages
        if not 0 < self.max_position_size <= 1:
            logger.error("max_position_size must be between 0 and 1")
            return False
        
        if not 0 < self.min_confidence <= 1:
            logger.error("min_confidence must be between 0 and 1")
            return False
        
        # Validate allocations
        total_allocation = sum(self.target_allocations.values())
        if abs(total_allocation - 1.0) > 0.01:
            logger.warning(f"Target allocations sum to {total_allocation:.3f}, normalizing...")
            # Normalize allocations
            self.target_allocations = {
                k: v / total_allocation 
                for k, v in self.target_allocations.items()
            }
        
        # Validate strategy weights
        total_weight = sum(self.strategy_weights.values())
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(f"Strategy weights sum to {total_weight:.3f}, normalizing...")
            # Normalize weights
            self.strategy_weights = {
                k: v / total_weight 
                for k, v in self.strategy_weights.items()
            }
        
        return True
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary"""
        config_dict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, dict):
                # Handle enum keys
                if key == 'strategy_weights':
                    config_dict[key] = {k.value if hasattr(k, 'value') else str(k): v for k, v in value.items()}
                else:
                    config_dict[key] = value
            else:
                config_dict[key] = value
        return config_dict
    
    def save(self, filepath: str) -> None:
        """Save configuration to file"""
        config_dict = self.to_dict()
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
        logger.info(f"Configuration saved to {filepath}")


def load_env_config() -> Dict[str, str]:
    """Load configuration from environment variables"""
    
    env_config = {}
    
    # API Keys
    env_config['recall_api_key'] = os.getenv('RECALL_API_KEY', '')
    env_config['recall_api_url'] = os.getenv('RECALL_API_URL', '')
    env_config['openai_api_key'] = os.getenv('OPENAI_API_KEY', '')
    
    # Trading parameters
    if os.getenv('MAX_POSITION_SIZE'):
        env_config['max_position_size'] = float(os.getenv('MAX_POSITION_SIZE'))
    
    if os.getenv('MIN_CONFIDENCE'):
        env_config['min_confidence'] = float(os.getenv('MIN_CONFIDENCE'))
    
    if os.getenv('MIN_TRADE_AMOUNT'):
        env_config['min_trade_amount'] = float(os.getenv('MIN_TRADE_AMOUNT'))
    
    # Environment
    env_config['environment'] = os.getenv('ENVIRONMENT', 'sandbox')
    env_config['dry_run'] = os.getenv('DRY_RUN', 'false').lower() == 'true'
    
    return {k: v for k, v in env_config.items() if v}


def load_config_file(filepath: str) -> Dict:
    """Load configuration from JSON file"""
    
    if not os.path.exists(filepath):
        logger.warning(f"Config file not found: {filepath}")
        return {}
    
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading config file {filepath}: {e}")
        return {}


def get_api_credentials(environment: str) -> Dict[str, str]:
    """Get API credentials for specified environment"""
    
    credentials = {
        'sandbox': {
            'recall_api_key': '656e7357490f3d74_c89f630440abcba2',
            'recall_api_url': 'https://sandbox-api.recall.network'
        },
        'production': {
            'recall_api_key': '8cb27f2ba7e5bef2_e6e06267b2a7dc35',
            'recall_api_url': 'https://api.recall.network'
        }
    }
    
    return credentials.get(environment, credentials['sandbox'])


def create_config(
    environment: str = "sandbox",
    config_path: Optional[str] = None,
    dry_run: bool = False
) -> TradingConfig:
    """Create configuration with precedence: CLI > env vars > config file > defaults"""
    
    # Start with defaults
    config_data = {}
    
    # Load from config file if provided
    if config_path:
        file_config = load_config_file(config_path)
        config_data.update(file_config)
    else:
        # Try default config file
        default_config_path = Path(__file__).parent / "portfolio_config.json"
        if default_config_path.exists():
            file_config = load_config_file(str(default_config_path))
            config_data.update(file_config)
    
    # Override with environment variables
    env_config = load_env_config()
    config_data.update(env_config)
    
    # Override with API credentials for environment
    api_creds = get_api_credentials(environment)
    config_data.update(api_creds)
    
    # Set environment and dry_run
    config_data['environment'] = environment
    config_data['dry_run'] = dry_run
    
    # Create config object
    try:
        # Handle strategy weights enum conversion
        if 'strategy_weights' in config_data:
            strategy_weights = {}
            for k, v in config_data['strategy_weights'].items():
                if isinstance(k, str):
                    # Convert string to enum
                    strategy_type = StrategyType(k)
                    strategy_weights[strategy_type] = v
                else:
                    strategy_weights[k] = v
            config_data['strategy_weights'] = strategy_weights
        
        config = TradingConfig(**config_data)
        
    except TypeError as e:
        logger.error(f"Configuration error: {e}")
        # Create with defaults if there's an error
        config = TradingConfig(
            recall_api_key=api_creds['recall_api_key'],
            recall_api_url=api_creds['recall_api_url'],
            environment=environment,
            dry_run=dry_run
        )
    
    return config


def validate_config(config: TradingConfig) -> bool:
    """Validate configuration and log warnings"""
    
    if not config.validate():
        return False
    
    # Additional validation warnings
    if config.max_position_size > 0.3:
        logger.warning(f"High max_position_size: {config.max_position_size:.1%}")
    
    if config.min_confidence < 0.5:
        logger.warning(f"Low min_confidence: {config.min_confidence:.1%}")
    
    if config.environment == "production" and not config.openai_api_key:
        logger.warning("No OpenAI API key - using fallback analysis only")
    
    return True


def create_default_config_file() -> None:
    """Create default configuration file"""
    
    config = TradingConfig(
        recall_api_key="your_api_key_here",
        recall_api_url="https://api.recall.network",
        openai_api_key="your_openai_key_here"
    )
    
    config_path = Path(__file__).parent / "portfolio_config.json"
    config.save(str(config_path))
    
    logger.info(f"Default config created at {config_path}")


if __name__ == "__main__":
    # Create default config file
    create_default_config_file()