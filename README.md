# 🤖 Recall Trading Agent

A sophisticated AI-powered trading agent designed for the Recall Network hackathon competition. Features multi-strategy analysis, advanced risk management, and real-time portfolio optimization.

## 🚀 Features

- **Multi-Strategy Trading**: Momentum, mean reversion, breakout, volatility, and ML ensemble strategies
- **Advanced Risk Management**: Kelly criterion position sizing, stop-loss, and portfolio limits
- **Real-time Portfolio Management**: Automated rebalancing and allocation drift monitoring
- **Competition Optimized**: Designed for consistent returns and capital preservation
- **Comprehensive Logging**: Full audit trail of all decisions and trades

## 📁 Project Structure

```
recall-trading-agent/
├── main.py                 # Main entry point
├── config/
│   ├── settings.py         # Configuration management
│   └── portfolio_config.json
├── src/
│   ├── core/              # Core data models and enums
│   ├── trading/           # Trading client and portfolio management
│   ├── strategies/        # Trading strategies
│   ├── indicators/        # Technical indicators
│   ├── agent/            # Main trading agent
│   └── utils/            # Utilities and helpers
├── tests/                # Unit tests
├── logs/                 # Log files
└── docs/                 # Documentation
```

## ⚡ Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/yourusername/recall-trading-agent.git
cd recall-trading-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Create a `.env` file:

```bash
# API Keys
OPENAI_API_KEY=your_openai_key_here  # Optional
RECALL_API_KEY=your_recall_key_here
RECALL_API_URL=your_recall_url_here

# Trading Parameters
MAX_POSITION_SIZE=0.22
MIN_CONFIDENCE=0.65
MIN_TRADE_AMOUNT=15.0
```

### 3. Run the Agent

```bash
# Sandbox mode (recommended for testing)
python main.py --env sandbox --mode competition --duration 24

# Production mode
python main.py --env production --mode competition --duration 24

# Single cycle test
python main.py --mode single

# Portfolio status
python main.py --mode status

# Performance report
python main.py --mode report
```

## 🎯 Competition Strategy

### Risk-First Approach
- **Conservative Position Sizing**: Maximum 22% per position
- **Kelly Criterion**: Dynamic position sizing based on signal confidence
- **Stop Loss**: 4% automatic stop loss on positions
- **Portfolio Limits**: Maximum 20% portfolio volatility

### Multi-Strategy Ensemble
- **Momentum (35%)**: Trend following with volume confirmation
- **Mean Reversion (25%)**: Bollinger bands and Z-score analysis  
- **Breakout (20%)**: Support/resistance level breaks
- **Volatility (10%)**: Volatility regime detection
- **ML Ensemble (10%)**: Machine learning signal aggregation

### Target Allocations
- **USDC (25%)**: Cash for flexibility and opportunities
- **WETH (40%)**: Stable large-cap exposure
- **SOL (35%)**: Higher growth potential

## 🔧 Configuration Options

### Trading Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_position_size` | 0.22 | Maximum position as % of portfolio |
| `min_confidence` | 0.65 | Minimum signal confidence to trade |
| `rebalance_threshold` | 0.04 | Portfolio rebalancing trigger |
| `min_trade_amount` | 15.0 | Minimum trade size (USD) |
| `max_slippage` | 0.015 | Maximum allowed slippage |

### Risk Management

| Parameter | Default | Description |
|-----------|---------|-------------|
| `stop_loss_percent` | 0.04 | Stop loss threshold (4%) |
| `take_profit_percent` | 0.12 | Take profit threshold (12%) |
| `kelly_fraction` | 0.2 | Kelly criterion sizing factor |
| `max_portfolio_volatility` | 0.20 | Maximum portfolio volatility |

## 📊 Strategies Overview

### Momentum Strategy
```python
# Signals generated when:
- Short SMA > Long SMA (bullish crossover)
- Price momentum > 3% with volume confirmation
- RSI not in extreme overbought/oversold territory
```

### Mean Reversion Strategy
```python
# Signals generated when:
- Price touches Bollinger Band extremes
- Z-score indicates oversold/overbought conditions
- RSI confirms contrarian signal
```

### Breakout Strategy
```python
# Signals generated when:
- Price breaks above resistance with volume
- Price breaks below support with volume
- Volume spike confirms breakout validity
```

### Volatility Strategy
```python
# Signals generated based on:
- Volatility regime changes
- High volatility reversal signals
- Low volatility trend continuation
```

### ML Ensemble Strategy
```python
# Combines multiple factors:
- Price momentum and trend analysis
- Volume patterns and confirmation
- Technical indicator convergence
- Market condition assessment
```

## 🛠️ Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_strategies.py
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

### Adding New Strategies

1. Create strategy class inheriting from `BaseStrategy`
2. Implement `generate_signal()` method
3. Add strategy to `TradingAgent._initialize_strategies()`
4. Update configuration weights

Example:
```python
from src.strategies.base import BaseStrategy

class CustomStrategy(BaseStrategy):
    def __init__(self, config):
        super().__init__(config, StrategyType.CUSTOM)
    
    async def generate_signal(self, symbol, market_data, portfolio):
        # Your strategy logic here
        return TradingSignal(...)
```

## 📈 Performance Monitoring

### Real-time Monitoring
- Portfolio value and allocation tracking
- P&L monitoring (realized and unrealized)
- Trade execution logs with reasoning
- Strategy performance attribution

### Performance Metrics
- Total return and annualized return
- Sharpe ratio and risk-adjusted returns
- Maximum drawdown and volatility
- Win rate and profit factor
- Trade statistics and success rates

### Logging
```bash
# View real-time logs
tail -f logs/trading_agent_YYYYMMDD.log

# Check error logs
grep "ERROR" logs/trading_agent_YYYYMMDD.log
```


## 🎛️ Environment Variables

```bash
# Required
RECALL_API_KEY=your_api_key_here
RECALL_API_URL=your_api_url_here

# Optional
OPENAI_API_KEY=your_openai_key_here
ENVIRONMENT=sandbox  # or production
DRY_RUN=false
LOG_LEVEL=INFO

# Trading Parameters
MAX_POSITION_SIZE=0.22
MIN_CONFIDENCE=0.65
REBALANCE_THRESHOLD=0.04
MIN_TRADE_AMOUNT=15.0
```

### Risk Disclaimers
- This software is for educational and competition purposes only
- Trading involves risk and past performance doesn't guarantee future results
- Use at your own risk and never trade more than you can afford to lose
- The agent makes autonomous decisions based on configured parameters

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
