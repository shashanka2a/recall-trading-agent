#!/usr/bin/env python3
"""
Automated setup script for Recall Trading Agent repository
Run this script to create the complete file structure with empty files
"""

import os
from pathlib import Path

def create_empty_file(filepath: str):
    """Create an empty file"""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create empty file
    path.touch()
    print(f"✅ Created: {filepath}")

def create_file_with_content(filepath: str, content: str):
    """Create a file with minimal content (only for essential files)"""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ Created: {filepath}")

def setup_repository():
    """Set up the complete repository structure with empty files"""
    
    print("🚀 Setting up Recall Trading Agent repository structure...")
    
    # Create directory structure
    directories = [
        "config",
        "src/core",
        "src/trading", 
        "src/strategies",
        "src/indicators",
        "src/agent",
        "src/utils",
        "tests",
        "logs",
        "docs",
        ".github/workflows",
        ".github/ISSUE_TEMPLATE"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {directory}")
    
    # Essential files with minimal content (needed for Python imports)
    essential_files = {
        "README.md": "# Recall Trading Agent\n\nAI-powered trading agent for Recall Network hackathon.\n",
        "requirements.txt": "# Add your dependencies here\naiohttp>=3.9.1\npandas>=2.1.4\nnumpy>=1.26.2\nopenai>=1.12.0\npython-dotenv>=1.0.0\n",
        ".env.example": "# Environment variables template\nRECALL_API_KEY=your_api_key_here\nRECALL_API_URL=your_api_url_here\nOPENAI_API_KEY=your_openai_key_here\n",
        ".gitignore": "__pycache__/\n*.pyc\n.env\nvenv/\nlogs/\n.DS_Store\n",
        "LICENSE": "MIT License\n\n# Add full license text here\n",
        
        # Python package files
        "src/__init__.py": '"""Recall Trading Agent"""\n__version__ = "2.0.0"\n',
        "src/core/__init__.py": '"""Core models and enums"""\n',
        "src/trading/__init__.py": '"""Trading components"""\n',
        "src/strategies/__init__.py": '"""Trading strategies"""\n',
        "src/indicators/__init__.py": '"""Technical indicators"""\n',
        "src/agent/__init__.py": '"""Main trading agent"""\n',
        "src/utils/__init__.py": '"""Utilities"""\n',
        "config/__init__.py": '"""Configuration"""\n',
        "tests/__init__.py": '"""Tests"""\n',
        
        # Git keep file
        "logs/.gitkeep": "# This file ensures the logs directory is tracked by git\n"
    }
    
    # All empty files to create
    empty_files = [
        # Root files
        "main.py",
        "setup.py",
        
        # Config files
        "config/settings.py",
        "config/portfolio_config.json",
        
        # Core module files
        "src/core/models.py",
        "src/core/enums.py", 
        "src/core/exceptions.py",
        
        # Trading module files
        "src/trading/client.py",
        "src/trading/portfolio_manager.py",
        "src/trading/risk_manager.py",
        
        # Strategy files
        "src/strategies/base.py",
        "src/strategies/momentum.py",
        "src/strategies/mean_reversion.py",
        "src/strategies/breakout.py",
        "src/strategies/volatility.py",
        "src/strategies/ml_ensemble.py",
        
        # Indicator files
        "src/indicators/technical.py",
        
        # Agent files
        "src/agent/trading_agent.py",
        
        # Utility files
        "src/utils/logger.py",
        "src/utils/helpers.py",
        
        # Test files
        "tests/test_strategies.py",
        "tests/test_indicators.py",
        "tests/test_portfolio.py",
        "tests/test_trading_agent.py",
        
        # Documentation files
        "docs/setup.md",
        "docs/strategies.md",
        "docs/api.md",
        
        # GitHub files
        ".github/workflows/tests.yml",
        ".github/ISSUE_TEMPLATE/bug_report.md",
        ".github/ISSUE_TEMPLATE/feature_request.md"
    ]
    
    # Create essential files with minimal content
    for filepath, content in essential_files.items():
        create_file_with_content(filepath, content)
    
    # Create all empty files
    for filepath in empty_files:
        create_empty_file(filepath)
    
    print(f"\n🎉 Repository structure setup complete!")
    print(f"📁 Created {len(essential_files)} essential files")
    print(f"📄 Created {len(empty_files)} empty files")
    print(f"📂 Created {len(directories)} directories")
    
    print(f"\n📋 Next steps:")
    print(f"1. Copy content from artifacts into the corresponding files")
    print(f"2. cp .env.example .env")
    print(f"3. Edit .env with your API keys")
    print(f"4. pip install -r requirements.txt")
    print(f"5. Copy main.py content from 'main_py' artifact")
    print(f"6. Copy other file contents from respective artifacts")
    print(f"7. git init && git add . && git commit -m 'Initial commit'")
    
    print(f"\n🗂️  File mapping to artifacts:")
    print(f"   main.py ← 'main_py' artifact")
    print(f"   config/settings.py ← 'config_settings' artifact")
    print(f"   src/core/models.py ← 'core_models' artifact")
    print(f"   src/core/enums.py ← 'core_enums' artifact")
    print(f"   src/trading/client.py ← 'trading_client' artifact")
    print(f"   src/strategies/base.py ← 'strategies_base' artifact")
    print(f"   src/strategies/momentum.py ← 'strategies_momentum' artifact")
    print(f"   src/indicators/technical.py ← 'technical_indicators' artifact")
    print(f"   src/agent/trading_agent.py ← 'trading_agent_main' artifact")
    print(f"   src/utils/logger.py ← 'utils_logger' artifact")
    print(f"   README.md ← 'readme_md' artifact")

if __name__ == "__main__":
    setup_repository()