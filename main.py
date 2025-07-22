#!/usr/bin/env python3
"""
Recall Trading Agent - Main Entry Point
"""

import asyncio
import argparse
import logging
import os
from typing import Optional

from config.settings import create_config, validate_config
from src.agent.trading_agent import TradingAgent
from src.utils.logger import setup_logging


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Recall Trading Agent')
    
    parser.add_argument(
        '--mode',
        choices=['single', 'competition', 'status', 'report'],
        default='competition',
        help='Trading mode (default: competition)'
    )
    
    parser.add_argument(
        '--env',
        choices=['sandbox', 'production'],
        default='sandbox',
        help='Environment (default: sandbox)'
    )
    
    parser.add_argument(
        '--duration',
        type=int,
        default=24,
        help='Competition duration in hours (default: 24)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Run in dry-run mode (no actual trades)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to custom config file'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    return parser.parse_args()


async def run_single_cycle(agent: TradingAgent) -> None:
    """Run a single trading cycle"""
    print("🔄 Running single trading cycle...")
    await agent.execute_trading_cycle()
    print("✅ Single cycle completed")


async def run_competition(agent: TradingAgent, duration_hours: int) -> None:
    """Run competition mode"""
    print(f"🏆 Starting {duration_hours}h competition...")
    await agent.run_competition(duration_hours)


async def show_status(agent: TradingAgent) -> None:
    """Show portfolio status"""
    print("📊 Fetching portfolio status...")
    await agent.show_portfolio_status()


async def generate_report(agent: TradingAgent) -> None:
    """Generate performance report"""
    print("📈 Generating performance report...")
    await agent.generate_performance_report()


async def main() -> None:
    """Main entry point"""
    
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    setup_logging(level=args.log_level)
    logger = logging.getLogger(__name__)
    
    try:
        # Create configuration
        config = create_config(
            environment=args.env,
            config_path=args.config,
            dry_run=args.dry_run
        )
        
        # Validate configuration
        if not validate_config(config):
            logger.error("❌ Configuration validation failed")
            return
        
        # Create trading agent
        agent = TradingAgent(config)
        
        # Display startup info
        print("🚀 Recall Trading Agent")
        print("=" * 40)
        print(f"Environment: {args.env.upper()}")
        print(f"Mode: {args.mode}")
        print(f"Dry Run: {args.dry_run}")
        print(f"API Key: {config.recall_api_key[:8]}...")
        print("=" * 40)
        
        # Execute based on mode
        if args.mode == 'single':
            await run_single_cycle(agent)
        elif args.mode == 'competition':
            await run_competition(agent, args.duration)
        elif args.mode == 'status':
            await show_status(agent)
        elif args.mode == 'report':
            await generate_report(agent)
        
    except KeyboardInterrupt:
        logger.info("👋 Stopped by user")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        raise


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")