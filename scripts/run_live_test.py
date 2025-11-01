"""
Run Live Trading Test

Test the continuous trading loop with live Hyperliquid data.
This does NOT execute real trades - it simulates the trading logic.

Usage:
    # Test with BTC and ETH only (10 iterations)
    python scripts/run_live_test.py --coins BTC ETH --iterations 10

    # Test with all coins (3 iterations)
    python scripts/run_live_test.py --iterations 3

    # Test with Claude
    python scripts/run_live_test.py --model anthropic --coins BTC ETH --iterations 5

    # Test with OpenAI
    python scripts/run_live_test.py --model openai --coins BTC ETH --iterations 5

    # Custom interval (every 5 minutes instead of 3)
    python scripts/run_live_test.py --coins BTC ETH --iterations 5 --interval 5
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.agents.llm_agent import LLMAgent
from src.trading.continuous_loop import ContinuousEvaluationLoop
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Test live trading loop with selected coins',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--model',
        type=str,
        default='anthropic',
        choices=['anthropic', 'openai', 'deepseek'],
        help='LLM model to use (default: anthropic)'
    )

    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='LLM temperature (0.0=deterministic, 0.7=creative, default: 0.7)'
    )

    parser.add_argument(
        '--coins',
        type=str,
        nargs='+',
        default=['BTC', 'ETH', 'SOL', 'BNB', 'XRP', 'DOGE'],
        help='Coins to trade (default: all 6 coins)'
    )

    parser.add_argument(
        '--iterations',
        type=int,
        default=10,
        help='Number of iterations to run (default: 10)'
    )

    parser.add_argument(
        '--interval',
        type=float,
        default=3.0,
        help='Minutes between iterations (default: 3.0)'
    )

    parser.add_argument(
        '--capital',
        type=float,
        default=10000.0,
        help='Starting capital in USD (default: 10000)'
    )

    parser.add_argument(
        '--duration',
        type=float,
        help='Maximum duration in hours (alternative to --iterations)'
    )

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("LIVE TRADING TEST")
    logger.info("=" * 80)
    logger.info(f"Model: {args.model.upper()}")
    logger.info(f"Temperature: {args.temperature}")
    logger.info(f"Coins: {', '.join(args.coins)}")
    logger.info(f"Starting Capital: ${args.capital:,.2f}")
    logger.info(f"Interval: {args.interval} minutes")
    if args.iterations:
        estimated_time = args.iterations * args.interval
        logger.info(f"Iterations: {args.iterations} (~{estimated_time:.0f} minutes total)")
    elif args.duration:
        logger.info(f"Duration: {args.duration} hours")
    logger.info("=" * 80)
    logger.info("NOTE: This uses LIVE data but does NOT execute real trades.")

    try:
        # Initialize LLM agent
        llm_agent = LLMAgent(provider=args.model, temperature=args.temperature)

        # Create continuous loop
        loop = ContinuousEvaluationLoop(
            llm_agent=llm_agent,
            starting_capital=args.capital,
            interval_seconds=int(args.interval * 60),
            coins=args.coins
        )

        # Run loop
        loop.run(
            max_iterations=args.iterations,
            max_duration_hours=args.duration
        )

    except KeyboardInterrupt:
        logger.info("Loop interrupted by user (Ctrl+C)")
        logger.info("Exiting gracefully...")

    return 0


if __name__ == "__main__":
    sys.exit(main())
