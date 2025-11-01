"""
Run Backtest

CLI script to run backtests on historical data.

Usage:
    # Fresh backtest (deterministic testing)
    python scripts/run_backtest.py \\
        --start 2025-10-18 \\
        --end 2025-10-20 \\
        --model claude \\
        --temperature 0.0 \\
        --checkpoint checkpoints/base_oct20.pkl

    # Resume from checkpoint
    python scripts/run_backtest.py \\
        --resume checkpoints/base_oct20.pkl \\
        --end 2025-10-26 \\
        --checkpoint checkpoints/claude_oct26.pkl

    # Non-determinism testing (temp=0.0 with --no-cache)
    python scripts/run_backtest.py \\
        --resume checkpoints/base_oct20.pkl \\
        --end 2025-10-26 \\
        --temperature 0.0 \\
        --no-cache \\
        --run-id 1
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.backtesting.backtest_engine import BacktestEngine
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Run backtests on historical trading data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fresh backtest
  python scripts/run_backtest.py --start 2025-10-18 --end 2025-10-20 --model claude

  # Resume from checkpoint
  python scripts/run_backtest.py --resume checkpoints/base_oct20.pkl --end 2025-10-26

  # Non-determinism testing (no cache)
  python scripts/run_backtest.py --resume checkpoints/base.pkl --no-cache --run-id 1
        """
    )

    # Date arguments
    parser.add_argument(
        '--start',
        type=str,
        help='Start date (YYYY-MM-DD), required for fresh backtest'
    )
    parser.add_argument(
        '--end',
        type=str,
        required=True,
        help='End date (YYYY-MM-DD), required'
    )

    # Model configuration
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
        default=0.0,
        help='LLM temperature (0.0=deterministic for testing, 0.7=creative, default: 0.0)'
    )

    # Coins to trade
    parser.add_argument(
        '--coins',
        type=str,
        nargs='+',
        default=['BTC', 'ETH', 'SOL', 'BNB', 'XRP', 'DOGE'],
        help='Coins to trade (default: all 6 Alpha Arena coins)'
    )

    # Interval
    parser.add_argument(
        '--interval',
        type=str,
        default='3m',
        choices=['1m', '3m', '4h'],
        help='Candle interval for decision-making (default: 3m)'
    )

    # Capital
    parser.add_argument(
        '--capital',
        type=float,
        default=10000.0,
        help='Starting capital in USD (default: 10000)'
    )

    # Checkpoint arguments
    parser.add_argument(
        '--resume',
        type=str,
        help='Resume from checkpoint file'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        help='Save final checkpoint to this file'
    )
    parser.add_argument(
        '--save-every',
        type=int,
        help='Save intermediate checkpoint every N iterations'
    )

    # LLM cache control
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Disable LLM response caching (for non-determinism testing)'
    )

    # Non-determinism testing
    parser.add_argument(
        '--run-id',
        type=int,
        help='Run ID for non-determinism testing'
    )

    # Data directories
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/historical',
        help='Directory with historical data (default: data/historical)'
    )
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='checkpoints',
        help='Directory for checkpoints (default: checkpoints)'
    )

    # Output
    parser.add_argument(
        '--output',
        type=str,
        help='Save results to JSON file'
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.resume and not args.start:
        parser.error("--start is required for fresh backtest (or use --resume)")

    # Determine start date
    if args.resume:
        # When resuming, start_date will be loaded from checkpoint
        start_date = args.start if args.start else "2025-10-18"  # Placeholder
        logger.info(f"Resuming from checkpoint: {args.resume}")
    else:
        start_date = args.start
        logger.info(f"Starting fresh backtest from {start_date}")

    # Print configuration
    logger.info("=" * 80)
    logger.info("BACKTEST CONFIGURATION")
    logger.info("=" * 80)
    logger.info(f"Date range: {start_date} to {args.end}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Temperature: {args.temperature}")
    logger.info(f"Coins: {', '.join(args.coins)}")
    logger.info(f"Interval: {args.interval}")
    logger.info(f"Starting capital: ${args.capital:,.2f}")
    logger.info(f"LLM cache: {'disabled' if args.no_cache else 'enabled'}")
    if args.run_id is not None:
        logger.info(f"Run ID: {args.run_id}")
    if args.resume:
        logger.info(f"Resume from: {args.resume}")
    if args.checkpoint:
        logger.info(f"Save checkpoint to: {args.checkpoint}")
    logger.info("=" * 80)

    try:
        # Initialize backtest engine
        engine = BacktestEngine(
            start_date=start_date,
            end_date=args.end,
            coins=args.coins,
            model=args.model,
            starting_capital=args.capital,
            checkpoint_path=args.resume,
            use_llm_cache=not args.no_cache,
            run_id=args.run_id,
            data_dir=args.data_dir,
            checkpoint_dir=args.checkpoint_dir,
            interval=args.interval,
            temperature=args.temperature
        )

        # Run backtest
        results = engine.run(
            checkpoint_path=args.checkpoint,
            save_every_n_iterations=args.save_every
        )

        # Print summary
        logger.info("\n" + "=" * 80)
        logger.info("BACKTEST RESULTS")
        logger.info("=" * 80)
        logger.info(f"\n📊 RISK-ADJUSTED METRICS")
        logger.info(f"  Sharpe Ratio:        {results['sharpe_ratio']:>8.3f}")
        logger.info(f"  Max Drawdown:        {results['max_drawdown_pct']:>7.2f}%")
        logger.info(f"  Win Rate:            {results['win_rate']*100:>7.1f}%")
        logger.info(f"  Profit Factor:       {results['profit_factor']:>8.2f}")

        logger.info(f"\n💰 RETURNS")
        logger.info(f"  Starting Capital:    ${results['starting_capital']:>11,.2f}")
        logger.info(f"  Final Value:         ${results['current_value']:>11,.2f}")
        logger.info(f"  Total Return:        ${results['total_return']:>11,.2f} ({results['total_return_pct']:>+6.2f}%)")
        logger.info(f"  Net Return:          ${results['net_return']:>11,.2f} ({results['net_return_pct']:>+6.2f}%)")

        logger.info(f"\n📈 TRADE STATISTICS")
        logger.info(f"  Total Trades:        {results['total_trades']:>8}")
        logger.info(f"  Winning Trades:      {results['winning_trades']:>8}")
        logger.info(f"  Losing Trades:       {results['losing_trades']:>8}")
        logger.info(f"  Avg Trade P&L:       ${results['avg_trade_pnl']:>10,.2f}")

        logger.info(f"\n💸 COSTS")
        logger.info(f"  Trading Fees:        ${results['total_fees_paid']:>11,.2f}")
        logger.info(f"  Total Costs:         ${results['total_costs']:>11,.2f}")

        logger.info(f"\n⚙️  EXECUTION")
        logger.info(f"  Model:               {results['model']}")
        logger.info(f"  Iterations:          {results['iterations']:>8}")
        logger.info(f"  LLM Cache Size:      {results['llm_cache_size']:>8}")
        if results.get('run_id') is not None:
            logger.info(f"  Run ID:              {results['run_id']:>8}")

        logger.info("\n" + "=" * 80)

        # Save results to file if requested
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)

            logger.info(f"Results saved to: {output_path}")

        logger.info("\n✅ BACKTEST COMPLETED SUCCESSFULLY\n")

        return 0

    except Exception as e:
        logger.error(f"\n❌ BACKTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
