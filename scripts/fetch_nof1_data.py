#!/usr/bin/env python3
"""
Fetch and log data from nof1.ai Alpha Arena

Usage:
    python scripts/fetch_nof1_data.py --action leaderboard
    python scripts/fetch_nof1_data.py --action trades --model deepseek-chat-v3.1
    python scripts/fetch_nof1_data.py --action positions --model gpt-5
    python scripts/fetch_nof1_data.py --action monitor --interval 60
"""

import argparse
import sys
from pathlib import Path
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.nof1_client import get_client
from src.utils.rotating_logger import setup_rotating_logger

# Set up logger with 5000 line rotation
logger = setup_rotating_logger(
    name="nof1_fetcher",
    log_dir="logs",
    max_lines=5000,
    log_level="INFO",
    console_output=True
)


def log_leaderboard(leaderboard):
    """Log formatted leaderboard"""
    logger.info("="*80)
    logger.info("ALPHA ARENA LEADERBOARD")
    logger.info("="*80)
    logger.info(f"{'Rank':<6} {'Model':<25} {'NAV':<12} {'Return %':<10} {'Positions':<10}")
    logger.info("-"*80)

    for i, model in enumerate(leaderboard, 1):
        rank_emoji = {1: "🥇", 2: "🥈", 3: "🥉"}.get(i, "  ")
        logger.info(
            f"{rank_emoji} {i:<3} {model['model_id']:<25} ${model['nav']:<11,.2f} "
            f"{model['return_pct']:>+8.2f}% {model['num_positions']:<10}"
        )

    logger.info("="*80)


def log_positions(model_id, positions_data):
    """Log formatted positions for a model"""
    if not positions_data:
        logger.warning(f"No data found for model: {model_id}")
        return

    logger.info("="*80)
    logger.info(f"POSITIONS: {model_id}")
    logger.info("="*80)
    logger.info(f"Realized PnL: ${positions_data['realized_pnl']:,.2f}")
    logger.info(f"Timestamp: {positions_data['timestamp']}")
    logger.info("-"*80)

    positions = positions_data.get('positions', {})

    if not positions:
        logger.info("No open positions")
    else:
        logger.info(f"{'Symbol':<8} {'Qty':<10} {'Entry':<12} {'Current':<12} {'Unrealized PnL':<15} {'Leverage'}")
        logger.info("-"*80)

        for symbol, pos in positions.items():
            logger.info(
                f"{symbol:<8} {pos['quantity']:<10.4f} ${pos['entry_price']:<11,.2f} "
                f"${pos['current_price']:<11,.2f} ${pos['unrealized_pnl']:>+13,.2f} {pos['leverage']}x"
            )

    logger.info("="*80)


def log_trades(model_id, trades):
    """Log formatted trades for a model"""
    if not trades:
        logger.warning(f"No trades found for model: {model_id}")
        return

    logger.info("="*80)
    logger.info(f"RECENT TRADES: {model_id} (Last 10)")
    logger.info("="*80)
    logger.info(f"{'Symbol':<8} {'Side':<6} {'Entry':<12} {'Exit':<12} {'PnL':<12} {'Date'}")
    logger.info("-"*80)

    for trade in trades[-10:]:  # Last 10 trades
        pnl = trade.get('realized_net_pnl', 0)
        pnl_str = f"${pnl:+,.2f}"
        date = trade.get('exit_human_time', '')[:10]  # Just the date

        logger.info(
            f"{trade['symbol']:<8} {trade['side']:<6} ${trade['entry_price']:<11,.2f} "
            f"${trade['exit_price']:<11,.2f} {pnl_str:<12} {date}"
        )

    logger.info("="*80)
    logger.info(f"Total Trades: {len(trades)}")

    # Summary stats
    total_pnl = sum(t.get('realized_net_pnl', 0) for t in trades)
    winning_trades = sum(1 for t in trades if t.get('realized_net_pnl', 0) > 0)
    win_rate = (winning_trades / len(trades) * 100) if trades else 0

    logger.info(f"Total PnL: ${total_pnl:,.2f}")
    logger.info(f"Win Rate: {win_rate:.1f}% ({winning_trades}/{len(trades)})")
    logger.info("="*80)


def main():
    parser = argparse.ArgumentParser(description="Fetch Alpha Arena data from nof1.ai")
    parser.add_argument(
        '--action',
        choices=['leaderboard', 'trades', 'positions', 'monitor', 'all'],
        default='leaderboard',
        help='Action to perform'
    )
    parser.add_argument('--model', type=str, help='Model ID (for trades/positions)')
    parser.add_argument('--interval', type=int, default=60, help='Monitoring interval in seconds')
    parser.add_argument('--cache', type=str, help='Cache directory for responses')
    parser.add_argument('--output', type=str, help='Output JSON file')

    args = parser.parse_args()

    logger.info(f"Starting nof1.ai data fetcher - Action: {args.action}")

    # Create client
    client = get_client(cache_dir=args.cache)

    try:
        if args.action == 'leaderboard':
            logger.info("Fetching leaderboard...")
            leaderboard = client.get_leaderboard()
            log_leaderboard(leaderboard)

            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(leaderboard, f, indent=2)
                logger.info(f"Saved to {args.output}")

        elif args.action == 'trades':
            if not args.model:
                logger.error("--model required for trades action")
                sys.exit(1)

            logger.info(f"Fetching trades for {args.model}...")
            trades = client.get_model_trades(args.model)
            log_trades(args.model, trades)

            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(trades, f, indent=2)
                logger.info(f"Saved to {args.output}")

        elif args.action == 'positions':
            if not args.model:
                logger.error("--model required for positions action")
                sys.exit(1)

            logger.info(f"Fetching positions for {args.model}...")
            positions = client.get_model_positions(args.model)
            log_positions(args.model, positions)

            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(positions, f, indent=2)
                logger.info(f"Saved to {args.output}")

        elif args.action == 'all':
            logger.info("📊 FETCHING ALL DATA FROM NOF1.AI...")

            # Leaderboard
            logger.info("Fetching leaderboard...")
            leaderboard = client.get_leaderboard()
            log_leaderboard(leaderboard)

            # Positions for top 3
            logger.info("📈 TOP 3 MODEL POSITIONS:")
            for model in leaderboard[:3]:
                logger.info(f"Fetching positions for {model['model_id']}...")
                positions = client.get_model_positions(model['model_id'])
                if positions:
                    log_positions(model['model_id'], positions)

            # Save everything
            if args.output:
                logger.info(f"Saving all data to {args.output}...")
                data = {
                    'leaderboard': leaderboard,
                    'trades': client.get_trades(),
                    'positions': client.get_account_totals(),
                    'prices': client.get_crypto_prices(),
                    'inception': client.get_since_inception_values()
                }
                with open(args.output, 'w') as f:
                    json.dump(data, f, indent=2)
                logger.info(f"✅ All data saved to {args.output}")

        elif args.action == 'monitor':
            def on_update(data):
                logger.info(f"[{data['timestamp']}] Update received")
                logger.info(
                    f"Active positions: {sum(len(a.get('positions', {})) for a in data['positions'])}"
                )
                logger.info(f"Total trades: {len(data['trades'])}")
                log_leaderboard(data['leaderboard'])

            logger.info(f"🔄 Monitoring nof1.ai every {args.interval} seconds (Ctrl+C to stop)...")
            client.monitor_updates(interval_seconds=args.interval, callback=on_update)

    except KeyboardInterrupt:
        logger.info("Stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
