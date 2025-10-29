"""
Analyze Wallet Fills with Readable Timestamps

Quick analysis script to view trading data with human-readable dates.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from datetime import datetime
from pathlib import Path
from collections import Counter
from src.utils.config import load_config

def analyze_wallet(model_name: str):
    """Analyze fills for a specific model"""

    file_path = Path(f"data/wallet_trades/{model_name}.json")

    if not file_path.exists():
        print(f"Error: {file_path} not found")
        return

    with open(file_path, 'r') as f:
        data = json.load(f)

    fills = data['fills']

    print("="*80)
    print(f"Analysis: {data['model']}")
    print(f"Address: {data['address']}")
    print("="*80)

    # Convert time and analyze
    for fill in fills:
        fill['datetime'] = datetime.fromtimestamp(fill['time'] / 1000)

    # Sort by time
    fills_sorted = sorted(fills, key=lambda x: x['time'])

    # Statistics
    print(f"\nTotal fills: {len(fills)}")
    print(f"Date range: {fills_sorted[0]['datetime'].date()} to {fills_sorted[-1]['datetime'].date()}")

    # Coins traded
    coins = Counter([f['coin'] for f in fills])
    print(f"\nCoins traded:")
    for coin, count in coins.most_common():
        print(f"  {coin}: {count} fills")

    # Direction analysis
    directions = Counter([f['dir'] for f in fills])
    print(f"\nTrading directions:")
    for direction, count in directions.most_common():
        print(f"  {direction}: {count}")

    # Calculate total P&L
    total_pnl = sum(float(f.get('closedPnl', 0)) for f in fills)
    total_fees = sum(float(f.get('fee', 0)) for f in fills)

    print(f"\nFinancials:")
    print(f"  Closed P&L: ${total_pnl:,.2f}")
    print(f"  Total Fees: ${total_fees:,.2f}")
    print(f"  Net P&L: ${total_pnl - total_fees:,.2f}")

    # Show recent trades
    print(f"\nMost Recent 5 Fills:")
    print("-"*80)
    print(f"{'Time':<20} {'Coin':<6} {'Side':<4} {'Price':<12} {'Size':<12} {'P&L':<12}")
    print("-"*80)

    for fill in fills_sorted[-5:]:
        time_str = fill['datetime'].strftime('%Y-%m-%d %H:%M:%S')
        coin = fill['coin']
        side = fill['side']
        px = f"${float(fill['px']):,.2f}"
        sz = f"{float(fill['sz']):,.2f}"
        pnl = f"${float(fill.get('closedPnl', 0)):,.2f}"

        print(f"{time_str:<20} {coin:<6} {side:<4} {px:<12} {sz:<12} {pnl:<12}")

    print("="*80)


def compare_all_models():
    """Compare trading patterns across all models"""

    # Load wallet config to get model names
    config = load_config('config/wallets.yaml')
    models = [name.replace(' ', '_').lower() for name in config['wallets'].keys()]

    print("\n" + "="*80)
    print("Model Comparison - P&L Analysis")
    print("="*80)
    print(f"\n{'Model':<25} {'Fills':<10} {'Closed P&L':<15} {'Fees':<15} {'Net P&L':<15}")
    print("-"*80)

    results = []

    for model in models:
        file_path = Path(f"data/wallet_trades/{model}.json")

        if not file_path.exists():
            continue

        with open(file_path, 'r') as f:
            data = json.load(f)

        fills = data['fills']
        model_name = data['model']

        # Calculate financials
        closed_pnl = sum(float(f.get('closedPnl', 0)) for f in fills)
        fees = sum(float(f.get('fee', 0)) for f in fills)
        net_pnl = closed_pnl - fees

        results.append({
            'model': model_name,
            'fills': len(fills),
            'closed_pnl': closed_pnl,
            'fees': fees,
            'net_pnl': net_pnl
        })

    # Sort by net P&L
    results.sort(key=lambda x: x['net_pnl'], reverse=True)

    for r in results:
        print(f"{r['model']:<25} {r['fills']:<10} "
              f"${r['closed_pnl']:>12,.2f}  "
              f"${r['fees']:>12,.2f}  "
              f"${r['net_pnl']:>12,.2f}")

    print("="*80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Analyze wallet trading fills')
    parser.add_argument('--model', type=str, help='Model to analyze (e.g., deepseek_chat_v3.1)')
    parser.add_argument('--compare', action='store_true', help='Compare all models')

    args = parser.parse_args()

    if args.compare:
        compare_all_models()
    elif args.model:
        analyze_wallet(args.model)
    else:
        # Default: show comparison
        compare_all_models()
