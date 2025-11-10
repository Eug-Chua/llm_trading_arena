"""
Benchmark Trading Strategies

Implements simple baseline strategies to compare against LLM performance:
1. Buy-and-Hold BTC
2. Equal-Weight Portfolio (all 6 coins)
3. Random Trading (control for luck)

These benchmarks help answer: "Are LLMs actually adding value, or would simple
strategies perform just as well?"
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data.indicators import load_indicator_config


def load_historical_prices(coin: str, start_date: str, end_date: str, interval: str = '4h') -> pd.DataFrame:
    """
    Load historical price data for a coin

    Args:
        coin: Coin symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        interval: Data interval

    Returns:
        DataFrame with OHLCV data
    """
    # Try archive location first
    data_dir = project_root / "data" / "archive" / "historical" / interval
    file_path = data_dir / f"{coin}.parquet"

    if not file_path.exists():
        # Try alternate location
        data_dir = project_root / "data" / "historical"
        file_path = data_dir / f"{coin}USD_{interval}.parquet"

    if not file_path.exists():
        raise FileNotFoundError(f"Historical data not found: {file_path}")

    df = pd.read_parquet(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Filter to date range
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    df = df[(df['timestamp'] >= start) & (df['timestamp'] <= end)]

    return df.sort_values('timestamp').reset_index(drop=True)


def calculate_returns_and_metrics(equity_curve: List[float], starting_capital: float = 10000) -> Dict:
    """
    Calculate performance metrics from equity curve

    Args:
        equity_curve: List of portfolio values over time
        starting_capital: Starting capital

    Returns:
        Dictionary with performance metrics
    """
    if len(equity_curve) < 2:
        return {
            'final_value': starting_capital,
            'total_return_pct': 0,
            'sharpe_ratio': 0,
            'max_drawdown_pct': 0,
            'volatility': 0
        }

    equity = np.array(equity_curve)
    returns = np.diff(equity) / equity[:-1]

    # Total return
    final_value = equity[-1]
    total_return_pct = ((final_value - starting_capital) / starting_capital) * 100

    # Sharpe ratio (assuming 4h intervals)
    # Annualization: 365 days * 6 intervals per day = 2190 intervals per year
    if len(returns) > 1 and np.std(returns) > 0:
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(2190)
    else:
        sharpe_ratio = 0

    # Max drawdown
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak * 100
    max_drawdown_pct = np.min(drawdown)

    # Volatility (annualized)
    volatility = np.std(returns) * np.sqrt(2190) * 100 if len(returns) > 1 else 0

    return {
        'final_value': final_value,
        'total_return_pct': total_return_pct,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown_pct': max_drawdown_pct,
        'volatility': volatility,
        'n_periods': len(equity_curve)
    }


def buy_and_hold_btc(start_date: str, end_date: str, starting_capital: float = 10000, interval: str = '4h') -> Dict:
    """
    Buy-and-Hold BTC strategy

    Buy BTC at the first timestamp, hold until the end.

    Args:
        start_date: Start date
        end_date: End date
        starting_capital: Starting capital
        interval: Data interval

    Returns:
        Performance metrics dictionary
    """
    print(f"\nRunning Buy-and-Hold BTC...")

    # Load BTC prices
    df = load_historical_prices('BTC', start_date, end_date, interval)

    if len(df) == 0:
        print("❌ No data found for BTC")
        return {}

    # Buy at first price
    entry_price = df.iloc[0]['close']
    btc_quantity = starting_capital / entry_price

    print(f"  Entry: {df.iloc[0]['timestamp']} at ${entry_price:,.2f}")
    print(f"  Bought {btc_quantity:.6f} BTC")

    # Track equity curve
    equity_curve = []
    for _, row in df.iterrows():
        current_value = btc_quantity * row['close']
        equity_curve.append(current_value)

    # Exit at last price
    exit_price = df.iloc[-1]['close']
    final_value = btc_quantity * exit_price

    print(f"  Exit: {df.iloc[-1]['timestamp']} at ${exit_price:,.2f}")
    print(f"  Final Value: ${final_value:,.2f}")

    # Calculate metrics
    metrics = calculate_returns_and_metrics(equity_curve, starting_capital)
    metrics['strategy'] = 'Buy-and-Hold BTC'
    metrics['entry_price'] = entry_price
    metrics['exit_price'] = exit_price
    metrics['btc_return_pct'] = ((exit_price - entry_price) / entry_price) * 100

    return metrics


def equal_weight_portfolio(start_date: str, end_date: str, coins: List[str],
                          starting_capital: float = 10000, interval: str = '4h') -> Dict:
    """
    Equal-Weight Portfolio strategy

    Allocate capital equally across all coins, rebalance at start only.

    Args:
        start_date: Start date
        end_date: End date
        coins: List of coins to include
        starting_capital: Starting capital
        interval: Data interval

    Returns:
        Performance metrics dictionary
    """
    print(f"\nRunning Equal-Weight Portfolio ({len(coins)} coins)...")

    # Load all coin prices
    coin_data = {}
    for coin in coins:
        try:
            df = load_historical_prices(coin, start_date, end_date, interval)
            if len(df) > 0:
                coin_data[coin] = df
        except FileNotFoundError:
            print(f"  ⚠️  Skipping {coin} - no data found")
            continue

    if not coin_data:
        print("❌ No data found for any coins")
        return {}

    # Allocate equally at start
    capital_per_coin = starting_capital / len(coin_data)
    positions = {}

    print(f"  Allocating ${capital_per_coin:,.2f} per coin")

    for coin, df in coin_data.items():
        entry_price = df.iloc[0]['close']
        quantity = capital_per_coin / entry_price
        positions[coin] = {
            'quantity': quantity,
            'entry_price': entry_price,
            'df': df
        }
        print(f"    {coin}: {quantity:.6f} @ ${entry_price:,.2f}")

    # Track equity curve (align all timestamps)
    # Get common timestamps across all coins
    all_timestamps = set(coin_data[list(coin_data.keys())[0]]['timestamp'])
    for coin, df in coin_data.items():
        all_timestamps = all_timestamps.intersection(set(df['timestamp']))

    all_timestamps = sorted(list(all_timestamps))

    equity_curve = []
    for timestamp in all_timestamps:
        portfolio_value = 0
        for coin, position in positions.items():
            df = position['df']
            price_row = df[df['timestamp'] == timestamp]
            if len(price_row) > 0:
                current_price = price_row.iloc[0]['close']
                portfolio_value += position['quantity'] * current_price

        equity_curve.append(portfolio_value)

    # Calculate final value
    final_value = equity_curve[-1] if equity_curve else starting_capital

    print(f"  Final Value: ${final_value:,.2f}")

    # Calculate metrics
    metrics = calculate_returns_and_metrics(equity_curve, starting_capital)
    metrics['strategy'] = f'Equal-Weight Portfolio ({len(coin_data)} coins)'
    metrics['coins'] = list(coin_data.keys())
    metrics['n_coins'] = len(coin_data)

    return metrics


def random_trading(start_date: str, end_date: str, coins: List[str],
                  starting_capital: float = 10000, interval: str = '4h',
                  n_trades: int = 20, seed: int = 42) -> Dict:
    """
    Random Trading strategy (control for luck)

    Make random buy/sell decisions to test if LLM performance is just luck.

    Args:
        start_date: Start date
        end_date: End date
        coins: List of coins to trade
        starting_capital: Starting capital
        interval: Data interval
        n_trades: Number of random trades to make
        seed: Random seed for reproducibility

    Returns:
        Performance metrics dictionary
    """
    print(f"\nRunning Random Trading (n_trades={n_trades}, seed={seed})...")

    np.random.seed(seed)

    # Load all coin prices
    coin_data = {}
    for coin in coins:
        try:
            df = load_historical_prices(coin, start_date, end_date, interval)
            if len(df) > 0:
                coin_data[coin] = df
        except FileNotFoundError:
            continue

    if not coin_data:
        print("❌ No data found for any coins")
        return {}

    # Get common timestamps
    all_timestamps = set(coin_data[list(coin_data.keys())[0]]['timestamp'])
    for coin, df in coin_data.items():
        all_timestamps = all_timestamps.intersection(set(df['timestamp']))

    all_timestamps = sorted(list(all_timestamps))

    # Simple random trading simulation
    cash = starting_capital
    positions = {}
    equity_curve = [starting_capital]

    trade_count = 0
    max_position_size = 0.3  # Max 30% of capital per position

    for i, timestamp in enumerate(all_timestamps[1:], 1):
        # Random decision: buy, sell, or hold
        action = np.random.choice(['buy', 'sell', 'hold'], p=[0.3, 0.2, 0.5])

        if action == 'buy' and cash > 100 and trade_count < n_trades:
            # Buy random coin
            coin = np.random.choice(list(coin_data.keys()))
            df = coin_data[coin]
            price_row = df[df['timestamp'] == timestamp]

            if len(price_row) > 0:
                price = price_row.iloc[0]['close']
                position_size = cash * np.random.uniform(0.1, max_position_size)
                quantity = position_size / price

                if coin not in positions:
                    positions[coin] = {'quantity': 0, 'entry_price': price}

                positions[coin]['quantity'] += quantity
                cash -= position_size
                trade_count += 1

        elif action == 'sell' and positions:
            # Sell random position
            coin = np.random.choice(list(positions.keys()))
            df = coin_data[coin]
            price_row = df[df['timestamp'] == timestamp]

            if len(price_row) > 0:
                price = price_row.iloc[0]['close']
                sell_quantity = positions[coin]['quantity'] * np.random.uniform(0.5, 1.0)
                cash += sell_quantity * price
                positions[coin]['quantity'] -= sell_quantity

                if positions[coin]['quantity'] < 0.0001:
                    del positions[coin]

                trade_count += 1

        # Calculate portfolio value
        portfolio_value = cash
        for coin, position in positions.items():
            df = coin_data[coin]
            price_row = df[df['timestamp'] == timestamp]
            if len(price_row) > 0:
                portfolio_value += position['quantity'] * price_row.iloc[0]['close']

        equity_curve.append(portfolio_value)

    final_value = equity_curve[-1]
    print(f"  Trades executed: {trade_count}")
    print(f"  Final Value: ${final_value:,.2f}")

    # Calculate metrics
    metrics = calculate_returns_and_metrics(equity_curve, starting_capital)
    metrics['strategy'] = f'Random Trading (seed={seed})'
    metrics['n_trades'] = trade_count

    return metrics


def run_all_benchmarks(start_date: str, end_date: str, interval: str = '4h') -> pd.DataFrame:
    """
    Run all benchmark strategies and compare

    Args:
        start_date: Start date
        end_date: End date
        interval: Data interval

    Returns:
        DataFrame with all benchmark results
    """
    coins = ['BTC', 'ETH', 'SOL', 'BNB', 'XRP', 'DOGE']
    starting_capital = 10000

    results = []

    print("=" * 80)
    print("RUNNING BENCHMARK STRATEGIES")
    print("=" * 80)
    print(f"Period: {start_date} to {end_date}")
    print(f"Starting Capital: ${starting_capital:,.2f}")
    print()

    # 1. Buy-and-Hold BTC
    btc_result = buy_and_hold_btc(start_date, end_date, starting_capital, interval)
    if btc_result:
        results.append(btc_result)

    # 2. Equal-Weight Portfolio
    equal_weight_result = equal_weight_portfolio(start_date, end_date, coins, starting_capital, interval)
    if equal_weight_result:
        results.append(equal_weight_result)

    # 3. Random Trading (multiple seeds for robustness)
    print("\nRunning Random Trading (3 different seeds)...")
    for seed in [42, 123, 999]:
        random_result = random_trading(start_date, end_date, coins, starting_capital, interval, n_trades=20, seed=seed)
        if random_result:
            results.append(random_result)

    # Create summary DataFrame
    summary_data = []
    for result in results:
        summary_data.append({
            'Strategy': result['strategy'],
            'Final Value ($)': f"{result['final_value']:,.2f}",
            'Total Return (%)': f"{result['total_return_pct']:.2f}",
            'Sharpe Ratio': f"{result['sharpe_ratio']:.3f}",
            'Max Drawdown (%)': f"{result['max_drawdown_pct']:.2f}",
            'Volatility (%)': f"{result['volatility']:.2f}"
        })

    df = pd.DataFrame(summary_data)

    return df, results


def main():
    """Run benchmark analysis"""

    # Load config for date range
    config = load_indicator_config()
    interval = config.get('data_interval', '4h')

    # Use same dates as backtests
    start_date = "2025-10-17"
    end_date = "2025-10-31"

    # Run benchmarks
    summary_df, detailed_results = run_all_benchmarks(start_date, end_date, interval)

    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print()

    # Save results
    output_dir = project_root / "results" / "benchmarks"
    output_dir.mkdir(exist_ok=True)

    summary_df.to_csv(output_dir / "benchmark_summary.csv", index=False)

    import json
    with open(output_dir / "benchmark_detailed.json", 'w') as f:
        json.dump(detailed_results, f, indent=2, default=str)

    print(f"✓ Results saved to: {output_dir}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
