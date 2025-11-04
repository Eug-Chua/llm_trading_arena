"""
Capture ratio calculation utilities for market timing analysis.

Calculates upside/downside capture ratios to measure how well the strategy
times its entries and exits relative to market movements.
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


def calculate_coin_capture_ratios(checkpoint: Dict, coins: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Calculate upside/downside capture ratios measuring market timing ability.

    For each coin, analyzes ALL 4h candles during the backtest period and measures:
    - Upside Capture: How much of the coin's up-moves did the strategy capture?
    - Downside Capture: How much of the coin's down-moves did the strategy suffer?

    Good market timing = High upside capture + Low downside capture

    The calculation works as follows:
    1. Load all 4h candles for the backtest period
    2. Build a position map showing when positions were held
    3. For each candle:
       - If holding: strategy_return = coin_return (captured the move)
       - If not holding: strategy_return = 0% (missed the move)
    4. Calculate capture ratio = (avg strategy return) / (avg benchmark return)

    Args:
        checkpoint: Checkpoint data with trade_history and metadata
        coins: List of coin symbols to analyze (e.g., ['BTC', 'ETH', 'SOL'])

    Returns:
        Dict mapping coin -> {
            'upside_capture': float,     # % of up-market gains captured
            'downside_capture': float,   # % of down-market losses suffered
            'periods_tracked': int       # Number of 4h candles analyzed
        }
        Capture ratios are percentages (e.g., 80% means captured 80% of benchmark moves)

    Example:
        If BTC went up 1% on average during up-candles, and the strategy was holding
        during half of them (capturing 0.5% on average), upside capture = 50%.
    """
    from src.backtesting.historical_loader import HistoricalDataLoader

    trade_log = checkpoint.get('trade_history', [])
    metadata = checkpoint.get('metadata', {})
    start_date = metadata.get('start_date')
    end_date = metadata.get('end_date')

    if not start_date or not end_date:
        return {}

    # Initialize historical data loader
    data_loader = HistoricalDataLoader()

    # Group trades by coin
    coin_results = {}

    for coin in coins:
        try:
            # Get ALL 4h candles for this coin during the backtest period
            all_candles = data_loader.load_data(
                coin=coin,
                interval='4h',
                start_date=start_date,
                end_date=end_date
            )

            if len(all_candles) < 1:
                continue

            # Build a map of when we held positions: timestamp -> leverage
            position_map = {}

            # Find all BUY/CLOSE pairs for this coin
            coin_trades = [t for t in trade_log if t.get('symbol') == coin]

            current_position = None
            for trade in sorted(coin_trades, key=lambda x: pd.to_datetime(x['timestamp'])):
                timestamp = pd.to_datetime(trade['timestamp'])

                if trade['action'] == 'BUY':
                    current_position = {
                        'leverage': trade.get('leverage', 1),
                        'entry_time': timestamp
                    }
                elif trade['action'] == 'CLOSE' and current_position:
                    # Mark all candles during this position as "holding"
                    entry_time = current_position['entry_time']
                    exit_time = timestamp
                    leverage = current_position['leverage']

                    # Find candles in this period
                    position_candles = all_candles[
                        (all_candles['timestamp'] >= entry_time) &
                        (all_candles['timestamp'] <= exit_time)
                    ]

                    for idx, candle in position_candles.iterrows():
                        position_map[candle['timestamp']] = leverage

                    current_position = None

            # Analyze each candle: up-market or down-market?
            upside_strategy_returns = []
            upside_benchmark_returns = []
            downside_strategy_returns = []
            downside_benchmark_returns = []

            for idx, candle in all_candles.iterrows():
                # Coin's % change for this candle (open to close)
                coin_return = ((candle['close'] - candle['open']) / candle['open']) * 100

                # Was the strategy holding a position during this candle?
                leverage = position_map.get(candle['timestamp'], 0)

                if leverage > 0:
                    # We were holding: strategy return = coin_return * leverage (include leverage effect)
                    strategy_return = coin_return * leverage
                else:
                    # We were NOT holding: strategy return = 0% (missed the move)
                    strategy_return = 0.0

                # Categorize as upside or downside based on coin's movement
                if coin_return > 0:
                    upside_benchmark_returns.append(coin_return)
                    upside_strategy_returns.append(strategy_return)
                elif coin_return < 0:
                    downside_benchmark_returns.append(coin_return)
                    downside_strategy_returns.append(strategy_return)

            # Calculate upside and downside capture ratios
            upside_capture = 0
            downside_capture = 0

            if upside_benchmark_returns and upside_strategy_returns:
                avg_upside_benchmark = np.mean(upside_benchmark_returns)
                avg_upside_strategy = np.mean(upside_strategy_returns)
                upside_capture = (avg_upside_strategy / avg_upside_benchmark) * 100 if avg_upside_benchmark != 0 else 0

            if downside_benchmark_returns and downside_strategy_returns:
                avg_downside_benchmark = np.mean(downside_benchmark_returns)
                avg_downside_strategy = np.mean(downside_strategy_returns)
                downside_capture = (avg_downside_strategy / avg_downside_benchmark) * 100 if avg_downside_benchmark != 0 else 0

            coin_results[coin] = {
                'upside_capture': upside_capture,
                'downside_capture': downside_capture,
                'periods_tracked': len(all_candles)
            }
        except Exception as e:
            logger.warning(f"Could not calculate capture ratios for {coin}: {e}")
            continue

    return coin_results
