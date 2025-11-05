"""
Data Loading Utilities for Coin Analysis

This module contains utility functions for loading coin data:
- OHLC historical data loading
- Trade extraction from checkpoints
"""

import pandas as pd
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))


def load_ohlc_data(coin: str, interval: str = '4h'):
    """Load historical OHLC data"""
    data_path = project_root / "data" / "historical" / interval / f"{coin}.parquet"

    if not data_path.exists():
        return None

    df = pd.read_parquet(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


def get_coin_trades(checkpoint, coin: str):
    """Extract trades for specific coin"""
    trade_log = checkpoint.get('trade_history', [])
    return [t for t in trade_log if t.get('symbol') == coin]
