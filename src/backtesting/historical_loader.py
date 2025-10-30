"""
Historical Data Loader

Loads historical OHLC candles from parquet files for backtesting.
Supports time-based windowing to get candles for specific lookback periods.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class HistoricalDataLoader:
    """
    Load historical OHLC data from parquet files

    Provides time-based windowing to fetch candles for backtesting.
    """

    def __init__(self, data_dir: str = "data/historical"):
        """
        Initialize historical data loader

        Args:
            data_dir: Path to directory containing parquet files
        """
        self.data_dir = Path(data_dir)
        self.cache: Dict[str, pd.DataFrame] = {}

        if not self.data_dir.exists():
            raise ValueError(f"Data directory not found: {self.data_dir}")

        logger.info(f"Initialized HistoricalDataLoader with data_dir: {self.data_dir}")

    def load_data(
        self,
        coin: str,
        interval: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load historical data for a coin

        Args:
            coin: Coin symbol (e.g., 'BTC', 'ETH')
            interval: Candle interval ('1m', '3m', '4h')
            start_date: Optional start date filter (YYYY-MM-DD)
            end_date: Optional end date filter (YYYY-MM-DD)

        Returns:
            DataFrame with columns: time, open, high, low, close, volume
        """
        cache_key = f"{coin}_{interval}"

        # Check cache first
        if cache_key in self.cache:
            df = self.cache[cache_key]
        else:
            # Load from parquet
            file_path = self.data_dir / interval / f"{coin}.parquet"

            if not file_path.exists():
                raise FileNotFoundError(
                    f"Historical data not found: {file_path}\n"
                    f"Run: python scripts/collect_historical_data.py"
                )

            df = pd.read_parquet(file_path)

            # Ensure timestamp column is datetime
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Cache for future use
            self.cache[cache_key] = df

        # Filter by date range if provided
        if start_date:
            start_dt = pd.to_datetime(start_date)
            df = df[df['timestamp'] >= start_dt]

        if end_date:
            end_dt = pd.to_datetime(end_date)
            df = df[df['timestamp'] <= end_dt]

        logger.debug(f"Loaded {len(df)} candles for {coin} ({interval})")
        return df.copy()

    def get_candles_at_time(
        self,
        coin: str,
        interval: str,
        timestamp: datetime,
        lookback_hours: int
    ) -> pd.DataFrame:
        """
        Get candles for a specific timestamp with lookback window

        Args:
            coin: Coin symbol
            interval: Candle interval ('3m', '4h')
            timestamp: Current timestamp
            lookback_hours: Hours to look back from timestamp

        Returns:
            DataFrame with candles from (timestamp - lookback) to timestamp
        """
        # Load all data for this coin/interval
        df = self.load_data(coin, interval)

        # Calculate lookback window
        start_time = timestamp - timedelta(hours=lookback_hours)

        # Filter to window
        window = df[
            (df['timestamp'] > start_time) &
            (df['timestamp'] <= timestamp)
        ].copy()

        if len(window) == 0:
            logger.warning(
                f"No candles found for {coin} {interval} at {timestamp} "
                f"(lookback {lookback_hours}h)"
            )

        return window

    def get_all_candles_at_time(
        self,
        coins: List[str],
        timestamp: datetime,
        lookback_3m: int = 3,
        lookback_4h: int = 240
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Get candles for all coins at a specific timestamp

        Args:
            coins: List of coin symbols
            timestamp: Current timestamp
            lookback_3m: Lookback hours for 3-minute candles
            lookback_4h: Lookback hours for 4-hour candles

        Returns:
            Dict mapping coin -> {'3m': df_3m, '4h': df_4h}
        """
        result = {}

        for coin in coins:
            try:
                candles_3m = self.get_candles_at_time(
                    coin, '3m', timestamp, lookback_3m
                )
                candles_4h = self.get_candles_at_time(
                    coin, '4h', timestamp, lookback_4h
                )

                result[coin] = {
                    '3m': candles_3m,
                    '4h': candles_4h
                }
            except FileNotFoundError as e:
                logger.warning(f"Skipping {coin}: {e}")
                continue

        return result

    def get_timestamps(
        self,
        coins: List[str],
        start_date: str,
        end_date: str,
        interval: str = '3m'
    ) -> List[datetime]:
        """
        Get all available timestamps in date range

        Args:
            coins: List of coins (uses first available)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Candle interval for timestamp alignment

        Returns:
            List of datetime timestamps
        """
        # Load data from first available coin
        for coin in coins:
            try:
                df = self.load_data(coin, interval, start_date, end_date)
                timestamps = df['timestamp'].tolist()
                logger.info(
                    f"Found {len(timestamps)} timestamps from "
                    f"{timestamps[0]} to {timestamps[-1]}"
                )
                return timestamps
            except FileNotFoundError:
                continue

        raise ValueError(
            f"No historical data found for any coin in {start_date} to {end_date}"
        )

    def get_date_range(self, coin: str, interval: str) -> Tuple[datetime, datetime]:
        """
        Get available date range for a coin

        Args:
            coin: Coin symbol
            interval: Candle interval

        Returns:
            Tuple of (start_date, end_date)
        """
        df = self.load_data(coin, interval)
        return df['timestamp'].min(), df['timestamp'].max()

    def clear_cache(self):
        """Clear cached data"""
        self.cache.clear()
        logger.info("Cleared data cache")
