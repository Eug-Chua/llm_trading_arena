"""
Historical Data Collection Script

Collects and stores time series data for the Alpha Arena competition period:
October 17, 2025 to November 3, 2025

Data stored in: data/historical/
Format: Parquet files (compressed, efficient for time series)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import json
from src.data.hyperliquid_client import HyperliquidClient
from src.utils.logger import setup_logger
from src.utils.config import load_config

logger = setup_logger(__name__)

# Load Alpha Arena coins from config
config = load_config("config/trading_rules.yaml")
ALPHA_ARENA_COINS = config['trading_rules']['symbols']


class HistoricalDataCollector:
    """Collects and stores historical market data"""

    def __init__(self, base_dir: str = "data/historical"):
        """
        Initialize collector

        Args:
            base_dir: Directory to store historical data
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.client = HyperliquidClient()

        # Alpha Arena competition dates
        self.start_date = datetime(2025, 10, 17)  # October 17, 2025
        self.end_date = datetime(2025, 11, 3)     # November 3, 2025

        logger.info(f"Initialized HistoricalDataCollector (storage: {self.base_dir})")

    def collect_candles(
        self,
        coins: list = None,
        intervals: list = None,
        force_refresh: bool = False,
        incremental: bool = True
    ):
        """
        Collect historical candle data

        Args:
            coins: List of coin symbols (default: Alpha Arena coins)
            intervals: List of intervals to collect (default: ['1m', '3m', '4h'])
            force_refresh: If True, re-download everything from scratch
            incremental: If True, append new candles to existing data (default)
        """
        if coins is None:
            coins = ALPHA_ARENA_COINS

        if intervals is None:
            intervals = ['1m', '3m', '4h']  # 1m for highest resolution, 3m/4h for Alpha Arena

        logger.info("="*80)
        logger.info("Starting historical data collection")
        logger.info(f"Period: {self.start_date.date()} to {self.end_date.date()}")
        logger.info(f"Coins: {', '.join(coins)}")
        logger.info(f"Intervals: {', '.join(intervals)}")

        # Convert to timestamps (milliseconds)
        start_ts = int(self.start_date.timestamp() * 1000)
        end_ts = int(self.end_date.timestamp() * 1000)

        for interval in intervals:
            logger.info(f"Collecting {interval} candles...")

            interval_dir = self.base_dir / interval
            interval_dir.mkdir(exist_ok=True)

            for coin in coins:
                file_path = interval_dir / f"{coin}.parquet"

                # Determine start time for fetch
                fetch_start_ts = start_ts
                existing_df = None

                if file_path.exists():
                    if force_refresh:
                        logger.info(f"{coin}: Force refresh - re-downloading all data")
                    elif incremental:
                        # Load existing data and fetch only new candles
                        existing_df = pd.read_parquet(file_path)
                        existing_df['timestamp'] = pd.to_datetime(existing_df['timestamp'])
                        last_timestamp = existing_df['timestamp'].max()
                        fetch_start_ts = int(last_timestamp.timestamp() * 1000)
                        logger.info(f"{coin}: Incremental update from {last_timestamp}")
                    else:
                        logger.info(f"{coin}: Already collected (skipping, use force_refresh=True or incremental=True)")
                        continue

                try:
                    logger.info(f"Fetching {coin} {interval} candles...")

                    # Hyperliquid API limit: ~5000 candles per request
                    # For 1m candles, this is ~3.5 days
                    # We need to fetch in chunks for longer periods

                    all_candles = []
                    chunk_start = fetch_start_ts

                    # Calculate chunk size based on interval
                    if interval == '1m':
                        chunk_days = 3  # 3 days = ~4320 candles
                    elif interval == '3m':
                        chunk_days = 10  # 10 days = ~4800 candles
                    else:
                        chunk_days = 30  # For 4h and longer, no chunking needed

                    chunk_size_ms = chunk_days * 24 * 60 * 60 * 1000

                    while chunk_start < end_ts:
                        chunk_end = min(chunk_start + chunk_size_ms, end_ts)

                        logger.debug(f"{coin}: Fetching chunk from {datetime.fromtimestamp(chunk_start/1000)} to {datetime.fromtimestamp(chunk_end/1000)}")

                        chunk_candles = self.client.get_candles(
                            coin=coin,
                            interval=interval,
                            start_time=chunk_start,
                            end_time=chunk_end
                        )

                        if chunk_candles:
                            all_candles.extend(chunk_candles)
                            logger.debug(f"{coin}: Got {len(chunk_candles)} candles in this chunk")

                        # Move to next chunk
                        chunk_start = chunk_end

                    candles = all_candles

                    if not candles:
                        logger.warning(f"{coin}: No data returned")
                        continue

                    logger.info(f"{coin}: Fetched {len(candles)} candles total")

                    # Convert to DataFrame
                    new_df = pd.DataFrame(candles)

                    # Add timestamp as datetime
                    new_df['timestamp'] = pd.to_datetime(new_df['t'], unit='ms')

                    # Rename columns for clarity
                    new_df = new_df.rename(columns={
                        't': 'timestamp_ms',
                        'T': 'end_timestamp_ms',
                        's': 'symbol',
                        'i': 'interval',
                        'o': 'open',
                        'c': 'close',
                        'h': 'high',
                        'l': 'low',
                        'v': 'volume',
                        'n': 'num_trades'
                    })

                    # Convert to numeric
                    for col in ['open', 'close', 'high', 'low', 'volume']:
                        new_df[col] = pd.to_numeric(new_df[col])

                    # Merge with existing data if incremental update
                    if existing_df is not None and incremental:
                        logger.info(f"{coin}: Merging {len(new_df)} new candles with {len(existing_df)} existing")

                        # Combine old and new data
                        df = pd.concat([existing_df, new_df], ignore_index=True)

                        # Remove duplicates (keep latest)
                        df = df.drop_duplicates(subset=['timestamp_ms'], keep='last')

                        # Sort by timestamp
                        df = df.sort_values('timestamp_ms').reset_index(drop=True)

                        logger.info(f"{coin}: After merge: {len(df)} total candles")
                    else:
                        df = new_df

                    # Save as Parquet (compressed, efficient for time series)
                    df.to_parquet(file_path, compression='gzip', index=False)

                    # Also save metadata
                    metadata = {
                        'coin': coin,
                        'interval': interval,
                        'start_date': self.start_date.isoformat(),
                        'end_date': self.end_date.isoformat(),
                        'num_candles': len(df),
                        'first_timestamp': df['timestamp'].min().isoformat(),
                        'last_timestamp': df['timestamp'].max().isoformat(),
                        'collected_at': datetime.now().isoformat()
                    }

                    metadata_path = interval_dir / f"{coin}_metadata.json"
                    with open(metadata_path, 'w') as f:
                        json.dump(metadata, f, indent=2)

                    logger.info(f"{coin}: Collected {len(df)} candles ({df['timestamp'].min()} to {df['timestamp'].max()})")

                except Exception as e:
                    logger.error(f"{coin}: Collection failed - {e}")

        logger.info("Historical data collection complete")
        self.show_summary()

    def show_summary(self):
        """Show summary of collected data"""
        logger.info("="*80)
        logger.info("Collection Summary")

        for interval_dir in sorted(self.base_dir.iterdir()):
            if not interval_dir.is_dir():
                continue

            interval = interval_dir.name
            parquet_files = list(interval_dir.glob("*.parquet"))

            if not parquet_files:
                logger.info(f"{interval}: No data collected")
                continue

            total_candles = 0
            total_size_mb = 0

            for file_path in sorted(parquet_files):
                coin = file_path.stem

                # Load metadata
                metadata_path = interval_dir / f"{coin}_metadata.json"
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)

                    size_mb = file_path.stat().st_size / (1024 * 1024)
                    total_size_mb += size_mb
                    total_candles += metadata['num_candles']

                    logger.info(f"{interval}/{coin}: {metadata['num_candles']} candles, {size_mb:.2f} MB")

            logger.info(f"{interval} TOTAL: {total_candles} candles, {total_size_mb:.2f} MB")

    def load_candles(self, coin: str, interval: str) -> pd.DataFrame:
        """
        Load historical candles from storage

        Args:
            coin: Coin symbol
            interval: Candle interval

        Returns:
            DataFrame with historical candles
        """
        file_path = self.base_dir / interval / f"{coin}.parquet"

        if not file_path.exists():
            raise FileNotFoundError(f"No data for {coin} {interval}. Run collect_candles() first.")

        logger.debug(f"Loading {coin} {interval} from {file_path}")
        df = pd.read_parquet(file_path)
        logger.info(f"Loaded {len(df)} candles for {coin} {interval}")
        return df

    def export_to_csv(self, coin: str, interval: str, output_path: str = None):
        """
        Export data to CSV (for external tools/analysis)

        Args:
            coin: Coin symbol
            interval: Candle interval
            output_path: Output CSV path (default: data/exports/)
        """
        df = self.load_candles(coin, interval)

        if output_path is None:
            export_dir = self.base_dir.parent / "exports"
            export_dir.mkdir(exist_ok=True)
            output_path = export_dir / f"{coin}_{interval}.csv"

        df.to_csv(output_path, index=False)
        logger.info(f"Exported {coin} {interval} to {output_path}")
        return output_path


def main():
    """Main collection script"""
    logger.info("Starting historical data collection script")

    collector = HistoricalDataCollector()

    # Collect data for Alpha Arena competition period
    collector.collect_candles(
        coins=ALPHA_ARENA_COINS,  # BTC, ETH, SOL, BNB, XRP, DOGE
        intervals=['1m', '3m', '4h'],  # 1m (last 3-4 days), 3m (Oct 18+), 4h (365+ days)
        force_refresh=False,       # Set True to re-download all data
        incremental=True           # Append new candles to existing data
    )

    logger.info("Script complete")


if __name__ == "__main__":
    main()
